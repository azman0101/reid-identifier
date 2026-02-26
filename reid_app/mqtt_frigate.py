import os
import json
import requests
from .utils import crop_image_from_box, update_frigate_description
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import logging
import threading
from datetime import datetime
from .config import settings
from .database.interface import ReIDRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_dhash(image, hash_size=8):
    """
    Computes a robust fuzzy difference hash (dHash) using purely OpenCV.
    Resizes to (hash_size + 1, hash_size), converts to grayscale, compares adjacent pixels.
    Produces a perfect identical hex string for visually identical/very similar inputs.
    """
    if image is None:
        return None
    try:
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = gray[:, 1:] > gray[:, :-1]
        return "{:0{width}x}".format(
            int("".join(["1" if x else "0" for x in diff.flatten()]), 2),
            width=(hash_size * hash_size) // 4,
        )
    except Exception as e:
        logger.warning(f"Failed to compute dhash: {e}")
        return None


class MQTTWorker:
    def __init__(self, reid_core, db_repo: ReIDRepository):
        self.reid_core = reid_core
        self.db_repo = db_repo
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.frigate_url = settings.frigate_url.rstrip("/")

        # Callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(
                f"Connected to MQTT Broker at {settings.mqtt_broker}:{settings.mqtt_port}"
            )
            client.subscribe("frigate/events")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            logger.warning(
                f"Unexpected disconnection from MQTT Broker with code {rc}. Reconnecting..."
            )

    def process_event(
        self, event_id, camera, existing_sub_label=None, box=None, data_box=None
    ):
        """Processes a single event in a separate thread."""
        try:
            snapshot_url = (
                f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg?crop=1"
            )
            logger.info(f"[{event_id}] Fetching cropped snapshot from: {snapshot_url}")
            response = requests.get(snapshot_url, timeout=10)

            if response.status_code == 200:
                logger.info(
                    f"[{event_id}] Successfully downloaded snapshot. Decoding image and running ReID..."
                )
                image_array = np.asarray(bytearray(response.content), dtype="uint8")
                image_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image_frame is None:
                    logger.warning(f"Failed to decode image for event {event_id}")
                    return

                # Manually crop if the image is full frame (Frigate API ?crop=1 often fails for historical events)
                image_frame = crop_image_from_box(image_frame, box, data_box)

                embedding = self.reid_core.get_embedding(image_frame)
                vector_bytes = embedding.tobytes()
                match, score = self.reid_core.find_match(embedding)
                logger.info(
                    f"[{event_id}] ReID inference complete. Target identified as: {match if match else 'UNKNOWN'} (Score: {score:.3f})"
                )

                # Determine label
                label = match if match else "unknown"

                snapshot_filename = f"{event_id}.jpg"

                # Check if we should update Frigate
                # We update if: 1) We have a match AND 2) The event doesn't currently have a sub_label
                # OR 3) The event DOES have a sub_label, but our score is extremely high (>0.85) and contradicts it
                HIGH_CONFIDENCE_THRESHOLD = 0.85

                should_update_frigate = False
                if match:
                    # Self-Learning Logic: if confidence is very high, learn this new appearance
                    if score >= settings.self_learning_threshold:
                        with self.reid_core.lock:
                            current_count = self.reid_core.gallery_labels.count(match)

                        if current_count < settings.max_gallery_per_identity:
                            logger.info(f"[{event_id}] Self-learning triggered for '{match}' (Score: {score:.3f} >= {settings.self_learning_threshold})")
                            self.reid_core.update_gallery(match, embedding, save_to_disk=True, frame=image_frame)

                    if not existing_sub_label:
                        should_update_frigate = True
                    elif (
                        existing_sub_label != match
                        and score >= HIGH_CONFIDENCE_THRESHOLD
                    ):
                        logger.info(
                            f"[{event_id}] OVERRIDE: Existing sub_label was '{existing_sub_label}' but we are highly confident ({score:.3f}) this is '{match}'."
                        )
                        should_update_frigate = True

                update_success = False
                if should_update_frigate:
                    # Known silhouette with sufficient confidence, update Frigate
                    # POST to sub_label endpoint
                    sub_label_url = (
                        f"{self.frigate_url}/api/events/{event_id}/sub_label"
                    )
                    logger.info(
                        f"[{event_id}] Informing Frigate API: Setting subLabel to '{match}' on {camera}"
                    )

                    payload = {
                        "subLabel": match,
                        "subLabelScore": 1.0,
                        "camera": camera,
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    resp = requests.post(sub_label_url, json=payload, headers=headers)

                    if resp.status_code == 200:
                        logger.info(
                            f"[{event_id}] ✅ Successfully updated Frigate with recognized identity: {match}"
                        )
                        update_success = True
                    else:
                        logger.error(
                            f"[{event_id}] Failed to update sub_label in Frigate HTTP {resp.status_code}: {resp.text}"
                        )

                    # We usually don't save the image locally if matched, unless we want to grow the gallery automatically.
                    snapshot_path_db = ""
                elif not match and not existing_sub_label:
                    # Unknown silhouette and not already labeled, save for backoffice
                    unknown_path = os.path.join(settings.unknown_dir, snapshot_filename)
                    cv2.imwrite(unknown_path, image_frame)
                    snapshot_path_db = snapshot_filename
                    logger.info(f"❓ Unknown saved: {event_id}.jpg")
                else:
                    # It was already sub_labeled manually, or we had low confidence. Don't save it as unknown.
                    snapshot_path_db = ""

                # --- UPDATE DESCRIPTION WITH STATUS ---
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_msg = ""
                display_score = 0.0

                if match:
                    display_score = score
                    if update_success:
                         status_msg = f"Auto-labeled '{match}'"
                    elif should_update_frigate and not update_success:
                         status_msg = f"Failed to auto-label '{match}' (API Error)"
                    elif existing_sub_label == match:
                         status_msg = f"Verified '{match}'"
                    else:
                         status_msg = f"Possible match '{match}' (conflicts with '{existing_sub_label}')"
                else:
                    # No match found above threshold
                    closest_label, closest_score = self.reid_core.find_closest_match(embedding)
                    if closest_label:
                        status_msg = f"Possible match '{closest_label}' - Low confidence"
                        display_score = closest_score
                    else:
                        status_msg = "No match found"
                        display_score = 0.0

                full_status_line = f"[ReID]: {current_time} - {status_msg} ({display_score * 100:.1f}%) [System]"
                update_frigate_description(event_id, full_status_line)
                # --------------------------------------

                # Compute fuzzy hash of the exact image going into the algorithm
                # (helps with extremely robust db deduplication later)
                image_hash = compute_dhash(image_frame)

                # Add to Database
                success = self.db_repo.add_event(
                    event_id=event_id,
                    camera=camera,
                    timestamp=datetime.now(),
                    label=label,
                    snapshot_path=snapshot_path_db,
                    image_hash=image_hash,
                    vector=vector_bytes,
                )

                # If adding to DB failed (likely duplicate), ensure we don't leave an orphaned file
                if not success and snapshot_path_db:
                    orphaned_path = os.path.join(settings.unknown_dir, snapshot_path_db)
                    if os.path.exists(orphaned_path):
                        try:
                            os.remove(orphaned_path)
                            logger.info(
                                f"🗑️ Removed duplicate/orphaned file: {orphaned_path}"
                            )
                        except OSError as e:
                            logger.error(
                                f"Failed to remove orphaned file {orphaned_path}: {e}"
                            )

                if match:
                    pass

            else:
                logger.warning(
                    f"Failed to fetch snapshot for {event_id}: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            after = payload.get("after", {})
            event_id = after.get("id", "unknown")
            camera = after.get("camera", "unknown")
            label = after.get("label", "unknown")
            has_snapshot = after.get("has_snapshot", False)
            sub_label = after.get("sub_label", None)

            # Additional bounding box info for manual cropping
            box = after.get("box", [])
            data_box = after.get("data", {}).get("box", [])

            # Print every single MQTT message about an event going through (useful for debugging)
            if after:
                logger.info(
                    f"[MQTT msg] incoming update for event {event_id} | cam: {camera} | label: {label} | has_snapshot: {has_snapshot} | sub_label: {sub_label}"
                )

            # Filter logic: only process when snapshot is available
            # Note: We now allow events with existing sub_label to pass through for high-confidence overrides
            if after and label == "person" and has_snapshot:
                # Prevent infinite loops: If the sub_label was already assigned by us (indicated by description)
                # or if the timestamp just updated but the sub_label is identical, avoid unnecessary re-inference
                # Unfortunately Frigate MQTT doesn't send 'description' in the event payload.
                # As a workaround to avoid infinite loops, we can track recent event_ids we've processed in memory,
                # but an easier way is to just let it re-evaluate and if the label matches our inference, it ignores it.

                logger.info(
                    f"[{event_id}] Match! Event meets all criteria for ReID. Spawning inference thread."
                )
                # Run processing in a separate thread to avoid blocking MQTT loop
                threading.Thread(
                    target=self.process_event,
                    args=(event_id, camera, sub_label, box, data_box),
                    daemon=True,
                ).start()

        except json.JSONDecodeError:
            logger.error("Failed to decode MQTT message payload")
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")

    def start(self):
        try:
            logger.info(f"Connecting to MQTT broker {settings.mqtt_broker}...")
            self.client.connect(settings.mqtt_broker, settings.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")


def start_mqtt(reid_core, db_repo):
    worker = MQTTWorker(reid_core, db_repo)
    worker.start()
    return worker
