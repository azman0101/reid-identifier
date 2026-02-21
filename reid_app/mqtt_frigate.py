import os
import json
import requests
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
                try:
                    h, w, _ = image_frame.shape

                    x1, y1, x2, y2 = 0, 0, w, h
                    cropped = False

                    if data_box and len(data_box) == 4:
                        # Frigate 0.14 events API data.box is [x, y, width, height] normalized (0 to 1)
                        nx, ny, nw, nh = data_box
                        x1 = int(nx * w)
                        y1 = int(ny * h)
                        x2 = int((nx + nw) * w)
                        y2 = int((ny + nh) * h)
                        cropped = True
                    elif box and len(box) == 4:
                        # Native Frigate MQTT 'box' is [xmin, ymin, xmax, ymax] absolute pixels
                        x1, y1, x2, y2 = (
                            int(box[0]),
                            int(box[1]),
                            int(box[2]),
                            int(box[3]),
                        )
                        cropped = True

                    if cropped:
                        # Ensure reasonable bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Add a 10% margin to avoid chopping heads/feet
                        margin_w = int((x2 - x1) * 0.10)
                        margin_h = int((y2 - y1) * 0.10)

                        cx1 = max(0, x1 - margin_w)
                        cy1 = max(0, y1 - margin_h)
                        cx2 = min(w, x2 + margin_w)
                        cy2 = min(h, y2 + margin_h)

                        # Only crop if it's actually smaller than the full frame
                        if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1 < w or cy2 - cy1 < h):
                            logger.info(
                                f"[{event_id}] Manually cropping image from {w}x{h} to bounding box [{cx1}:{cx2}, {cy1}:{cy2}]"
                            )
                            image_frame = image_frame[cy1:cy2, cx1:cx2]
                        else:
                            logger.info(
                                f"[{event_id}] Skipping manual crop (box covers entire frame or invalid)."
                            )

                except Exception as e:
                    logger.warning(
                        f"[{event_id}] Failed to crop image manually: {e}. Proceeding with original image."
                    )

                embedding = self.reid_core.get_embedding(image_frame)
                match, score = self.reid_core.find_match(embedding)
                logger.info(
                    f"[{event_id}] ReID inference complete. Target identified as: {match if match else 'UNKNOWN'} (Score: {score:.3f})"
                )

                # Determine label
                label = match if match else "unknown"

                # Save snapshot path (relative)
                # If matched, we don't save to unknown dir, but we should track where it went?
                # Actually, if matched, we don't save the image file in our system usually (unless we want to add it to gallery).
                # But for history, we might want to keep a reference.
                # The current logic only saves if unknown.

                snapshot_filename = f"{event_id}.jpg"

                # Check if we should update Frigate
                # We update if: 1) We have a match AND 2) The event doesn't currently have a sub_label
                # OR 3) The event DOES have a sub_label, but our score is extremely high (>0.85) and contradicts it
                HIGH_CONFIDENCE_THRESHOLD = 0.85

                should_update_frigate = False
                if match:
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

                        # Add a description noting that this script added the label
                        desc_url = (
                            f"{self.frigate_url}/api/events/{event_id}/description"
                        )
                        desc_payload = {
                            "description": f"Auto-labeled as '{match}' by ReID-Identifier (OpenVINO)"
                        }
                        desc_resp = requests.post(
                            desc_url, json=desc_payload, headers=headers
                        )

                        if desc_resp.status_code == 200:
                            logger.info(
                                f"[{event_id}] ✅ Successfully updated Frigate event description."
                            )
                        else:
                            logger.error(
                                f"[{event_id}] Failed to update description in Frigate HTTP {desc_resp.status_code}: {desc_resp.text}"
                            )

                    else:
                        logger.error(
                            f"[{event_id}] Failed to update sub_label in Frigate HTTP {resp.status_code}: {resp.text}"
                        )

                    # We usually don't save the image locally if matched, unless we want to grow the gallery automatically.
                    # For now, let's say we don't save the file locally to save space,
                    # OR we could save it to a 'history' folder?
                    # The prompt asked to track camera and date.
                    # If we don't save the file, we can't show it in history if Frigate deletes it.
                    # Let's stick to current logic: save only if unknown.
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

                # Compute fuzzy hash of the exact image going into the algorithm
                # (helps with extremely robust db deduplication later)
                image_hash = compute_dhash(image_frame)

                # Add to Database
                self.db_repo.add_event(
                    event_id=event_id,
                    camera=camera,
                    timestamp=datetime.now(),
                    label=label,
                    snapshot_path=snapshot_path_db,
                    image_hash=image_hash,
                )

                if match:
                    # If matched automatically, we might want to record a system history entry?
                    # The interface has 'add_event' which sets current_label.
                    # We don't need to call update_label unless it changed from something else.
                    # Initial insert is enough.
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
