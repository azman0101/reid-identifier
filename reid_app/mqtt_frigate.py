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

class MQTTWorker:
    def __init__(self, reid_core, db_repo: ReIDRepository):
        self.reid_core = reid_core
        self.db_repo = db_repo
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.frigate_url = settings.frigate_url.rstrip('/')

        # Callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT Broker at {settings.mqtt_broker}:{settings.mqtt_port}")
            client.subscribe("frigate/events")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT Broker with code {rc}. Reconnecting...")

    def process_event(self, event_id, camera):
        """Processes a single event in a separate thread."""
        try:
            snapshot_url = f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg?crop=1"
            response = requests.get(snapshot_url, timeout=10)

            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype="uint8")
                image_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image_frame is None:
                    logger.warning(f"Failed to decode image for event {event_id}")
                    return

                embedding = self.reid_core.get_embedding(image_frame)
                match = self.reid_core.find_match(embedding)

                # Determine label
                label = match if match else "unknown"

                # Save snapshot path (relative)
                # If matched, we don't save to unknown dir, but we should track where it went?
                # Actually, if matched, we don't save the image file in our system usually (unless we want to add it to gallery).
                # But for history, we might want to keep a reference.
                # The current logic only saves if unknown.

                snapshot_filename = f"{event_id}.jpg"

                if match:
                    # Known silhouette, update Frigate
                    # POST to sub_label endpoint
                    sub_label_url = f"{self.frigate_url}/api/events/{event_id}/sub_label"
                    payload = {"subLabel": match, "subLabelScore": 1.0}
                    resp = requests.post(sub_label_url, json=payload)

                    if resp.status_code == 200:
                        logger.info(f"✅ Recognized: {match} (Event: {event_id})")
                    else:
                        logger.error(f"Failed to update sub_label for {event_id}: {resp.text}")

                    # We usually don't save the image locally if matched, unless we want to grow the gallery automatically.
                    # For now, let's say we don't save the file locally to save space,
                    # OR we could save it to a 'history' folder?
                    # The prompt asked to track camera and date.
                    # If we don't save the file, we can't show it in history if Frigate deletes it.
                    # Let's stick to current logic: save only if unknown.
                    snapshot_path_db = ""
                else:
                    # Unknown silhouette, save for backoffice
                    unknown_path = os.path.join(settings.unknown_dir, snapshot_filename)
                    cv2.imwrite(unknown_path, image_frame)
                    snapshot_path_db = snapshot_filename
                    logger.info(f"❓ Unknown saved: {event_id}.jpg")

                # Add to Database
                self.db_repo.add_event(
                    event_id=event_id,
                    camera=camera,
                    timestamp=datetime.now(),
                    label=label,
                    snapshot_path=snapshot_path_db
                )

                if match:
                    # If matched automatically, we might want to record a system history entry?
                    # The interface has 'add_event' which sets current_label.
                    # We don't need to call update_label unless it changed from something else.
                    # Initial insert is enough.
                    pass

            else:
                logger.warning(f"Failed to fetch snapshot for {event_id}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            after = payload.get("after", {})

            # Filter logic: only process when snapshot is available and no sub_label exists yet
            if (after and
                after.get("label") == "person" and
                after.get("has_snapshot") and
                not after.get("sub_label")):

                event_id = after.get("id")
                camera = after.get("camera")
                # Run processing in a separate thread to avoid blocking MQTT loop
                threading.Thread(target=self.process_event, args=(event_id, camera), daemon=True).start()

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
