import os
import json
import requests
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import logging
import threading
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MQTTWorker:
    def __init__(self, reid_core):
        self.reid_core = reid_core
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

    def process_event(self, event_id):
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
                else:
                    # Unknown silhouette, save for backoffice
                    unknown_path = os.path.join(settings.unknown_dir, f"{event_id}.jpg")
                    cv2.imwrite(unknown_path, image_frame)
                    logger.info(f"❓ Unknown saved: {event_id}.jpg")
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
                # Run processing in a separate thread to avoid blocking MQTT loop
                threading.Thread(target=self.process_event, args=(event_id,), daemon=True).start()

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

def start_mqtt(reid_core):
    worker = MQTTWorker(reid_core)
    worker.start()
    return worker
