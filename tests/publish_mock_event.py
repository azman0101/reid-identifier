import paho.mqtt.client as mqtt
import json
import time
import argparse

# Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

def publish_event(event_id, camera):
    try:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)

        payload = {
            "type": "update",
            "after": {
                "id": event_id,
                "label": "person",
                "camera": camera,
                "has_snapshot": True,
                "sub_label": None
            }
        }

        print(f"Publishing mock event: {event_id} on camera {camera}...")
        client.publish("frigate/events", json.dumps(payload))
        client.disconnect()
        print("✅ Event published successfully.")
    except Exception as e:
        print(f"❌ MQTT Error: {e}")
        print(f"Make sure Mosquitto is running on {MQTT_BROKER}:{MQTT_PORT}.")
        print("You can start it with: docker-compose -f docker-compose.yml up -d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a mock Frigate MQTT event.")
    parser.add_argument("--event-id", type=str, default=f"test_event_{int(time.time())}",
                        help="The unique ID of the mock event.")
    parser.add_argument("--camera", type=str, default="front_door",
                        help="The camera name for the mock event.")

    args = parser.parse_args()
    publish_event(args.event_id, args.camera)
