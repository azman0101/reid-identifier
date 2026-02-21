import requests
import json
import paho.mqtt.client as mqtt
import time
import argparse
import os


def replay_real_events(
    frigate_url, mqtt_broker, mqtt_port, cameras, limit, debug=False
):
    # 1. Fetch real historical data from Frigate API
    cams_param = ",".join(cameras) if cameras else ""
    url = f"{frigate_url}/api/events?labels=person&limit={limit}&has_snapshot=1"
    if cams_param:
        url += f"&cameras={cams_param}"

    print(f"Fetching historical events from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json()
    except requests.RequestException as e:
        print(f"Error connecting to Frigate API: {e}")
        return

    if debug:
        print("\n--- DEBUG: Raw API Response ---")
        print(json.dumps(events, indent=2))
        print("-------------------------------\n")

    if not events:
        print("No historical person events found for these cameras.")
        return

    print(f"Found {len(events)} events to replay. Connecting to MQTT...")

    # 2. Connect to your MQTT Broker
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(mqtt_broker, mqtt_port)
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        return

    for event in events:
        event_id = event["id"]
        camera_name = event.get("camera", "unknown")
        print(f"\nRepublishing real event: {event_id} from {camera_name}")

        # 3. Construct the MQTT payload
        # Your ReID script listens for 'new' or 'update' types
        payload = {"type": "new", "before": event, "after": event}

        # 4. Publish to the topic your ReID script is watching
        client.publish("frigate/events", json.dumps(payload))
        print(f"  [Sent] ReID script should now be processing snapshot for {event_id}")

        # Gap to allow OpenVINO to finish each inference before the next event
        time.sleep(2)

    client.disconnect()
    print("\nFinished replaying events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay historical Frigate events via MQTT for testing."
    )
    parser.add_argument(
        "--frigate-url",
        default=os.environ.get("FRIGATE_URL", "http://192.168.1.XX:5000"),
        help="Frigate API base URL",
    )
    parser.add_argument(
        "--mqtt-broker",
        default=os.environ.get("MQTT_BROKER", "192.168.1.XX"),
        help="MQTT Broker IP/Host",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.environ.get("MQTT_PORT", 1883)),
        help="MQTT Broker Port",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=["camera", "camera1terrasse"],
        help="List of cameras to filter by (space separated)",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of recent events to replay"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw Frigate API response for debugging",
    )

    args = parser.parse_args()

    replay_real_events(
        frigate_url=args.frigate_url.rstrip("/"),
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        cameras=args.cameras,
        limit=args.limit,
        debug=args.debug,
    )
