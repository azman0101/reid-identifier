"""
Replay Historical Frigate Events via MQTT

This script fetches recent 'person' detection events from a real Frigate API
and republishes them to your local MQTT broker as if they just happened.
This is extremely useful for testing the ReID-Identifier application pipeline
without having to walk in front of a camera.

Usage Examples:
---------------
1. Run with default environment variables/arguments:
   python3 tests/replay_real_events.py

2. Replay the last 10 events from a specific camera:
   python3 tests/replay_real_events.py --cameras cam1terrasse --limit 10

3. Specify custom Frigate and MQTT broker IPs:
   python3 tests/replay_real_events.py --frigate-url http://192.168.1.50:5000 --mqtt-broker 192.168.1.50

4. Enable extremely verbose API response parsing to debug raw data:
   python3 tests/replay_real_events.py --debug

5. Replay specific events by ID (overrides cameras and limit filter):
   python3 tests/replay_real_events.py --events 1771847434.876006-yvqoaj 1771847321.606909-j6y6v2
"""

import requests
import json
import paho.mqtt.client as mqtt
import time
import argparse
import os


def replay_real_events(
    frigate_url,
    mqtt_broker,
    mqtt_port,
    cameras,
    limit,
    debug=False,
    specific_events=None,
):
    events = []

    # 1. Fetch real historical data from Frigate API
    if specific_events:
        print(f"Fetching specific events: {', '.join(specific_events)}")
        for evt_id in specific_events:
            url = f"{frigate_url}/api/events/{evt_id}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    events.append(response.json())
                else:
                    print(
                        f"Warning: Event {evt_id} not found (HTTP {response.status_code})"
                    )
            except requests.RequestException as e:
                print(f"Error connecting to Frigate API for event {evt_id}: {e}")
    else:
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
        default=os.environ.get("FRIGATE_URL", "http://192.168.1.22:5000"),
        help="Frigate API base URL",
    )
    parser.add_argument(
        "--mqtt-broker",
        default=os.environ.get("MQTT_BROKER", "192.168.1.22"),
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
        "--events",
        nargs="*",
        help="List of specific event IDs to replay (overrides limit and cameras)",
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
        specific_events=args.events,
    )
