import http.server
import socketserver
import threading
import time
import json
import os
import cv2
import numpy as np

# Configuration
FRIGATE_PORT = 5000
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Create a dummy image
img = np.zeros((300, 300, 3), dtype=np.uint8)
# Draw a green square
cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)
cv2.imwrite("dummy_snapshot.jpg", img)

class FrigateHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if "snapshot.jpg" in self.path:
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            try:
                with open("dummy_snapshot.jpg", "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                print(f"Error serving image: {e}")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if "sub_label" in self.path:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            print(f"Received sub_label update: {post_data.decode('utf-8')}")
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress logging for cleanliness
        pass

def run_frigate_mock():
    # Allow reuse address
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", FRIGATE_PORT), FrigateHandler) as httpd:
            print(f"Mock Frigate serving at port {FRIGATE_PORT}")
            httpd.serve_forever()
    except OSError as e:
        print(f"Failed to bind port {FRIGATE_PORT}: {e}")

if __name__ == "__main__":
    print("Starting Mock Frigate HTTP Server...")
    print("Run `python tests/publish_mock_event.py` in another terminal to publish events.")
    run_frigate_mock()

