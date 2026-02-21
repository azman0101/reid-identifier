# Frigate Person Re-Identification (ReID) with OpenVINO

## Overview

This project implements a **Person Re-Identification (ReID)** system designed to work alongside [Frigate NVR](https://frigate.video/). It enhances your security camera setup by identifying specific individuals across different cameras and events.

Unlike simple object detection (which tells you "this is a person"), this system tells you **"this is John"** or **"this is Alice"**.

It uses **OpenVINO™** for high-performance inference on Intel hardware (CPUs and iGPUs) and provides a modern "Active Learning" web interface to easily label unknown people.

## Features

- **🚀 High Performance**: Powered by OpenVINO for fast inference on Intel CPUs and iGPUs (with automatic fallback).
- **🧠 Active Learning**: A user-friendly web interface allows you to label unknown silhouettes. The system learns instantly without retraining.
- **📡 Seamless Integration**: Listens to Frigate MQTT events in real-time and updates Frigate event sub-labels automatically.
- **🐳 Modern Stack**: Built with **FastAPI**, **Docker** (optimized with `uv`), and **Pydantic** for robust configuration.
- **📦 Self-Contained**: Automatically downloads necessary Open Model Zoo models on startup.

## Architecture

1.  **Detection**: Frigate detects a person and sends an event via MQTT.
2.  **Processing**: This service receives the event, fetches the snapshot from Frigate, and generates a unique "embedding" (fingerprint) of the person's appearance using the `person-reidentification-retail-0288` model.
3.  **Matching**: The embedding is compared against a gallery of known individuals.
4.  **Action**:
    - **Match Found**: The system updates the Frigate event with the person's name (sub-label).
    - **No Match**: The snapshot is saved to the "Unknown" gallery.
5.  **Labeling**: You visit the web dashboard to label unknown snapshots. These are moved to the gallery, and the system immediately recognizes this person in future events.

## Project Structure

```
.
├── Dockerfile              # Multi-stage Docker build using uv
├── docker-compose.yml      # Deployment configuration
├── pyproject.toml          # Python dependencies and project metadata
├── README.md               # Project documentation
├── models/                 # Data directory (mounted volume)
│   ├── gallery/            # Labeled images (the "knowledge base")
│   └── unknown/            # Unlabeled images awaiting review
├── reid_app/               # Application source code
│   ├── main.py             # FastAPI web server and entry point
│   ├── reid_engine.py      # Core AI logic (OpenVINO, embeddings)
│   ├── mqtt_frigate.py     # MQTT listener and Frigate API client
│   ├── model_manager.py    # Automatic model downloader
│   ├── config.py           # Configuration management
│   └── templates/          # Web interface templates
└── tests/                  # Tests and mock scripts
```

## Installation & Usage

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed.
- A running instance of [Frigate](https://frigate.video/) with MQTT enabled.
- (Optional) Intel GPU for hardware acceleration.

### Quick Start

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/azman0101/reid-identifier.git
    cd reid-identifier
    ```

2.  **Configure Environment**:
    Edit `docker-compose.yml` to match your setup:
    - `MQTT_BROKER`: IP address of your MQTT broker.
    - `FRIGATE_URL`: URL of your Frigate instance.
    - `DEVICE_NAME`: `GPU` for Intel iGPU or `CPU` for processor only.

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up -d --build
    ```

4.  **Access the Dashboard**:
    Open `http://localhost:8000` in your browser.

### Configuration

The application is configured via environment variables (defined in `docker-compose.yml`):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MQTT_BROKER` | `localhost` | MQTT Broker IP address |
| `MQTT_PORT` | `1883` | MQTT Broker Port |
| `FRIGATE_URL` | `http://localhost:5000` | Base URL for Frigate API |
| `DEVICE_NAME` | `GPU` | OpenVINO device (`CPU`, `GPU`, `AUTO`) |

## Development

To run tests or develop locally:

1.  **Install dependencies** (using [uv](https://github.com/astral-sh/uv)):
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

2.  **Run Tests**:
    ```bash
    python -m unittest discover tests
    ```

3.  **Run Mock Environment**:
    To simulate Frigate events without a real camera setup:

    a. Start the local MQTT broker:
    ```bash
    docker-compose -f docker-compose.yml up -d
    ```

    b. Activate the virtual environment (if not already done):
    ```bash
    source .venv/bin/activate
    ```

    c. Start the mock Frigate HTTP server:
    ```bash
    python tests/mock_frigate_env.py
    ```

    d. Publish a test event to trigger ReID (in a new terminal):
    ```bash
    source .venv/bin/activate
    python tests/publish_mock_event.py --event-id test_123
    ```

## License

MIT License
