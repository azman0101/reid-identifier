import os
import shutil
import json
import logging
import sys
import threading
from .utils import crop_image_from_box
import cv2
import numpy as np
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from .config import settings
from .reid_engine import ReIDCore
from .model_manager import ensure_models_exist
from .mqtt_frigate import start_mqtt
from .database.sqlite_repo import SQLiteRepository

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global instances
reid_core = None
mqtt_worker = None
db_repo = None


def run_backfill(core, repo):
    """Backfill vectors for events that are missing them."""
    logger.info("Starting vector backfill process in background...")
    try:
        events = repo.get_all_events()
        count = 0
        total = 0

        # Pre-scan gallery to make lookups faster
        gallery_map = {} # event_id -> filename
        if os.path.exists(settings.gallery_dir):
            for f in os.listdir(settings.gallery_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                     # format: Label_ID.jpg
                     parts = f.rsplit('.', 1)[0].split('_')
                     if len(parts) >= 2:
                         evt_id = parts[-1]
                         gallery_map[evt_id] = f

        for event in events:
            # Check if vector is missing (None or empty bytes)
            if event.get("vector"):
                continue

            total += 1
            event_id = event["id"]
            img_path = None

            # Check unknown dir
            unknown_path = os.path.join(settings.unknown_dir, f"{event_id}.jpg")
            if os.path.exists(unknown_path):
                img_path = unknown_path

            # Check gallery
            elif event_id in gallery_map:
                img_path = os.path.join(settings.gallery_dir, gallery_map[event_id])

            if img_path:
                img = cv2.imread(img_path)
                if img is not None:
                    embedding = core.get_embedding(img)
                    if embedding is not None:
                        repo.update_vector(event_id, embedding.tobytes())
                        count += 1

        if count > 0:
            logger.info(f"Backfill complete: Updated {count} vectors out of {total} missing.")
        else:
            logger.info("Backfill complete: No vectors needed update.")

    except Exception as e:
        logger.error(f"Backfill process failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ReID App...")

    # Initialize DB
    global db_repo
    db_repo = SQLiteRepository()
    db_repo.init_db()

    # Ensure models exist
    try:
        ensure_models_exist()
    except Exception as e:
        logger.critical(f"Failed to ensure models exist: {e}")
        sys.exit(1)

    # Initialize Core
    global reid_core
    try:
        reid_core = ReIDCore()
    except Exception as e:
        logger.critical(f"Failed to initialize ReID Core: {e}")
        sys.exit(1)

    # Start MQTT
    global mqtt_worker
    if reid_core:
        mqtt_worker = start_mqtt(reid_core, db_repo)

    # Start Backfill
    if reid_core and db_repo:
        threading.Thread(target=run_backfill, args=(reid_core, db_repo), daemon=True).start()

    yield

    # Shutdown
    if mqtt_worker:
        mqtt_worker.client.loop_stop()
        mqtt_worker.client.disconnect()
    logger.info("Shutting down ReID App...")


app = FastAPI(lifespan=lifespan)

# Load Version Info
version_info = {'build_date': 'Unknown', 'git_sha': 'Unknown'}
try:
    if os.path.exists('reid_app/version.json'):
        with open('reid_app/version.json', 'r') as f:
            version_info = json.load(f)
    elif os.path.exists('/app/reid_app/version.json'):
        with open('/app/reid_app/version.json', 'r') as f:
            version_info = json.load(f)
except Exception as e:
    logging.warning(f'Could not load version info: {e}')
templates = Jinja2Templates(directory="reid_app/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="reid_app/static"), name="static")
app.mount("/gallery_imgs", StaticFiles(directory=settings.gallery_dir), name="gallery")
app.mount("/unknown_imgs", StaticFiles(directory=settings.unknown_dir), name="unknown")


@app.get("/snapshot/{event_id}")
async def fetch_snapshot(event_id: str):
    """Fetches a snapshot from local disk or Frigate API."""
    try:
        filename = f"{event_id}.jpg"
        unknown_path = os.path.join(settings.unknown_dir, filename)

        # 1. Check local unknown dir
        if os.path.exists(unknown_path):
            return FileResponse(unknown_path)

        # 2. Check local gallery dir
        # We need to find matching file: Label_EventID.jpg
        # Only scan if not in unknown
        if os.path.exists(settings.gallery_dir):
            for f in os.listdir(settings.gallery_dir):
                if f.endswith(f"_{event_id}.jpg") or f == filename:
                     return FileResponse(os.path.join(settings.gallery_dir, f))

        # 3. Fetch from Frigate
        # First get event details for bounding box
        event_url = f"{settings.frigate_url}/api/events/{event_id}"
        snapshot_url = f"{settings.frigate_url}/api/events/{event_id}/snapshot.jpg?crop=1"

        def _download_and_crop():
            try:
                # Get Event Data for Box
                box = None
                data_box = None
                try:
                    ev_resp = requests.get(event_url, timeout=5)
                    if ev_resp.status_code == 200:
                        data = ev_resp.json()
                        data_box = data.get("data", {}).get("box")
                        if not data_box:
                            box = data.get("box")
                except Exception as e:
                    logger.warning(f"Could not fetch event details for {event_id}: {e}")

                # Get Image
                resp = requests.get(snapshot_url, timeout=10)
                if resp.status_code != 200:
                    return False

                # Decode Image
                image_array = np.asarray(bytearray(resp.content), dtype="uint8")
                image_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image_frame is None:
                    return False

                # Crop Logic (using utility)
                image_frame = crop_image_from_box(image_frame, box, data_box)
                # Save to unknown dir
                cv2.imwrite(unknown_path, image_frame)
                return True
            except Exception as e:
                logger.error(f"Error processing snapshot download: {e}")
                return False

        success = await run_in_threadpool(_download_and_crop)
        if success:
            return FileResponse(unknown_path)
        else:
            return JSONResponse({"status": "error", "message": "Snapshot not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error fetching snapshot {event_id}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        # Check if directories exist
        if not os.path.exists(settings.unknown_dir):
            os.makedirs(settings.unknown_dir, exist_ok=True)
        if not os.path.exists(settings.gallery_dir):
            os.makedirs(settings.gallery_dir, exist_ok=True)

        # Retrieve events from DB for metadata
        unknown_events = db_repo.get_events_by_label("unknown")

        # Build list of unknown events
        unknowns_data = []
        for event in unknown_events:
            filename = event["snapshot_path"]
            if not filename:
                filename = f"{event['id']}.jpg"

            full_path = os.path.join(settings.unknown_dir, filename)
            is_local = os.path.exists(full_path)

            if is_local:
                 img_src = f"/unknown_imgs/{filename}"
            else:
                 img_src = f"/snapshot/{event['id']}"

            unknowns_data.append(
                {
                    "filename": filename,
                    "event_id": event["id"],
                    "camera": event["camera"],
                    "timestamp": event["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "is_local": is_local,
                    "img_src": img_src
                }
            )
        # Also catch files on disk that might not be in DB (orphaned)
        disk_files = set(
            f
            for f in os.listdir(settings.unknown_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        db_files = set(u["filename"] for u in unknowns_data)
        for f in disk_files:
            if f not in db_files:
                unknowns_data.append(
                    {
                        "filename": f,
                        "event_id": f.split(".")[0],
                        "camera": "Unknown",
                        "timestamp": "Unknown",
                    }
                )

        # For Gallery
        gallery_files = sorted(
            [
                f
                for f in os.listdir(settings.gallery_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        gallery_data = []
        for f in gallery_files:
            # Format: Label_ID.jpg
            # Try to extract ID
            parts = f.rsplit(".", 1)[0].split("_")

            # If named correctly, last part is ID
            if len(parts) >= 2:
                event_id = parts[-1]
                event = db_repo.get_event(event_id)
                if event:
                    gallery_data.append(
                        {
                            "filename": f,
                            "label": event["current_label"],
                            "camera": event["camera"],
                            "timestamp": event["timestamp"].strftime("%Y-%m-%d %H:%M"),
                        }
                    )
                else:
                    # File exists but no event record
                    gallery_data.append(
                        {
                            "filename": f,
                            "label": parts[0],
                            "camera": "-",
                            "timestamp": "-",
                        }
                    )
            else:
                gallery_data.append(
                    {"filename": f, "label": f, "camera": "-", "timestamp": "-"}
                )

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        unknowns_data = []
        gallery_data = []

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "version": version_info, "unknowns": unknowns_data, "gallery": gallery_data},
    )


@app.get("/db_viewer", response_class=HTMLResponse)
async def db_viewer(request: Request):
    try:
        events = db_repo.get_all_events()
        history = db_repo.get_all_label_history()
    except Exception as e:
        logger.error(f"Error retrieving database records: {e}")
        events = []
        history = []

    return templates.TemplateResponse(
        "db_viewer.html",
        {
            "request": request, "version": version_info,
            "events": events,
            "history": history,
            "external_url": settings.external_url.rstrip("/"),
        },
    )

@app.get("/visualization", response_class=HTMLResponse)
async def visualization(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request, "version": version_info})

@app.get("/api/scatter")
async def get_scatter_data():
    """Returns 2D PCA projection of vectors."""
    try:
        events = db_repo.get_all_vectors()
        if len(events) < 2:
             return []

        # Prepare data for PCA
        ids = []
        labels = []
        vectors = []
        snapshots = []
        timestamps = []

        for e in events:
            if not e.get("vector"): continue
            vec = np.frombuffer(e["vector"], dtype=np.float32)
            if vec.shape != (256,): continue

            ids.append(e["id"])
            labels.append(e["current_label"])
            vectors.append(vec)

            snapshots.append(f"/snapshot/{e['id']}")
            timestamps.append(e["timestamp"].strftime("%Y-%m-%d %H:%M") if e["timestamp"] else "")

        if len(vectors) < 2:
            return []

        # PCA
        data_matrix = np.array(vectors, dtype=np.float32)
        mean, eigenvectors = cv2.PCACompute(data_matrix, mean=None, maxComponents=2)
        projected = cv2.PCAProject(data_matrix, mean, eigenvectors)

        # Generate colors
        def get_color(label):
            if label == "unknown": return "#888888"
            hash_val = sum(ord(c) for c in label)
            hue = (hash_val * 137) % 360
            return f"hsl({hue}, 70%, 50%)"

        result = []
        for i in range(len(ids)):
            result.append({
                "id": ids[i],
                "x": float(projected[i, 0]),
                "y": float(projected[i, 1]),
                "label": labels[i],
                "color": get_color(labels[i]),
                "snapshot_url": snapshots[i],
                "timestamp": timestamps[i]
            })

        return result
    except Exception as e:
        logger.error(f"Error computing scatter data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/label")
async def label_image(
    filename: str = Form(...), new_label: str = Form(...), source: str = Form(...)
):
    """Moves an image from 'unknown' to 'gallery' or renames in 'gallery'."""
    try:
        if not reid_core:
            return JSONResponse(
                {"status": "error", "message": "ReID Core not initialized"},
                status_code=500,
            )

        # Sanitize label
        clean_label = "".join([c for c in new_label if c.isalnum()]).capitalize()
        if not clean_label:
            return JSONResponse(
                {"status": "error", "message": "Invalid label"}, status_code=400
            )

        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)

        src_dir = settings.unknown_dir if source == "unknown" else settings.gallery_dir
        src_path = os.path.join(src_dir, filename)

        if not os.path.exists(src_path):
            return JSONResponse(
                {"status": "error", "message": "Source file not found"}, status_code=404
            )

        # Determine Event ID and New Filename
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        event_id = None

        if source == "unknown":
            # Filename is event_id.jpg
            event_id = base_name
            # Update DB for this specific event
            db_repo.update_label(event_id, clean_label, source="manual")

            # New filename: Label_EventID.jpg
            new_filename = f"{clean_label}_{event_id}{ext}"
            dest_path = os.path.join(settings.gallery_dir, new_filename)
            shutil.move(src_path, dest_path)

        elif source == "gallery":
            # Filename is OldLabel_EventID.jpg
            if "_" in base_name:
                parts = base_name.split("_")
                old_label = parts[0]
                event_id = parts[-1]

                # If renaming entire identity (e.g. Voisin -> Martine)
                if old_label != clean_label:
                    # Update DB for ALL events with old_label
                    db_repo.rename_identity(old_label, clean_label, source="manual")

                    # Rename ALL matching files on disk
                    renamed_files = []
                    for f in os.listdir(settings.gallery_dir):
                        if f.startswith(old_label + "_"):
                            # Construct new name
                            # Preserve the ID part
                            f_base = os.path.splitext(f)[0]
                            f_ext = os.path.splitext(f)[1]
                            f_suffix = f_base.split("_", 1)[1]

                            new_f = f"{clean_label}_{f_suffix}{f_ext}"
                            shutil.move(
                                os.path.join(settings.gallery_dir, f),
                                os.path.join(settings.gallery_dir, new_f),
                            )
                            if f == filename:
                                new_filename = new_f

                    if "new_filename" not in locals():
                        new_filename = f"{clean_label}_{base_name}{ext}"  # Fallback
                else:
                    # Same label, nothing to do?
                    new_filename = filename
            else:
                # No underscore? Just rename file
                new_filename = f"{clean_label}_{base_name}{ext}"
                dest_path = os.path.join(settings.gallery_dir, new_filename)
                shutil.move(src_path, dest_path)

        # Reload gallery
        reid_core.reload_gallery()

        return {"status": "success", "new_filename": new_filename}
    except Exception as e:
        logger.error(f"Error labeling image: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/delete")
async def delete_image(filename: str = Form(...), source: str = Form(...)):
    """Deletes an image."""
    try:
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        folder = settings.unknown_dir if source == "unknown" else settings.gallery_dir
        path = os.path.join(folder, filename)

        def _delete_sync():
            if os.path.exists(path):
                os.remove(path)
                if source == "gallery" and reid_core:
                    reid_core.reload_gallery()
                return True
            return False

        deleted = await run_in_threadpool(_delete_sync)

        if deleted:
            return {"status": "deleted"}
        else:
            return JSONResponse(
                {"status": "error", "message": "File not found"}, status_code=404
            )
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
