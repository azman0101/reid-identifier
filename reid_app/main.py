import os
import shutil
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates

from .config import settings
from .reid_engine import ReIDCore
from .model_manager import ensure_models_exist
from .mqtt_frigate import start_mqtt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
reid_core = None
mqtt_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ReID App...")

    # Ensure models exist (blocking call, but fast if files exist)
    try:
        ensure_models_exist()
    except Exception as e:
        logger.error(f"Failed to ensure models exist: {e}")
        # Depending on criticality, maybe exit? But let's proceed and hope for the best or manual fix.

    # Initialize Core
    global reid_core
    try:
        reid_core = ReIDCore()
    except Exception as e:
        logger.error(f"Failed to initialize ReID Core: {e}")
        # Without core, app is useless, but maybe we can serve error page?

    # Start MQTT
    global mqtt_worker
    if reid_core:
        mqtt_worker = start_mqtt(reid_core)

    yield

    # Shutdown
    if mqtt_worker:
        mqtt_worker.client.loop_stop()
        mqtt_worker.client.disconnect()
    logger.info("Shutting down ReID App...")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="reid_app/templates")

# Mount static files
app.mount("/gallery_imgs", StaticFiles(directory=settings.gallery_dir), name="gallery")
app.mount("/unknown_imgs", StaticFiles(directory=settings.unknown_dir), name="unknown")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        # Check if directories exist, create if not
        if not os.path.exists(settings.unknown_dir):
            os.makedirs(settings.unknown_dir, exist_ok=True)
        if not os.path.exists(settings.gallery_dir):
            os.makedirs(settings.gallery_dir, exist_ok=True)

        unknown_files = [f for f in os.listdir(settings.unknown_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        gallery_files = [f for f in os.listdir(settings.gallery_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        unknown_files.sort()
        gallery_files.sort()
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        unknown_files = []
        gallery_files = []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "unknowns": unknown_files,
        "gallery": gallery_files
    })

@app.post("/label")
async def label_image(
    filename: str = Form(...),
    new_label: str = Form(...),
    source: str = Form(...)
):
    """Moves an image from 'unknown' to 'gallery' or renames in 'gallery'."""
    try:
        if not reid_core:
            return JSONResponse({"status": "error", "message": "ReID Core not initialized"}, status_code=500)

        # Sanitize label (Alphanumeric only)
        clean_label = "".join([c for c in new_label if c.isalnum()]).capitalize()

        if not clean_label:
             return JSONResponse({"status": "error", "message": "Invalid label"}, status_code=400)

        src_dir = settings.unknown_dir if source == "unknown" else settings.gallery_dir
        src_path = os.path.join(src_dir, filename)

        if not await run_in_threadpool(os.path.exists, src_path):
             return JSONResponse({"status": "error", "message": "Source file not found"}, status_code=404)

        # Construct new filename
        # If filename has underscore, assume suffix follows. Else use filename as suffix.
        if '_' in filename:
            # e.g., "unknown_123.jpg" -> "Label_123.jpg"
            # or "OldLabel_123.jpg" -> "Label_123.jpg"
            suffix = filename.split('_', 1)[1]
            new_filename = f"{clean_label}_{suffix}"
        else:
            # e.g. "123.jpg" -> "Label_123.jpg"
            new_filename = f"{clean_label}_{filename}"

        dest_path = os.path.join(settings.gallery_dir, new_filename)

        # Move (rename is same as move if on same fs)
        # shutil.move handles cross-fs moves if volumes are mounted differently (unlikely here but safe)
        await run_in_threadpool(shutil.move, src_path, dest_path)

        # Reload gallery to update embeddings
        await run_in_threadpool(reid_core.reload_gallery)

        return {"status": "success", "new_filename": new_filename}
    except Exception as e:
        logger.error(f"Error labeling image: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/delete")
async def delete_image(filename: str = Form(...), source: str = Form(...)):
    """Deletes an image."""
    try:
        folder = settings.unknown_dir if source == "unknown" else settings.gallery_dir
        path = os.path.join(folder, filename)

        if await run_in_threadpool(os.path.exists, path):
            await run_in_threadpool(os.remove, path)
            if source == "gallery" and reid_core:
                await run_in_threadpool(reid_core.reload_gallery)
            return {"status": "deleted"}
        else:
            return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
