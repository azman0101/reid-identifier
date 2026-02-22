import logging
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

def crop_image_from_box(image_frame, box=None, data_box=None):
    """
    Crops an image based on a bounding box with a 10% margin.

    Args:
        image_frame: The image as a numpy array (H, W, C).
        box: Frigate MQTT box [xmin, ymin, xmax, ymax] (absolute pixels).
        data_box: Frigate API box [x, y, w, h] (normalized 0-1).

    Returns:
        The cropped image frame, or the original if cropping fails/is invalid.
    """
    try:
        if image_frame is None:
            logger.error("Utils: Image frame is None!")
            return None

        h, w, _ = image_frame.shape
        logger.info(f"Utils: Processing image {w}x{h}")
        logger.info(f"Utils: Input Box: {box}")
        logger.info(f"Utils: Input Data Box: {data_box}")

        x1, y1, x2, y2 = 0, 0, w, h
        cropped = False

        if data_box and len(data_box) == 4:
            logger.info("Utils: Using data_box (normalized)")
            nx, ny, nw, nh = data_box
            x1 = int(nx * w)
            y1 = int(ny * h)
            x2 = int((nx + nw) * w)
            y2 = int((ny + nh) * h)
            logger.info(f"Utils: Calculated initial crop: [{x1}:{x2}, {y1}:{y2}]")
            cropped = True
        elif box and len(box) == 4:
            logger.info("Utils: Using box (absolute)")
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            logger.info(f"Utils: Using provided crop: [{x1}:{x2}, {y1}:{y2}]")
            cropped = True
        else:
            logger.warning("Utils: No valid box found. Skipping crop.")

        if cropped:
            # Ensure reasonable bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            logger.info(f"Utils: Clamped crop: [{x1}:{x2}, {y1}:{y2}]")

            # Add a 10% margin to avoid chopping heads/feet
            margin_w = int((x2 - x1) * 0.10)
            margin_h = int((y2 - y1) * 0.10)
            logger.info(f"Utils: Margins: w={margin_w}, h={margin_h}")

            cx1 = max(0, x1 - margin_w)
            cy1 = max(0, y1 - margin_h)
            cx2 = min(w, x2 + margin_w)
            cy2 = min(h, y2 + margin_h)
            logger.info(f"Utils: Final crop with margin: [{cx1}:{cx2}, {cy1}:{cy2}]")

            # Only crop if it's actually smaller than the full frame
            if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1 < w or cy2 - cy1 < h):
                logger.info(
                    f"Utils: Applying crop [{cx1}:{cx2}, {cy1}:{cy2}]"
                )
                return image_frame[cy1:cy2, cx1:cx2]
            else:
                logger.info(
                    "Utils: Skipping crop because box covers entire frame or invalid dimensions."
                )
                return image_frame

        return image_frame

    except Exception as e:
        logger.warning(
            f"Utils: Failed to crop image manually: {e}. Proceeding with original image."
        )
        return image_frame
