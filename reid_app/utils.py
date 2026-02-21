import logging

# We assume this function is used where 'logger' is passed, or we can configure a default one
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
        h, w, _ = image_frame.shape
        x1, y1, x2, y2 = 0, 0, w, h
        cropped = False

        if data_box and len(data_box) == 4:
            # Frigate 0.14 events API data.box is [x, y, width, height] normalized (0 to 1)
            nx, ny, nw, nh = data_box
            x1 = int(nx * w)
            y1 = int(ny * h)
            x2 = int((nx + nw) * w)
            y2 = int((ny + nh) * h)
            cropped = True
        elif box and len(box) == 4:
            # Native Frigate MQTT 'box' is [xmin, ymin, xmax, ymax] absolute pixels
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped = True

        if cropped:
            # Ensure reasonable bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Add a 10% margin to avoid chopping heads/feet
            margin_w = int((x2 - x1) * 0.10)
            margin_h = int((y2 - y1) * 0.10)

            cx1 = max(0, x1 - margin_w)
            cy1 = max(0, y1 - margin_h)
            cx2 = min(w, x2 + margin_w)
            cy2 = min(h, y2 + margin_h)

            # Only crop if it's actually smaller than the full frame
            if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1 < w or cy2 - cy1 < h):
                logger.info(
                    f"Manually cropping image from {w}x{h} to bounding box [{cx1}:{cx2}, {cy1}:{cy2}]"
                )
                return image_frame[cy1:cy2, cx1:cx2]
            else:
                logger.info(
                    "Skipping manual crop (box covers entire frame or invalid)."
                )
                return image_frame

        return image_frame

    except Exception as e:
        logger.warning(
            f"Failed to crop image manually: {e}. Proceeding with original image."
        )
        return image_frame
