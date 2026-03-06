import logging
import requests
from .config import settings

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
        logger.debug(f"Utils: Processing image {w}x{h}")
        logger.debug(f"Utils: Input Box: {box}")
        logger.debug(f"Utils: Input Data Box: {data_box}")

        x1, y1, x2, y2 = 0, 0, w, h
        cropped = False

        if box and len(box) == 4:
            logger.debug("Utils: Using box (absolute)")
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Check if image is already cropped
            if bx2 > w or by2 > h:
                logger.info(
                    "Utils: Image appears already cropped (box falls outside bounds). Returning as-is."
                )
                return image_frame
            else:
                x1, y1, x2, y2 = bx1, by1, bx2, by2
                logger.debug(f"Utils: Using provided crop: [{x1}:{x2}, {y1}:{y2}]")
                cropped = True

        elif data_box and len(data_box) == 4:
            logger.debug("Utils: Using data_box (normalized)")
            nx, ny, nw, nh = data_box

            # If the image is extremely small or already portrait mode, it's likely already cropped
            if (h >= w and w < 600) or w < 300:
                logger.info(
                    "Utils: Image appears already cropped (small/portrait dimensions). Returning as-is."
                )
                return image_frame

            x1 = int(nx * w)
            y1 = int(ny * h)
            x2 = int((nx + nw) * w)
            y2 = int((ny + nh) * h)
            logger.debug(f"Utils: Calculated initial crop: [{x1}:{x2}, {y1}:{y2}]")
            cropped = True
        else:
            logger.warning("Utils: No valid box found. Skipping crop.")

        if cropped:
            # Ensure reasonable bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            logger.debug(f"Utils: Clamped crop: [{x1}:{x2}, {y1}:{y2}]")

            # Add a 10% margin to avoid chopping heads/feet
            margin_w = int((x2 - x1) * 0.10)
            margin_h = int((y2 - y1) * 0.10)

            # Apply margins and clamp strictly to image boundaries
            cx1 = max(0, x1 - margin_w)
            cy1 = max(0, y1 - margin_h)
            cx2 = min(w, x2 + margin_w)
            cy2 = min(h, y2 + margin_h)

            # Only crop if valid dimensions and it's actually smaller than the full frame
            if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1 < w or cy2 - cy1 < h):
                logger.info(f"Utils: Applying crop [{cx1}:{cx2}, {cy1}:{cy2}]")
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


def update_frigate_description(event_id, new_status_line):
    """
    Updates the Frigate event description, preserving existing user notes
    and replacing only the [ReID] status line.

    Args:
        event_id: The Frigate event ID.
        new_status_line: The complete new status line (e.g., "[ReID]: ...").
    """
    try:
        url = f"{settings.frigate_url}/api/events/{event_id}"

        # 1. Fetch current description
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            logger.error(
                f"Failed to fetch event {event_id} for description update: {resp.status_code}"
            )
            return False

        data = resp.json()
        current_desc = data.get("description", "")
        if current_desc is None:
            current_desc = ""

        # 2. Parse and filter lines
        lines = current_desc.split("\n")
        new_lines = []
        for line in lines:
            if not line.strip().startswith("[ReID]:"):
                new_lines.append(line)

        # Remove trailing empty lines to keep it clean
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()

        # 3. Append new status
        new_lines.append(new_status_line)

        final_desc = "\n".join(new_lines)

        if final_desc == current_desc:
            logger.info(f"Description for {event_id} is unchanged. Skipping update.")
            return True

        # 4. POST update
        desc_url = f"{settings.frigate_url}/api/events/{event_id}/description"
        payload = {"description": final_desc}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        post_resp = requests.post(desc_url, json=payload, headers=headers, timeout=5)

        if post_resp.status_code == 200:
            logger.info(f"Updated description for {event_id}")
            return True
        else:
            logger.error(
                f"Failed to update description for {event_id}: {post_resp.status_code} {post_resp.text}"
            )
            return False

    except Exception as e:
        logger.error(f"Error updating description for {event_id}: {e}")
        return False
