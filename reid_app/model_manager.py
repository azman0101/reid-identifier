import os
import time
import requests
import logging
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_XML_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml"
MODEL_BIN_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.bin"


def download_file(url, filepath, max_retries=5, base_delay=1):
    # Check if file exists
    if os.path.exists(filepath):
        logger.info(f"File {filepath} already exists. Skipping download.")
        return

    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} to {filepath} (Attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded {filepath}")
            return
        except Exception as e:
            last_exception = e
            logger.error(f"Failed to download {url}: {e}")
            # Clean up partial download if failed
            if os.path.exists(filepath):
                os.remove(filepath)

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    logger.critical(f"Exhausted all {max_retries} retries for {url}.")
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Download failed for {url}")


def ensure_models_exist():
    model_xml_path = settings.model_path
    if not model_xml_path.endswith(".xml"):
        # Just in case the config path is directory or something else
        # Assuming the config points to the xml file
        pass

    model_bin_path = model_xml_path.replace(".xml", ".bin")

    model_dir = os.path.dirname(model_xml_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    download_file(MODEL_XML_URL, model_xml_path)
    download_file(MODEL_BIN_URL, model_bin_path)


if __name__ == "__main__":
    ensure_models_exist()
