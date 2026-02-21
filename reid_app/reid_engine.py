import os
import cv2
import numpy as np
import logging
import threading
from openvino import Core
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReIDCore:
    def __init__(self):
        self.ie = Core()
        self.model_path = settings.model_path
        self.gallery_dir = settings.gallery_dir
        self.known_silhouettes = {}
        self.compiled_model = None
        self.output_layer = None
        self.lock = threading.Lock()  # Protects shared state (known_silhouettes)

        self._load_model()
        self.reload_gallery()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run model_manager.py first.")

        logger.info(f"Loading model from {self.model_path}...")
        model = self.ie.read_model(model=self.model_path)

        try:
            logger.info(f"Compiling model for device: {settings.device_name}")
            self.compiled_model = self.ie.compile_model(model=model, device_name=settings.device_name)
        except RuntimeError as e:
            logger.warning(f"Failed to compile model on {settings.device_name}: {e}. Falling back to CPU.")
            self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")

        self.output_layer = self.compiled_model.output(0)
        logger.info("Model loaded successfully.")

    def get_embedding(self, image_frame):
        """
        Takes an image frame (BGR), resizes it to 128x256, and returns the embedding vector.
        Thread-safe inference call (OpenVINO compiled_model is reentrant).
        """
        if image_frame is None or image_frame.size == 0:
            return np.zeros((256,), dtype=np.float32)

        # Resize to 128x256 (Width, Height)
        resized_image = cv2.resize(image_frame, (128, 256))

        # HWC to CHW
        input_tensor = resized_image.transpose((2, 0, 1))

        # Add batch dimension [1, C, H, W]
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        result = self.compiled_model([input_tensor])[self.output_layer]

        return result.flatten()

    def reload_gallery(self):
        """Reloads identities from the gallery directory into memory."""
        logger.info("Reloading gallery...")

        new_gallery = {}
        if os.path.exists(self.gallery_dir):
            for filename in os.listdir(self.gallery_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if '_' in filename:
                        label = filename.split('_')[0]
                    else:
                        label = os.path.splitext(filename)[0]

                    filepath = os.path.join(self.gallery_dir, filename)
                    img = cv2.imread(filepath)

                    if img is not None:
                        try:
                            # Note: get_embedding is thread-safe so we can call it here without lock,
                            # but we are modifying local new_gallery.
                            embedding = self.get_embedding(img)
                            if label not in new_gallery:
                                new_gallery[label] = []
                            new_gallery[label].append(embedding)
                        except Exception as e:
                            logger.error(f"Error processing {filename}: {e}")
                    else:
                        logger.warning(f"Could not read image: {filename}")

        with self.lock:
            self.known_silhouettes = new_gallery
            logger.info(f"Gallery reloaded. Known identities: {list(self.known_silhouettes.keys())}")

    def find_match(self, embedding, threshold=0.65):
        """
        Finds the best match for the given embedding.
        Returns the label if the similarity score is above the threshold, else None.
        Thread-safe access to known_silhouettes.
        """
        best_match = None
        best_score = -1.0

        # Normalize the input embedding once
        norm_embedding = np.linalg.norm(embedding)
        if norm_embedding == 0:
            return None

        with self.lock:
            # Iterate over a snapshot/copy or under lock
            # Iterating under lock is safer and fast enough since it's just dot products
            for label, embeddings in self.known_silhouettes.items():
                for known_emb in embeddings:
                    norm_known = np.linalg.norm(known_emb)
                    if norm_known == 0:
                        continue

                    score = np.dot(embedding, known_emb) / (norm_embedding * norm_known)

                    if score > best_score:
                        best_score = score
                        best_match = label

        logger.debug(f"Best match: {best_match} with score: {best_score}")

        if best_score >= threshold:
            return best_match
        return None
