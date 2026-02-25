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
        self.gallery_embeddings = np.empty((0, 256), dtype=np.float32)
        self.gallery_labels = []
        self.compiled_model = None
        self.output_layer = None
        self.lock = threading.Lock()  # Protects shared state (known_silhouettes)

        self._load_model()
        self.reload_gallery()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run model_manager.py first."
            )

        logger.info(f"Loading model from {self.model_path}...")
        model = self.ie.read_model(model=self.model_path)

        try:
            logger.info(f"Compiling model for device: {settings.device_name}")
            self.compiled_model = self.ie.compile_model(
                model=model, device_name=settings.device_name
            )
        except RuntimeError as e:
            logger.warning(
                f"Failed to compile model on {settings.device_name}: {e}. Falling back to CPU."
            )
            self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")

        self.output_layer = self.compiled_model.output(0)
        logger.info("Model loaded successfully.")

    def get_embedding(self, image_frame):
        """
        Takes an image frame (BGR), resizes it to 128x256 and returns the embedding vector.
        Thread-safe inference call (OpenVINO compiled_model is reentrant).
        """
        if image_frame is None or image_frame.size == 0:
            return np.zeros((256,), dtype=np.float32)

        # Convert BGR (OpenCV default) to RGB (standard for deep learning models)
        rgb_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)

        # Resize to 128x256 (Width, Height)
        resized_image = cv2.resize(rgb_image, (128, 256))

        # HWC to CHW
        input_tensor = resized_image.transpose((2, 0, 1))

        # Keep as float32 or let compiled_model handle it, but standard is float32
        input_tensor = input_tensor.astype(np.float32)

        # Add batch dimension [1, C, H, W]
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        result = self.compiled_model([input_tensor])[self.output_layer]

        return result.flatten()

    def reload_gallery(self):
        """Reloads identities from the gallery directory into memory."""
        logger.info("Reloading gallery...")

        new_embeddings = []
        new_labels = []

        if os.path.exists(self.gallery_dir):
            for filename in os.listdir(self.gallery_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    if "_" in filename:
                        label = filename.split("_")[0]
                    else:
                        label = os.path.splitext(filename)[0]

                    filepath = os.path.join(self.gallery_dir, filename)
                    img = cv2.imread(filepath)

                    if img is not None:
                        try:
                            # Note: get_embedding is thread-safe so we can call it here without lock.
                            embedding = self.get_embedding(img)
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                normalized_emb = embedding / norm
                                new_embeddings.append(normalized_emb)
                                new_labels.append(label)
                        except Exception as e:
                            logger.error(f"Error processing {filename}: {e}")
                    else:
                        logger.warning(f"Could not read image: {filename}")

        with self.lock:
            if new_embeddings:
                self.gallery_embeddings = np.array(new_embeddings, dtype=np.float32)
                self.gallery_labels = new_labels
            else:
                self.gallery_embeddings = np.empty((0, 256), dtype=np.float32)
                self.gallery_labels = []

            unique_labels = list(set(self.gallery_labels))
            logger.info(f"Gallery reloaded. Known identities: {unique_labels}")

    def _compute_best_match(self, embedding):
        """
        Internal helper: Computes the best match and score against the gallery.
        Returns: (best_match_label, best_score_float)
        If gallery is empty or embedding invalid, returns (None, 0.0)
        """
        if self.gallery_embeddings.shape[0] == 0:
            return None, 0.0

        # Normalize the input embedding once
        norm_embedding = np.linalg.norm(embedding)
        if norm_embedding == 0:
            return None, 0.0

        normalized_query = embedding / norm_embedding

        with self.lock:
            # Vectorized dot product against all known embeddings simultaneously
            # shape of self.gallery_embeddings is (N, D), normalized_query is (D,)
            # result is an array of shape (N,) containing all cosine similarity scores
            scores = np.dot(self.gallery_embeddings, normalized_query)

            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_match = self.gallery_labels[best_idx]

        return best_match, best_score

    def find_match(self, embedding, threshold=0.55):
        """
        Finds the best match for the given embedding.
        Returns a tuple: (label, score).
        label is None if the similarity score is below the threshold.
        Thread-safe access to known_silhouettes.
        """
        best_match, best_score = self._compute_best_match(embedding)
        logger.debug(f"Best match: {best_match} with score: {best_score}")

        if best_match and best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def find_closest_match(self, embedding):
        """
        Finds the absolute closest match regardless of threshold.
        Returns: (label, score).
        """
        return self._compute_best_match(embedding)
