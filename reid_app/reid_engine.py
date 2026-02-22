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
        # Cached numpy structures for fast vectorized lookup
        self._known_embeddings_matrix = np.empty((0, 256), dtype=np.float32)
        self._known_labels_list = []
        self._known_norms = np.empty((0,), dtype=np.float32)

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
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    if "_" in filename:
                        label = filename.split("_")[0]
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

        # Flatten the gallery for vectorization
        all_embeddings = []
        all_labels = []

        for label, embeddings in new_gallery.items():
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:  # Pre-filter zero-norm embeddings
                    all_embeddings.append(emb)
                    all_labels.append(label)

        # Convert to numpy arrays
        if all_embeddings:
            embeddings_matrix = np.stack(all_embeddings).astype(np.float32)
            norms_array = np.linalg.norm(embeddings_matrix, axis=1)
        else:
            embeddings_matrix = np.empty((0, 256), dtype=np.float32)
            norms_array = np.empty((0,), dtype=np.float32)

        with self.lock:
            self.known_silhouettes = new_gallery
            self._known_embeddings_matrix = embeddings_matrix
            self._known_labels_list = all_labels
            self._known_norms = norms_array

            logger.info(
                f"Gallery reloaded. Known identities: {list(self.known_silhouettes.keys())}. "
                f"Total embeddings: {len(self._known_labels_list)}"
            )

    def find_match(self, embedding, threshold=0.65):
        """
        Finds the best match for the given embedding.
        Returns a tuple: (label, score).
        label is None if the similarity score is below the threshold.
        Thread-safe access to known_silhouettes.
        """
        # Normalize the input embedding once
        norm_embedding = np.linalg.norm(embedding)
        if norm_embedding == 0:
            return None, 0.0

        with self.lock:
            if self._known_embeddings_matrix.shape[0] == 0:
                return None, 0.0

            # Vectorized Cosine Similarity
            # scores = (A . B) / (|A| * |B|)
            # matrix dot embedding -> shape (N,)
            dot_products = np.dot(self._known_embeddings_matrix, embedding)

            # Divide by norms
            # _known_norms contains non-zero norms (filtered during reload)
            # norm_embedding is non-zero (checked above)
            # We can use np.true_divide or simply /
            scores = dot_products / (self._known_norms * norm_embedding)

            # Find best match
            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_match = self._known_labels_list[best_idx]

        logger.debug(f"Best match: {best_match} with score: {best_score}")

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score
