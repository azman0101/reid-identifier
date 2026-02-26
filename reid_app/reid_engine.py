import os
import cv2
import numpy as np
import logging
import threading
import time
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

    def _letterbox_resize(self, image, target_size=(128, 256)):
        """Preserves aspect ratio with neutral gray padding."""
        ih, iw = image.shape[:2]
        tw, th = target_size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        image_resized = cv2.resize(image, (nw, nh))
        # Neutral gray background (128) instead of black to help the model
        canvas = np.full((th, tw, 3), 128, dtype=np.uint8)

        # Centering
        dx, dy = (tw - nw) // 2, (th - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = image_resized
        return canvas

    def get_embedding(self, image_frame, use_tta=None):
        """
        Extraction with Letterboxing and TTA option.
        """
        if use_tta is None:
            use_tta = settings.use_tta

        if image_frame is None or image_frame.size == 0:
            return np.zeros((256,), dtype=np.float32)

        # 1. Aspect Ratio Padding
        processed_img = self._letterbox_resize(image_frame)

        def inference(img):
            # HWC to CHW
            tensor = img.transpose((2, 0, 1))
            # Batch dimension
            tensor = tensor[np.newaxis, ...].astype(np.float32)
            return self.compiled_model([tensor])[self.output_layer].flatten()

        # 2. Test Time Augmentation (Flip)
        emb = inference(processed_img)

        if use_tta:
            flipped_img = cv2.flip(processed_img, 1)
            emb_flipped = inference(flipped_img)
            # Average and renormalization
            emb = (emb + emb_flipped) / 2.0

        # Final L2 Normalization
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _k_reciprocal_dist(self, query_emb, gallery_embs, k=5):
        """
        Simplified reranking by neighborhood similarity (Jaccard).
        """
        # 1. Initial cosine distances
        original_scores = np.dot(gallery_embs, query_emb) # (N,)

        # 2. Find k-nearest neighbors of query in gallery
        # argsort is ascending, so for high scores (similarity) we use -original_scores
        query_neighbors = np.argsort(-original_scores)[:k]

        # 3. For each candidate in gallery, look at its own neighbors
        reranked_scores = np.zeros_like(original_scores)

        # Optimizable, but loop is fine for small gallery sizes
        # For larger galleries, we might need a pre-computed distance matrix for the gallery itself
        # But here we compute on the fly as gallery changes only on reload/update.
        # Actually, if gallery is large, computing 'np.dot(gallery_embs, gallery_embs[i])' inside loop is O(N^2).
        # We can precompute gallery_gallery_scores if needed, but let's stick to the snippet for now.

        for i in range(len(gallery_embs)):
            # Neighbors of candidate i
            cand_scores = np.dot(gallery_embs, gallery_embs[i])
            cand_neighbors = np.argsort(-cand_scores)[:k]

            # Intersection (Reciprocity)
            intersection = np.intersect1d(query_neighbors, cand_neighbors)
            # Simplified Jaccard Score: intersection size / k
            reranked_scores[i] = (len(intersection) / k) * 0.3 + original_scores[i] * 0.7

        return reranked_scores

    def update_gallery(self, label, new_embedding, save_to_disk=False, frame=None):
        """
        Dynamically adds a new outfit/embedding to an existing cluster.
        """
        with self.lock:
            # 1. Update RAM
            # Ensure new_embedding is (1, 256) for vstack
            if new_embedding.ndim == 1:
                new_embedding = new_embedding.reshape(1, -1)

            self.gallery_embeddings = np.vstack([self.gallery_embeddings, new_embedding])
            self.gallery_labels.append(label)

        logger.info(f"Dynamically updated cluster for: {label} (Total embeddings: {len(self.gallery_labels)})")

        # 2. Optional: Physical save so cluster survives restart
        if save_to_disk and frame is not None:
            # Generate unique name to avoid overwriting original
            timestamp = int(time.time())
            # Clean label for filename just in case
            safe_label = "".join([c for c in label if c.isalnum() or c in ('_', '-')])
            filename = f"{safe_label}_auto_{timestamp}.jpg"
            filepath = os.path.join(self.gallery_dir, filename)
            try:
                cv2.imwrite(filepath, frame)
                logger.info(f"Saved auto-learning image to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save auto-learning image: {e}")

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
                            # get_embedding now returns normalized embedding.
                            embedding = self.get_embedding(img)
                            # Verify normalization just in case
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                # It should already be normalized, but ensure no zero vectors
                                new_embeddings.append(embedding)
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

    def _compute_best_match(self, embedding, use_rerank=None):
        """
        Internal helper: Computes the best match and score against the gallery.
        Returns: (best_match_label, best_score_float)
        If gallery is empty or embedding invalid, returns (None, 0.0)
        """
        if use_rerank is None:
            use_rerank = settings.use_rerank

        if self.gallery_embeddings.shape[0] == 0:
            return None, 0.0

        # Normalize the input embedding once (get_embedding returns normalized, but safe to check)
        norm_embedding = np.linalg.norm(embedding)
        if norm_embedding == 0:
            return None, 0.0

        # If it's already normalized (norm ~ 1.0), this division is fine.
        # If get_embedding didn't normalize properly, this fixes it.
        normalized_query = embedding / norm_embedding

        with self.lock:
            if use_rerank and self.gallery_embeddings.shape[0] > settings.rerank_k:
                scores = self._k_reciprocal_dist(normalized_query, self.gallery_embeddings, k=settings.rerank_k)
            else:
                # Vectorized dot product against all known embeddings simultaneously
                scores = np.dot(self.gallery_embeddings, normalized_query)

            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_match = self.gallery_labels[best_idx]

        return best_match, best_score

    def find_match(self, embedding, threshold=0.55, use_rerank=None):
        """
        Finds the best match for the given embedding.
        Returns a tuple: (label, score).
        label is None if the similarity score is below the threshold.
        Thread-safe access to known_silhouettes.
        """
        best_match, best_score = self._compute_best_match(embedding, use_rerank=use_rerank)
        logger.debug(f"Best match: {best_match} with score: {best_score}")

        if best_match and best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def find_closest_match(self, embedding, use_rerank=None):
        """
        Finds the absolute closest match regardless of threshold.
        Returns: (label, score).
        """
        return self._compute_best_match(embedding, use_rerank=use_rerank)
