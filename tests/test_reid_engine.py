import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock settings before importing reid_engine
# We set env vars before import, but settings might have been loaded.
# We will patch settings object directly in tests.

from reid_app.reid_engine import ReIDCore
from reid_app.config import settings


class TestReIDCore(unittest.TestCase):
    @patch("reid_app.reid_engine.Core")
    def test_initialization_cpu_fallback(self, mock_core_class):
        # Setup mock openvino core
        mock_ie = MagicMock()
        mock_core_class.return_value = mock_ie

        # Mock read_model
        mock_model = MagicMock()
        mock_ie.read_model.return_value = mock_model

        # Mock compile_model to raise RuntimeError for GPU then succeed for CPU
        mock_compiled_model_cpu = MagicMock()

        # Side effect: First call raises Error, second returns CPU model
        def side_effect(model, device_name):
            if device_name == "GPU":
                raise RuntimeError("GPU failed")
            return mock_compiled_model_cpu

        mock_ie.compile_model.side_effect = side_effect

        # Patch settings and os.path.exists
        with (
            patch.object(settings, "device_name", "GPU"),
            patch.object(settings, "model_path", "dummy.xml"),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            # Initialize engine
            engine = ReIDCore()

            # Verify calls
            mock_ie.read_model.assert_called_with(model="dummy.xml")

            # compile_model should be called with GPU first
            mock_ie.compile_model.assert_any_call(model=mock_model, device_name="GPU")

            # Then with CPU
            mock_ie.compile_model.assert_called_with(
                model=mock_model, device_name="CPU"
            )

            # Ensure engine uses the CPU model
            self.assertEqual(engine.compiled_model, mock_compiled_model_cpu)

    @patch("reid_app.reid_engine.Core")
    def test_get_embedding(self, mock_core_class):
        mock_ie = MagicMock()
        mock_core_class.return_value = mock_ie

        mock_compiled_model = MagicMock()
        mock_ie.compile_model.return_value = mock_compiled_model

        # Mock output layer object
        mock_output_layer = MagicMock()
        mock_compiled_model.output.return_value = mock_output_layer

        # Mock inference result
        expected_embedding = np.zeros(256, dtype=np.float32)

        # Mock the __call__ of compiled_model
        # It takes a list of tensors and returns a dict
        mock_compiled_model.side_effect = lambda inputs: {
            mock_output_layer: expected_embedding
        }

        with (
            patch.object(settings, "model_path", "dummy.xml"),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            engine = ReIDCore()

            # Create dummy image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            embedding = engine.get_embedding(img)

            self.assertEqual(embedding.shape, (256,))
            np.testing.assert_array_equal(embedding, expected_embedding)

    @patch("reid_app.reid_engine.Core")
    def test_find_match_vectorized(self, mock_core_class):
        # Setup mocks to avoid model loading
        mock_ie = MagicMock()
        mock_core_class.return_value = mock_ie
        mock_ie.read_model.return_value = MagicMock()
        mock_ie.compile_model.return_value = MagicMock()

        with (
            patch.object(settings, "model_path", "dummy.xml"),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),  # Don't load real gallery
        ):
            engine = ReIDCore()

            # Create synthetic gallery
            # Identity A: [1, 0, 0, ...]
            # Identity B: [0, 1, 0, ...]

            emb_a = np.zeros(256, dtype=np.float32)
            emb_a[0] = 1.0

            emb_b = np.zeros(256, dtype=np.float32)
            emb_b[1] = 1.0

            # Populate gallery manually via internal structures
            # Case 1: Populated gallery
            engine.gallery_embeddings = np.stack([emb_a, emb_b]).astype(np.float32)
            engine.gallery_labels = ["A", "B"]

            # Query close to A
            query_a = np.zeros(256, dtype=np.float32)
            query_a[0] = 0.9
            query_a[1] = 0.1

            label, score = engine.find_match(query_a, threshold=0.5)
            self.assertEqual(label, "A")
            self.assertGreater(score, 0.8)

            # Query close to B
            query_b = np.zeros(256, dtype=np.float32)
            query_b[0] = 0.1
            query_b[1] = 0.9

            label, score = engine.find_match(query_b, threshold=0.5)
            self.assertEqual(label, "B")
            self.assertGreater(score, 0.8)

            # Query orthogonal
            query_z = np.zeros(256, dtype=np.float32)
            query_z[2] = 1.0

            label, score = engine.find_match(query_z, threshold=0.9)
            self.assertIsNone(label)
            self.assertEqual(score, 0.0)

    @patch("reid_app.reid_engine.Core")
    def test_zero_norm_handling(self, mock_core_class):
        # Setup mocks
        mock_ie = MagicMock()
        mock_core_class.return_value = mock_ie
        mock_ie.read_model.return_value = MagicMock()
        mock_ie.compile_model.return_value = MagicMock()

        with (
            patch.object(settings, "model_path", "dummy.xml"),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            engine = ReIDCore()

            # Empty gallery
            engine.gallery_embeddings = np.empty((0, 256), dtype=np.float32)
            engine.gallery_labels = []

            label, score = engine.find_match(np.random.rand(256).astype(np.float32))
            self.assertIsNone(label)
            self.assertEqual(score, 0.0)

            # Test that find_match handles zero-norm QUERY.
            engine.gallery_embeddings = np.ones((1, 256), dtype=np.float32)
            engine.gallery_labels = ["A"]

            zero_query = np.zeros(256, dtype=np.float32)
            label, score = engine.find_match(zero_query)
            self.assertIsNone(label)
            self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
