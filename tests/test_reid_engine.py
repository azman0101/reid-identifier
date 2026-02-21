import os
import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Mock settings before importing reid_engine
# We set env vars before import, but settings might have been loaded.
# We will patch settings object directly in tests.

from reid_app.reid_engine import ReIDCore
from reid_app.config import settings

class TestReIDCore(unittest.TestCase):

    @patch('reid_app.reid_engine.Core')
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
        with patch.object(settings, 'device_name', 'GPU'),              patch.object(settings, 'model_path', 'dummy.xml'),              patch('os.path.exists', return_value=True):

            # Initialize engine
            engine = ReIDCore()

            # Verify calls
            mock_ie.read_model.assert_called_with(model='dummy.xml')

            # compile_model should be called with GPU first
            mock_ie.compile_model.assert_any_call(model=mock_model, device_name="GPU")

            # Then with CPU
            mock_ie.compile_model.assert_called_with(model=mock_model, device_name="CPU")

            # Ensure engine uses the CPU model
            self.assertEqual(engine.compiled_model, mock_compiled_model_cpu)

    @patch('reid_app.reid_engine.Core')
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
        mock_compiled_model.side_effect = lambda inputs: {mock_output_layer: expected_embedding}

        with patch.object(settings, 'model_path', 'dummy.xml'),              patch('os.path.exists', return_value=True):

            engine = ReIDCore()

            # Create dummy image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            embedding = engine.get_embedding(img)

            self.assertEqual(embedding.shape, (256,))
            np.testing.assert_array_equal(embedding, expected_embedding)

if __name__ == '__main__':
    unittest.main()
