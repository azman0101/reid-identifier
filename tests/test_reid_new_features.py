import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from reid_app.reid_engine import ReIDCore
from reid_app.config import settings

class TestReIDNewFeatures(unittest.TestCase):
    def setUp(self):
        # Patch settings
        self.settings_patcher = patch.object(settings, "model_path", "dummy.xml")
        self.settings_patcher.start()

        # Patch Core to avoid OpenVINO loading
        self.core_patcher = patch("reid_app.reid_engine.Core")
        self.mock_core = self.core_patcher.start()
        self.mock_ie = MagicMock()
        self.mock_core.return_value = self.mock_ie
        self.mock_ie.read_model.return_value = MagicMock()
        self.mock_ie.compile_model.return_value = MagicMock()

        # Patch os.path.exists/listdir to avoid filesystem
        self.os_exists_patcher = patch("os.path.exists", return_value=True)
        self.os_exists_patcher.start()
        self.os_listdir_patcher = patch("os.listdir", return_value=[])
        self.os_listdir_patcher.start()

    def tearDown(self):
        self.settings_patcher.stop()
        self.core_patcher.stop()
        self.os_exists_patcher.stop()
        self.os_listdir_patcher.stop()

    def test_letterbox_resize(self):
        engine = ReIDCore()
        # Input image: 100x200 (tall)
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        # Target: 128x256
        resized = engine._letterbox_resize(img, target_size=(128, 256))

        self.assertEqual(resized.shape, (256, 128, 3))
        # Check padding color
        # Since 100x200 scales to 128x256 perfectly (1:2 aspect ratio), no padding might be needed?
        # 100/200 = 0.5. 128/256 = 0.5. Matches exactly.

        # Try wide image: 200x100
        img_wide = np.zeros((100, 200, 3), dtype=np.uint8)
        resized_wide = engine._letterbox_resize(img_wide, target_size=(128, 256))
        self.assertEqual(resized_wide.shape, (256, 128, 3))
        # Should have gray padding at top/bottom
        # Scale: 128/200 = 0.64. H=100*0.64=64.
        # Padding: (256-64)/2 = 96 pixels top/bottom.
        # Check top pixel (should be 128)
        self.assertTrue(np.all(resized_wide[0, 0] == 128))

    def test_update_gallery(self):
        engine = ReIDCore()
        engine.gallery_embeddings = np.empty((0, 256), dtype=np.float32)
        engine.gallery_labels = []

        new_emb = np.ones((1, 256), dtype=np.float32)
        engine.update_gallery("new_person", new_emb)

        self.assertEqual(len(engine.gallery_labels), 1)
        self.assertEqual(engine.gallery_labels[0], "new_person")
        self.assertEqual(engine.gallery_embeddings.shape, (1, 256))

    def test_reranking_logic(self):
        engine = ReIDCore()

        # Create a gallery where A and B are distinct clusters
        # A1, A2 close to each other
        # B1, B2 close to each other
        # Query is close to A1

        # We simulate dot products by manually creating embeddings
        # A1 = [1, 0, ...], A2 = [0.9, 0.4, ...], B1 = [0, 1, ...], B2 = [0.1, 0.9, ...]

        A1 = np.zeros(256, dtype=np.float32); A1[0] = 1.0
        A2 = np.zeros(256, dtype=np.float32); A2[0] = 0.9; A2[1] = 0.4; A2 = A2/np.linalg.norm(A2)
        B1 = np.zeros(256, dtype=np.float32); B1[10] = 1.0
        B2 = np.zeros(256, dtype=np.float32); B2[10] = 0.9; B2[11] = 0.4; B2 = B2/np.linalg.norm(B2)

        engine.gallery_embeddings = np.vstack([A1, A2, B1, B2])
        engine.gallery_labels = ["A", "A", "B", "B"]

        # Query close to A1
        Q = np.zeros(256, dtype=np.float32); Q[0] = 0.95; Q[1] = 0.1; Q = Q/np.linalg.norm(Q)

        # Use _k_reciprocal_dist directly
        # k=2 because small gallery
        reranked = engine._k_reciprocal_dist(Q, engine.gallery_embeddings, k=2)

        # Expect higher scores for index 0 and 1 (A)
        self.assertGreater(reranked[0], reranked[2])
        self.assertGreater(reranked[1], reranked[3])

if __name__ == "__main__":
    unittest.main()
