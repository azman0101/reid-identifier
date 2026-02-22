import unittest
import os
import shutil
from unittest.mock import MagicMock, patch
from reid_app.mqtt_frigate import MQTTWorker
from reid_app.config import settings

class TestMQTTOrphanedFile(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_mqtt"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        settings.unknown_dir = self.test_dir

        self.mock_core = MagicMock()
        self.mock_db = MagicMock()
        self.worker = MQTTWorker(self.mock_core, self.mock_db)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("reid_app.mqtt_frigate.requests")
    @patch("reid_app.mqtt_frigate.cv2")
    @patch("reid_app.mqtt_frigate.compute_dhash")
    def test_orphaned_file_deletion(self, mock_dhash, mock_cv2, mock_requests):
        event_id = "test_event"
        filename = f"{event_id}.jpg"
        filepath = os.path.join(self.test_dir, filename)

        def side_effect_imwrite(path, img):
            with open(path, "w") as f:
                f.write("dummy")
        mock_cv2.imwrite.side_effect = side_effect_imwrite

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake_image_bytes"
        mock_requests.get.return_value = mock_resp

        mock_cv2.imdecode.return_value = MagicMock()
        self.mock_core.find_match.return_value = (None, 0.0)

        # Simulate DUPLICATE -> add_event returns False
        self.mock_db.add_event.return_value = False

        self.worker.process_event(event_id, "cam1")

        self.mock_db.add_event.assert_called_once()
        self.assertFalse(os.path.exists(filepath), "Orphaned file should be deleted on duplicate")

    @patch("reid_app.mqtt_frigate.requests")
    @patch("reid_app.mqtt_frigate.cv2")
    @patch("reid_app.mqtt_frigate.compute_dhash")
    def test_success_file_kept(self, mock_dhash, mock_cv2, mock_requests):
        event_id = "test_event_success"
        filename = f"{event_id}.jpg"
        filepath = os.path.join(self.test_dir, filename)

        def side_effect_imwrite(path, img):
            with open(path, "w") as f:
                f.write("dummy")
        mock_cv2.imwrite.side_effect = side_effect_imwrite

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake_image_bytes"
        mock_requests.get.return_value = mock_resp
        mock_cv2.imdecode.return_value = MagicMock()
        self.mock_core.find_match.return_value = (None, 0.0)

        # Simulate SUCCESS -> add_event returns True
        self.mock_db.add_event.return_value = True

        self.worker.process_event(event_id, "cam1")

        self.assertTrue(os.path.exists(filepath), "File should be kept on success")

if __name__ == "__main__":
    unittest.main()
