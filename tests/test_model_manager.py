import unittest
import os
from unittest.mock import MagicMock, patch, mock_open
from reid_app.model_manager import download_file

class TestModelManager(unittest.TestCase):
    @patch("reid_app.model_manager.requests.get")
    @patch("reid_app.model_manager.os.path.exists")
    @patch("reid_app.model_manager.open", new_callable=mock_open)
    @patch("reid_app.model_manager.time.sleep")
    def test_download_file_retry_success(self, mock_sleep, mock_file, mock_exists, mock_get):
        # Setup
        url = "http://example.com/model.xml"
        filepath = "/tmp/model.xml"
        mock_exists.return_value = False

        # Mock response to fail twice then succeed
        mock_response_success = MagicMock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.iter_content.return_value = [b"chunk1", b"chunk2"]

        # We need requests.get to return different responses or raise exception directly
        # If requests.get raises exception (e.g. connection error)
        mock_get.side_effect = [Exception("Network Down"), Exception("Network Down"), mock_response_success]

        # Execute
        download_file(url, filepath, max_retries=5, base_delay=0.1)

        # Verify
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2) # slept after 1st and 2nd failure

        # Verify sleep times (exponential backoff)
        # 1st fail (attempt 0): sleep(0.1 * 2^0) = 0.1
        # 2nd fail (attempt 1): sleep(0.1 * 2^1) = 0.2
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)

    @patch("reid_app.model_manager.requests.get")
    @patch("reid_app.model_manager.os.path.exists")
    @patch("reid_app.model_manager.time.sleep")
    def test_download_file_retry_exhausted(self, mock_sleep, mock_exists, mock_get):
        # Setup
        url = "http://example.com/model.xml"
        filepath = "/tmp/model.xml"
        mock_exists.return_value = False

        # Always fail
        mock_get.side_effect = Exception("Permanent Failure")

        # Execute & Verify
        with self.assertRaises(Exception) as cm:
            download_file(url, filepath, max_retries=3, base_delay=0.1)

        self.assertEqual(str(cm.exception), "Permanent Failure")
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2) # sleeps after attempt 0 and 1. Attempt 2 is last, no sleep after it.
