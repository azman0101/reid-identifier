import unittest
import os
import shutil
import sqlite3
from reid_app.database.sqlite_repo import SQLiteRepository
from reid_app.config import settings
from datetime import datetime

class TestSQLiteRepo(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_db"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

        self.db_path = os.path.join(self.test_dir, "test.db")
        self.repo = SQLiteRepository(self.db_path)
        self.repo.init_db()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_event_returns_boolean(self):
        # 1. Add first event -> Should return True
        res1 = self.repo.add_event(
            event_id="e1",
            camera="cam1",
            timestamp=datetime.now(),
            label="unknown",
            snapshot_path="e1.jpg",
            image_hash="hash1",
            vector=b"vec"
        )
        self.assertTrue(res1, "First insert should return True")

        # 2. Add duplicate event (same hash) -> Should return False
        res2 = self.repo.add_event(
            event_id="e2",
            camera="cam1",
            timestamp=datetime.now(),
            label="unknown",
            snapshot_path="e2.jpg",
            image_hash="hash1", # Same hash
            vector=b"vec"
        )
        self.assertFalse(res2, "Duplicate insert should return False")

        # 3. Add another unique event -> Should return True
        res3 = self.repo.add_event(
            event_id="e3",
            camera="cam1",
            timestamp=datetime.now(),
            label="unknown",
            snapshot_path="e3.jpg",
            image_hash="hash2",
            vector=b"vec"
        )
        self.assertTrue(res3, "Unique insert should return True")

if __name__ == "__main__":
    unittest.main()
