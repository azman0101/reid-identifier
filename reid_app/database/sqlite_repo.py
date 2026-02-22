import sqlite3
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from .interface import ReIDRepository
from ..config import settings
import os

logger = logging.getLogger(__name__)


class SQLiteRepository(ReIDRepository):
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to storing in the mounted models directory
            db_path = os.path.join(os.path.dirname(settings.model_path), "reid.db")
        self.db_path = db_path
        logger.info(f"Initializing SQLite database at {self.db_path}")

    def _connect(self):
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def init_db(self):
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id TEXT PRIMARY KEY,
                        camera TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        snapshot_path TEXT NOT NULL,
                        current_label TEXT NOT NULL
                    )
                """)

                # Handle older schemas dynamically by adding columns
                cursor.execute("PRAGMA table_info(events)")
                columns = [info[1] for info in cursor.fetchall()]

                if "image_hash" not in columns:
                    cursor.execute("ALTER TABLE events ADD COLUMN image_hash TEXT")
                    cursor.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_image_hash ON events(image_hash)"
                    )

                if "vector" not in columns:
                    cursor.execute("ALTER TABLE events ADD COLUMN vector BLOB")

                # Label History table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS label_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        old_label TEXT NOT NULL,
                        new_label TEXT NOT NULL,
                        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        source TEXT NOT NULL,
                        FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
                    )
                """)

                # Index for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_label ON events(current_label)"
                )

                conn.commit()
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def add_event(
        self,
        event_id: str,
        camera: str,
        timestamp: datetime,
        label: str,
        snapshot_path: str,
        image_hash: str = None,
        vector: bytes = None,
    ):
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO events (id, camera, timestamp, snapshot_path, current_label, image_hash, vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (event_id, camera, timestamp, snapshot_path, label, image_hash, vector),
                )
                conn.commit()
                logger.debug(f"Event {event_id} added successfully.")
        except sqlite3.IntegrityError as e:
            logger.debug(
                f"Event {event_id} (or visual duplicate) already exists (skipping). Details: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to add event {event_id}: {e}")

    def update_vector(self, event_id: str, vector: bytes):
        """Update vector for a single event (used for backfill)"""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE events SET vector = ? WHERE id = ?",
                    (vector, event_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update vector for {event_id}: {e}")

    def get_all_vectors(self) -> List[Dict[str, Any]]:
        """Retrieve all events that have a vector stored."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, current_label, vector, snapshot_path, timestamp FROM events WHERE vector IS NOT NULL")
            return [dict(row) for row in cursor.fetchall()]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_label(self, event_id: str, new_label: str, source: str = "manual"):
        """Update label for a single event"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Get old label
            cursor.execute("SELECT current_label FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Event {event_id} not found for update.")
                return
            old_label = row[0]

            if old_label == new_label:
                return

            # Update event
            cursor.execute(
                "UPDATE events SET current_label = ? WHERE id = ?",
                (new_label, event_id),
            )

            # Record history
            cursor.execute(
                """
                INSERT INTO label_history (event_id, old_label, new_label, source, changed_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (event_id, old_label, new_label, source, datetime.now()),
            )

            conn.commit()
            logger.info(
                f"Updated event {event_id}: {old_label} -> {new_label} ({source})"
            )

    def rename_identity(self, old_label: str, new_label: str, source: str = "manual"):
        """Bulk update label for all events associated with old_label"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Find all events with old_label
            cursor.execute(
                "SELECT id FROM events WHERE current_label = ?", (old_label,)
            )
            events = cursor.fetchall()

            if not events:
                logger.info(f"No events found for label '{old_label}'.")
                return

            # Execute manually in loop to ensure history is recorded for EACH event
            # Although a bulk insert is more efficient, strict history tracking suggests explicit logging per row is safer logic-wise
            # But we can do it in one transaction block

            current_time = datetime.now()

            # Update all events
            cursor.execute(
                "UPDATE events SET current_label = ? WHERE current_label = ?",
                (new_label, old_label),
            )

            # Insert history for all affected events
            history_entries = [
                (event_id[0], old_label, new_label, source, current_time)
                for event_id in events
            ]
            cursor.executemany(
                """
                INSERT INTO label_history (event_id, old_label, new_label, source, changed_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                history_entries,
            )

            conn.commit()
            logger.info(
                f"Renamed {len(events)} events from {old_label} to {new_label} ({source})"
            )

    def get_events_by_label(self, label: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM events WHERE current_label = ? ORDER BY timestamp DESC",
                (label,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_label_history(self, event_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM label_history WHERE event_id = ? ORDER BY changed_at DESC",
                (event_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_events(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_label_history(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM label_history ORDER BY changed_at DESC")
            return [dict(row) for row in cursor.fetchall()]
