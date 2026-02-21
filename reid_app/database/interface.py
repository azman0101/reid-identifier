from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

class ReIDRepository(ABC):
    """
    Abstract interface for database operations.
    Follows the Repository Pattern to decouple business logic from data storage.
    """

    @abstractmethod
    def init_db(self):
        """Initialize the database schema."""
        pass

    @abstractmethod
    def add_event(self, event_id: str, camera: str, timestamp: datetime, label: str, snapshot_path: str):
        """
        Record a new detection event.
        :param event_id: Frigate event ID (PK)
        :param camera: Name of the camera
        :param timestamp: Detection timestamp
        :param label: Initial label (e.g., 'unknown' or 'Martine')
        :param snapshot_path: Relative path to the stored image
        """
        pass

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single event by ID."""
        pass

    @abstractmethod
    def update_label(self, event_id: str, new_label: str, source: str = "manual"):
        """
        Update the label of a specific event and record history.
        :param event_id: The event to update
        :param new_label: The new label
        :param source: 'manual' (user) or 'auto' (system correction)
        """
        pass

    @abstractmethod
    def rename_identity(self, old_label: str, new_label: str, source: str = "manual"):
        """
        Rename an identity globally (e.g., 'unknown' -> 'Martine', or 'Voisin' -> 'Martine').
        Updates all associated events and logs history for each change.
        :param old_label: The label to be replaced
        :param new_label: The new label
        :param source: 'manual' (user) or 'auto' (system correction)
        """
        pass

    @abstractmethod
    def get_events_by_label(self, label: str) -> List[Dict[str, Any]]:
        """Retrieve all events associated with a label."""
        pass

    @abstractmethod
    def get_label_history(self, event_id: str) -> List[Dict[str, Any]]:
        """Retrieve the history of label changes for an event."""
        pass
