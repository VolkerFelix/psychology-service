"""Factory for creating storage service instances."""

from typing import Optional, Union

from app.services.storage.db_storage import DatabaseStorage


class StorageFactory:
    """Factory for creating storage service instances."""

    @staticmethod
    def create_storage_service(
        storage_type: Optional[str] = None, db_url: Optional[str] = None
    ) -> Union[DatabaseStorage, None]:
        """
        Create a storage service instance based on the specified type.

        Args:
            storage_type: Type of storage service to create
                Options: 'database', 'memory'
                If None, use the default from settings.
            db_url: Optional database URL to use

        Returns:
            Storage service instance or None if type is not supported.
        """
        if storage_type is None:
            # Default to database storage for production
            storage_type = "database"

        if storage_type == "database":
            return DatabaseStorage(db_url=db_url)
        elif storage_type == "memory":
            # Create a DatabaseStorage with SQLite in-memory database
            return DatabaseStorage(db_url="sqlite:///:memory:")

        # Add other storage types as needed
        return None
