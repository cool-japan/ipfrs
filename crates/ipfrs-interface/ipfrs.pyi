"""
Type stubs for IPFRS Python bindings

This file provides type hints for the IPFRS Python module,
enabling IDE autocomplete, type checking with mypy, and better
developer experience.
"""

from typing import Optional, Union
from types import TracebackType

__version__: str
__author__: str

class Client:
    """
    Python client for IPFRS operations.

    This class provides a Python interface to IPFRS (InterPlanetary File & Reasoning System).
    Supports context manager protocol for automatic resource cleanup.

    Example:
        >>> with Client() as client:
        ...     cid = client.add(b"Hello, IPFRS!")
        ...     data = client.get(cid)
        ...     exists = client.has(cid)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Create a new IPFRS client.

        Args:
            config_path: Optional path to configuration file

        Raises:
            IOError: If client initialization fails
        """
        ...

    def add(self, data: bytes) -> str:
        """
        Add data to IPFRS and return its Content Identifier (CID).

        Args:
            data: Data to store (must be non-empty bytes)

        Returns:
            Content Identifier (CID) of the stored data

        Raises:
            ValueError: If data is empty or invalid
            IOError: If storage operation fails

        Example:
            >>> cid = client.add(b"Hello, IPFRS!")
            >>> print(cid)
            bafkreidummy0000000000000d
        """
        ...

    def get(self, cid: str) -> bytes:
        """
        Retrieve data from IPFRS by its CID.

        Args:
            cid: Content Identifier of the data to retrieve

        Returns:
            Retrieved data as bytes

        Raises:
            ValueError: If CID is empty or invalid
            IOError: If block not found or retrieval fails

        Example:
            >>> data = client.get("bafkreidummy0000000000000d")
            >>> print(data.decode())
            Hello, IPFRS!
        """
        ...

    def has(self, cid: str) -> bool:
        """
        Check if a block exists by its CID.

        Args:
            cid: Content Identifier to check

        Returns:
            True if block exists, False otherwise

        Raises:
            ValueError: If CID is empty or invalid
            IOError: If lookup operation fails

        Example:
            >>> exists = client.has("bafkreidummy0000000000000d")
            >>> print(exists)
            True
        """
        ...

    def version(self) -> str:
        """
        Get version information.

        Returns:
            Version string

        Example:
            >>> print(client.version())
            ipfrs-interface 0.1.0
        """
        ...

    def __enter__(self) -> "Client":
        """Context manager entry."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        """
        Context manager exit with automatic resource cleanup.

        Returns:
            False (does not suppress exceptions)
        """
        ...

    def __repr__(self) -> str:
        """String representation for debugging."""
        ...

    def __str__(self) -> str:
        """Human-readable string representation."""
        ...


class BlockInfo:
    """
    Information about a block in IPFRS.

    Attributes:
        cid: Content Identifier of the block
        size: Size of the block in bytes
    """

    cid: str
    size: int

    def __init__(self, cid: str, size: int) -> None:
        """
        Create new block information.

        Args:
            cid: Content Identifier
            size: Block size in bytes
        """
        ...

    def __repr__(self) -> str:
        """String representation showing cid and size."""
        ...
