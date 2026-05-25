from abc import ABC, abstractmethod
from pathlib import Path

from ..models import RetrievedChunk


class BaseRAGClient(ABC):
    """Pluggable interface — implement this to add a new platform."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def ingest_file(
        self,
        file_path: Path,
        doc_id: str,
        metadata: dict | None = None,
    ) -> str:
        """Upload a file and return the platform-assigned document ID."""
        ...

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, tag: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for the query.

        If tag is provided, restrict results to documents with that tag.
        """
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Remove a document — called during test cleanup."""
        ...

    async def __aenter__(self) -> "BaseRAGClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass
