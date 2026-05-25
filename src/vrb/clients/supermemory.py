import mimetypes
from pathlib import Path

import aiofiles
import httpx

from ..config import settings
from ..models import RetrievedChunk
from .base import BaseRAGClient

_MIME_FALLBACKS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def _mime(path: Path) -> str:
    detected, _ = mimetypes.guess_type(str(path))
    return detected or _MIME_FALLBACKS.get(path.suffix.lower(), "application/octet-stream")


class SuperMemoryClient(BaseRAGClient):
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.supermemory_base_url,
            headers={"Authorization": f"Bearer {settings.supermemory_api_key}"},
            timeout=120.0,
        )

    @property
    def name(self) -> str:
        return "supermemory"

    async def ingest_file(
        self,
        file_path: Path,
        doc_id: str,
        metadata: dict | None = None,
    ) -> str:
        mime = _mime(file_path)
        async with aiofiles.open(file_path, "rb") as fh:
            file_bytes = await fh.read()

        if mime.startswith("text/"):
            content = file_bytes.decode("utf-8", errors="replace")
        else:
            import base64
            b64 = base64.b64encode(file_bytes).decode()
            content = f"data:{mime};base64,{b64}"

        tags = (metadata or {}).get("tags", ["benchmark"])
        payload = {
            "content": content,
            "customId": doc_id,
            "containerTags": tags,
        }
        resp = await self._http.post("/documents", json=payload)

        resp.raise_for_status()
        # Return customId so delete_document can find it by the same key
        return doc_id

    async def search(
        self, query: str, top_k: int = 5, tag: str | None = None
    ) -> list[RetrievedChunk]:
        payload: dict = {"q": query, "limit": top_k}
        if tag:
            payload["containerTags"] = [tag]
        resp = await self._http.post("/search", json=payload)
        resp.raise_for_status()
        body = resp.json()

        chunks: list[RetrievedChunk] = []
        for item in body.get("results", []):
            for chunk in item.get("chunks", []):
                chunks.append(RetrievedChunk(
                    content=chunk.get("content", ""),
                    score=float(chunk.get("score", item.get("score", 0.0))),
                    metadata=item.get("metadata", {}),
                ))
        return chunks

    async def delete_document(self, doc_id: str) -> None:
        resp = await self._http.delete(f"/documents/{doc_id}")
        resp.raise_for_status()

    async def __aenter__(self) -> "SuperMemoryClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()
