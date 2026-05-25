from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import respx
import httpx

from vrb.clients.supermemory import SuperMemoryClient
from vrb.models import RetrievedChunk


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    # Re-import to pick up monkeypatched env
    import importlib
    import vrb.config as cfg
    importlib.reload(cfg)
    return SuperMemoryClient()


@respx.mock
@pytest.mark.asyncio
async def test_search_returns_chunks(client):
    respx.get("https://api.supermemory.ai/v3/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"content": "hello world", "score": 0.9, "metadata": {}}]},
        )
    )
    chunks = await client.search("test query")
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert chunks[0].score == 0.9


@respx.mock
@pytest.mark.asyncio
async def test_search_empty_results(client):
    respx.get("https://api.supermemory.ai/v3/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    chunks = await client.search("nothing here")
    assert chunks == []


@respx.mock
@pytest.mark.asyncio
async def test_delete_document(client):
    respx.delete("https://api.supermemory.ai/v3/documents/abc123").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    await client.delete_document("abc123")  # should not raise
