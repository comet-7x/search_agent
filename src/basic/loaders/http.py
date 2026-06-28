from typing import Any

from httpx import AsyncClient

from .base import BaseLoader, BlobData, guess_mime_type


class HttpLoader(BaseLoader):
    """加载 HTTP / HTTPS 资源。"""

    def __init__(self, http_kwargs: dict[str, Any] | None = None) -> None:
        self._client = AsyncClient(**(http_kwargs or {"timeout": 30}))

    def can_handle(self, ref: str) -> bool:
        return ref.startswith(("http://", "https://"))

    async def load(self, ref: str) -> BlobData | None:
        try:
            response = await self._client.get(ref)
            response.raise_for_status()
            content_type = guess_mime_type(
                response.content,
                fallback_content_type=response.headers.get("content-type"),
            )
            return BlobData(content=response.content, content_type=content_type)
        except Exception:
            return None
