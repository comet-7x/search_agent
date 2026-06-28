import asyncio
from urllib.parse import urlparse

from minio import Minio

from .base import BaseLoader, BlobData, guess_mime_type


class MinioLoader(BaseLoader):
    """加载 MinIO 对象存储中的内容。

    ref 格式：minio://bucket-name/object/key/path
    示例：minio://product-images/2024/photo.jpg
    """

    def __init__(self, client: Minio) -> None:
        self._client = client

    def can_handle(self, ref: str) -> bool:
        return ref.startswith("minio://")

    async def load(self, ref: str) -> BlobData | None:
        try:
            parsed = urlparse(ref)
            bucket = parsed.netloc
            object_name = parsed.path.lstrip("/")

            response = await asyncio.to_thread(
                self._client.get_object, bucket, object_name
            )
            try:
                raw_bytes = response.data
            finally:
                response.close()
                response.release_conn()

            content_type = guess_mime_type(raw_bytes, fallback_url=ref)
            return BlobData(content=raw_bytes, content_type=content_type)
        except Exception:
            return None
