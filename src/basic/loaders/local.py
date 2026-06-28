import mimetypes
from pathlib import Path

from .base import BaseLoader, BlobData, guess_mime_type


class LocalLoader(BaseLoader):
    """加载本地文件系统内容，支持 file:// URI 和绝对路径。"""

    def can_handle(self, ref: str) -> bool:
        return ref.startswith("file://") or Path(ref).is_absolute()

    async def load(self, ref: str) -> BlobData | None:
        try:
            path = Path(ref.removeprefix("file://"))
            content = path.read_bytes()
            content_type = (
                guess_mime_type(content, fallback_url=str(path))
                or mimetypes.guess_type(str(path))[0]
                or "application/octet-stream"
            )
            return BlobData(content=content, content_type=content_type)
        except Exception:
            return None
