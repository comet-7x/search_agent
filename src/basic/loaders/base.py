import mimetypes
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from ..types import BlobData


def guess_mime_type(
    raw_data: bytes,
    *,
    fallback_content_type: str | None = None,
    fallback_url: str | None = None,
) -> str:
    """按优先级检测内容 MIME 类型：魔数 → 响应头 → URL 扩展名 → 默认值。

    Args:
        raw_data:              原始字节，用于魔数检测
        fallback_content_type: 服务器响应头中的 Content-Type（第二优先级）
        fallback_url:          资源 URL，用于扩展名推断（第三优先级）
    """
    # 1. 魔数检测
    if len(raw_data) >= 12:
        # 图片
        if raw_data[:4] == b"\x89PNG":
            return "image/png"
        if raw_data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        h6 = raw_data[:6]
        if h6 in (b"GIF89a", b"GIF87a"):
            return "image/gif"
        if raw_data[:2] == b"BM":
            return "image/bmp"
        if raw_data[:4] == b"RIFF" and raw_data[8:12] == b"WEBP":
            return "image/webp"
        h4 = raw_data[:4]
        if h4 in (b"II*\x00", b"MM\x00*"):
            return "image/tiff"
        if h4 == b"\x00\x00\x01\x00":
            return "image/x-icon"
        # 文档
        if raw_data[:4] == b"%PDF":
            return "application/pdf"
        if raw_data[:4] == b"PK\x03\x04":
            return "application/zip"

    # 2. 响应头 Content-Type
    if fallback_content_type:
        mime = fallback_content_type.split(";")[0].strip()
        if mime and mime != "application/octet-stream":
            return mime

    # 3. URL 扩展名推断
    if fallback_url:
        path = urlparse(fallback_url).path
        mime = mimetypes.guess_type(path)[0] or ""
        if mime:
            return mime

    return "application/octet-stream"


class BaseLoader(ABC):
    """内容加载器抽象基类。

    每个实现负责一种来源（HTTP、本地文件、MinIO 等），
    通过 can_handle 声明自己能处理哪类 URI。
    """

    @abstractmethod
    def can_handle(self, ref: str) -> bool:
        """返回该加载器是否能处理给定的 URI。"""
        ...

    @abstractmethod
    async def load(self, ref: str) -> BlobData | None:
        """加载内容，失败时返回 None（调用方负责降级处理）。"""
        ...


class CompositeLoader(BaseLoader):
    """按顺序尝试多个加载器，由第一个能处理的负责执行。"""

    def __init__(self, loaders: list[BaseLoader]) -> None:
        self._loaders = loaders

    def can_handle(self, ref: str) -> bool:
        return any(loader.can_handle(ref) for loader in self._loaders)

    async def load(self, ref: str) -> BlobData | None:
        for loader in self._loaders:
            if loader.can_handle(ref):
                return await loader.load(ref)
        return None
