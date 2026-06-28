from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingTokenizerResponse(BaseModel):
    count: int = Field(..., description="tokens数量")
    max_model_len: int = Field(..., description="模型最大 embedding 数量")
    tokens: list[int] = Field(..., description="tokens列表")


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def aembed(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def embed_list(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def aembed_list(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def aembed_text(self, text: str) -> list[float]:
        """将纯文本编码为稠密向量。"""
        ...

    @abstractmethod
    async def aembed_image(self, image_data_url: str) -> list[float]:
        """将图片（data URL 或 HTTP URL）编码为稠密向量。"""
        ...

    @abstractmethod
    async def aembed_multimodal(self, text: str, image_data_url: str | None) -> list[float]:
        """将文本 + 图片联合编码为稠密向量；image_data_url=None 时退化为纯文本。"""
        ...
