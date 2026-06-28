import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass
class AnnRequestConfig:
    """单路 ANN 搜索请求的配置，对应 Milvus 中一个向量字段。

    Attributes:
        field:         Milvus 集合中的向量字段名
        search_params: 索引搜索参数，例如 {"metric_type": "COSINE", "params": {"ef": 200}}
        query_type:    "embed"  → 用 embedder 将 query 编码为稠密向量（FLOAT_VECTOR）
                       "text"   → 直接将 query 文本传入（BM25 稀疏向量，服务端自动转换）
        limit:         本路单独召回数量；None 则使用 HybridRetriever 的全局 recall_limit
    """

    field: str
    search_params: dict[str, Any]
    query_type: Literal["embed_text", "embed_image", "embed_multimodal", "text"] = "embed_text"
    limit: int | None = None


class RetrievalResult(BaseModel):
    """单条检索结果，是整个召回链路的统一输出格式。

    Attributes:
        text:       主文本内容（来自 text_field 对应的字段）
        score:      最终分数（若有 Reranker 则为精排分，否则为召回分）
        raw_score:  召回阶段的原始分数（仅在 Reranker 存在时填充，调试用）
        metadata:   除 text 字段外的其余字段（包含主键 id、来源等）
    """

    text: str
    score: float
    raw_score: float | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


class BaseRetriever(ABC):
    """召回链路的抽象基类。

    子类只需实现 aretrieve，retrieve 默认通过 asyncio.run 托底。
    若子类运行在已有事件循环的环境（FastAPI / async 框架），请直接使用 aretrieve。
    """

    @abstractmethod
    async def aretrieve(
        self,
        query: str,
        image: Any = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]: ...

    def retrieve(
        self,
        query: str,
        image: Any = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """同步入口，不可在已运行的事件循环中调用（请改用 aretrieve）。"""
        return asyncio.run(self.aretrieve(query, image, top_k))
