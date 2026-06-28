from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..retrieval.base import BaseRetriever

if TYPE_CHECKING:
    from .query_rewriter import QueryRewriter


@dataclass
class AgentContext:
    """Agent 运行时共享的依赖容器，通过 RunContextWrapper 注入工具函数。

    Attributes:
        retriever:       召回器，工具函数通过它执行 RAG 检索
        retrieval_top_k: 每次检索返回的最大结果数
        query_rewriter:  可选的查询改写器；设置后 search 工具会自动扩展查询以提升召回
        expand_n:        query_rewriter 每次扩展的变体数量
    """

    retriever: BaseRetriever
    retrieval_top_k: int = field(default=5)
    query_rewriter: QueryRewriter | None = field(default=None)
    expand_n: int = field(default=3)
