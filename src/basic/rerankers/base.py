import asyncio
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, Sequence, TypeVar


class RerankCandidate(Protocol):
    """与存储层解耦的候选抽象，VectorSearchContent 天然满足此协议。"""

    record: dict[str, Any]
    score: float


T = TypeVar("T")


class BaseReranker(ABC, Generic[T]):
    """多模态 Reranker 基类，采用模板方法模式将输入构造与打分分离。

    子类必须实现：
      · _build_input  将原始候选转换为本 Reranker 所需的输入格式（可含异步 IO）
      · score         同步打分
      · ascore        异步打分


    调用方（HybridRetriever）只需调用 ascore_candidates，无需感知内部输入类型 T。
    """

    @abstractmethod
    async def _build_input(self, candidate: RerankCandidate) -> T:
        """将单条原始候选构造为打分所需的输入（可含图片获取等异步 IO）。"""
        ...

    @abstractmethod
    def score(self, query: str, inputs: list[T], **kwargs) -> list[float]:
        """对已构造好的输入同步打分，返回与 inputs 等长的分数列表。"""
        ...

    @abstractmethod
    async def ascore(self, query: str, inputs: list[T], **kwargs) -> list[float]:
        """对已构造好的输入异步打分，返回与 inputs 等长的分数列表。"""
        ...

    async def ascore_candidates(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        """模板方法：批量构造输入 → 异步打分。HybridRetriever 调用此方法。"""
        inputs = [await self._build_input(c) for c in candidates]
        return await self.ascore(query, inputs)

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        """ascore_candidates 的同步包装，不可在已运行的事件循环中调用。"""
        return asyncio.run(self.ascore_candidates(query, candidates))
