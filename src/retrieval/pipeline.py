import asyncio
import logging
from typing import Any
from collections.abc import Callable

from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.client.abstract import BaseRanker

from ..basic.database.vector.milvus import AsyncMilvusStorage, VectorSearchContent
from ..basic.embeddings.base import BaseEmbeddingModel
from ..basic.types import BlobData
from ..basic.rerankers.base import BaseReranker

from .base import AnnRequestConfig, BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """通用多路召回 + 可选精排的检索管道。

    召回路数由 ann_requests 决定：
      · 1 条配置 → 单路搜索（search）
      · 多条配置 → Hybrid Search（hybrid_search + Ranker 融合）

    每路配置通过 AnnRequestConfig.query_type 区分数据来源：
      · "embed" → 调用 embedder 编码 query 为稠密向量
      · "text"  → 直接将 query 文本传入（BM25 服务端自动转换）

    Args:
        storage:              AsyncMilvusStorage 实例
        embedder:             稠密向量的编码模型（仅在有 "embed" 类型请求时调用）
        collection:           目标集合名
        ann_requests:         各路搜索配置列表，顺序不影响结果
        output_fields:        搜索时额外返回的字段；None 则返回所有字段
        text_extractor:       从 record dict 中提取文本的函数，写入 RetrievalResult.text；
                              None 则 text 为空字符串，所有字段均在 metadata 中
        reranker:             精排模型，接收原始 VectorSearchContent 列表（含多模态字段）；
                              None 则跳过精排
        ranker:               多路融合策略，默认 RRFRanker（单路时不生效）
        recall_limit:         全局召回数量上限，各路未单独指定 limit 时使用此值
        score_threshold:      绝对阈值过滤：低于该分数的结果直接丢弃；None 表示不过滤
        score_gap_threshold:  断崖阈值过滤：相邻结果分差超过该值时截断后续所有结果；
                              None 表示不过滤。两者可同时启用，先过绝对阈值再过断崖
    """

    def __init__(
        self,
        storage: AsyncMilvusStorage,
        embedder: BaseEmbeddingModel,
        collection: str,
        ann_requests: list[AnnRequestConfig],
        *,
        output_fields: list[str] | None = None,
        text_extractor: Callable[[dict[str, Any]], str] | None = None,
        reranker: BaseReranker | None = None,
        ranker: BaseRanker | None = None,
        recall_limit: int = 100,
        score_threshold: float | None = None,
        score_gap_threshold: float | None = None,
    ) -> None:
        if not ann_requests:
            raise ValueError("ann_requests 不能为空，至少需要一路搜索配置。")

        self._storage = storage
        self._embedder = embedder
        self._collection = collection
        self._ann_requests = ann_requests
        self._output_fields = output_fields
        self._text_extractor = text_extractor
        self._reranker = reranker
        self._ranker: BaseRanker = ranker or RRFRanker()
        self._recall_limit = recall_limit
        self._score_threshold = score_threshold
        self._score_gap_threshold = score_gap_threshold

    # 公开入口
    async def aretrieve(
        self,
        query: str,
        image: BlobData | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """执行完整检索流程：召回 → 精排（可选）→ 返回 Top-K。

        Args:
            query: 文本查询
            image: 可选图片输入，用于 embed_image / embed_multimodal 路
            top_k: 最终返回条数
        """
        candidates = await self._recall(query, image)

        if not candidates:
            return []

        if self._reranker:
            scored = await self._rerank(query, candidates, self._reranker)
            results = self._build_results(scored, has_reranker=True)
        else:
            results = self._build_results(
                [(c.score, c) for c in candidates],
                has_reranker=False,
            )

        return self._apply_thresholds(results)[:top_k]

    # 召回
    async def _recall(self, query: str, image: BlobData | None) -> list[VectorSearchContent]:
        """按配置路数选择单路 search 或多路 hybrid_search。"""
        vectors = await self._encode_query(query, image)

        if len(self._ann_requests) == 1:
            return await self._single_recall(query, vectors, self._ann_requests[0])
        return await self._hybrid_recall(query, vectors)

    async def _encode_query(
        self,
        query: str,
        image: BlobData | None,
    ) -> dict[str, list[float]]:
        """按需并行编码各路向量，避免重复推理。

        Returns:
            key 为 query_type（embed_text / embed_image / embed_multimodal），
            value 为对应稠密向量。
        """
        needed = {cfg.query_type for cfg in self._ann_requests}
        image_data_url = image.to_data_url() if image else None

        coros: dict[str, Any] = {}
        if "embed_text" in needed:
            coros["embed_text"] = self._embedder.aembed_text(query)
        if "embed_image" in needed:
            if image_data_url is None:
                raise RuntimeError(
                    "AnnRequestConfig 包含 embed_image 路，但 aretrieve 未传入 image。"
                )
            coros["embed_image"] = self._embedder.aembed_image(image_data_url)
        if "embed_multimodal" in needed:
            coros["embed_multimodal"] = self._embedder.aembed_multimodal(query, image_data_url)

        if not coros:
            return {}

        keys = list(coros.keys())
        results = await asyncio.gather(*coros.values())
        return dict(zip(keys, results))

    async def _single_recall(
        self,
        query: str,
        vectors: dict[str, list[float]],
        cfg: AnnRequestConfig,
    ) -> list[VectorSearchContent]:
        data = self._resolve_data(query, vectors, cfg)
        limit = cfg.limit or self._recall_limit
        logger.debug(
            f"Single recall: collection='{self._collection}', field='{cfg.field}', limit={limit}"
        )
        return await self._storage.search(
            collection=self._collection,
            data=data,
            anns_field=cfg.field,
            search_params=cfg.search_params,
            limit=limit,
            output_fields=self._output_fields,
        )

    async def _hybrid_recall(
        self,
        query: str,
        vectors: dict[str, list[float]],
    ) -> list[VectorSearchContent]:
        requests = [
            AnnSearchRequest(
                data=self._resolve_data(query, vectors, cfg),
                anns_field=cfg.field,
                param=cfg.search_params,
                limit=cfg.limit or self._recall_limit,
            )
            for cfg in self._ann_requests
        ]
        logger.debug(
            f"Hybrid recall: collection='{self._collection}', "
            f"routes={len(requests)}, recall_limit={self._recall_limit}"
        )
        return await self._storage.hybrid_search(
            collection=self._collection,
            requests=requests,
            ranker=self._ranker,
            limit=self._recall_limit,
            output_fields=self._output_fields,
        )

    @staticmethod
    def _resolve_data(
        query: str,
        vectors: dict[str, list[float]],
        cfg: AnnRequestConfig,
    ) -> list[Any]:
        """根据 query_type 从 vectors 中取出该路对应的 data。"""
        if cfg.query_type == "text":
            return [query]
        vec = vectors.get(cfg.query_type)
        if vec is None:
            raise RuntimeError(
                f"字段 '{cfg.field}' 需要 '{cfg.query_type}' 向量，但未找到——"
                "请检查 aretrieve 是否传入了 image，或 AnnRequestConfig 配置是否正确。"
            )
        return [vec]

    # 精排
    async def _rerank(
        self,
        query: str,
        candidates: list[VectorSearchContent],
        reranker: BaseReranker,
    ) -> list[tuple[float, VectorSearchContent]]:
        """模板方法精排：_build_input 构造输入 → ascore 打分，返回降序列表。"""
        scores = await reranker.ascore_candidates(query, candidates)
        logger.debug(f"Reranked {len(candidates)} candidates.")
        return list(zip(scores, candidates))

    # 阈值过滤
    def _apply_thresholds(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """先做绝对阈值过滤，再做断崖截断，结果已按 score 降序排列。"""
        if self._score_threshold is not None:
            results = [r for r in results if r.score >= self._score_threshold]

        if self._score_gap_threshold is not None:
            for i in range(len(results) - 1):
                if results[i].score - results[i + 1].score >= self._score_gap_threshold:
                    results = results[: i + 1]
                    break

        return results

    # 结果构建
    def _build_results(
        self,
        scored: list[tuple[float, VectorSearchContent]],
        *,
        has_reranker: bool,
    ) -> list[RetrievalResult]:
        results = []
        for final_score, candidate in sorted(scored, key=lambda x: x[0], reverse=True):
            text = self._text_extractor(candidate.record) if self._text_extractor else ""
            results.append(
                RetrievalResult(
                    text=text,
                    score=final_score,
                    raw_score=candidate.score if has_reranker else None,
                    metadata=candidate.record,
                )
            )
        return results
