"""
retrieval 模块示例

演示：
  1. 单路召回，无精排
  2. 单路召回 + 纯文本精排
  3. 单路召回 + 多模态精排（text + MinIO 图片）
  4. 阈值过滤：绝对阈值 / 断崖阈值 / 组合使用
  5. 三路多模态召回（text / image / multimodal 各一路）
"""

import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from src.basic.database.vector.milvus import AsyncMilvusStorage
from src.basic.embeddings.qwen3_vl_embedding import Qwen3VLEmbeddingModel
from src.basic.loaders import CompositeLoader, MinioLoader
from src.basic.rerankers.qwen_vl_reranker import Qwen3VLReranker
from src.retrieval.base import AnnRequestConfig, RetrievalResult
from src.retrieval.pipeline import HybridRetriever

load_dotenv(find_dotenv())

# ── 服务地址 ─────────────────────────────────────────────────────────────────
MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_DB = os.environ["MILVUS_DB"]

EMBEDDING_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_KEY = os.environ["EMBEDDING_API_KEY"]

RERANKER_URL = os.environ["RERANKER_BASE_URL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
RERANKER_KEY = os.environ["RERANKER_API_KEY"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# 检索参数
COLLECTION = "commercial"
TEXT_FIELD = "text"
IMAGE_FIELD = "image_url"
OUTPUT_FIELDS = [TEXT_FIELD, IMAGE_FIELD, "file_id", "page_index"]
QUERY = "2000年我国水产品对外贸易总量"
TOP_K = 5
RECALL_LIMIT = 20
TEST_IMAGE = "minio://test-bucket/test_multi_rag/output/7/2000/page_images/page_1.jpg"


# 工厂
def _minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE
    )


def _minio_loader() -> CompositeLoader:
    return CompositeLoader([MinioLoader(_minio_client())])


def _embedder() -> Qwen3VLEmbeddingModel:
    return Qwen3VLEmbeddingModel(
        base_url=EMBEDDING_URL, model_name=EMBEDDING_MODEL, api_key=EMBEDDING_KEY
    )


def _storage() -> AsyncMilvusStorage:
    return AsyncMilvusStorage(uri=MILVUS_URI, db_name=MILVUS_DB)


def _text_ann() -> AnnRequestConfig:
    return AnnRequestConfig(
        field="text_embedding",
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        query_type="embed_text",
    )


def _reranker(with_image: bool) -> Qwen3VLReranker:
    return Qwen3VLReranker(
        base_url=RERANKER_URL,
        model_name=RERANKER_MODEL,
        api_key=RERANKER_KEY,
        text_field=TEXT_FIELD,
        image_field=IMAGE_FIELD if with_image else None,
        image_loader=_minio_loader() if with_image else None,
    )


def _retriever(
    ann_requests: list[AnnRequestConfig] | None = None,
    reranker: Qwen3VLReranker | None = None,
    score_threshold: float | None = None,
    score_gap_threshold: float | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        storage=_storage(),
        embedder=_embedder(),
        collection=COLLECTION,
        ann_requests=ann_requests or [_text_ann()],
        output_fields=OUTPUT_FIELDS,
        text_extractor=lambda r: r.get(TEXT_FIELD, ""),
        reranker=reranker,
        recall_limit=RECALL_LIMIT,
        score_threshold=score_threshold,
        score_gap_threshold=score_gap_threshold,
    )


# 输出工具
def _show(results: list[RetrievalResult], label: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {label}  ({len(results)} 条)")
    print(f"{'═' * 60}")
    for i, r in enumerate(results, 1):
        raw = f"  raw={r.raw_score:.4f}" if r.raw_score is not None else ""
        print(f"  [{i}] score={r.score:.4f}{raw}  {r.text[:60].strip()}…")


async def main() -> None:
    print(f"Query: {QUERY!r}\n")

    # 1. 单路召回，无精排
    r1 = await _retriever().aretrieve(QUERY, top_k=TOP_K)
    _show(r1, "场景 1 — 单路召回（无精排）")

    # 2. 单路召回 + 纯文本精排
    r2 = await _retriever(reranker=_reranker(with_image=False)).aretrieve(QUERY, top_k=TOP_K)
    _show(r2, "场景 2 — 单路召回 + 纯文本精排")

    # 3. 单路召回 + 多模态精排
    r3 = await _retriever(reranker=_reranker(with_image=True)).aretrieve(QUERY, top_k=TOP_K)
    _show(r3, "场景 3 — 单路召回 + 多模态精排")

    # 4. 阈值过滤对比
    r_base = await _retriever().aretrieve(QUERY, top_k=10)
    r_abs = await _retriever(score_threshold=0.84).aretrieve(QUERY, top_k=10)
    r_gap = await _retriever(score_gap_threshold=0.02).aretrieve(QUERY, top_k=10)
    r_both = await _retriever(score_threshold=0.75, score_gap_threshold=0.01).aretrieve(
        QUERY, top_k=10
    )

    print(f"\n{'═' * 60}")
    print("  场景 4 — 阈值过滤对比")
    print(f"{'═' * 60}")
    print(f"  无阈值 (top10)          : {[f'{r.score:.4f}' for r in r_base]}")
    print(f"  绝对阈值 ≥0.84          : {[f'{r.score:.4f}' for r in r_abs]}  ({len(r_abs)} 条)")
    print(f"  断崖阈值 gap>0.02       : {[f'{r.score:.4f}' for r in r_gap]}  ({len(r_gap)} 条)")
    print(f"  组合 ≥0.75 + gap>0.01  : {[f'{r.score:.4f}' for r in r_both]}  ({len(r_both)} 条)")

    # 5. 三路多模态召回
    # 需要集合中存在 image_embedding / multimodal_embedding 字段，否则注释掉
    loader = _minio_loader()
    image_blob = await loader.load(TEST_IMAGE)
    assert image_blob is not None and image_blob.is_image

    three_channel = _retriever(
        ann_requests=[
            AnnRequestConfig(
                field="text_embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                query_type="embed_text",
            ),
            # 以下两路需要集合中存在对应向量字段，当前 commercial 集合暂无，已注释
            # AnnRequestConfig(
            #     field="image_embedding",
            #     search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            #     query_type="embed_image",
            # ),
            # AnnRequestConfig(
            #     field="multimodal_embedding",
            #     search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            #     query_type="embed_multimodal",
            # ),
        ]
    )
    r5 = await three_channel.aretrieve(QUERY, image=image_blob, top_k=TOP_K)
    _show(r5, "场景 5 — 三路多模态召回（当前仅 text 路）")


if __name__ == "__main__":
    asyncio.run(main())
