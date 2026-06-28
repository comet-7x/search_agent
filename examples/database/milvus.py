"""
database.vector.milvus 模块示例

演示：
  1. list_collections        — 列举所有集合
  2. has_collection          — 判断集合是否存在
  3. describe_collection     — 查看集合 Schema
  4. get_collection_stats    — 查看行数统计
  5. search                  — 单路向量检索
  6. hybrid_search           — 稠密+稀疏混合检索（BM25）
"""

import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from pymilvus import AnnSearchRequest
from pymilvus.client.abstract import RRFRanker

from src.basic.database.vector.milvus import AsyncMilvusStorage
from src.basic.embeddings.qwen3_vl_embedding import Qwen3VLEmbeddingModel

load_dotenv(find_dotenv())

# 配置
MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_DB = os.environ["MILVUS_DB"]
COLLECTION = "commercial"
OUTPUT_FIELDS = ["text", "image_url", "file_id", "page_index"]

EMBEDDING_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_KEY = os.environ["EMBEDDING_API_KEY"]

QUERY = "2000年我国水产品对外贸易总量"


async def main() -> None:
    embedder = Qwen3VLEmbeddingModel(
        base_url=EMBEDDING_URL, model_name=EMBEDDING_MODEL, api_key=EMBEDDING_KEY
    )

    async with AsyncMilvusStorage(uri=MILVUS_URI, db_name=MILVUS_DB) as storage:
        # 1. list_collections
        print("── 1. list_collections ──────────────────────────────")
        collections = await storage.list_collections()
        print(f"  {collections}")

        # 2. has_collection
        print("\n── 2. has_collection ────────────────────────────────")
        exists = await storage.has_collection(COLLECTION)
        print(f"  '{COLLECTION}' 存在: {exists}")

        # 3. describe_collection
        print("\n── 3. describe_collection ───────────────────────────")
        info = await storage.describe_collection(COLLECTION)
        if info:
            print(f"  description : {info.get('description', '-')}")
            fields = info.get("fields", [])
            print(f"  字段数量    : {len(fields)}")
            for f in fields:
                dtype = f.get("type", "?")
                print(f"    - {f.get('name', '?'):25} type={dtype}")

        # 4. get_collection_stats
        print("\n── 4. get_collection_stats ──────────────────────────")
        stats = await storage.get_collection_stats(COLLECTION)
        print(f"  row_count = {stats}")

        # 5. search — 单路向量检索
        print("\n── 5. search（单路向量）─────────────────────────────")
        vec: list[float] = await embedder.aembed_text(QUERY)
        results = await storage.search(
            collection=COLLECTION,
            data=[vec],
            anns_field="text_embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5,
            output_fields=OUTPUT_FIELDS,
        )
        print(f"  共 {len(results)} 条结果")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r.score:.4f}  {r.record.get('text', '')[:60]}…")

        # 6. hybrid_search — 稠密 + 稀疏 BM25 混合检索
        print("\n── 6. hybrid_search（dense + BM25）──────────────────")
        ann_dense = AnnSearchRequest(
            data=[vec],
            anns_field="text_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=20,
        )
        ann_sparse = AnnSearchRequest(
            data=[QUERY],
            anns_field="text_sparse_embedding",
            param={"metric_type": "BM25"},
            limit=20,
        )
        hybrid_results = await storage.hybrid_search(
            collection=COLLECTION,
            requests=[ann_dense, ann_sparse],
            ranker=RRFRanker(k=60),
            limit=5,
            output_fields=OUTPUT_FIELDS,
        )
        print(f"  共 {len(hybrid_results)} 条结果")
        for i, r in enumerate(hybrid_results, 1):
            print(f"  [{i}] score={r.score:.4f}  {r.record.get('text', '')[:60]}…")


if __name__ == "__main__":
    asyncio.run(main())
