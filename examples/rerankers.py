"""
rerankers 模块示例

演示：
  1. 纯文本精排（text-only）
  2. 多模态精排（text + MinIO 图片）
  3. score_candidates（同步调用）
  4. 自定义 ScoreMultiModalParam 直接打分（ascore）
"""
import asyncio
import os
from dataclasses import dataclass
from typing import Any

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from src.basic.loaders import CompositeLoader, MinioLoader
from src.basic.rerankers.qwen_vl_reranker import (
    ChatCompletionContentPartTextParam,
    Qwen3VLReranker,
    ScoreMultiModalParam,
)

load_dotenv(find_dotenv())

# 配置
RERANKER_URL   = os.environ["RERANKER_BASE_URL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
RERANKER_KEY   = os.environ["RERANKER_API_KEY"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET   = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE   = os.getenv("MINIO_SECURE", "false").lower() == "true"

QUERY = "2000年我国水产品对外贸易总量"

CANDIDATES_RAW = [
    {
        "text": "2000年，我国水产品对外贸易总量达到了历史新高，出口量大幅增长。",
        "image_url": "minio://test-bucket/test_multi_rag/output/7/2000/page_images/page_0.jpg",
        "score": 0.91,
    },
    {
        "text": "该报告分析了1990-2010年间我国农业出口结构的演变趋势。",
        "image_url": "minio://test-bucket/test_multi_rag/output/7/2000/page_images/page_1.jpg",
        "score": 0.87,
    },
    {
        "text": "卡尔曼滤波算法在导航系统中的应用已有数十年历史。",
        "image_url": None,
        "score": 0.72,
    },
]


@dataclass
class Candidate:
    """满足 RerankCandidate 协议的简单容器。"""
    record: dict[str, Any]
    score: float


def make_candidates() -> list[Candidate]:
    return [Candidate(record={"text": c["text"], "image_url": c["image_url"]}, score=c["score"])
            for c in CANDIDATES_RAW]


def make_minio_loader() -> CompositeLoader:
    return CompositeLoader([
        MinioLoader(Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE))
    ])


def demo_sync() -> None:
    """score_candidates 是同步包装，只能在没有运行中的事件循环时调用。"""
    print("── 3. score_candidates（同步调用）─────────────────────")
    text_reranker = Qwen3VLReranker(
        base_url=RERANKER_URL,
        model_name=RERANKER_MODEL,
        api_key=RERANKER_KEY,
        text_field="text",
        image_field=None,
    )
    scores_sync = text_reranker.score_candidates(QUERY, make_candidates())
    print(f"  同步分数: {[round(s, 4) for s in scores_sync]}")


async def main() -> None:
    text_reranker = Qwen3VLReranker(
        base_url=RERANKER_URL,
        model_name=RERANKER_MODEL,
        api_key=RERANKER_KEY,
        text_field="text",
        image_field=None,
    )
    candidates = make_candidates()

    # 1. 纯文本精排
    print("── 1. 纯文本精排（text-only）──────────────────────────")
    scores = await text_reranker.ascore_candidates(QUERY, candidates)
    for c, s in zip(candidates, scores):
        print(f"  [{s:.4f}] {c.record['text'][:50]}…")

    # 2. 多模态精排（text + MinIO 图片）
    print("\n── 2. 多模态精排（text + image）───────────────────────")
    mm_reranker = Qwen3VLReranker(
        base_url=RERANKER_URL,
        model_name=RERANKER_MODEL,
        api_key=RERANKER_KEY,
        text_field="text",
        image_field="image_url",
        image_loader=make_minio_loader(),
    )
    scores_mm = await mm_reranker.ascore_candidates(QUERY, candidates)
    for c, s in zip(candidates, scores_mm):
        print(f"  [{s:.4f}] {c.record['text'][:50]}… img={'有' if c.record.get('image_url') else '无'}")

    # 4. 直接构造 ScoreMultiModalParam 调用 ascore
    print("\n── 4. 直接调用 ascore（手动构造输入）──────────────────")
    inputs = [
        ScoreMultiModalParam(content=[ChatCompletionContentPartTextParam(text=c["text"])])
        for c in CANDIDATES_RAW
    ]
    scores_direct = await text_reranker.ascore(QUERY, inputs)
    print(f"  直接打分: {[round(s, 4) for s in scores_direct]}")
    assert scores_direct == scores, "直接打分与 ascore_candidates 结果不一致！"
    print("  与 ascore_candidates 结果一致 ✓")


if __name__ == "__main__":
    # 场景 3 在同步上下文中运行
    demo_sync()
    print()
    asyncio.run(main())
