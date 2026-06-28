"""
agent 模块示例

演示：
  1. 非流式问答（单轮）
  2. 流式问答（单轮）
  3. 多轮对话（BufferMemory 维护历史）
  4. QueryRewriter 多路扩展检索
"""
import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from src.agent import AgentContext, AgentRunner, BufferMemory, QueryRewriter, StreamChunk, make_search_agent
from src.basic.database.vector.milvus import AsyncMilvusStorage
from src.basic.embeddings.qwen3_vl_embedding import Qwen3VLEmbeddingModel
from src.basic.loaders import CompositeLoader, MinioLoader
from src.basic.rerankers.qwen_vl_reranker import Qwen3VLReranker
from src.retrieval.base import AnnRequestConfig
from src.retrieval.pipeline import HybridRetriever

load_dotenv(find_dotenv())

# ── 配置 ──────────────────────────────────────────────────────────────────────
MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_DB  = os.environ["MILVUS_DB"]

EMBEDDING_URL   = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_KEY   = os.environ["EMBEDDING_API_KEY"]

RERANKER_URL   = os.environ["RERANKER_BASE_URL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
RERANKER_KEY   = os.environ["RERANKER_API_KEY"]

LLM_URL   = os.environ["LLM_BASE_URL"]
LLM_KEY   = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET   = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE   = os.getenv("MINIO_SECURE", "false").lower() == "true"

COLLECTION    = "commercial"
TEXT_FIELD    = "text"
IMAGE_FIELD   = "image_url"
OUTPUT_FIELDS = [TEXT_FIELD, IMAGE_FIELD, "file_id", "page_index"]


# ── 构建检索器 ────────────────────────────────────────────────────────────────
def build_retriever() -> HybridRetriever:
    minio_loader = CompositeLoader([
        MinioLoader(Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE))
    ])
    return HybridRetriever(
        storage=AsyncMilvusStorage(uri=MILVUS_URI, db_name=MILVUS_DB),
        embedder=Qwen3VLEmbeddingModel(base_url=EMBEDDING_URL, model_name=EMBEDDING_MODEL, api_key=EMBEDDING_KEY),
        collection=COLLECTION,
        ann_requests=[
            AnnRequestConfig(
                field="text_embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                query_type="embed_text",
            )
        ],
        output_fields=OUTPUT_FIELDS,
        text_extractor=lambda r: r.get(TEXT_FIELD, ""),
        reranker=Qwen3VLReranker(
            base_url=RERANKER_URL,
            model_name=RERANKER_MODEL,
            api_key=RERANKER_KEY,
            text_field=TEXT_FIELD,
            image_field=IMAGE_FIELD,
            image_loader=minio_loader,
        ),
        recall_limit=20,
        score_threshold=0.3,
    )


async def main() -> None:
    retriever = build_retriever()
    context   = AgentContext(retriever=retriever, retrieval_top_k=5)
    agent     = make_search_agent(model=LLM_MODEL, base_url=LLM_URL, api_key=LLM_KEY)
    runner    = AgentRunner(agent=agent, context=context, max_turns=10)

    # 1. 非流式单轮问答
    print("── 1. 非流式单轮问答 ───────────────────────────────────")
    q1 = "2000年我国水产品对外贸易总量是多少？"
    print(f"Q: {q1}")
    answer = await runner.run(q1)
    print(f"A: {answer}\n")

    # 2. 流式单轮问答
    print("── 2. 流式单轮问答 ─────────────────────────────────────")
    q2 = "请简要介绍一下我国水产品加工行业的发展情况。"
    q2 = "2023年全国水产品贸易平均单价约为多少美元/吨？"
    thinking_buf: list[str] = []
    print("<thinking>", end="", flush=True)
    async for chunk in runner.stream_detailed(q2):
        if chunk.type == "thinking":
            thinking_buf.append(chunk.content)
            print(chunk.content, end="", flush=True)
        else:
            if thinking_buf:  # 第一个 text chunk 前关闭 thinking 块
                print("</thinking>\n\nA: ", end="", flush=True)
                thinking_buf.clear()
            print(chunk.content, end="", flush=True)
    print("\n")

    # 3. 多轮对话（BufferMemory 维护历史）
    print("── 3. 多轮对话 ─────────────────────────────────────────")
    memory = BufferMemory(max_items=60)

    turns = [
        "2000年我国水产品出口额大约是多少？",
        "那进口额呢？",
        "请把上面两个数字做个对比总结。",
    ]
    for i, q in enumerate(turns, 1):
        print(f"Turn {i} Q: {q}")
        print(f"Turn {i} A: ", end="", flush=True)
        async for chunk in runner.stream(q, memory=memory):
            print(chunk, end="", flush=True)
        print(f"\n（历史条目数: {memory.item_count}）\n")

    # 4. QueryRewriter 多路扩展检索
    print("── 4. QueryRewriter 多路扩展检索 ───────────────────────")
    rewriter = QueryRewriter(model=LLM_MODEL, base_url=LLM_URL, api_key=LLM_KEY)
    rw_context = AgentContext(
        retriever=retriever,
        retrieval_top_k=5,
        query_rewriter=rewriter,
        expand_n=3,
    )
    rw_agent  = make_search_agent(model=LLM_MODEL, base_url=LLM_URL, api_key=LLM_KEY)
    rw_runner = AgentRunner(agent=rw_agent, context=rw_context, max_turns=10)

    # 演示：改写器会把这个模糊问题扩展为多个检索角度
    q4 = "近年来我国渔业经济的整体发展趋势如何？"
    print(f"Q: {q4}")
    expanded = await rewriter.expand(q4, n=3)
    print("扩展查询：")
    for eq in expanded:
        print(f"  · {eq}")
    print("\nA: ", end="", flush=True)
    async for chunk in rw_runner.stream(q4):
        print(chunk, end="", flush=True)
    print("\n")


    # 5. stream_detailed() — 同时展示思考过程和最终回答
    print("── 5. stream_detailed（思考 + 回答）────────────────────")
    q5 = "2023年我国水产品进出口是顺差还是逆差？"
    print(f"Q: {q5}")

    thinking_buf: list[str] = []
    print("<thinking>", end="", flush=True)
    async for chunk in runner.stream_detailed(q5):
        if chunk.type == "thinking":
            thinking_buf.append(chunk.content)
            print(chunk.content, end="", flush=True)
        else:
            if thinking_buf:  # 第一个 text chunk 前关闭 thinking 块
                print("</thinking>\n\nA: ", end="", flush=True)
                thinking_buf.clear()
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
