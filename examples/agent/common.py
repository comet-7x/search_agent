"""
共享配置与工具函数，供各 agent 示例文件导入。
运行方式：从项目根目录执行 python examples/agent/<test_file>.py
"""

import os

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from agents import ModelSettings

from src.agent import AgentContext, AgentRunner, BufferMemory, QueryRewriter, make_search_agent
from src.basic.database.vector.milvus import AsyncMilvusStorage
from src.basic.embeddings.qwen3_vl_embedding import Qwen3VLEmbeddingModel
from src.basic.loaders import CompositeLoader, MinioLoader
from src.basic.rerankers.qwen_vl_reranker import Qwen3VLReranker
from src.retrieval.base import AnnRequestConfig
from src.retrieval.pipeline import HybridRetriever

load_dotenv(find_dotenv())

# ── 服务端点 ──────────────────────────────────────────────────────────────────
MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_DB = os.environ["MILVUS_DB"]

EMBEDDING_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_KEY = os.environ["EMBEDDING_API_KEY"]

RERANKER_URL = os.environ["RERANKER_BASE_URL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
RERANKER_KEY = os.environ["RERANKER_API_KEY"]

LLM_URL = os.environ["LLM_BASE_URL"]
LLM_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

COLLECTION = "commercial"
TEXT_FIELD = "text"
IMAGE_FIELD = "image_url"
OUTPUT_FIELDS = [TEXT_FIELD, IMAGE_FIELD, "file_id", "page_index"]


# ── 工厂函数 ──────────────────────────────────────────────────────────────────
def build_retriever() -> HybridRetriever:
    minio_loader = CompositeLoader(
        [
            MinioLoader(
                Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS,
                    secret_key=MINIO_SECRET,
                    secure=MINIO_SECURE,
                )
            )
        ]
    )
    return HybridRetriever(
        storage=AsyncMilvusStorage(uri=MILVUS_URI, db_name=MILVUS_DB),
        embedder=Qwen3VLEmbeddingModel(
            base_url=EMBEDDING_URL, model_name=EMBEDDING_MODEL, api_key=EMBEDDING_KEY
        ),
        collection=COLLECTION,
        ann_requests=[
            AnnRequestConfig(
                field="text_embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                query_type="embed_text",
                limit=50,
            ),
            AnnRequestConfig(
                field="text_sparse_embedding",
                search_params={"metric_type": "BM25"},
                query_type="text",
                limit=50,
            ),
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
        recall_limit=25,
        score_threshold=0.2,
    )


def build_runner(
    retriever: HybridRetriever,
    *,
    query_rewriter: QueryRewriter | None = None,
    expand_n: int = 3,
    max_turns: int = 10,
) -> AgentRunner:
    context = AgentContext(
        retriever=retriever,
        retrieval_top_k=5,
        query_rewriter=query_rewriter,
        expand_n=expand_n,
    )
    agent = make_search_agent(
        model=LLM_MODEL,
        base_url=LLM_URL,
        api_key=LLM_KEY,
        model_settings=ModelSettings(parallel_tool_calls=True),
    )
    return AgentRunner(agent=agent, context=context, max_turns=max_turns)


def build_query_rewriter() -> QueryRewriter:
    return QueryRewriter(model=LLM_MODEL, base_url=LLM_URL, api_key=LLM_KEY)


# ── 流式打印辅助 ──────────────────────────────────────────────────────────────
async def print_stream(
    runner: AgentRunner,
    query: str,
    memory: BufferMemory | None = None,
) -> None:
    """流式运行并打印思考、工具调用、工具输出与最终回答。

    输出格式：
        ── [思考] ──
        <推理内容>
        ── [工具调用] search ──
        query = "..."
        ── [工具输出] ──
        [1] score=0.95  <摘要>...
        ── [回答] ──
        <最终回答>
    """
    prev_type: str | None = None
    text_started = False  # [回答] 标题是否已打开

    async for chunk in runner.stream_detailed(query, memory):
        t = chunk.type

        if t == "thinking":
            if prev_type != "thinking":
                print("\n[思考]", flush=True)
            print(chunk.content, end="", flush=True)

        elif t == "tool_call":
            text_started = False
            print(f"\n[工具调用] {chunk.tool_name}", flush=True)
            if chunk.tool_args:
                for k, v in chunk.tool_args.items():
                    print(f"  {k} = {v!r}", flush=True)

        elif t == "tool_output":
            text_started = False
            print(f"\n[工具输出] call_id={chunk.tool_call_id}", flush=True)
            preview = chunk.content[:20].replace("\n", " ")
            print(f"  {preview}{'...' if len(chunk.content) > 20 else ''}", flush=True)

        elif t == "text":
            if not text_started:
                if not chunk.content.strip():
                    # ① 回答尚未开始时的空白 delta（模型换行），丢弃且不更新 prev_type
                    continue
                print("\n[回答]", flush=True)
                text_started = True
            print(chunk.content, end="", flush=True)

        elif t == "thinking_done":
            # ③ RunItemStreamEvent 比文本 delta 晚触发时，prev_type 已是 "text"，说明是迟到事件
            if prev_type == "thinking":
                print("\n[思考结束]", flush=True)

        elif t == "message_done":
            text_started = False
            # ② 工具调用前的 MessageOutputItem 内容为空，不打印
            if chunk.content.strip():
                print("\n[回答结束]", flush=True)

        prev_type = t

    print()
