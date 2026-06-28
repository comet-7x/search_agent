import asyncio

from agents import RunContextWrapper, function_tool

from ...retrieval.base import RetrievalResult
from ..context import AgentContext


def _dedup(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    """按 metadata['id'] 去重，保留每个文档的最高分，返回按分数降序的 top_k 条。"""
    seen: dict[int, RetrievalResult] = {}
    for r in results:
        doc_id: int = r.metadata["id"]
        if doc_id not in seen or r.score > seen[doc_id].score:
            seen[doc_id] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]


@function_tool(
    name_override="search",
    description_override=(
        "在知识库中检索与用户问题最相关的文档片段。"
        "当需要回答具体问题、查找事实数据或引用文献时调用此工具。"
        "返回按相关度排序的文本片段，每条包含相关性得分。"
        "若结果质量不佳（分数偏低或内容无关），应尝试换用不同关键词再次调用。"
    ),
)
async def search_tool(context: RunContextWrapper[AgentContext], query: str) -> str:
    """
    Args:
        query: 针对问题提炼的检索关键词或短句，应简洁精准
    """
    ctx = context.context
    top_k = ctx.retrieval_top_k

    if ctx.query_rewriter is not None:
        queries = await ctx.query_rewriter.expand(query, n=ctx.expand_n)
        print("\n扩展查询：")
        for eq in queries:
            marker = "▶ [原始]" if eq == query else "  ·"
            print(f"  {marker} {eq}")
        all_results = await asyncio.gather(
            *[ctx.retriever.aretrieve(q, top_k=top_k) for q in queries]
        )
        merged = [r for batch in all_results for r in batch]
        results = _dedup(merged, top_k)
    else:
        results = await ctx.retriever.aretrieve(query, top_k=top_k)

    if not results:
        return "知识库中未找到与该查询相关的内容。请尝试换用不同的关键词重新检索。"

    parts = [f"[{i}] 相关度={r.score:.3f}\n{r.text}" for i, r in enumerate(results, 1)]
    return "\n\n---\n\n".join(parts)

