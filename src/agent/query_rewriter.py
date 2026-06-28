from openai import AsyncOpenAI
from pydantic import BaseModel


class QueryExpansion(BaseModel):
    queries: list[str]


class QueryRewriter:
    """用 LLM 将检索查询扩展为多个角度，提升召回率。

    使用 Pydantic structured output，schema 自动生成，响应自动反序列化。
    实例通常挂载在 AgentContext.query_rewriter 上，由 search_tool 透明使用。

    Args:
        model:    用于改写的 LLM 模型名称
        base_url: OpenAI 兼容接口地址
        api_key:  接口密钥
    """

    _EXPAND_SYSTEM = (
        "你是一名专业的向量检索查询改写助手。\n"
        "任务：将用户的原始问题改写为多个不同视角的检索查询，"
        "帮助克服基于向量距离相似度搜索的局限性，从多个角度提升文档召回覆盖。\n\n"
        "改写策略：\n"
        "- 使用不同的措辞、同义词或相关概念，同时保留原始查询的核心意图\n"
        "- 替换上位概念或领域术语（如'水产品'→'渔业'，'平均单价'→'价格'）\n"
        "- 提取核心实体，构造简短精炼的关键词组合\n"
        "- 将复杂问题拆解为可独立检索的子问题"
    )

    _NO_THINK_BODY: dict[str, object] = {"chat_template_kwargs": {"enable_thinking": False}}

    def __init__(self, *, model: str, base_url: str, api_key: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def expand(self, query: str, n: int = 3) -> list[str]:
        """将原始查询扩展为 n 个检索变体（含原始查询本身）。

        Args:
            query: 原始检索查询
            n:     期望生成的变体数量（不含原始查询）

        Returns:
            [原始查询] + [最多 n 个变体]，共 1~n+1 条
        """
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": self._EXPAND_SYSTEM},
                {"role": "user", "content": f"将以下问题改写为{n}个不同的检索查询：\n{query}"},
            ],
            temperature=0.3,
            max_tokens=256,
            response_format=QueryExpansion,
            extra_body=self._NO_THINK_BODY,
        )
        parsed = response.choices[0].message.parsed
        variants = (parsed.queries if parsed else [])[:n]
        return [query] + [v for v in variants if v != query]
