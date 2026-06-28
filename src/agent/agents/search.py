from agents import Agent, ModelSettings, Tool, set_tracing_disabled

from ..context import AgentContext
from ..tools.retrieval import search_tool
from ..tools.get_time import time_tool
from ..tools.caculator import calculate as calculator_tool
from .base import DEFAULT_INSTRUCTIONS, make_openai_model

set_tracing_disabled(disabled=True)


def make_search_agent(
    *,
    model: str,
    base_url: str,
    api_key: str,
    name: str = "SearchAgent",
    instructions: str = DEFAULT_INSTRUCTIONS,
    model_settings: ModelSettings | None = None,
    extra_tools: list[Tool] | None = None,
) -> Agent[AgentContext]:
    """构建 RAG 问答 Agent。

    Args:
        model:          LLM 模型名称
        base_url:       OpenAI 兼容接口地址（含 /v1）
        api_key:        接口密钥
        name:           Agent 名称
        instructions:   系统提示词，默认使用内置 RAG 助手 prompt
        model_settings: 额外模型参数（temperature、top_p 等），None 使用模型默认值
        extra_tools:    在 search 工具之外追加的工具列表
    """
    tools: list[Tool] = [search_tool, time_tool, calculator_tool, *(extra_tools or [])]

    return Agent[AgentContext](
        name=name,
        instructions=instructions,
        model=make_openai_model(model=model, base_url=base_url, api_key=api_key),
        model_settings=model_settings or ModelSettings(),
        tools=tools,
    )
