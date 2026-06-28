import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from agents import Agent, ItemHelpers, RawResponsesStreamEvent, RunItemStreamEvent, Runner, set_tracing_disabled
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem, ReasoningItem
from openai.types.responses import ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent

from .context import AgentContext
from .memory.base import BaseMemory

set_tracing_disabled(disabled=True)

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """流式输出块，包含类型标记与可选的工具调用元数据。

    type 取值：
      "thinking"      模型推理过程（逐 token 增量）
      "text"          最终回答（逐 token 增量）
      "tool_call"     工具调用请求（完整，非增量）
      "tool_output"   工具返回结果（完整）
      "thinking_done" 模型思考结束的停止信号
      "message_done"  模型回答结束的停止信号

    工具调用附加字段（仅 tool_call / tool_output 时有值）：
      tool_name    工具名称，如 "search"
      tool_args    已解析的参数 dict，如 {"query": "2000年水产品出口额"}
      tool_call_id 用于关联同一次调用的 call / output
    """

    content: str
    type: Literal["text", "thinking", "tool_call", "tool_output", "thinking_done","message_done"] = "text"
    tool_name: str | None = field(default=None)
    tool_args: dict[str, Any] | None = field(default=None)
    tool_call_id: str | None = field(default=None)


class AgentRunner:
    """Agent 运行器，统一封装流式与非流式调用，并自动维护对话历史。

    Args:
        agent:     已配置好工具和模型的 Agent 实例
        context:   本次会话的 AgentContext（含 retriever 等共享依赖）
        max_turns: 单次 run 允许的最大 Agent 轮次，防止工具调用死循环
    """

    def __init__(
        self,
        agent: Agent[AgentContext],
        context: AgentContext,
        max_turns: int = 15,
    ) -> None:
        self._agent = agent
        self._context = context
        self._max_turns = max_turns

    async def run(
        self,
        query: str,
        memory: BaseMemory | None = None,
    ) -> str:
        """非流式运行，等待完整结果后返回。

        Args:
            query:  本轮用户输入
            memory: 对话记忆，传入时自动读取历史并在结束后写入

        Returns:
            Agent 的最终文本输出
        """
        input_data: str | list[Any] = memory.get_input(query) if memory else query
        result = await Runner.run(
            self._agent,
            input_data,
            context=self._context,
            max_turns=self._max_turns,
        )
        if memory:
            memory.add_turn(result.to_input_list())
        output = result.final_output
        return str(output) if output is not None else ""

    async def stream_detailed(
        self,
        query: str,
        memory: BaseMemory | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式运行，按顺序 yield 思考、工具调用、工具输出、最终回答。

        事件顺序示例（单次检索）：
          thinking chunk(s)  →  tool_call  →  tool_output  →  thinking chunk(s)  →  text chunk(s)

        对话历史在流完全结束（或中断）后写入 memory。

        Args:
            query:  本轮用户输入
            memory: 对话记忆，传入时自动读取历史并在结束后写入

        Yields:
            StreamChunk，通过 .type 区分内容类别
        """
        input_data: str | list[Any] = memory.get_input(query) if memory else query
        streaming = Runner.run_streamed(
            self._agent,
            input_data,
            context=self._context,
            max_turns=self._max_turns,
        )
        try:
            async for event in streaming.stream_events():
                # ── 低级 token 事件 ────────────────────────────────────────
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        yield StreamChunk(content=event.data.delta, type="text")
                    elif isinstance(event.data, ResponseReasoningTextDeltaEvent):
                        yield StreamChunk(content=event.data.delta, type="thinking")

                # ── 高级 item 事件（工具调用完整包） ────────────────────────
                elif isinstance(event, RunItemStreamEvent):
                    item = event.item

                    if isinstance(item, ToolCallItem):
                        raw = item.raw_item
                        call_id = getattr(raw, "call_id", None)
                        tool_name = getattr(raw, "name", None)
                        args_str = getattr(raw, "arguments", None) or "{}"
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            args = {"_raw": args_str}
                        yield StreamChunk(
                            content=args_str,
                            type="tool_call",
                            tool_name=tool_name,
                            tool_args=args,
                            tool_call_id=call_id,
                        )

                    elif isinstance(item, ToolCallOutputItem):
                        raw = item.raw_item
                        call_id = (
                            raw.get("call_id")
                            if isinstance(raw, dict)
                            else getattr(raw, "call_id", None)
                        )
                        yield StreamChunk(
                            content=str(item.output),
                            type="tool_output",
                            tool_call_id=call_id,
                        )
                    elif isinstance(item, ReasoningItem):
                        text = "".join(c.text for c in (item.raw_item.content or []))
                        yield StreamChunk(content=text, type="thinking_done")
                    elif isinstance(item, MessageOutputItem):
                        yield StreamChunk(content=ItemHelpers.text_message_output(item), type="message_done")
        finally:
            if memory and streaming.is_complete:
                memory.add_turn(streaming.to_input_list())

    async def stream(
        self,
        query: str,
        memory: BaseMemory | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式运行，只 yield 最终回答的文本增量（忽略思考和工具事件）。

        向后兼容接口；需要工具事件或思考内容时请用 stream_detailed()。

        Args:
            query:  本轮用户输入
            memory: 对话记忆，传入时自动读取历史并在结束后写入

        Yields:
            最终回答的文本增量（delta）
        """
        async for chunk in self.stream_detailed(query, memory):
            if chunk.type == "text":
                yield chunk.content
