from typing import Any

from .base import BaseMemory


class BufferMemory(BaseMemory):
    """滑动窗口对话历史记忆。

    存储 result.to_input_list() 的完整输出，当历史条目超过 max_items 时
    从头部丢弃最旧的条目，始终保留最近的上下文。

    Args:
        max_items: 保留的最大历史条目数。每轮对话通常产生 3–6 条，
                   默认 60 条约等于 10–20 轮对话。None 表示不限制。
    """

    def __init__(self, max_items: int | None = 60) -> None:
        self._max_items = max_items
        self._history: list[Any] = []
        self._turn_count: int = 0

    def add_turn(self, history: list[Any]) -> None:
        self._history = list(history)
        if self._max_items is not None and len(self._history) > self._max_items:
            self._history = self._history[-self._max_items :]
        self._turn_count += 1

    def get_input(self, new_message: str) -> str | list[Any]:
        if not self._history:
            return new_message
        return self._history + [{"role": "user", "content": new_message}]

    def clear(self) -> None:
        self._history = []
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def item_count(self) -> int:
        """当前历史中的条目数，用于调试。"""
        return len(self._history)
