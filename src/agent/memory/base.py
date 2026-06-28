from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """对话记忆抽象基类。

    子类负责维护对话历史，并将其转换为下一轮 Runner.run() 所需的输入格式。
    """

    @abstractmethod
    def add_turn(self, history: list[Any]) -> None:
        """在一轮对话结束后，将 result.to_input_list() 的结果写入记忆。"""
        ...

    @abstractmethod
    def get_input(self, new_message: str) -> str | list[Any]:
        """构造下一轮的输入：历史消息列表 + 新用户消息，空记忆时直接返回字符串。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有历史记录，重置为初始状态。"""
        ...

    @property
    @abstractmethod
    def turn_count(self) -> int:
        """已完成的对话轮数。"""
        ...
