"""
多轮对话测试

测试场景：
  - 使用 BufferMemory 维护对话历史
  - 每轮均通过 stream_detailed 显示思考过程和回答
  - 验证跨轮指代（"那进口额呢？"）是否被正确理解
"""
import asyncio

from common import build_retriever, build_runner, print_stream
from src.agent import BufferMemory

TURNS = [
    "2000年我国水产品出口额大约是多少？",
    "那进口额呢？",
    "请把上面两个数字做个对比总结。",
]


async def main() -> None:
    retriever = build_retriever()
    runner    = build_runner(retriever)
    memory    = BufferMemory(max_items=60)

    print("═" * 60)
    print("多轮对话（BufferMemory + stream_detailed）")
    print("═" * 60)

    for i, q in enumerate(TURNS, 1):
        print(f"\n── Turn {i} ──────────────────────────────────────────────")
        print(f"Q: {q}")
        await print_stream(runner, q, memory=memory)
        print(f"（已存储历史条目数: {memory.item_count}）")


if __name__ == "__main__":
    asyncio.run(main())
