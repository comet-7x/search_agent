"""
单轮问答测试

测试场景：
  - 非流式（runner.run）：等待完整答案后输出
  - 流式（runner.stream_detailed）：实时显示思考过程和最终回答
"""

import asyncio

from common import build_retriever, build_runner, print_stream

QUESTIONS = [
    # "2000年我国水产品对外贸易总量是多少？",
    # "2023年全国水产品贸易平均单价约为多少美元/吨？",
    # "2023年我国水产品进出口是顺差还是逆差？",
    # # 数值统计与计算类
    # "2023年全国淡水养殖甲壳类中，虾占甲壳类产量的比例是多少？",
    # "2023年淡水虾中，克氏原螯虾占比是多少？",
    # "2023年海水养殖甲壳类产量与淡水养殖甲壳类产量相差多少？",
    "2023年全国水产品贸易平均单价约为多少美元/吨？",
    # 法规类
    # "进口厄瓜多尔冷冻南美白虾是否属于《进口厄瓜多尔冷冻南美白虾检验检疫要求》的适用范围？该要求参照了哪些法律法规和双边文件？产品准入、企业注册和检验检疫合规中最具有决定性的因素分别是什么？该公告是哪年发布的，当前是否仍可作为进口准入判断依据？",
]


async def main() -> None:
    retriever = build_retriever()
    runner = build_runner(retriever)

    # ── 非流式 ────────────────────────────────────────────────────────────────
    # print("═" * 60)
    # print("非流式单轮问答（runner.run）")
    # print("═" * 60)
    # for q in QUESTIONS:
    #     print(f"\nQ: {q}")
    #     answer = await runner.run(q)
    #     print(f"A: {answer}")

    # ── 流式（含思考） ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("流式单轮问答（runner.stream_detailed）")
    print("═" * 60)
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        await print_stream(runner, q)


if __name__ == "__main__":
    asyncio.run(main())
