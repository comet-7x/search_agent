"""
QueryRewriter 多路扩展检索测试

测试场景：
  - 展示原始查询被扩展为 N 个变体
  - 并行检索所有变体后合并去重
  - 通过 stream_detailed 显示思考过程和回答
  - 对比：有 / 无 rewriter 的检索结果差异
"""

import asyncio

from common import build_query_rewriter, build_retriever, build_runner, print_stream

QUESTIONS = [
    "2000年我国水产品对外贸易总量是多少？",
    # "2023年全国水产品贸易平均单价约为多少美元/吨？",
    # "2023年我国水产品进出口是顺差还是逆差？",
    # 数值统计与计算类
    # "2023年全国淡水养殖甲壳类中，虾占甲壳类产量的比例是多少？",
    # "2023年淡水虾中，克氏原螯虾占比是多少？",
    # "2023年海水养殖甲壳类产量与淡水养殖甲壳类产量相差多少？",
    # "2010年到2019年的全国渔民人均纯收入",
    # "2015年-2025年中甲壳类生产养殖每年产量是多少？增长率是多少？",
    # "搜索最近10年的全国渔业经济总产值",
    # 法规类
    # "进口厄瓜多尔冷冻南美白虾是否属于《进口厄瓜多尔冷冻南美白虾检验检疫要求》的适用范围？该要求参照了哪些法律法规和双边文件？产品准入、企业注册和检验检疫合规中最具有决定性的因素分别是什么？该公告是哪年发布的，当前是否仍可作为进口准入判断依据？",
]


async def main() -> None:
    retriever = build_retriever()
    rewriter = build_query_rewriter()

    runner_plain = build_runner(retriever)
    runner_rw = build_runner(retriever, query_rewriter=rewriter, expand_n=3)

    for q in QUESTIONS:
        print("═" * 60)
        print(f"Q: {q}")

        # 展示扩展查询
        # expanded = await rewriter.expand(q, n=3)
        # print("\n扩展查询：")
        # for eq in expanded:
        #     marker = "▶ [原始]" if eq == q else "  ·"
        #     print(f"  {marker} {eq}")

        # # 无 rewriter
        # print("\n── [无 rewriter] ──────────────────────────────────────")
        # await print_stream(runner_plain, q)

        # 有 rewriter（并行多路检索）
        print("\n── [有 rewriter，并行 4 路检索] ───────────────────────")
        await print_stream(runner_rw, q)
        print()


if __name__ == "__main__":
    asyncio.run(main())
