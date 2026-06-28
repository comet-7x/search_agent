from datetime import datetime
from zoneinfo import ZoneInfo

from agents import function_tool


@function_tool(
    name_override="get_time",
    description_override=("获取当前的准确时间（亚洲/上海时区），精确到秒，包含英文星期"),
)
def time_tool():
    """获取当前的准确时间（亚洲/上海时区），精确到秒，包含英文星期"""
    asian_tz = ZoneInfo("Asia/Shanghai")
    now = datetime.now(asian_tz)

    # %A 代表完整的星期名称，比如 Wednesday
    return f"The current time is: {now.strftime('%Y-%m-%d %H:%M:%S, %A')}"
