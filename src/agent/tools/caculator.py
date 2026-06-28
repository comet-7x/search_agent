import ast
import json
import math
import operator
import statistics

from agents import function_tool

# 允许的二元运算符
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 防止指数爆炸(如 2**99999 卡死/内存炸)
_MAX_POW_EXPONENT = 1000


def _safe_pow(base: float, exp: float) -> float:
    if abs(exp) > _MAX_POW_EXPONENT:
        raise ValueError(f"指数 {exp} 超出允许范围 (|exp| <= {_MAX_POW_EXPONENT})")
    return operator.pow(float(base), float(exp))


# 允许调用的白名单函数
_FUNCTIONS = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": _safe_pow,
    "sqrt": math.sqrt,
    "mean": statistics.fmean,
}


def _eval(node: ast.AST):
    """递归求值,仅放行白名单节点;遇到其它节点一律拒绝。"""
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        # 仅允许数值常量,拒绝 bool / str / bytes / None 等
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量: {node.value!r}")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return _safe_pow(left, right)
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ValueError(f"不支持的函数: {name}")
        if node.keywords:
            raise ValueError("不支持关键字参数,请使用位置参数")
        args = [_eval(a) for a in node.args]
        return _FUNCTIONS[node.func.id](*args)

    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@function_tool(
    name_override="calculator",
    description_override=(
        "对纯数字算术表达式求值，返回精确结果。"
        "凡是需要计算的数值（均价、占比、增长率、合计等）必须用本工具，禁止心算或估算。"
        "调用前先把所有量换算到基本单位（元、吨、个）的纯数字再代入表达式，"
        "不要在表达式中保留'万''亿'等中文单位。"
    ),
)
def calculate(expression: str) -> str:
    r"""对纯数字算术表达式求值并返回精确结果。任何需要计算的数值都必须用本工具,禁止心算。

    调用前必须先把所有量换算到**基本单位**(如 元、吨、个)的纯数字再代入表达式,
    不要在表达式里保留"万""亿"等中文单位(单位换算出错是常见错误来源)。
    结果的单位与文字解释由你在回答中补充说明。

    支持的运算:
    - 四则运算 ``+ - * / // %``、幂 ``**``、括号、一元负号
    - 函数:``round(x, n)``、``abs(x)``、``min(...)``、``max(...)``、
      ``sum([...])``、``pow(x, y)``、``sqrt(x)``、``mean([...])``

    Args:
        expression (str): 纯数字算术表达式。示例:
            - 均价(美元/吨): ``"44237000000 / 10560500"`` → 4188.9
            - 占比(%): ``"round(255.97 / 399.46 * 100, 2)"`` → 64.08
            - 年均增长率(%): ``"round(((21108.29 / 8962.81) ** (1 / 9) - 1) * 100, 2)"`` → 10.0
            - 多年合计: ``"sum([8962.81, 10011.65, 11256.08])"``

    Returns:
        str: JSON 字符串。成功时为 ``{"expression": "...", "result": 4188.9}``;
             失败时为 ``{"expression": "...", "error": "错误描述"}``。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        if isinstance(result, list):
            raise ValueError("表达式结果必须是单个数值,而非列表")
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "除数为零"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)}, ensure_ascii=False)
