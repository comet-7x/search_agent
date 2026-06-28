# `@function_tool` 学习笔记

> 基于 openai-agents SDK（`.venv/lib/python3.13/site-packages/agents/tool.py`）源码阅读整理。

---

## 一、它做了什么

`@function_tool` 是一个装饰器，将普通 Python 函数变成 Agent 可以调用的工具。核心工作分两阶段：

**装饰时**（模块加载时执行一次）

- 用 `inspect.signature` 扫描函数签名，生成 JSON Schema
- 将签名映射为 Pydantic 模型（`params_pydantic_model`），用于运行时验证
- 如果第一参数类型是 `RunContextWrapper[X]`，标记 `takes_context=True`

**工具被调用时**（每次 LLM 触发执行）

```
LLM 返回 JSON 字符串
  → json.loads
  → Pydantic 验证 (params_pydantic_model(**json_data))
  → to_call_args → 展开成 (args, kwargs)
  → 调用实际函数
  → 返回值转字符串，送回 LLM
```

---

## 二、函数签名规则

### 2.1 普通参数 → LLM 的输入字段

```python
@function_tool
def search(query: str, top_k: int = 5) -> str:
    ...
```

LLM 看到的 JSON Schema：

```json
{
  "name": "search",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "integer"}
    },
    "required": ["query", "top_k"]   ← strict_mode=True 时所有参数均 required
  }
}
```

> **注意**：`strict_mode=True`（默认）下带默认值的参数仍会出现在 `required` 中。
> 真正的"可选参数"应写成 `Optional[str] = None`，而非 `str = "default"`。

### 2.2 `RunContextWrapper` — 注入上下文，不暴露给 LLM

```python
from agents import RunContextWrapper, function_tool

@function_tool
async def search_tool(context: RunContextWrapper[AgentContext], query: str) -> str:
    ctx = context.context   # 拿到 AgentContext 实例
    results = await ctx.retriever.aretrieve(query)
    ...
```

- 第一参数必须是 `RunContextWrapper[MyContextType]`
- SDK 在生成 Schema 时自动跳过该参数（LLM 不知道它的存在）
- 调用时由 SDK 自动注入，开发者无需操心

### 2.3 同步 vs 异步

```python
# 同步函数 → SDK 内部走 asyncio.to_thread，不阻塞事件循环
@function_tool
def calculate(expression: str) -> str:
    ...

# 异步函数 → 直接 await
@function_tool
async def search_tool(context: RunContextWrapper[AgentContext], query: str) -> str:
    ...
```

两种写法均可，SDK 在 `_on_invoke_tool_impl` 里通过 `inspect.iscoroutinefunction` 自动分支。

---

## 三、所有参数详解

```python
@function_tool(
    name_override=None,
    description_override=None,
    docstring_style=None,
    use_docstring_info=True,
    failure_error_function=...,
    strict_mode=True,
    is_enabled=True,
    needs_approval=False,
    tool_input_guardrails=None,
    tool_output_guardrails=None,
    timeout=None,
    timeout_behavior="error_as_result",
    timeout_error_function=None,
    defer_loading=False,
    custom_data_extractor=None,
)
```

| 参数                       | 类型                                               | 说明                                                                                     |
| -------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `name_override`          | `str \| None`                                     | 工具名（LLM 看到的），不填则用函数名                                                     |
| `description_override`   | `str \| None`                                     | 工具描述，不填则取 docstring 第一段。**建议始终填写**，描述直接影响 LLM 的调用决策 |
| `docstring_style`        | `"google" \| "numpy" \| "sphinx" \| None`           | docstring 风格，`None` 时自动检测                                                      |
| `use_docstring_info`     | `bool`                                           | `False` 时完全忽略 docstring（不提取描述和参数说明）                                   |
| `strict_mode`            | `bool`                                           | OpenAI Structured Outputs 严格模式，**强烈建议保持 True**                          |
| `is_enabled`             | `bool \| (ctx, agent) -> bool`                    | 动态控制工具可见性；返回`False` 时该工具对 LLM 隐藏                                    |
| `needs_approval`         | `bool \| (ctx, args, call_id) -> Awaitable[bool]` | 调用前暂停，等待人工审批（`RunState.approve/reject`）                                  |
| `tool_input_guardrails`  | `list[ToolInputGuardrail]`                       | 参数级 guardrail，可在执行前拦截危险调用                                                 |
| `tool_output_guardrails` | `list[ToolOutputGuardrail]`                      | 输出级 guardrail，可过滤敏感返回值                                                       |
| `timeout`                | `float \| None`                                   | 工具执行超时秒数                                                                         |
| `timeout_behavior`       | `"error_as_result" \| "raise_exception"`          | 超时后是将错误信息返给 LLM，还是直接抛异常终止 run                                       |
| `failure_error_function` | `(ctx, error) -> str \| None`                     | 自定义异常 → 错误文本的格式化逻辑；传`None` 则异常直接上抛                            |
| `defer_loading`          | `bool`                                           | Responses API 工具搜索懒加载，通常无需关心                                               |
| `custom_data_extractor`  | `callback \| None`                                | 在`ToolCallOutputItem` 上附加仅 SDK 可见的元数据，不发给 LLM                           |

---

## 四、docstring 驱动参数描述

工具的**参数描述**同样影响 LLM 的调用质量。SDK 从 docstring `Args:` 节提取：

```python
@function_tool(name_override="search")
async def search_tool(context: RunContextWrapper[AgentContext], query: str) -> str:
    """
    Args:
        query: 针对问题提炼的检索关键词或短句，应简洁精准
    """
```

生成的 schema：

```json
"query": {
  "type": "string",
  "description": "针对问题提炼的检索关键词或短句，应简洁精准"
}
```

> 参数描述越具体，LLM 越能构造高质量的调用参数。

---

## 五、本项目中的三个工具对照

### `search_tool` — 异步 + 上下文注入

```python
@function_tool(
    name_override="search",
    description_override="在知识库中检索...",
)
async def search_tool(context: RunContextWrapper[AgentContext], query: str) -> str:
    ctx = context.context
    results = await ctx.retriever.aretrieve(query)
    ...
```

要点：需要访问 `AgentContext`（retriever、query_rewriter 等），必须用 `RunContextWrapper`。

### `time_tool` — 同步 + 无上下文

```python
@function_tool(
    name_override="get_time",
    description_override="获取当前的准确时间（亚洲/上海时区）...",
)
def time_tool() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return now.strftime("%Y-%m-%d %H:%M:%S, %A")
```

要点：纯计算，无需上下文，同步即可。SDK 自动 `asyncio.to_thread` 包装。

### `calculator_tool` — 同步 + 安全沙箱

```python
@function_tool(
    name_override="calculator",
    description_override="对纯数字算术表达式求值...",
)
def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)   # 白名单 AST 求值，无 eval() 注入风险
    return json.dumps({"expression": expression, "result": result})
```

要点：用 AST 白名单代替 `eval()`，避免代码注入；返回 JSON 字符串，让 LLM 能解析结果和错误。

---

## 六、常见陷阱

**1. `description_override=""` 空字符串**

LLM 不知道该工具的用途，几乎不会主动调用它。始终填写有意义的描述。

**2. strict_mode + 默认值**

```python
# 错误：LLM 仍需填写 top_k，"默认值"对 LLM 无意义
def search(query: str, top_k: int = 5) -> str: ...

# 正确：明确声明可选
def search(query: str, top_k: Optional[int] = None) -> str:
    top_k = top_k or 5
```

**3. 同步函数中的阻塞 I/O**

SDK 确实会用 `asyncio.to_thread` 包装，但如果你在同步函数里调用了异步库（如 `asyncio.run`），会死锁。I/O 密集的工具应写成 `async def`。

**4. 返回值类型**

工具返回值会被 `str()` 转换后送给 LLM。如果返回复杂对象，应主动序列化成可读字符串（JSON 或纯文本），而不是依赖默认的 `__repr__`。
