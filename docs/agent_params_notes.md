# Agent 参数笔记

## Agent 完整参数列表

### 继承自 AgentBase

| 参数                    | 类型                | 默认值   | 说明                                                |
| ----------------------- | ------------------- | -------- | --------------------------------------------------- |
| `name`                | `str`             | 必填     | Agent 名称                                          |
| `handoff_description` | `str \| None`      | `None` | 被其他 Agent 作为 handoff 目标时的功能描述          |
| `tools`               | `list[Tool]`      | `[]`   | 可调用的工具列表                                    |
| `mcp_servers`         | `list[MCPServer]` | `[]`   | MCP 服务器列表，需手动管理生命周期                  |
| `mcp_config`          | `MCPConfig`       | `{}`   | MCP 配置（strict schema、错误处理、工具命名前缀等） |

### Agent 自身参数

| 参数                  | 类型                                      | 默认值              | 说明                                                |
| --------------------- | ----------------------------------------- | ------------------- | --------------------------------------------------- |
| `instructions`      | `str \| Callable \| None`                 | `None`            | 系统提示词，支持静态字符串或动态函数                |
| `prompt`            | `Prompt \| DynamicPromptFunction \| None` | `None`            | OpenAI 平台托管的提示词（仅 Responses API）         |
| `handoffs`          | `list[Agent \| Handoff]`                 | `[]`              | 可移交的子 Agent 列表                               |
| `model`             | `str \| Model \| None`                    | `None`            | 使用的模型，None 则用全局默认                       |
| `model_settings`    | `ModelSettings`                         | `ModelSettings()` | 模型参数（temperature、top_p、max_tokens 等）       |
| `input_guardrails`  | `list[InputGuardrail]`                  | `[]`              | 输入护栏，与首次响应并行执行，只对第一个 Agent 生效 |
| `output_guardrails` | `list[OutputGuardrail]`                 | `[]`              | 输出护栏，在最终输出后执行                          |
| `output_type`       | `type \| AgentOutputSchemaBase \| None`   | `None`            | 结构化输出类型，None 则输出纯字符串                 |
| `hooks`             | `AgentHooks \| None`                     | `None`            | 生命周期回调钩子                                    |
| `tool_use_behavior` | 见下表                                    | `"run_llm_again"` | 工具调用后的行为控制                                |
| `reset_tool_choice` | `bool`                                  | `True`            | 工具调用后是否重置 tool_choice，防止无限循环        |

#### tool_use_behavior 取值

| 值                                        | 含义                                                    |
| ----------------------------------------- | ------------------------------------------------------- |
| `"run_llm_again"`                       | 工具结果返回给 LLM，继续生成响应（默认）                |
| `"stop_on_first_tool"`                  | 第一个工具输出直接作为最终输出                          |
| `StopAtTools(stop_at_tool_names=[...])` | 遇到指定工具时停止，该工具输出作为最终输出              |
| 自定义函数                                | `(context, tool_results) -> ToolsToFinalOutputResult` |

---

## instructions vs prompt 的区别

### instructions — 本地定义

```python
agent = Agent(
    name="助手",
    instructions="你是一个专业的客服，只回答关于产品的问题。"
)
```

- 提示词存在代码里
- 每次调用 API 时作为 system 消息随 payload 发送
- 支持所有模型
- 动态化：传函数 `(context, agent) -> str`

### prompt — OpenAI 平台托管

```python
agent = Agent(
    name="助手",
    prompt={"id": "pmpt_abc123", "version": "2"}
)
```

- 提示词预先存储在 OpenAI Prompt 管理平台
- 代码里只传 id，服务端取内容
- 仅支持 OpenAI Responses API
- 动态化：传函数 `(data) -> Prompt`

### 对比

|          | `instructions`   | `prompt`                     |
| -------- | ------------------ | ------------------------------ |
| 内容存放 | 本地代码           | OpenAI 平台远端                |
| 传输方式 | 每次发送全文       | 只发送 id                      |
| 适用模型 | 所有模型           | 仅 OpenAI Responses API        |
| 版本管理 | 代码版本控制       | 平台独立管理                   |
| 典型场景 | 开发调试、灵活定制 | 生产环境、非技术人员维护提示词 |

### prompt 的限制

- 不支持 Chat Completions API（会警告或报错）
- 不支持非 OpenAI 模型
- 属于 OpenAI 付费 API 平台功能（按 token 计费）
- 无法被其他厂商平替（除非自定义 Model 实现）

两者在运行时分开传入模型（`run_loop.py`）：

```python
system_prompt, prompt_config = await asyncio.gather(
    execution_agent.get_system_prompt(context_wrapper),  # instructions
    execution_agent.get_prompt(context_wrapper),         # prompt
)
```

---

## grep 读代码技巧

### 基本格式

```bash
grep [选项] "要找的内容" 文件或目录
```

### 常用选项

| 选项                 | 作用                   |
| -------------------- | ---------------------- |
| `-r`               | 递归搜索子目录         |
| `-n`               | 显示行号               |
| `-i`               | 忽略大小写             |
| `-l`               | 只显示文件名           |
| `--include="*.py"` | 只搜索特定后缀         |
| `-v`               | 排除匹配行（反向搜索） |

### 常用组合

```bash
# 基本搜索（几乎每次都用这个）
grep -rn "关键词" src/ --include="*.py"

# 限制输出行数，避免刷屏
grep -rn "关键词" src/ | head -20

# 排除干扰项
grep -rn "prompt" src/ | grep -v "system_prompt"

# 精确匹配单词（\b 是单词边界）
grep -rn "\.prompt\b" src/
```

### 读代码的固定追踪套路

遇到不认识的东西，按顺序执行：

```bash
# 1. 找定义
grep -rn "class Foo\|def Foo" src/

# 2. 找 import 来源
grep -rn "import.*Foo" src/

# 3. 找调用方
grep -rn "\.foo\b" src/ --include="*.py"
```

### 三个核心动作

**定位 → 追踪 → 验证**

- 定位：从行为差异出发，找到入口（类、函数、参数）
- 追踪：顺着 import 往上找类型来源，顺着调用往下找执行路径
- 验证：用自己的话解释，能讲清楚说明真正理解了
