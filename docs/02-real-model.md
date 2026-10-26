# 第 2 课：用真实模型驱动第一课的 harness

本课继续完成相同的文件统计任务。主要变化是把模拟模型替换为真实的 DeepSeek API；文件工具和执行循环继续使用第一课的实现。

新增代码：

- [deepseek_model.py](../deepseek_model.py)：配置、工具说明、HTTP 请求、模型响应解析和工具结果配对。
- [run_real.py](../run_real.py)：创建新的练习目录，启动真实任务，保存协议轨迹。
- [test_deepseek_model.py](../test_deepseek_model.py)：离线验证接口边界，不消耗 API 用量。
- [.env.example](../.env.example)：可公开的配置模板。

## 第 1 步：把连接配置与代码分开

本地 `.env` 中使用这些配置名：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT=30
```

把密钥填在本地文件里，或者设置进程环境变量；真实 `.env` 已被 Git 忽略。代码通过 `load_config` 读取，环境变量优先。

配置对象中的关键一行是：

```python
api_key: str = field(repr=False)
```

这样打印配置对象时会显示地址和模型名，不显示密钥。

## 第 2 步：告诉模型有哪些工具

第一课的 Python 工具函数存在于本机，模型无法直接看见它们。我们用 JSON Schema 描述工具用途和参数。下面是读取工具的协议形状：

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "读取练习目录内的 UTF-8 文本文件。",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "相对于练习目录的路径"}
      },
      "required": ["path"],
      "additionalProperties": false
    }
  }
}
```

`tool_definitions()` 为三个工具生成这些说明。它们帮助模型生成请求；请求收到后，我们仍然检查工具名、必需字段和参数类型，执行时继续检查路径边界。

## 第 3 步：向 API 发送完整的当前记录

`DeepSeekModel.__call__` 组装请求体：

```python
payload = {
    "model": self.config.model,
    "messages": self.history.copy(),
    "tools": tool_definitions(),
    "thinking": {"type": "disabled"},
    "max_tokens": 1024,
    "stream": False,
}
```

`request_json` 向 `base_url + "/chat/completions"` 发送 POST 请求。认证通过 HTTP 请求头传递，密钥不会出现在模型的对话上下文里。

本课显式关闭 thinking，与官方非思考模式工具调用示例一致。先学清楚工具协议，后续可以再研究不同推理设置。

请求有超时上限。网络或接口错误会明确失败，不自动重试，避免重复发起可能已经处理过的付费请求。错误信息中的完整密钥会被替换。

## 第 4 步：把协议响应变回第一课的对象

模型的工具参数是 JSON 字符串，适配器把它解析成字典：

```python
name = call["function"]["name"]
arguments = json.loads(call["function"]["arguments"])
```

验证字段后构造 `ToolCall(name, arguments)`，第一课的 `run` 就能继续执行。

模型也可能一次提出多个工具请求。本课将它们放入 `_pending` 队列，由 harness 按顺序执行。必须收到全部对应结果之后，才发下一次 API 请求。

工具结果必须带回原始请求编号：

```python
self.history.append({
    "role": "tool", "tool_call_id": call_id,
    "content": json.dumps(message["content"], ensure_ascii=False),
})
```

`tool_call_id` 让服务端知道这个结果回答的是哪一个工具请求。API 返回的原始 assistant 消息已经保存在 `history` 中，因此不会重复加入第一课记录的 assistant 请求。

这解释了为什么“执行步数”和“API 调用次数”可能不同：一次模型响应可以包含多个工具请求。

## 第 5 步：在新的练习目录里启动

`run_real.py` 每次创建新的运行目录，复制练习输入，并复用第一课的循环：

```python
model = DeepSeekModel(config)
run(model, workspace)
```

实际实现还会在 `finally` 中保存 `trace.json`，其中包含模型名、API 调用次数、token 用量、协议消息以及验收是否通过。轨迹位于工具可写的 workspace 之外，不包含 HTTP 请求头。

这样也解决了第一课的一个限制：新的目录里没有旧报告，模型必须在这次运行中生成产物才能通过当前验收。

```powershell
python -X utf8 run_real.py
python -X utf8 -m unittest -v
```

## 实跑时出现的一次失败

第一次真实调用中，模型正确读到了 4 个非空行，却写出了这样的报告：

```json
{
  "file": "notes.txt",
  "total_lines": 5,
  "non_empty_lines": 4
}
```

验收器要求 `source` 和 `non_empty_lines`，于是拒绝结束。检查后发现，任务文本只说“统计后保存”，没有告诉真实模型约定的字段；模拟模型则在代码中写死了正确格式，掩盖了这个遗漏。

修正是把输出约定加入 `TASK`：

```python
TASK = (
    "读取 notes.txt，统计非空行数，将结果保存到 output/report.json。"
    "输出必须是 JSON 对象，且只能包含两个字段：source 固定为字符串 notes.txt，"
    "non_empty_lines 为实际统计得到的非空行数整数。不要添加其他字段。"
)
```

保留字段和计数验收，同时加入回归用例，确保“计数正确但字段错误”的报告仍被拒绝。代码审查还发现 Python 会认为 `3.0 == 3`，因此补上严格的整数类型检查及其回归用例。

这个过程说明：用真实模型验证执行闭环时，还要检查任务描述与验收标准是否一致。

## 修正后的真实运行结果

2026-10-26 实跑 `deepseek-flash`：

| API 调用 | 模型提出的行动 | harness 执行结果 |
| --- | --- | --- |
| 1 | 读取 `notes.txt` | 反馈实际文件内容 |
| 2 | 写入 `output/report.json` | 反馈写入成功 |
| 3 | 请求结束 | 独立验收通过 |

实际产物：

```json
{"source": "notes.txt", "non_empty_lines": 4}
```

该成功运行的 API 响应报告了 2027 个输入 token、152 个输出 token，共 2179 个 token。这些用量属于该次运行，后续运行会变化。

真实模型这次直接读取已知文件，省略了模拟版固定的查看目录步骤。模型自主选择了工具行动，harness 继续控制执行与验收。

当前仍只支持这个固定练习任务。任意任务的验收、代码搜索和编辑、受控测试命令、上下文整理、持久检查点与恢复将在后续逐步实现。保存轨迹不等于已经具备断点恢复。

## 官方对照资料

- [DeepSeek：首次 API 调用与模型配置](https://api-docs.deepseek.com/quick_start/agent_integrations/)
- [DeepSeek：工具调用](https://api-docs.deepseek.com/guides/tool_calls/)
- [DeepSeek：Chat Completions 接口](https://api-docs.deepseek.com/api/create-chat-completion/)
