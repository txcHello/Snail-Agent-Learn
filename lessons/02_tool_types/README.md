# 第二课：五类工具与事件入口

本课在 [第一课](../01_minimal_langgraph/README.md) 的 LangGraph 循环上演进。先保持图结构不变，再增加工具；所有代码和样例都在本课目录内，不导入第一课的实现。

按照你给出的分类，本课各提供一个例子：

| 类型 | 本课实现 | 谁发起 | 可以观察到什么 |
| --- | --- | --- | --- |
| 感知 | `read_file(path)` | Agent | 返回练习文件的真实内容 |
| 执行 | `write_file(path, content)` | Agent | 创建本次运行的摘要文件 |
| 协作 | `delegate_summary(path)` | 主 Agent | 摘要子 Agent 独立读文件并返回摘要 |
| 事件触发 | `trigger.py` 接收 JSON | 外部输入 | 事件转换成任务，然后启动 Agent |
| 用户沟通 | `notify_user(message)` | Agent | 终端显示消息，本地收件箱收到记录 |

因此，模型看到 **4 个主动工具**；事件触发是独立入口。这五类描述的是职责，同一实现技术可以承担不同职责，例如创建报告和投递本地消息都涉及文件写入。

## 运行与观察

所有命令都从仓库根目录执行，共享根目录的 uv 环境与本地 `.env`：

```powershell
uv sync --locked

# 由命令行任务启动
uv run python -X utf8 lessons/02_tool_types/agent.py

# 由示例 JSON 事件启动；二者选择一个即可观察完整流程
uv run python -X utf8 lessons/02_tool_types/trigger.py
```

默认任务明确要求四种主动工具各使用一次：读取笔记 → 委托摘要 → 写入摘要 → 确认写入成功后通知本地用户 → 最终回答。摘要子 Agent 会额外调用自己的读取工具。

每次运行创建独立目录：

```text
.runs/02_tool_types/<运行标识>/
├── output/
│   └── summary.md       # 执行工具创建的任务产物
└── inbox.jsonl          # 用户沟通工具投递的本地消息
```

运行标识会显示在终端。实际摘要和通知内容由模型生成。`.runs/` 被 Git 忽略，不上传生成结果。

## 本次实测结果

使用示例 JSON 事件和真实 DeepSeek API，观察到以下顺序：

```text
事件入口接收 JSON，启动主 Agent
主 Agent 第 1 轮：read_file + delegate_summary
  摘要子 Agent 第 1 轮：read_file
  摘要子 Agent 第 2 轮：返回三点摘要
主 Agent 第 2 轮：write_file
主 Agent 第 3 轮：notify_user
主 Agent 第 4 轮：最终回答
```

共 6 次真实模型调用。四种主动工具在主 Agent 中各使用一次；额外的文件读取发生在子 Agent 的独立上下文中。已检查摘要文件包含三点内容，本地收件箱包含一条投递记录。读取和委托在同一个模型响应中提出，工具按本课配置顺序执行。后续运行的分组、措辞和调用次数可能不同。

## 第一步：感知已有信息

先看 [tools.py](tools.py) 的 `read_file`。它保留第一课的基本行为：

```python
target = resolve_inside(SOURCE_ROOT, path)
with target.open(encoding="utf-8") as source:
    content = source.read(MAX_CHARS + 1)
```

`SOURCE_ROOT` 只指向本课的 `demo_workspace/`。相对路径会先解析并检查边界，读取超过 16000 字符会报错。工具结果通过 `ToolNode` 回到主 Agent 的消息历史中。

## 第二步：执行改变环境的行动

`write_file` 使用不同的目录：

```python
target = resolve_inside(run_dir / "output", path)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as destination:
    destination.write(content)
```

模型只填写 `summary.md` 这样的输出相对路径。`"x"` 表示只创建新文件，已有文件会报错；输入目录不会被这项工具改写。与感知工具相比，这次调用留下了新的任务产物。

## 第三步：委托一个有独立上下文的子 Agent

主 Agent 调用的 `delegate_summary` 只把文件路径交给摘要执行器：

```python
return summarize(path)
```

摘要执行器由 [agent.py](agent.py) 的 `create_summary_agent` 创建：

```python
graph = build_graph(config, [read_file], max_turns=3, label="摘要子 Agent")
```

每次委托都以新的 system/user 消息和 `model_calls=0` 调用子图。子 Agent 自己决定调用 `read_file`，看到读取结果后生成三点摘要。

- 主 Agent 有四个工具，负责组织任务、保存产物和通知用户。
- 摘要子 Agent 在同一 Python 进程中运行另一张图，只拥有读取工具和自己的消息历史。
- 子图最后的回答作为 `delegate_summary` 的工具结果交还主 Agent。

主 Agent 的完整聊天记录、写入能力和通知能力不会作为子 Agent 的工具或任务上下文传入。两者都调用真实 DeepSeek，终端用不同标签区分。

## 第四步：从外部事件启动任务

打开 [events/file_received.json](events/file_received.json)：

```json
{
  "id": "notes-received-001",
  "type": "file_received",
  "path": "notes.txt"
}
```

[trigger.py](trigger.py) 先检查字段、事件类型和文件路径，再做以下两步：

```python
task = event_to_task(json.loads(content))
print(run(task, config))
```

任务出现之前，Agent 还没有运行；入口先收到事件，再启动图。因此 `trigger.py` 不使用 `@tool`，也不会出现在 `bind_tools` 的列表中。

本课由命令行将 JSON 事件交付给入口，演示“外部事件 → Agent”。每次处理一个事件即退出。尚未接入邮件订阅、定时调度、文件监听或 Webhook 服务；重复提交同一事件会产生新的运行，不提供去重。

也可以指定另一份同格式的本地事件文件：

```powershell
uv run python -X utf8 lessons/02_tool_types/trigger.py lessons/02_tool_types/events/file_received.json
```

## 第五步：主动沟通执行结果

`notify_user` 将消息同时显示在终端，并追加到本次运行的 `inbox.jsonl`：

```python
destination.write(json.dumps({"channel": "local", "message": message}, ensure_ascii=False) + "\n")
```

摘要文件是用户要的产物，收件箱则是用户了解进展的渠道。这里的“已投递”明确表示投递到了本地文件，未发送邮件、短信或外部聊天消息。消息最长 2000 字符。

## 对比第一课

| 第一课 | 第二课 | 为什么变化 |
| --- | --- | --- |
| `agent.py` 内定义一个读取工具 | `tools.py` 集中展示四类主动工具 | 逐项比较工具职责 |
| `build_graph(config)` 绑定固定读取工具 | `build_graph(config, tools, ...)` 接收工具列表 | 主、子 Agent 使用同一种循环，拥有不同能力 |
| 一个 Agent 的 `messages` | 主、子图各自的 `messages` | 学习委托与上下文边界 |
| 只有命令行任务入口 | 命令行入口和 JSON 事件入口 | 学习“被事件启动” |
| 只读文件 | 输入只读，产物和消息写入独立运行目录 | 可以观察执行与沟通留下的记录 |

主图仍是 `START → model → tools → model`，模型没有工具请求时结束。`run()` 里的 `for` 只消费图执行事件，不手写决策循环。

## 验证和当前边界

本课保留可重复运行的离线测试，不消耗 API：

```powershell
uv run python -X utf8 -m unittest discover -s lessons/02_tool_types -v
```

测试覆盖工具参数边界、输出创建与防覆盖、本地消息投递、事件格式、子 Agent 上下文隔离、图调用预算和完整工具协作路径。

- 主 Agent 最多调用模型 6 次；每次摘要委托最多调用模型 3 次，两者分别计数，尚无共享总预算。
- 主、子图使用 `max_concurrency=1` 顺序执行本课工具，便于观察写入与消息顺序。
- HTTP 默认超时 30 秒、SDK 不自动重试；工具异常回传模型，配置或最终图执行失败以非零状态退出。
- 仍然没有通用答案验收器、持久记忆或断点恢复；目录检查不是操作系统级沙箱。
- 摘要文件不会覆盖已有产物，但本课未提供事件去重、消息去重或失败后的自动回滚。

框架对照：[LangGraph 图与状态](https://docs.langchain.com/oss/python/langgraph/graph-api)、[工具定义](https://docs.langchain.com/oss/python/langchain/tools)。
