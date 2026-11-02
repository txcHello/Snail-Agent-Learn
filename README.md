# Snail-Agent：按课程学习 Agent 与 harness

使用 Python、LangGraph 和真实 DeepSeek 模型，逐步学习 Agent 的状态、工具、循环与外部交互。

## 课程目录

每次新的学习需求都新增一个编号文件夹。每课包含代码、说明文档和样例，基于上一课演进，已完成课程保留供比较。

| 目录 | 学习内容 |
| --- | --- |
| [01_minimal_langgraph](lessons/01_minimal_langgraph/README.md) | 原有 LangGraph 最小 Agent 快照：一个读取工具、模型节点、工具节点和条件边 |
| [02_tool_types](lessons/02_tool_types/README.md) | 五类交互：感知、执行、子 Agent 协作、外部事件触发、本地用户沟通 |
| [03_debugger_startup](lessons/03_debugger_startup/README.md) | 基于第二课修复 PyCharm / runpy 启动时的同目录模块导入，并保留回归测试 |
| [04_learning_paradigms](lessons/04_learning_paradigms/README.md) | 概念学习文档：后训练、上下文学习、外部化学习，以及与文件 Agent 和 harness 的关系 |
| [05_agent_context](lessons/05_agent_context/README.md) | 图解 Agent 上下文的五类信息、工具调用消息变化、程序状态与消融实验 |
| [06_react_loop](lessons/06_react_loop/README.md) | 实现 ReAct 式多币种汇总：真实模型、工具反馈循环、公开轨迹和报告验收 |
| [07_loop_engineering](lessons/07_loop_engineering/README.md) | 在 ReAct 外增加任务循环：发现、输入快照、验收、持久状态、有限重试与中断恢复 |

第一课的代码保持归档时原样，文档中的启动路径已更新。第二课独立保存完整实现，不导入第一课代码。根目录只保留课程导航和共享的项目配置。

使用 PyCharm AI Debugger 时请运行第三课；第二课保留修复前快照，可能出现 `No module named 'tools'`。第三课文档解释原因与逐步改动。

## 环境与配置

使用 uv 管理环境和依赖，Python 兼容要求为 3.10+，开发解释器选择为 3.14：

```powershell
uv sync --locked
```

首次运行，将根目录 `.env.example` 复制为根目录 `.env`，填入自己的 API Key。已有配置可以继续使用。所有课程共用这份本地配置，不把密钥复制到课程文件夹中。

## 选择一课运行

以下命令均在仓库根目录执行：

```powershell
# 第一课：原始最小循环；uv 将根目录 .env 加入进程环境
uv run --env-file .env python -X utf8 lessons/01_minimal_langgraph/agent.py

# 第二课：由用户任务启动，使用四种主动工具
uv run python -X utf8 lessons/02_tool_types/agent.py

# 第二课：由本地 JSON 事件启动同一个 Agent
uv run python -X utf8 lessons/02_tool_types/trigger.py

# 第二课：离线测试，不消耗 API
uv run python -X utf8 -m unittest discover -s lessons/02_tool_types -v

# 第三课：兼容 IDE / runpy 的最新版本
uv run python -X utf8 lessons/03_debugger_startup/agent.py
uv run python -X utf8 -m unittest discover -s lessons/03_debugger_startup -v

# 第六课：运行可观察的 ReAct 循环
uv run --locked python -X utf8 lessons/06_react_loop/agent.py
uv run --locked python -X utf8 -m unittest discover -s lessons/06_react_loop -v

# 第七课：外层任务循环，每次最多执行一项，重复启动观察恢复与去重
uv run --locked python -X utf8 lessons/07_loop_engineering/loop.py --max-executions 1
uv run --locked python -X utf8 -m unittest discover -s lessons/07_loop_engineering -v
```

第二课的任务入口与事件入口各自运行一次完整任务，选择一个即可体验。产物和本地消息保存在根目录 `.runs/02_tool_types/<运行标识>/`，实际路径会显示在终端。

## 如何比较

先读第一课的 `agent.py`，再读第二课的 `agent.py` 和 `tools.py`。每课 README 展示新增代码、作用和运行方式；第二课文档还提供了逐项变化表。

第四、五课是概念学习文档，包含配图、对照表、示例、自测题和参考答案，无需运行程序。

第七课保留第六课的 LangGraph 内层，外层用本地任务目录和 JSON 状态演示 Loop 工程。默认状态目录固定在 `.runs/07_loop_engineering/default/`，重复启动会延续已有任务；详情见本课文档。

后续新需求从 `08_...` 继续编号。uv 配置在根目录共享，新增依赖使用 `uv add`，移除使用 `uv remove`，提交 `uv.lock`。`.env`、`.venv/`、`.runs/` 和本地记忆不会上传。
