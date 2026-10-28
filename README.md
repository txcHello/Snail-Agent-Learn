# Snail-Agent：按课程学习 Agent 与 harness

使用 Python、LangGraph 和真实 DeepSeek 模型，逐步学习 Agent 的状态、工具、循环与外部交互。

## 课程目录

每次新的学习需求都新增一个编号文件夹。每课包含代码、说明文档和样例，基于上一课演进，已完成课程保留供比较。

| 目录 | 学习内容 |
| --- | --- |
| [01_minimal_langgraph](lessons/01_minimal_langgraph/README.md) | 原有 LangGraph 最小 Agent 快照：一个读取工具、模型节点、工具节点和条件边 |
| [02_tool_types](lessons/02_tool_types/README.md) | 五类交互：感知、执行、子 Agent 协作、外部事件触发、本地用户沟通 |
| [03_debugger_startup](lessons/03_debugger_startup/README.md) | 基于第二课修复 PyCharm / runpy 启动时的同目录模块导入，并保留回归测试 |

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
```

第二课的任务入口与事件入口各自运行一次完整任务，选择一个即可体验。产物和本地消息保存在根目录 `.runs/02_tool_types/<运行标识>/`，实际路径会显示在终端。

## 如何比较

先读第一课的 `agent.py`，再读第二课的 `agent.py` 和 `tools.py`。每课 README 展示新增代码、作用和运行方式；第二课文档还提供了逐项变化表。

后续新需求从 `04_...` 继续编号。uv 配置在根目录共享，新增依赖使用 `uv add`，移除使用 `uv remove`，提交 `uv.lock`。`.env`、`.venv/`、`.runs/` 和本地记忆不会上传。
