# 第三课：修复 IDE 启动时的本地模块导入

本课复制第二课的代码、样例和行为测试，只修复启动路径，并使用独立的运行产物目录。第一、二课保留原样。

## 第一步：复现并理解问题

第二课有 `from tools import create_tools, read_file`。`tools.py` 就在同一目录，但 Python 按 `sys.path` 搜索模块，不会因为两个文件相邻就自动找到它。

终端执行 `python path/to/agent.py` 时，解释器通常会把脚本目录放入搜索路径。这次 PyCharm AI Debugger 的堆栈显示，它通过 `runpy.run_path(script_path, run_name="__main__")` 执行文件；这个入口不会自动添加脚本的父目录。

在仓库根目录复现旧版问题，不调用模型：

```powershell
uv run python -X utf8 -c "import runpy; runpy.run_path('lessons/02_tool_types/agent.py')"
```

实际结果：`ModuleNotFoundError: No module named 'tools'`。文件存在，本次问题是搜索路径，不需要安装名为 `tools` 的依赖。

参考 [Python 官方 runpy 文档](https://docs.python.org/3/library/runpy.html#runpy.run_path)。

## 第二步：在导入之前补齐路径

本课 `agent.py` 的关键改动：

```python
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

# 必须放在 from tools 之前。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import create_tools, read_file
```

- `__file__` 是当前脚本路径，避免依赖 IDE 的工作目录。
- `.resolve().parent` 得到本课代码所在的绝对目录。
- `sys.path.insert(0, ...)` 让导入系统能找到本课的 `tools.py`。
- 条件判断避免终端已加入目录时重复添加。

`trigger.py` 同样在 `from agent import ...` 之前添加自己的脚本目录，避免事件入口又出现 `No module named 'agent'`。

本课保留教学脚本结构，适合在独立 Python 进程中选择一课运行。不要在同一进程中连续载入多课的同名 `agent` / `tools`：Python 还会缓存已经导入的模块。正式打包和包内相对导入可以在后续学习。

## 第三步：运行修复版本

将 PyCharm 当前运行配置的脚本路径改为：

```text
D:\myself_project\agent\lessons\03_debugger_startup\agent.py
```

解释器继续使用 `D:\myself_project\agent\.venv\Scripts\python.exe`，工作目录可设为仓库根目录。运行事件入口时，把脚本换为本课的 `trigger.py`。本次没有修改本地 `.idea/` 配置。

第二课保留修复前代码，因此继续启动第二课仍可能遇到原错误。

终端运行（在仓库根目录，两种入口任选一个）：

```powershell
uv run --locked python -X utf8 lessons/03_debugger_startup/agent.py
uv run --locked python -X utf8 lessons/03_debugger_startup/trigger.py
```

所有课程继续共享根目录 `.env` 与 uv 环境。输出位于 `.runs/03_debugger_startup/<运行标识>/`，不上传 Git。

## 第四步：验证结果

```powershell
uv run --locked python -X utf8 -m unittest discover -s lessons/03_debugger_startup -v
```

共 14 项测试：沿用第二课 11 项行为测试，新增 3 项启动测试。新增测试在独立子进程中清除 `PYTHONPATH`，覆盖两个入口、仓库根目录和其他工作目录，以及 `runpy` 的导入与主入口执行。离线主入口测试把 API Key 环境变量置空，预期在配置校验时退出，避免消耗 API。

修复前新增测试有 6 个子用例失败，分别报缺失 `tools` 或 `agent`；修复后全部通过。测试保留在 `test_startup.py`。

还通过堆栈中的 `runpy` 主入口方式调用了真实 DeepSeek：

```powershell
uv run --locked python -u -X utf8 -c "import runpy, sys; sys.argv = ['trigger.py']; runpy.run_path('lessons/03_debugger_startup/trigger.py', run_name='__main__')"
```

实际结果：主 Agent 调用模型 4 次、摘要子 Agent 2 次，读取、委托摘要、写入和本地通知全部完成，生成并核对 `output/summary.md` 与 `inbox.jsonl`。本次未在 PyCharm 的 AI Debugger 界面中复测，验证覆盖了堆栈中的 Python 执行方式。

工具职责与边界沿用[第二课说明](../02_tool_types/README.md)：四个主动工具、一个外部事件入口；通知仅投递本地收件箱，目录检查并非操作系统沙箱。
