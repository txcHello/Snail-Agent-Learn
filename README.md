# Snail-Agent：最小 Agent 循环

当前只学习一件事：**调用真实模型 → 执行工具 → 回传结果 → 再次调用模型**。

Python 3.10+，仅使用标准库。全部实现都在 [agent.py](agent.py)，只有一个只读工具 `read_file`。

## 运行

保留已有的本地 `.env` 即可。首次配置时，将 `.env.example` 复制为 `.env`，填入自己的 `DEEPSEEK_API_KEY`。

```powershell
python -X utf8 agent.py
```

默认任务：读取 `demo_workspace/notes.txt`，统计非空行数并回答。样例有 **4 个非空行**。

本次真实 API 运行共调用模型 2 次，终端输出摘录如下（省略文件正文和回答中的逐行说明）：

```text
[1] 调用模型
    工具请求：read_file {"path": "notes.txt"}
[2] 调用模型
非空行数统计：4 行
```

也可以指定任务：

```powershell
python -X utf8 agent.py "读取 notes.txt，概括文件内容"
```

路径相对于 `demo_workspace/`。程序会显示模型调用轮数、工具请求、工具结果和最终回答。每次启动是一个新任务。

## 按顺序读代码

1. **`TOOLS`**：告诉模型工具叫什么、做什么、接受什么参数。
2. **`load_config()`**：读取 `.env`，环境变量优先，不在源码中存放密钥。
3. **`call_model()`**：发送一次 DeepSeek 请求，返回一条 assistant 消息。
4. **`execute_tool()`**：检查工具请求并真正读取文件，返回内容或错误。
5. **`run()`**：将上述步骤串成循环。

`messages` 是本次运行的消息历史：

```text
system     告诉模型如何使用工具
user       用户任务
assistant  模型请求 read_file
tool       程序返回真实文件内容，tool_call_id 对应请求的 id
assistant  模型基于读取结果回答，或者继续请求工具
```

核心是：先把模型返回的消息追加到 `messages`，再追加每个工具的执行结果。下一轮 API 调用会收到这些历史消息。一次返回多个工具请求时，依次执行，并回传全部结果后再调用模型。

## 本步边界

- 使用真实 `deepseek-flash`，无模拟模型、无额外适配类。
- 工具只读练习目录中的 UTF-8 文件，每次最多读取 16000 字符；目录检查不是操作系统级沙箱。
- 最多调用模型 6 轮，每次 HTTP 请求默认超时 30 秒；网络错误直接退出，不自动重试。
- 模型不再请求工具并返回非空文本时，循环结束。这只是结束条件，本步没有独立的答案验收器。
- 仅展示终端过程；消息保存在内存中。本步没有报告写入、持久轨迹或断点恢复。
- 旧课程代码已从当前版本移除，可从 Git 历史找回。本地密钥、已有运行记录和 IDE 文件不纳入提交。

工具调用协议参考 [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/tool_calls/)。
