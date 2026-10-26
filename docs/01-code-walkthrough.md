# 第 1 课：逐步读懂最小 harness

本课把实现拆成五步讲解。完整源码在 [mini_harness.py](../mini_harness.py)；讲解中的代码片段都来自这个文件。先掌握数据如何流动，再关注路径检查等细节。

本课任务：读取 `notes.txt`，统计非空行数，写入 `output/report.json`。

## 第 1 步：让“下一步行动”有明确的结构

```python
@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class FinalAnswer:
    text: str
```

`dataclass` 自动生成初始化方法，所以可以这样构造一个请求：

```python
ToolCall("read_file", {"path": "notes.txt"})
```

这里的 `name` 选工具，`arguments` 提供参数。`frozen=True` 禁止直接重新赋值实例字段；它不会把字段内的字典变成不可变对象。

这个对象只是一个请求。文件读取发生在下一步，模型不直接获得文件系统接口。

`FinalAnswer("已完成")` 表示模型希望结束，是否成功由第 5 步判断。

## 第 2 步：执行工具，并把结果统一包装

`execute_tool(root, call)` 根据请求分发到三种操作。以读取文件的实际代码为例：

```python
elif call.name == "read_file":
    with target.open(encoding="utf-8") as source:
        data = source.read(MAX_CHARS + 1)
    if len(data) > MAX_CHARS:
        raise ValueError("文件超过本课的读取上限")
```

`target` 是前面已经解析和检查过的文件路径。最多读取上限加一个字符，就能发现超限文件，避免把整个大文件装进内存。

所有工具执行前都会经过这些检查：

```python
if call.name not in {"list_files", "read_file", "write_file"}:
    raise ValueError(f"不支持的工具：{call.name}")
raw_path = call.arguments["path"]
if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
    raise ValueError("path 必须是相对路径字符串")
target = (root / raw_path).resolve()
if not target.is_relative_to(root):
    raise PermissionError("路径超出了练习目录")
```

例如 `../outside.txt` 会在路径解析之后被拒绝。写入还额外限制在练习目录的 `output` 子目录。

成功返回的结构是：

```python
return {"ok": True, "data": data}
```

失败返回的结构是：

```python
except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
    # 失败也作为观察结果交回模型，让下一轮有机会调整行动。
    return {"ok": False, "error": str(error)}
```

这样下一轮既能看到读取到的文本，也能看到“文件不存在”这样的错误。错误反馈让模型有机会调整行动；本课的模拟模型遇错就结束，尚未实现修复策略。

## 第 3 步：把决策、执行和反馈连成循环

`run` 首先创建上下文，并限制最多执行多少轮：

```python
messages: list[dict] = [{"role": "user", "content": TASK}]
for step in range(1, max_steps + 1):
    decision = model(messages)
```

`messages` 是本次运行的记录，最开始只有用户目标。`model(messages)` 表示根据当前记录提出下一步。

对于工具请求，循环记录请求、执行工具，再记录执行结果：

```python
messages.append({
    "role": "assistant", "tool": decision.name, "arguments": decision.arguments,
})
observation = execute_tool(root, decision)
print(f"    工具结果：{json.dumps(observation, ensure_ascii=False)}")
messages.append({"role": "tool", "name": decision.name, "content": observation})
```

下一次进入 `model(messages)` 时，刚刚加入的工具结果已经在上下文里。理解这一次追加，就理解了行动怎样影响下一轮决策。

模型提出结束请求时，进入另一个分支：

```python
if isinstance(decision, FinalAnswer):
    print(f"[{step}] 模型请求结束：{decision.text}")
    if not verify_report(root):
        raise RuntimeError("模型请求结束，但独立验收未通过")
    messages.append({"role": "assistant", "content": decision.text})
    print("[验收] report.json 与源文件一致，任务通过。")
    return messages
```

如果一直没有成功结束，循环用尽后会抛出异常。这个上限约束的是模型决策次数，最后一次结束请求也计入一轮。

## 第 4 步：先用模拟模型观察反馈怎样被使用

`demo_model` 先提取已经发生的工具观察：

```python
observations = [message["content"] for message in messages if message["role"] == "tool"]
```

还没发生工具调用时，返回查看目录的请求；查看完目录后，返回读取文件的请求：

```python
if len(observations) == 0:
    return ToolCall("list_files", {"path": "."})
if len(observations) == 1:
    return ToolCall("read_file", {"path": "notes.txt"})
```

读取完成后，取出实际文件内容，计算结果并请求写入：

```python
if len(observations) == 2:
    text = observations[-1]["data"]
    report = {
        "source": "notes.txt",
        "non_empty_lines": sum(bool(line.strip()) for line in text.splitlines()),
    }
    return ToolCall("write_file", {
        "path": "output/report.json",
        "content": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    })
```

`splitlines()` 拆分行，`strip()` 去除空白，`bool(...)` 判断是否非空，`sum(...)` 统计结果。

本课行动顺序由代码固定，只用于观察 harness。接入真实大模型后，会由模型根据目标、工具说明和观察选择动作；还需要增加 API 适配、工具参数校验和超时处理。

## 第 5 步：用产物验收任务

`verify_report` 独立重新读取源文件与结果文件，按预先规定的格式比较：

```python
source = (root / "notes.txt").read_text(encoding="utf-8")
report = json.loads((root / "output/report.json").read_text(encoding="utf-8"))
return report == {
    "source": "notes.txt",
    "non_empty_lines": sum(bool(line.strip()) for line in source.splitlines()),
}
```

这里只验证结果是否符合这个练习任务。真实编程任务的验收可能是测试命令、文件差异或其他可检查的条件，需要按具体任务定义。

本课还没有区分不同运行的产物，因此正确的旧文件也可以通过验收。后续会加入运行标识、执行记录和检查点，学习这个边界怎样收紧。

## 跑起来，并读懂这四轮

```powershell
python mini_harness.py
python -m unittest -v
```

| 轮次 | 模拟模型返回 | harness 的动作 | 下一轮新增的观察 |
| --- | --- | --- | --- |
| 1 | `list_files` 请求 | 检查路径并列目录 | `notes.txt` 等目录项 |
| 2 | `read_file` 请求 | 读取练习文件 | 文件的真实文本 |
| 3 | `write_file` 请求 | 把统计结果写入 output | 写入成功的反馈 |
| 4 | `FinalAnswer` | 独立检查报告 | 通过后返回 |

接下来按 README 的三个动手实验修改输入、限制轮数和制造错误结果。能解释每次变化由哪一段代码造成，就掌握了第一课。
