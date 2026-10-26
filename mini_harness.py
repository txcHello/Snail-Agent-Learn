"""第一课：用模拟模型观察 harness。运行：python mini_harness.py。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json


TASK = "读取 notes.txt，统计非空行数，将结果保存到 output/report.json。"
MAX_CHARS = 16_000


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class FinalAnswer:
    text: str


def execute_tool(root: Path, call: ToolCall) -> dict:
    """工具由程序执行；模型只能提交请求。文件访问只面向练习目录。"""
    try:
        if call.name not in {"list_files", "read_file", "write_file"}:
            raise ValueError(f"不支持的工具：{call.name}")
        raw_path = call.arguments["path"]
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            raise ValueError("path 必须是相对路径字符串")
        target = (root / raw_path).resolve()
        if not target.is_relative_to(root):
            raise PermissionError("路径超出了练习目录")

        if call.name == "list_files":
            names = []
            for entry in target.iterdir():
                names.append(entry.name)
                if len(names) > 100:
                    raise ValueError("本课最多列出 100 个目录项")
            data = sorted(names)
        elif call.name == "read_file":
            with target.open(encoding="utf-8") as source:
                data = source.read(MAX_CHARS + 1)
            if len(data) > MAX_CHARS:
                raise ValueError("文件超过本课的读取上限")
        else:
            output_root = (root / "output").resolve()
            if output_root == root or not output_root.is_relative_to(root):
                raise PermissionError("output 必须位于练习目录内")
            if not target.is_relative_to(output_root):
                raise PermissionError("只能写入 output 目录")
            content = call.arguments["content"]
            if not isinstance(content, str) or len(content) > MAX_CHARS:
                raise ValueError("content 必须是最多 16000 字符的字符串")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            data = f"已写入 {target.relative_to(root).as_posix()}"
        return {"ok": True, "data": data}
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        # 失败也作为观察结果交回模型，让下一轮有机会调整行动。
        return {"ok": False, "error": str(error)}


def verify_report(root: Path) -> bool:
    """验收规则由人预先定义；不依据模型的结束语判断成功。"""
    try:
        source = (root / "notes.txt").read_text(encoding="utf-8")
        report = json.loads((root / "output/report.json").read_text(encoding="utf-8"))
        return report == {
            "source": "notes.txt",
            "non_empty_lines": sum(bool(line.strip()) for line in source.splitlines()),
        }
    except (OSError, ValueError, UnicodeError):
        return False


def run(
    model: Callable[[list[dict]], ToolCall | FinalAnswer],
    workspace: Path,
    max_steps: int = 8,
) -> list[dict]:
    """harness 的最小闭环：模型决策 → 工具执行 → 反馈 → 再决策。"""
    if max_steps < 1:
        raise ValueError("max_steps 必须至少为 1")
    root = workspace.resolve(strict=True)
    messages: list[dict] = [{"role": "user", "content": TASK}]
    for step in range(1, max_steps + 1):
        decision = model(messages)
        if isinstance(decision, FinalAnswer):
            print(f"[{step}] 模型请求结束：{decision.text}")
            if not verify_report(root):
                raise RuntimeError("模型请求结束，但独立验收未通过")
            messages.append({"role": "assistant", "content": decision.text})
            print("[验收] report.json 与源文件一致，任务通过。")
            return messages
        if not isinstance(decision, ToolCall):
            raise TypeError("模型必须返回 ToolCall 或 FinalAnswer")
        print(f"[{step}] 请求工具：{decision.name} {decision.arguments}")
        messages.append({
            "role": "assistant", "tool": decision.name, "arguments": decision.arguments,
        })
        observation = execute_tool(root, decision)
        print(f"    工具结果：{json.dumps(observation, ensure_ascii=False)}")
        messages.append({"role": "tool", "name": decision.name, "content": observation})
    raise RuntimeError(f"达到 {max_steps} 轮上限，任务尚未完成")


def demo_model(messages: list[dict]) -> ToolCall | FinalAnswer:
    """模拟模型：行动顺序写死，用来调试 harness；不具备自主推理能力。"""
    observations = [message["content"] for message in messages if message["role"] == "tool"]
    if observations and not observations[-1]["ok"]:
        return FinalAnswer("工具调用失败，无法完成任务。")
    if len(observations) == 0:
        return ToolCall("list_files", {"path": "."})
    if len(observations) == 1:
        return ToolCall("read_file", {"path": "notes.txt"})
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
    return FinalAnswer("已生成非空行数统计。")


if __name__ == "__main__":
    print("模式：模拟模型（不调用网络，不需要 API Key）")
    print(f"任务：{TASK}")
    try:
        run(demo_model, Path(__file__).parent / "demo_workspace")
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise SystemExit(f"运行失败：{error}") from error
