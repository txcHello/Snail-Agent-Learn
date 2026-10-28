"""四类主动工具：感知、执行、协作、用户沟通。事件入口见 trigger.py。"""

from collections.abc import Callable
import json
from pathlib import Path

from langchain_core.tools import tool

SOURCE_ROOT = (Path(__file__).resolve().parent / "demo_workspace").resolve()
MAX_CHARS = 16_000


def resolve_inside(root: Path, path: str) -> Path:
    """将相对路径限制在指定目录内，防止越过练习目录。"""
    root = root.resolve()
    if not path.strip() or Path(path).is_absolute():
        raise ValueError("请使用非空的相对路径")
    target = (root / path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("路径超出了允许的目录")
    return target


# 1. 感知：读取已有信息。
@tool
def read_file(path: str) -> str:
    """读取练习输入目录内的 UTF-8 文件，path 例如 notes.txt。"""
    target = resolve_inside(SOURCE_ROOT, path)
    with target.open(encoding="utf-8") as source:
        content = source.read(MAX_CHARS + 1)
    if len(content) > MAX_CHARS:
        raise ValueError("文件超过 16000 字符的读取上限")
    return content


def create_tools(run_dir: Path, summarize: Callable[[str], str]) -> list:
    """通过闭包注入输出目录和子 Agent；这些参数不会暴露给模型。"""
    run_dir = run_dir.resolve()

    # 2. 执行：创建任务产物，每次运行使用自己的输出目录。
    @tool
    def write_file(path: str, content: str) -> str:
        """在本次运行的 output 目录创建 UTF-8 文件，例如 summary.md；不覆盖已有文件。"""
        if len(content) > MAX_CHARS:
            raise ValueError("写入内容超过 16000 字符")
        target = resolve_inside(run_dir / "output", path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as destination:
            destination.write(content)
        return f"已创建：{target}"

    # 3. 协作：主 Agent 只传任务，子 Agent 自己读取文件并生成摘要。
    @tool
    def delegate_summary(path: str) -> str:
        """委托独立的摘要子 Agent 读取练习文件并返回三点摘要，path 例如 notes.txt。"""
        resolve_inside(SOURCE_ROOT, path)
        return summarize(path)

    # 5. 用户沟通：将消息投递给本地用户；第 4 类事件入口见 trigger.py。
    @tool
    def notify_user(message: str) -> str:
        """向本地用户发送消息：显示在终端并追加到本次运行的 inbox.jsonl 收件箱。"""
        if not message.strip() or len(message) > 2_000:
            raise ValueError("用户消息必须是 1 到 2000 字符的非空文本")
        run_dir.mkdir(parents=True, exist_ok=True)
        inbox = run_dir / "inbox.jsonl"
        with inbox.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps({"channel": "local", "message": message}, ensure_ascii=False) + "\n")
        print(f"[本地用户消息] {message}")
        return f"已投递到本地收件箱：{inbox}"

    return [read_file, write_file, delegate_summary, notify_user]
