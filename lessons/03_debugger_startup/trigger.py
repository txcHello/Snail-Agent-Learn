"""事件触发入口：由宿主传入 JSON，随后启动 Agent，不注册为模型工具。"""

import argparse
import json
from pathlib import Path
import sys

# 事件入口也可能被 IDE 单独通过 runpy 执行。
LESSON_ROOT = Path(__file__).resolve().parent
if str(LESSON_ROOT) not in sys.path:
    sys.path.insert(0, str(LESSON_ROOT))

from agent import ROOT, load_config, run, task_for_file
from tools import SOURCE_ROOT, resolve_inside


def event_to_task(event: dict) -> str:
    """只接受一个明确的本地 file_received 事件格式。"""
    if not isinstance(event, dict) or set(event) != {"id", "type", "path"}:
        raise ValueError("事件必须且只能包含 id、type、path 三个字段")
    if not isinstance(event["id"], str) or not event["id"].strip() or len(event["id"]) > 100:
        raise ValueError("事件 id 必须是 1 到 100 字符的非空字符串")
    if event["type"] != "file_received" or not isinstance(event["path"], str):
        raise ValueError("仅支持 file_received 事件，path 必须是字符串")
    target = resolve_inside(SOURCE_ROOT, event["path"])
    if not target.is_file():
        raise ValueError("事件指定的练习文件不存在")
    return f"收到文件事件 {json.dumps(event['id'], ensure_ascii=False)}。\n" + task_for_file(event["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="提交一个本地 JSON 事件并启动 Agent")
    parser.add_argument("event", nargs="?", default=str(ROOT / "events" / "file_received.json"))
    args = parser.parse_args()
    config = {}
    try:
        with Path(args.event).open(encoding="utf-8-sig") as source:
            content = source.read(16_001)
        if len(content) > 16_000:
            raise ValueError("事件载荷超过 16000 字符")
        task = event_to_task(json.loads(content))
        print("[事件入口] JSON 事件已接收，开始执行 Agent")
        config = load_config()
        print(run(task, config))
    except Exception as error:
        detail = str(error)
        if config:
            detail = detail.replace(config["api_key"], "[redacted]")
        raise SystemExit(f"事件处理失败：{detail}") from None


if __name__ == "__main__":
    main()
