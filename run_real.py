"""第二课入口：python -X utf8 run_real.py。只调用真实模型，不回退到模拟模型。"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

from deepseek_model import DeepSeekModel, ModelError, load_config
from mini_harness import run


def main() -> int:
    project = Path(__file__).resolve().parent
    config = load_config(project)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = project / ".runs" / run_id
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_bytes((project / "demo_workspace/notes.txt").read_bytes())
    model = DeepSeekModel(config)
    verified = False
    try:
        print(f"模式：真实 API，model={config.model}，base_url={config.base_url}")
        print(f"运行目录：{run_dir}")
        run(model, workspace)
        verified = True
        return 0
    finally:
        # trace 位于工具可写的 workspace 之外；只记录协议消息和用量，不保存请求头。
        trace = {
            "run_id": run_id, "model": config.model, "verified": verified,
            "api_calls": model.calls, "usage": model.usage, "messages": model.history,
        }
        (run_dir / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(f"轨迹已保存：{run_dir / 'trace.json'}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ModelError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise SystemExit(f"运行失败：{error}") from None
