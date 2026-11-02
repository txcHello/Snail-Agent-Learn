"""第七课的外层：发现 → 选择 → ReAct → 验收 → 记账 → 再选择。"""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APIConnectionError, InternalServerError, RateLimitError

from agent import DEFAULT_TASK, PROJECT_ROOT, load_config, record, run, save_json
from tools import VerificationError, calculate_expected, validate_report

FILES = ("income.csv", "rates.json")
DEFAULT_INBOX = ROOT / "demo_inbox"
DEFAULT_WORK_DIR = PROJECT_ROOT / ".runs" / "07_loop_engineering" / "default"
MAX_ATTEMPTS = 2  # 每项任务总共最多两次，跨进程重启也不重置。


@contextmanager
def single_runner(work_dir: Path):
    """同一个状态目录只允许一个调度进程；进程退出后操作系统释放锁。"""
    with (work_dir / "runner.lock").open("a+b") as file:
        file.seek(0)
        if os.name == "nt":
            import msvcrt
            if os.fstat(file.fileno()).st_size == 0:
                file.write(b"0")
                file.flush()
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("已有进程在使用这个状态目录") from error
        else:
            import fcntl
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("已有进程在使用这个状态目录") from error
        try:
            yield
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


def persist(work_dir: Path, state: dict) -> None:
    """先写临时文件再替换，避免直接覆盖 state.json 时留下半份 JSON。"""
    temporary = work_dir / "state.tmp"
    save_json(temporary, state)
    temporary.replace(work_dir / "state.json")


def transition(work_dir: Path, state: dict, task_id: str, status: str, **updates):
    task = state["tasks"][task_id]
    task.update(status=status, **updates)
    persist(work_dir, state)  # 状态先落盘，日志不承担恢复依据。
    record(work_dir, "task_status", task_id=task_id, **task)
    print(f"[任务] {task['name']} → {status}（已尝试 {task['attempts']}/{MAX_ATTEMPTS}）")


def discover(inbox: Path, work_dir: Path, state: dict) -> None:
    """一个子目录是一项任务；名称和两份文件内容共同决定任务身份。"""
    if not inbox.is_dir():
        raise ValueError("收件目录不存在")
    for folder in sorted(inbox.iterdir()):
        if not folder.is_dir():
            continue
        if not all((folder / name).is_file() for name in FILES):
            print(f"[忽略未就绪目录] {folder.name}：需要 income.csv 和 rates.json")
            continue
        content = {name: (folder / name).read_bytes() for name in FILES}
        identity = [folder.name, *[sha256(content[name]).hexdigest() for name in FILES]]
        task_id = sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
        if task_id in state["tasks"]:
            continue
        snapshot = work_dir / "tasks" / task_id / "input"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, data in content.items():
            (snapshot / name).write_bytes(data)
        state["tasks"][task_id] = {"name": folder.name, "status": "pending", "attempts": 0,
                                    "last_error": None, "report": None}
        transition(work_dir, state, task_id, "pending")


def attempt_dir(work_dir: Path, task_id: str, attempt: int) -> Path:
    return work_dir / "tasks" / task_id / f"attempt-{attempt}"


def reconcile(work_dir: Path, state: dict) -> None:
    """上次中断或已完成的任务，先看真实产物，再决定下一步。"""
    for task_id, task in state["tasks"].items():
        if task["status"] not in {"running", "completed", "pending"} or not task["attempts"]:
            continue
        directory = attempt_dir(work_dir, task_id, task["attempts"])
        snapshot = work_dir / "tasks" / task_id / "input"
        try:
            validation = validate_report(directory, snapshot)
        except (ValueError, OSError, KeyError, ArithmeticError) as error:
            if task["status"] == "pending":
                continue
            # 中断任务可重跑；曾验收的产物被删改则交给人，不能悄悄重做。
            status = "pending" if task["status"] == "running" and task["attempts"] < MAX_ATTEMPTS else "failed"
            transition(work_dir, state, task_id, status, report=None,
                       last_error=f"恢复检查未通过：{type(error).__name__}")
        else:
            if task["status"] != "completed":
                transition(work_dir, state, task_id, "completed", report=validation["report"],
                           total_cny=validation["total_cny"], last_error=None)
                print("[恢复] 已有报告通过验收，无需重新调用模型")


def run_loop(config: dict, inbox: Path = DEFAULT_INBOX, work_dir: Path = DEFAULT_WORK_DIR,
             max_executions: int = 2, worker=None) -> dict:
    """处理当前可见任务；没有待办或达到本轮执行预算就退出。"""
    if max_executions < 1:
        raise ValueError("max_executions 必须至少为 1")
    worker = worker or run
    inbox, work_dir = inbox.resolve(), work_dir.resolve()
    if inbox == work_dir or inbox in work_dir.parents or work_dir in inbox.parents:
        raise ValueError("收件目录和状态目录必须相互独立")
    work_dir.mkdir(parents=True, exist_ok=True)
    with single_runner(work_dir):
        state_file = work_dir / "state.json"
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"version": 1, "tasks": {}}
        if state.get("version") != 1 or not isinstance(state.get("tasks"), dict):
            raise ValueError("无法识别状态文件；保留文件并停止")
        persist(work_dir, state)
        executions = 0
        while True:
            reconcile(work_dir, state)
            discover(inbox, work_dir, state)
            pending = [key for key, task in state["tasks"].items() if task["status"] == "pending"]
            if not pending or executions >= max_executions:
                reason = "execution_budget" if pending else "no_pending_tasks"
                break
            # 先做尚未尝试的任务，避免失败任务独占本轮预算。
            task_id = min(pending, key=lambda key: (state["tasks"][key]["attempts"], state["tasks"][key]["name"], key))
            task = state["tasks"][task_id]
            snapshot = work_dir / "tasks" / task_id / "input"
            try:
                calculate_expected(snapshot)  # 输入错误直接失败，不花模型调用来修文件。
            except (ValueError, OSError, KeyError, ArithmeticError) as error:
                transition(work_dir, state, task_id, "failed", last_error=f"输入无效：{type(error).__name__}")
                continue
            if task["attempts"]:
                time.sleep(1)  # 最多重试一次；教学用固定退避。
            directory = attempt_dir(work_dir, task_id, task["attempts"] + 1)
            transition(work_dir, state, task_id, "running", attempts=task["attempts"] + 1)
            executions += 1
            feedback = f" 上一次尝试未完成，反馈：{task['last_error']}" if task["last_error"] else ""
            try:
                worker(config, task=DEFAULT_TASK + feedback, run_dir=directory, source_root=snapshot)
                validation = validate_report(directory, snapshot)  # 不信任 worker 返回的“成功”。
            except Exception as error:
                retryable = isinstance(error, (VerificationError, APIConnectionError, InternalServerError, RateLimitError))
                status = "pending" if retryable and task["attempts"] < MAX_ATTEMPTS else "failed"
                # 不把网络响应体、请求头或密钥复制到持久状态与下一轮提示中。
                detail = str(error) if isinstance(error, VerificationError) else type(error).__name__
                transition(work_dir, state, task_id, status, last_error=detail)
            else:
                transition(work_dir, state, task_id, "completed", report=validation["report"],
                           total_cny=validation["total_cny"], last_error=None)
        counts = {status: sum(task["status"] == status for task in state["tasks"].values())
                  for status in ("pending", "running", "completed", "failed")}
        summary = {"stop_reason": reason, "executions": executions, "counts": counts}
        save_json(work_dir / "summary.json", summary)
        record(work_dir, "loop_stopped", **summary)
        print(f"[循环停止] {json.dumps(summary, ensure_ascii=False)}")
        print(f"状态文件：{state_file}")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--max-executions", type=int, default=2, help="本次最多启动几次任务尝试")
    args = parser.parse_args()
    summary = run_loop(load_config(), args.inbox, args.work_dir, args.max_executions)
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # 原始 API 错误不直接打印，避免泄露服务端响应中的配置。
        raise SystemExit(f"循环未完成：{type(error).__name__}；请检查配置、目录与状态文件") from None
