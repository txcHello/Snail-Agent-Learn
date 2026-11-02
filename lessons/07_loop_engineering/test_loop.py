"""外层任务循环的离线集成测试：真实文件与状态，替换昂贵的模型调用。"""

import contextlib
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

import agent
import loop
from tools import calculate_expected, create_tools

CONFIG = {"api_key": "test-secret-only", "model": "deepseek-flash",
          "base_url": "https://api.deepseek.com", "timeout": 30}


def write_real_report(config, *, task, run_dir, source_root):
    """替换模型决策，但保留真实计算、写文件和外层验收。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    tools = {item.name: item for item in create_tools(run_dir, source_root)}
    tools["calculate_totals"].invoke({})
    tools["write_report"].invoke({})
    return {"validation": {"status": "claimed_success"}}


class LoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inbox, self.work = self.root / "inbox", self.root / "work"
        shutil.copytree(loop.DEFAULT_INBOX, self.inbox)
        self.output = StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        sleeper = patch("loop.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def execute(self, worker=write_real_report, budget=2):
        return loop.run_loop(CONFIG, self.inbox, self.work, budget, worker)

    def state(self):
        return json.loads((self.work / "state.json").read_text(encoding="utf-8"))

    def one_task(self):
        shutil.rmtree(self.inbox / "02_november")

    def test_two_tasks_complete_and_restart_calls_no_worker(self):
        worker = Mock(side_effect=write_real_report)
        result = self.execute(worker)
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertEqual(sorted(task["total_cny"] for task in self.state()["tasks"].values()), ["2004.00", "244.00"])
        resumed = self.execute(worker)
        self.assertEqual(resumed["executions"], 0)
        self.assertEqual(worker.call_count, 2)
        self.assertNotIn(CONFIG["api_key"], (self.work / "state.json").read_text(encoding="utf-8"))

    def test_execution_budget_preserves_pending_task_for_next_start(self):
        first = self.execute(budget=1)
        self.assertEqual(first["stop_reason"], "execution_budget")
        self.assertEqual(first["counts"]["pending"], 1)
        second = self.execute(budget=1)
        self.assertEqual(second["counts"]["completed"], 2)
        self.assertEqual(second["executions"], 1)

    def test_same_name_changed_content_creates_new_identity_and_keeps_old_snapshot(self):
        self.one_task()
        self.execute()
        old_id = next(iter(self.state()["tasks"]))
        income = self.inbox / "01_october" / "income.csv"
        income.write_text(income.read_text(encoding="utf-8") + "extra,CNY,1.00\n", encoding="utf-8")
        result = self.execute()
        self.assertEqual(result["executions"], 1)
        self.assertEqual(result["counts"]["completed"], 2)
        frozen = self.work / "tasks" / old_id / "input"
        self.assertEqual(calculate_expected(frozen)["total_cny"], "2004.00")
        self.assertEqual(sorted(task["total_cny"] for task in self.state()["tasks"].values()), ["2004.00", "2005.00"])

    def test_claimed_success_without_artifact_retries_then_fails_persistently(self):
        self.one_task()
        worker = Mock(return_value={"validation": {"status": "passed"}})
        self.assertEqual(self.execute(worker, budget=1)["counts"]["pending"], 1)
        self.assertEqual(self.execute(worker, budget=1)["counts"]["failed"], 1)
        self.assertEqual(self.execute(worker)["executions"], 0)
        self.assertEqual(worker.call_count, 2)

    def test_failed_validation_feedback_reaches_new_attempt_and_can_recover(self):
        self.one_task()
        calls = []

        def worker(config, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return write_real_report(config, **kwargs)
            # 无报告，模拟 Agent 提前声称结束。
            return {}

        result = self.execute(worker)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertIn("报告不存在", calls[1]["task"])
        self.assertNotEqual(calls[0]["run_dir"], calls[1]["run_dir"])

    def test_interrupted_running_without_report_restarts_with_next_attempt(self):
        self.one_task()
        worker = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(worker)
        task = next(iter(self.state()["tasks"].values()))
        self.assertEqual(task["status"], "running")
        result = self.execute()
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(next(iter(self.state()["tasks"].values()))["attempts"], 2)

    def test_report_written_before_interrupt_is_adopted_without_second_worker(self):
        self.one_task()

        def interrupted(config, **kwargs):
            write_real_report(config, **kwargs)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.execute(interrupted)
        worker = Mock(side_effect=AssertionError("不应重复执行"))
        result = self.execute(worker)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["executions"], 0)
        worker.assert_not_called()

    def test_invalid_input_fails_without_model_and_does_not_block_next_task(self):
        (self.inbox / "01_october" / "income.csv").write_text("id,currency,amount\nx,CNY,NaN\n", encoding="utf-8")
        worker = Mock(side_effect=write_real_report)
        result = self.execute(worker)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(worker.call_count, 1)

    def test_non_retryable_error_is_recorded_without_sensitive_exception_body(self):
        self.one_task()
        worker = Mock(side_effect=PermissionError(CONFIG["api_key"]))
        result = self.execute(worker)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(worker.call_count, 1)
        task = next(iter(self.state()["tasks"].values()))
        self.assertEqual(task["last_error"], "PermissionError")
        self.assertNotIn(CONFIG["api_key"], (self.work / "trace.jsonl").read_text(encoding="utf-8"))

    def test_connection_error_retries_with_persisted_budget_and_safe_feedback(self):
        self.one_task()
        calls = []

        def worker(config, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise loop.APIConnectionError(message=CONFIG["api_key"], request=Mock())
            return write_real_report(config, **kwargs)

        self.assertEqual(self.execute(worker, budget=1)["counts"]["pending"], 1)
        self.assertEqual(self.execute(worker, budget=1)["counts"]["completed"], 1)
        self.assertIn("APIConnectionError", calls[1]["task"])
        self.assertNotIn(CONFIG["api_key"], calls[1]["task"])

    def test_valid_report_after_worker_error_is_adopted_before_retry(self):
        self.one_task()

        def interrupted(config, **kwargs):
            write_real_report(config, **kwargs)
            raise loop.APIConnectionError(request=Mock())

        worker = Mock(side_effect=interrupted)
        result = self.execute(worker)
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["executions"], 1)
        worker.assert_called_once()

    def test_repeated_interruptions_do_not_reset_attempt_budget(self):
        self.one_task()
        worker = Mock(side_effect=KeyboardInterrupt)
        for _ in range(2):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(worker)
        result = self.execute(worker)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["executions"], 0)
        self.assertEqual(worker.call_count, 2)

    def test_completed_report_tampering_is_detected_without_silent_reexecution(self):
        self.one_task()
        self.execute()
        report = Path(next(iter(self.state()["tasks"].values()))["report"])
        report.write_text("{}", encoding="utf-8")
        worker = Mock()
        result = self.execute(worker)
        self.assertEqual(result["counts"]["failed"], 1)
        worker.assert_not_called()

    def test_empty_inbox_stops_without_work_and_incomplete_folder_is_ignored(self):
        shutil.rmtree(self.inbox)
        (self.inbox / "uploading").mkdir(parents=True)
        worker = Mock()
        result = self.execute(worker)
        self.assertEqual(result["stop_reason"], "no_pending_tasks")
        self.assertEqual(result["executions"], 0)
        worker.assert_not_called()

    def test_corrupt_state_is_preserved_instead_of_starting_over(self):
        self.work.mkdir()
        state_file = self.work / "state.json"
        state_file.write_text("not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(state_file.read_text(encoding="utf-8"), "not json")

    def test_os_lock_rejects_second_process_then_releases(self):
        self.work.mkdir()
        code = ("import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(loop.ROOT)!r}); import loop; "
                "guard=loop.single_runner(Path(sys.argv[1])); guard.__enter__(); guard.__exit__(None,None,None)")
        command = [sys.executable, "-X", "utf8", "-c", code, str(self.work)]
        with loop.single_runner(self.work):
            blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("已有进程", blocked.stderr)
        released = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(released.returncode, 0, released.stderr)

    def test_outer_loop_invokes_real_langgraph_with_task_specific_input(self):
        self.one_task()
        (self.inbox / "01_october" / "income.csv").write_text("id,currency,amount\nx,CNY,12.34\n", encoding="utf-8")
        requests = [AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": str(index), "type": "tool_call"}])
                    for index, name in enumerate(("calculate_totals", "write_report"))]
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [*requests, AIMessage(content="完成")]
        with patch("agent.ChatDeepSeek", return_value=model):
            result = self.execute(worker=agent.run)
        self.assertEqual(result["counts"]["completed"], 1)
        task = next(iter(self.state()["tasks"].values()))
        self.assertEqual(task["total_cny"], "12.34")
        self.assertEqual(model.invoke.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
