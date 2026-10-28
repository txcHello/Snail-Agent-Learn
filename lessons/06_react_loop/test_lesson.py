"""离线验证 ReAct 反馈循环、金额计算、结束验收和可查看的轨迹。"""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

import agent
from tools import calculate_expected, create_tools, validate_report

CONFIG = {"api_key": "test-secret-only", "model": "deepseek-flash", "base_url": "https://api.deepseek.com", "timeout": 30}


def request(name, args=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}])


class ReActTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tools = {item.name: item for item in create_tools(self.root)}
        output = contextlib.redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def model(self, responses):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = responses
        return model

    def test_sample_calculation_matches_hand_checked_amounts(self):
        result = calculate_expected()
        self.assertEqual({item["currency"]: item["cny"] for item in result["items"]},
                         {"CNY": "300.00", "USD": "1080.00", "EUR": "624.00"})
        self.assertEqual(result["total_cny"], "2004.00")

    def test_rounding_is_per_currency_after_grouping(self):
        (self.root / "income.csv").write_text("id,currency,amount\na,USD,0.005\nb,USD,0.005\n", encoding="utf-8")
        (self.root / "rates.json").write_text('{"base":"CNY","note":"test","cny_per_unit":{"USD":"1"}}', encoding="utf-8")
        self.assertEqual(calculate_expected(self.root)["total_cny"], "0.01")
        (self.root / "income.csv").write_text("id,currency,amount\na,USD,NaN\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            calculate_expected(self.root)

    def test_tool_boundaries_and_calculate_before_write(self):
        for path in ("../.env", "agent.py", str(self.root / "income.csv")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools["read_file"].invoke({"path": path})
        with self.assertRaisesRegex(ValueError, "先调用"):
            self.tools["write_report"].invoke({})
        self.tools["calculate_totals"].invoke({})
        self.tools["write_report"].invoke({})
        self.assertEqual(validate_report(self.root)["total_cny"], "2004.00")
        with self.assertRaises(FileExistsError):
            self.tools["write_report"].invoke({})
        for item in self.tools.values():
            self.assertNotIn("run_dir", item.args_schema.model_json_schema()["properties"])

    def test_full_cycle_preserves_call_ids_observations_and_growing_context(self):
        snapshots = []
        responses = iter([
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "income.csv"}, "id": "income", "type": "tool_call"},
                {"name": "read_file", "args": {"path": "rates.json"}, "id": "rates", "type": "tool_call"}]),
            request("calculate_totals", call_id="calc"), request("write_report", call_id="write"),
            AIMessage(content="报告已生成，总额为 2004.00 CNY。"),
        ])

        def invoke(messages):
            snapshots.append(deepcopy(messages))
            if len(snapshots) == 2:
                self.assertEqual([message.tool_call_id for message in messages[-2:]], ["income", "rates"])
                self.assertIn("sale-001", messages[-2].content)
            if len(snapshots) == 3:
                self.assertEqual(json.loads(messages[-1].content)["total_cny"], "2004.00")
            return next(responses)

        model = self.model(invoke)
        with patch("agent.ChatDeepSeek", return_value=model):
            result = agent.run(CONFIG, run_dir=self.root)
        self.assertEqual([len(items) for items in snapshots], [2, 5, 7, 9])
        self.assertEqual(result["validation"]["status"], "passed")
        trajectory = json.loads((self.root / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual([item["type"] for item in trajectory], ["human", "ai", "tool", "tool", "ai", "tool", "ai", "tool", "ai"])
        self.assertEqual(json.loads((self.root / "prefix.json").read_text(encoding="utf-8"))["system_prompt"], agent.SYSTEM_PROMPT)
        records = [json.loads(line) for line in (self.root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[-1]["event"], "validation_passed")
        self.assertNotIn(CONFIG["api_key"], "".join(path.read_text(encoding="utf-8") for path in self.root.glob("*.json*")))

    def test_failure_observation_allows_model_to_correct_next_action(self):
        model = self.model([request("write_report"), request("calculate_totals", call_id="calc"),
                            request("write_report", call_id="write"), AIMessage(content="完成")])
        with patch("agent.ChatDeepSeek", return_value=model):
            result = agent.run(CONFIG, run_dir=self.root)
        failure = result["messages"][3]
        self.assertEqual(failure.status, "error")
        self.assertIn("先调用", failure.content)
        self.assertEqual(result["validation"]["status"], "passed")

    def test_ending_without_report_is_rejected(self):
        model = self.model([AIMessage(content="报告已经生成。")])
        with patch("agent.ChatDeepSeek", return_value=model), self.assertRaisesRegex(ValueError, "报告不存在"):
            agent.run(CONFIG, run_dir=self.root)
        self.assertFalse((self.root / "validation.json").exists())
        self.assertIn("validation_failed", (self.root / "trace.jsonl").read_text(encoding="utf-8"))

    def test_changed_report_is_rejected(self):
        self.tools["calculate_totals"].invoke({})
        self.tools["write_report"].invoke({})
        path = self.root / "output/report.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["total_cny"] = "9999.00"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不一致"):
            validate_report(self.root)

    def test_budget_limits_repeated_tools(self):
        model = self.model(lambda _: request("read_file", {"path": "income.csv"}))
        with patch("agent.ChatDeepSeek", return_value=model), self.assertRaisesRegex(RuntimeError, "2 次"):
            agent.run(CONFIG, run_dir=self.root, max_turns=2)
        self.assertEqual(model.invoke.call_count, 2)

    def test_truncated_response_stops_before_tools(self):
        message = request("calculate_totals")
        message.response_metadata = {"finish_reason": "length"}
        model = self.model([message])
        with patch("agent.ChatDeepSeek", return_value=model), self.assertRaisesRegex(RuntimeError, "未完整"):
            agent.run(CONFIG, run_dir=self.root)
        trace = (self.root / "trace.jsonl").read_text(encoding="utf-8")
        self.assertNotIn('"event": "tool_result"', trace)

    def test_runpy_startup_imports_sibling_without_changing_working_directory(self):
        script = Path(agent.__file__).resolve()
        command = [sys.executable, "-X", "utf8", "-c", f"import runpy; runpy.run_path({str(script)!r})"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
