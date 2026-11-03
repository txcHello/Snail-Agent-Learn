"""验证指标分子分母、金标准评分、缺失费用及评估到 LangGraph 的集成。"""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

import agent
import evaluate
from metrics import score_trial, summarize
from tools import calculate_expected

CONFIG = {"api_key": "test-secret-only", "model": "deepseek-flash", "base_url": "https://api.deepseek.com", "timeout": 30}
PRICE = {"currency": "USD", "basis": "test arithmetic only", "input_uncached_per_million": "2",
         "input_cached_per_million": "0.2", "output_per_million": "4"}


def response(calls=(), *, usage=True, finish="tool_calls"):
    result = AIMessage(content="" if calls else "完成", tool_calls=[
        {"name": name, "args": args, "id": identifier, "type": "tool_call"} for name, args, identifier in calls],
        response_metadata={"finish_reason": finish})
    if usage:
        result.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                                 "input_token_details": {"cache_read": 20}}
    return result


def normal_responses():
    return [response([("read_file", {"path": "income.csv"}, "read1"), ("read_file", {"path": "rates.json"}, "read2")]),
            response([("calculate_totals", {}, "calc")]), response([("write_report", {}, "write")]),
            response(finish="stop")]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "trial"
        self.source = self.directory / "input"
        self.source.mkdir(parents=True)
        self.suite = json.loads((evaluate.ROOT / "cases.json").read_text(encoding="utf-8"))
        self.case = self.suite["cases"][0]
        self.prepare(self.case)
        redirect = contextlib.redirect_stdout(StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def prepare(self, case):
        (self.source / "income.csv").write_text(case["income_csv"], encoding="utf-8")
        agent.save_json(self.source / "rates.json", self.suite["rates"])

    def run_graph(self, responses, case=None):
        case = case or self.case
        self.prepare(case)
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = responses
        error_type = None
        with patch("agent.ChatDeepSeek", return_value=model):
            try:
                agent.run(CONFIG, run_dir=self.directory, source_root=self.source,
                          fail_write_once=case.get("fail_write_once", False))
            except Exception as error:
                error_type = type(error).__name__
        return score_trial(case, self.suite["rates"], self.directory, 2.0, error_type, PRICE)

    def test_all_manual_goldens_match_hand_checked_fixtures(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                self.prepare(case)
                expected = {"base": "CNY", "rate_note": self.suite["rates"]["note"], **case["expected"]}
                self.assertEqual(calculate_expected(self.source), expected)

    def test_full_graph_scores_success_and_cached_token_cost(self):
        row = self.run_graph(normal_responses())
        self.assertTrue(row["passed"])
        self.assertEqual(row["requests"], 4)
        self.assertEqual(row["tool_calls"], 4)
        self.assertEqual(row["input_tokens_observed"], 400)
        self.assertEqual(row["cached_tokens_observed"], 80)
        self.assertEqual(row["estimated_cost"], "0.000816")

    def test_missing_usage_is_unknown_cost_not_zero(self):
        messages = normal_responses()
        messages[-1].usage_metadata = None
        row = self.run_graph(messages)
        self.assertTrue(row["passed"])
        self.assertFalse(row["usage_complete"])
        self.assertIsNone(row["estimated_cost"])

    def test_missing_cache_breakdown_with_different_rates_is_unknown(self):
        messages = normal_responses()
        for message in messages:
            message.usage_metadata.pop("input_token_details")
        row = self.run_graph(messages)
        self.assertTrue(row["usage_complete"])
        self.assertIsNone(row["estimated_cost"])

    def test_truncated_response_is_recorded_and_its_cost_is_not_lost(self):
        row = self.run_graph([response(finish="length")])
        self.assertFalse(row["passed"])
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertEqual(row["estimated_cost"], "0.000204")

    def test_network_error_keeps_unknown_usage_and_failure(self):
        row = self.run_graph([RuntimeError("network test")])
        self.assertFalse(row["passed"])
        self.assertEqual(row["requests"], 1)
        self.assertFalse(row["usage_complete"])
        self.assertIsNone(row["estimated_cost"])

    def test_injected_failure_is_separate_from_other_tool_errors_and_recovers(self):
        case = self.suite["cases"][2]
        messages = normal_responses()
        messages.insert(3, response([("write_report", {}, "retry")]))
        row = self.run_graph(messages, case)
        self.assertTrue(row["passed"])
        self.assertTrue(row["recovered"])
        self.assertEqual(row["injected_errors"], 1)
        self.assertEqual(row["tool_errors"], 0)

    def test_valid_artifact_without_required_reads_fails_workflow(self):
        row = self.run_graph(normal_responses()[1:])
        self.assertTrue(row["artifact_pass"])
        self.assertFalse(row["workflow_pass"])
        self.assertFalse(row["passed"])

    def test_unknown_tool_and_invalid_arguments_are_counted(self):
        messages = [response([("nonexistent", {}, "bad1")]),
                    response([("read_file", {}, "bad2")]), *normal_responses()]
        row = self.run_graph(messages)
        self.assertTrue(row["artifact_pass"])
        self.assertFalse(row["passed"])
        self.assertEqual(row["unknown_tools"], 1)
        self.assertEqual(row["invalid_args"], 1)
        self.assertEqual(row["tool_errors"], 2)

    def test_unparseable_tool_call_is_counted_and_stops_the_trial(self):
        message = response()
        message.invalid_tool_calls = [{"name": "read_file", "args": "{bad-json", "id": "bad", "error": "parse error", "type": "invalid_tool_call"}]
        row = self.run_graph([message])
        self.assertFalse(row["passed"])
        self.assertEqual(row["invalid_calls"], 1)
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(row["estimated_cost"], "0.000204")

    def test_read_boundary_request_fails_even_if_later_artifact_is_correct(self):
        row = self.run_graph([response([("read_file", {"path": "../.env"}, "unsafe")]), *normal_responses()])
        self.assertTrue(row["artifact_pass"])
        self.assertFalse(row["passed"])
        self.assertEqual(row["boundary_requests"], 1)

    def test_corrupted_artifact_is_rejected_against_literal_golden(self):
        row = self.run_graph(normal_responses())
        report = self.directory / "output/report.json"
        data = json.loads(report.read_text(encoding="utf-8"))
        data["total_cny"] = "999.00"
        agent.save_json(report, data)
        row = score_trial(self.case, self.suite["rates"], self.directory, 1, None, PRICE)
        self.assertFalse(row["artifact_pass"])
        self.assertFalse(row["passed"])

    def test_weighted_ratios_percentiles_and_failed_trial_cost(self):
        first = {"model_id": "candidate", **self.run_graph(normal_responses())}
        second = deepcopy(first)
        second.update(passed=False, tool_calls=1, invalid_args=1, elapsed_seconds=10.0)
        summary = summarize([first, second])[0]
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["valid_tool_call_rate"], 4 / 5)
        self.assertEqual(summary["p50_seconds"], 6)
        self.assertEqual(summary["p95_seconds"], 10)
        self.assertEqual(summary["estimated_cost_per_success"], "0.001632")
        self.assertIsNone(summary["recovery_rate"])
        second.update(injected_errors=1, tool_results=2, tool_errors=1)
        self.assertEqual(summarize([first, second])[0]["non_injected_tool_error_rate"], 1 / 5)
        second["estimated_cost"] = None
        self.assertIsNone(summarize([first, second])[0]["estimated_total_cost"])

    def test_zero_success_has_no_cost_per_success(self):
        row = {"model_id": "candidate", **self.run_graph([response(finish="stop")])}
        summary = summarize([row])[0]
        self.assertEqual(summary["success_rate"], 0)
        self.assertIsNone(summary["estimated_cost_per_success"])
        self.assertIsNone(summary["valid_tool_call_rate"])

    def test_evaluator_runs_real_graph_and_writes_public_reports(self):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = normal_responses()
        output = self.root / "evaluation"
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": CONFIG["api_key"]}), patch("agent.ChatDeepSeek", return_value=model):
            evaluate.evaluate(evaluate.ROOT / "models.example.json", model_ids=["deepseek_flash"],
                              case_ids=["basic"], output=output)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(summary["success_rate"], 1)
        self.assertEqual(json.loads((output / "progress.json").read_text(encoding="utf-8")),
                         {"planned": 1, "scored": 1, "complete": True})
        for path in output.rglob("*"):
            if path.is_file():
                self.assertNotIn(CONFIG["api_key"], path.read_text(encoding="utf-8-sig"), path.name)
        for filename in ("trials.csv", "trials.jsonl", "report.md", "manifest.json"):
            self.assertTrue((output / filename).is_file())
        with self.assertRaises(FileExistsError):
            evaluate.evaluate(evaluate.ROOT / "models.example.json", model_ids=["deepseek_flash"], output=output)

    def test_invalid_configuration_is_rejected_before_any_api_request(self):
        specs = json.loads((evaluate.ROOT / "models.example.json").read_text(encoding="utf-8"))
        specs["models"][0]["api_key"] = "never-store-a-key-here"
        path = self.root / "bad.json"
        agent.save_json(path, specs)
        with self.assertRaises(ValueError):
            evaluate.load_models(path)

    def test_openai_compatible_driver_does_not_receive_deepseek_options(self):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = normal_responses()
        with patch("agent.ChatOpenAI", return_value=model) as factory:
            agent.run({**CONFIG, "driver": "openai_compatible"}, run_dir=self.directory, source_root=self.source)
        self.assertNotIn("extra_body", factory.call_args.kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
