"""验证 harness 的行为边界；只使用临时目录。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from mini_harness import FinalAnswer, ToolCall, demo_model, execute_tool, run, verify_report


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "notes.txt").write_text("一行\n\n二行\n   \n三行\n", encoding="utf-8")

    def test_demo_reads_actual_file_and_writes_verified_result(self):
        with redirect_stdout(StringIO()):
            messages = run(demo_model, self.root)
        report = json.loads((self.root / "output/report.json").read_text(encoding="utf-8"))
        self.assertEqual(report, {"source": "notes.txt", "non_empty_lines": 3})
        tools = [message["name"] for message in messages if message["role"] == "tool"]
        self.assertEqual(tools, ["list_files", "read_file", "write_file"])

    def test_tool_requests_cannot_escape_workspace(self):
        for path in ("../outside.txt", str(self.root.parent / "outside.txt")):
            with self.subTest(path=path):
                result = execute_tool(self.root, ToolCall("read_file", {"path": path}))
                self.assertFalse(result["ok"])

    def test_model_cannot_overwrite_source_file(self):
        source = self.root / "notes.txt"
        original = source.read_bytes()
        result = execute_tool(self.root, ToolCall("write_file", {
            "path": "notes.txt", "content": "修改原始资料",
        }))
        self.assertFalse(result["ok"])
        self.assertEqual(source.read_bytes(), original)

    def test_missing_file_is_returned_as_feedback(self):
        received_feedback = []

        def model(messages):
            if len(messages) > 1:
                received_feedback.append(messages[-1]["content"])
            return ToolCall("read_file", {"path": "missing.txt"})

        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "轮上限"):
            run(model, self.root, max_steps=2)
        self.assertEqual(len(received_feedback), 1)
        self.assertFalse(received_feedback[0]["ok"])

    def test_claiming_done_does_not_pass_verification(self):
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "验收未通过"):
            run(lambda messages: FinalAnswer("我已完成全部工作"), self.root)

    def test_incorrect_artifact_fails_verification(self):
        (self.root / "output").mkdir()
        (self.root / "output/report.json").write_text(
            '{"source": "notes.txt", "non_empty_lines": 999}', encoding="utf-8",
        )
        self.assertFalse(verify_report(self.root))

    def test_unknown_tool_is_rejected(self):
        result = execute_tool(self.root, ToolCall("run_shell", {"path": "."}))
        self.assertFalse(result["ok"])

    def test_correct_count_with_wrong_report_fields_is_rejected(self):
        # 真实模型曾返回这个形状：计数正确，但输出字段不符合约定。
        (self.root / "output").mkdir()
        (self.root / "output/report.json").write_text(
            '{"file": "notes.txt", "total_lines": 5, "non_empty_lines": 3}', encoding="utf-8",
        )
        self.assertFalse(verify_report(self.root))

    def test_report_count_must_be_an_integer(self):
        (self.root / "output").mkdir()
        (self.root / "output/report.json").write_text(
            '{"source": "notes.txt", "non_empty_lines": 3.0}', encoding="utf-8",
        )
        self.assertFalse(verify_report(self.root))


if __name__ == "__main__":
    unittest.main()
