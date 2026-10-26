"""第二课离线测试：模拟 HTTP 响应，不调用付费接口。"""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import json
import unittest

from deepseek_model import DeepSeekModel, ModelConfig, ModelError, load_config, request_json
from mini_harness import FinalAnswer, TASK, ToolCall, run


def response(calls=None, text=None, finish_reason=None):
    return {"choices": [{
        "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
        "message": {"role": "assistant", "content": text, "tool_calls": calls},
    }], "usage": {"prompt_tokens": 20, "completion_tokens": 10}}


def tool(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


class DeepSeekTests(unittest.TestCase):
    def test_environment_overrides_local_config_and_secret_is_not_in_repr(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
            "DEEPSEEK_MODEL": "environment-model",
        }, clear=True):
            root = Path(directory)
            (root / ".env").write_text(
                "DEEPSEEK_API_KEY=local-test-secret\nDEEPSEEK_MODEL=file-model\n", encoding="utf-8",
            )
            config = load_config(root)
        self.assertEqual(config.model, "environment-model")
        self.assertNotIn("local-test-secret", repr(config))

    def test_missing_key_and_invalid_endpoint_are_rejected(self):
        with TemporaryDirectory() as directory:
            for values in ({}, {"DEEPSEEK_API_KEY": "test", "DEEPSEEK_BASE_URL": "http://example.com"}):
                with self.subTest(values=values), patch.dict("os.environ", values, clear=True):
                    with self.assertRaises(ModelError):
                        load_config(Path(directory))

    def test_request_sends_expected_endpoint_headers_and_timeout(self):
        with patch("deepseek_model.urlopen", return_value=BytesIO(b'{"choices": []}')) as send:
            self.assertEqual(request_json(ModelConfig("test-secret"), {"model": "deepseek-flash"}),
                             {"choices": []})
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(send.call_args.kwargs["timeout"], 30)

    def test_http_errors_redact_key_and_are_not_retried(self):
        error = HTTPError("https://api.deepseek.com", 401, "Unauthorized", {},
                          BytesIO(b'{"error": "invalid test-secret"}'))
        with patch("deepseek_model.urlopen", side_effect=error) as send:
            with self.assertRaises(ModelError) as caught:
                request_json(ModelConfig("test-secret"), {})
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertEqual(send.call_count, 1)

    def test_network_failure_is_explicit_and_not_retried(self):
        with patch("deepseek_model.urlopen", side_effect=URLError("timed out")) as send:
            with self.assertRaisesRegex(ModelError, "网络请求失败"):
                request_json(ModelConfig("test-secret"), {})
        self.assertEqual(send.call_count, 1)

    def test_multiple_calls_keep_ids_and_wait_for_all_results(self):
        replies = [response(calls=[tool("one", "list_files", {"path": "."}),
                                   tool("two", "read_file", {"path": "notes.txt"})]),
                   response(text="已完成")]
        messages = [{"role": "user", "content": TASK}]
        with patch("deepseek_model.request_json", side_effect=replies) as send, redirect_stdout(StringIO()):
            model = DeepSeekModel(ModelConfig("test-secret"))
            self.assertEqual(model(messages), ToolCall("list_files", {"path": "."}))
            messages.extend([{"role": "assistant", "tool": "list_files"},
                             {"role": "tool", "name": "list_files", "content": {"ok": True}}])
            self.assertEqual(model(messages), ToolCall("read_file", {"path": "notes.txt"}))
            self.assertEqual(send.call_count, 1)
            messages.extend([{"role": "assistant", "tool": "read_file"},
                             {"role": "tool", "name": "read_file", "content": {"ok": True, "data": "x"}}])
            self.assertEqual(model(messages), FinalAnswer("已完成"))
        second_payload = send.call_args.args[1]
        tool_results = [item for item in second_payload["messages"] if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in tool_results], ["one", "two"])
        self.assertEqual(model.usage, {"prompt_tokens": 40, "completion_tokens": 20})
        self.assertEqual(second_payload["thinking"], {"type": "disabled"})

    def test_bad_tool_arguments_are_rejected_before_execution(self):
        for arguments in (["notes.txt"], {"path": 123}, {"path": "notes.txt", "extra": "x"}):
            with self.subTest(arguments=arguments):
                reply = response(calls=[tool("one", "read_file", arguments)])
                with patch("deepseek_model.request_json", return_value=reply), redirect_stdout(StringIO()):
                    with self.assertRaises(ModelError):
                        DeepSeekModel(ModelConfig("test-secret"))([{"role": "user", "content": TASK}])

    def test_truncated_response_does_not_execute_partial_tool_call(self):
        reply = response(calls=[tool("one", "read_file", {"path": "notes.txt"})], finish_reason="length")
        with patch("deepseek_model.request_json", return_value=reply), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ModelError, "未完整结束"):
                DeepSeekModel(ModelConfig("test-secret"))([{"role": "user", "content": TASK}])

    def test_adapter_and_original_harness_complete_file_task(self):
        report = json.dumps({"source": "notes.txt", "non_empty_lines": 2})
        replies = [response(calls=[tool("read", "read_file", {"path": "notes.txt"})]),
                   response(calls=[tool("write", "write_file", {
                       "path": "output/report.json", "content": report,
                   })]), response(text="已完成")]
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            (root / "notes.txt").write_text("a\n\nb\n", encoding="utf-8")
            with patch("deepseek_model.request_json", side_effect=replies) as send:
                run(DeepSeekModel(ModelConfig("test-secret")), root)
            self.assertEqual(send.call_count, 3)
            self.assertEqual(json.loads((root / "output/report.json").read_text(encoding="utf-8")),
                             {"source": "notes.txt", "non_empty_lines": 2})


if __name__ == "__main__":
    unittest.main()
