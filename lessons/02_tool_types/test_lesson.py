"""第二课的离线行为测试：不调用网络，不需要真实密钥。"""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

import agent
from tools import create_tools, read_file, SOURCE_ROOT
from trigger import event_to_task
from trigger import main as event_main

CONFIG = {"api_key": "test-key", "base_url": "https://api.deepseek.com", "model": "deepseek-flash", "timeout": 30}


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class LessonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name).resolve()
        self.summarize = Mock(return_value="三点摘要")
        self.tools = {item.name: item for item in create_tools(self.run_dir, self.summarize)}
        output = contextlib.redirect_stdout(StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_four_active_tools_hide_runtime_and_credentials(self):
        self.assertEqual(set(self.tools), {"read_file", "write_file", "delegate_summary", "notify_user"})
        for item in self.tools.values():
            properties = item.args_schema.model_json_schema()["properties"]
            self.assertFalse({"config", "api_key", "run_dir", "summarize"} & set(properties))

    def test_perception_reads_actual_file_and_blocks_escape(self):
        content = read_file.invoke({"path": "notes.txt"})
        self.assertEqual(sum(bool(line.strip()) for line in content.splitlines()), 4)
        for path in ("../agent.py", str(SOURCE_ROOT / "notes.txt"), "missing.txt"):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                read_file.invoke({"path": path})

    def test_execution_creates_output_without_overwriting_input_or_existing_file(self):
        writer = self.tools["write_file"]
        writer.invoke({"path": "summary.md", "content": "学习摘要"})
        self.assertEqual((self.run_dir / "output/summary.md").read_text(encoding="utf-8"), "学习摘要")
        with self.assertRaises(FileExistsError):
            writer.invoke({"path": "summary.md", "content": "覆盖"})
        for path in ("../notes.txt", str(self.run_dir / "elsewhere.txt")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                writer.invoke({"path": path, "content": "禁止越界"})
        with self.assertRaises(ValueError):
            writer.invoke({"path": "big.txt", "content": "x" * 16001})

    def test_collaboration_delegates_only_the_requested_path(self):
        self.assertEqual(self.tools["delegate_summary"].invoke({"path": "notes.txt"}), "三点摘要")
        self.summarize.assert_called_once_with("notes.txt")
        with self.assertRaises(ValueError):
            self.tools["delegate_summary"].invoke({"path": "../.env"})
        self.summarize.assert_called_once()

    def test_communication_delivers_local_json_messages(self):
        notifier = self.tools["notify_user"]
        notifier.invoke({"message": "摘要已生成"})
        notifier.invoke({"message": "请查看产物"})
        entries = [json.loads(line) for line in (self.run_dir / "inbox.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([entry["message"] for entry in entries], ["摘要已生成", "请查看产物"])
        self.assertTrue(all(entry["channel"] == "local" for entry in entries))
        for message in (" ", "x" * 2001):
            with self.subTest(message_length=len(message)), self.assertRaises(ValueError):
                notifier.invoke({"message": message})

    def test_event_is_input_and_validated_before_agent_execution(self):
        event = {"id": "notes-001", "type": "file_received", "path": "notes.txt"}
        task = event_to_task(event)
        self.assertIn("notes-001", task)
        self.assertIn("delegate_summary", task)
        invalid = [None, {**event, "type": "unknown"}, {**event, "path": "../.env"}, {**event, "path": "missing.txt"}, {**event, "id": ""}, {**event, "task": "任意指令"}]
        for value in invalid:
            with self.subTest(event=value), self.assertRaises(ValueError):
                event_to_task(value)

    def test_main_and_subagent_execute_with_independent_contexts(self):
        models = []
        run_dir = self.run_dir

        class ScriptedModel:
            def __init__(self, **config):
                self.messages = []
                models.append(self)

            def bind_tools(self, tools):
                self.names = [item.name for item in tools]
                return self

            def invoke(self, messages):
                self.messages.append(deepcopy(messages))
                turn = len(self.messages)
                if len(self.names) == 1:
                    if turn == 1:
                        return call("read_file", {"path": "notes.txt"}, "child-read")
                    assert isinstance(messages[-1], ToolMessage)
                    assert "harness" in messages[-1].content
                    return AIMessage(content="子 Agent 的三点摘要")
                if turn == 1:
                    return call("read_file", {"path": "notes.txt"}, "parent-read")
                if turn == 2:
                    return call("delegate_summary", {"path": "notes.txt"}, "delegate")
                if turn == 3:
                    return call("write_file", {"path": "summary.md", "content": messages[-1].content}, "write")
                if turn == 4:
                    assert (run_dir / "output/summary.md").is_file()
                    return call("notify_user", {"message": "摘要已生成"}, "notify")
                return AIMessage(content="任务完成")

        with patch("agent.ChatDeepSeek", ScriptedModel):
            self.assertEqual(agent.run(agent.DEFAULT_TASK, CONFIG, self.run_dir), "任务完成")
        child, parent = models
        self.assertEqual([len(child.messages), len(parent.messages)], [2, 5])
        self.assertEqual([message.type for message in child.messages[0]], ["system", "human"])
        self.assertNotIn("notify_user", child.messages[0][1].content)
        self.assertEqual(child.names, ["read_file"])
        self.assertEqual((self.run_dir / "output/summary.md").read_text(encoding="utf-8"), "子 Agent 的三点摘要")
        self.assertEqual(json.loads((self.run_dir / "inbox.jsonl").read_text(encoding="utf-8"))["message"], "摘要已生成")

    def test_invalid_event_does_not_load_credentials_or_start_agent(self):
        event_file = self.run_dir / "bad_event.json"
        event_file.write_text('{"type": "unknown"}', encoding="utf-8")
        with patch("sys.argv", ["trigger.py", str(event_file)]), patch("trigger.load_config") as load, patch("trigger.run") as run:
            with self.assertRaises(SystemExit):
                event_main()
        load.assert_not_called()
        run.assert_not_called()

    def test_main_and_subagent_budgets_are_enforced(self):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = lambda _: call("read_file", {"path": "notes.txt"}, "read")
        with patch("agent.ChatDeepSeek", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "1 次模型调用上限"):
                agent.run("持续读取", CONFIG, self.run_dir, max_turns=1)
            self.assertEqual(model.invoke.call_count, 1)
            model.invoke.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "3 次模型调用上限"):
                agent.create_summary_agent(CONFIG)("notes.txt")
            self.assertEqual(model.invoke.call_count, 3)

    def test_tool_failure_returns_to_model(self):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [call("write_file", {"path": "../outside.txt", "content": "bad"}, "bad-write"), AIMessage(content="写入失败")]
        with patch("agent.ChatDeepSeek", return_value=model):
            graph = agent.build_graph(CONFIG, list(self.tools.values()))
            result = graph.invoke({"messages": [{"role": "user", "content": "测试越界"}], "model_calls": 0})
        self.assertEqual(result["messages"][-2].status, "error")
        self.assertFalse((self.run_dir / "outside.txt").exists())

    def test_invalid_response_stops_before_tool_execution(self):
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.return_value = AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"path": "bad.txt", "content": "bad"}, "id": "truncated", "type": "tool_call"}], response_metadata={"finish_reason": "length"})
        with patch("agent.ChatDeepSeek", return_value=model), self.assertRaises(RuntimeError):
            agent.run("测试截断", CONFIG, self.run_dir)
        self.assertFalse((self.run_dir / "output/bad.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
