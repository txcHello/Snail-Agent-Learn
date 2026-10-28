"""在全新解释器中复现 IDE 的 runpy 启动方式；不调用真实 API。"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LESSON = Path(__file__).resolve().parent
PROJECT_ROOT = LESSON.parents[1]


class StartupTests(unittest.TestCase):
    def run_python(self, args, cwd):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        # 环境变量优先于 .env；让主入口在配置校验处退出，不发送网络请求。
        env["DEEPSEEK_API_KEY"] = ""
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, "-X", "utf8", *args], cwd=cwd, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_runpy_imports_siblings_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            for cwd in (PROJECT_ROOT, elsewhere):
                for entry in ("agent.py", "trigger.py"):
                    with self.subTest(cwd=str(cwd), entry=entry):
                        script = str(LESSON / entry)
                        code = (
                            "import runpy, sys; from pathlib import Path; "
                            f"lesson = Path({str(LESSON)!r}); "
                            "assert str(lesson) not in sys.path; "
                            f"runpy.run_path({script!r}); "
                            "import tools; "
                            "assert Path(tools.__file__).resolve() == lesson / 'tools.py'"
                        )
                        result = self.run_python(["-c", code], cwd)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runpy_main_reaches_configuration_without_api_calls(self):
        for entry in ("agent.py", "trigger.py"):
            with self.subTest(entry=entry):
                code = (
                    "import runpy, sys; sys.argv = ['probe']; "
                    f"runpy.run_path({str(LESSON / entry)!r}, run_name='__main__')"
                )
                result = self.run_python(["-c", code], PROJECT_ROOT)
                self.assertEqual(result.returncode, 1)
                self.assertIn("请在根目录 .env 中设置", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_direct_script_still_reaches_configuration_from_other_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            for entry in ("agent.py", "trigger.py"):
                with self.subTest(entry=entry):
                    result = self.run_python([str(LESSON / entry)], elsewhere)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("请在根目录 .env 中设置", result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
