"""固定任务与 Harness，对模型候选做可复现的本地评估。"""

import argparse
import contextlib
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import DEFAULT_TASK, PROJECT_ROOT, SYSTEM_PROMPT, run, save_json
from metrics import score_trial, summarize


def load_models(path: Path, selected: list[str] | None = None) -> tuple[list, list]:
    env = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip("\"'")
    env.update(os.environ)
    specs = json.loads(path.read_text(encoding="utf-8"))["models"]
    ids = [item["id"] for item in specs]
    if len(set(ids)) != len(ids) or (selected and set(selected) - set(ids)):
        raise ValueError("模型 id 重复或所选 id 不存在")
    public, private = [], []
    for spec in specs:
        if selected and spec["id"] not in selected:
            continue
        allowed = {"id", "driver", "model", "base_url", "key_env", "temperature", "reasoning_effort", "pricing"}
        if set(spec) - allowed or not re.fullmatch(r"[a-zA-Z0-9_-]+", spec["id"]):
            raise ValueError("模型配置字段或 id 无效；配置只引用环境变量，不填写密钥")
        url = urlsplit(spec["base_url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url 必须是无账号和查询参数的 HTTPS 地址")
        if spec["driver"] not in {"deepseek", "openai_compatible"}:
            raise ValueError("暂只支持 deepseek / openai_compatible 驱动")
        key = env.get(spec["key_env"], "").strip()
        if not key:
            raise ValueError(f"缺少环境变量：{spec['key_env']}")
        pricing = spec.get("pricing", {})
        if set(pricing) - {"currency", "basis", "source", "input_uncached_per_million", "input_cached_per_million", "output_per_million"}:
            raise ValueError("pricing 含不支持的字段")
        for field in ("input_uncached_per_million", "input_cached_per_million", "output_per_million"):
            value = pricing.get(field)
            if value is not None:
                number = Decimal(str(value))
                if not number.is_finite() or number < 0:
                    raise ValueError("单价必须是非负有限数值或 null")
        if any(pricing.get(field) is not None for field in ("input_uncached_per_million", "input_cached_per_million", "output_per_million")):
            if not pricing.get("currency") or not pricing.get("basis"):
                raise ValueError("配置价格时必须注明 currency 和 basis")
        public.append(spec)
        private.append({**{key: value for key, value in spec.items() if key not in {"id", "key_env", "pricing"}},
                        "api_key": key, "timeout": 30})
    if not public:
        raise ValueError("没有选择任何模型")
    return public, private


def save_reports(output: Path, rows: list[dict]) -> None:
    summary = summarize(rows)
    save_json(output / "summary.json", summary)
    planned = len(json.loads((output / "manifest.json").read_text(encoding="utf-8"))["schedule"])
    save_json(output / "progress.json", {"planned": planned, "scored": len(rows), "complete": len(rows) == planned})
    import csv
    with (output / "trials.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 模型评估结果", "", f"已评分 {len(rows)}/{planned} 项。固定任务与 Harness 的小样本评估；费用按配置价格估算，不等于账单。", "",
             "| 候选 | 成功数/总数 | 完成率 | 平均秒数 | P95秒数 | 平均请求数 | 有效工具调用率 | 恢复率 | 估算成本/成功 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def percentage(value):
        return f"{value:.1%}" if value is not None else "未知/不适用"
    for item in summary:
        cost = item["estimated_cost_per_success"]
        formatted = f"{Decimal(cost):.6f} {item['currency']}" if cost is not None else "未知/无成功"
        lines.append(f"| {item['model_id']} | {item['passed']}/{item['trials']} | {percentage(item['success_rate'])} | "
                     f"{item['mean_seconds']:.2f} | {item['p95_seconds']:.2f} | {item['mean_requests']:.2f} | "
                     f"{percentage(item['valid_tool_call_rate'])} | {percentage(item['recovery_rate'])} | {formatted} |")
    lines += ["", "完整口径、费用依据及缺失状态见 summary.json；逐样例轨迹与判断见 trials.jsonl。",
              "少量样例不支持模型排名或稳定性结论；请按任务类别、失败轨迹和多轮结果判断。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(models_file: Path, repeats: int = 1, model_ids=None, case_ids=None,
             output: Path | None = None, seed: int = 42, worker=None) -> Path:
    if not 1 <= repeats <= 20:
        raise ValueError("repeats 范围为 1 到 20")
    public, configs = load_models(models_file, model_ids)
    suite = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    cases = [case for case in suite["cases"] if not case_ids or case["id"] in case_ids]
    if not cases or (case_ids and set(case_ids) - {case["id"] for case in cases}):
        raise ValueError("样例 id 不存在")
    output = (output or PROJECT_ROOT / ".runs/09_model_evaluation" /
              (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)  # 一次实验独立目录，禁止混入旧轨迹。
    schedule = [(repeat, case, index) for repeat in range(1, repeats + 1) for case in cases for index in range(len(public))]
    random.Random(seed).shuffle(schedule)
    code_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ("agent.py", "tools.py", "metrics.py", "evaluate.py", "cases.json")}
    save_json(output / "manifest.json", {"models": public, "suite": suite, "seed": seed,
        "repeats": repeats, "system_prompt": SYSTEM_PROMPT, "max_turns": 8, "max_tokens": 1024,
        "code_sha256": code_hashes,
        "packages": {name: importlib.metadata.version(name) for name in ("langgraph", "langchain-deepseek", "langchain-openai")},
        "schedule": [{"repeat": repeat, "case": case["id"], "model_id": public[index]["id"]} for repeat, case, index in schedule]})
    print(f"评估目录：{output}；计划 {len(schedule)} 次任务，最多 {len(schedule) * 8} 次模型请求")
    worker, rows = worker or run, []
    for position, (repeat, case, index) in enumerate(schedule, 1):
        spec, config = public[index], configs[index]
        directory = output / f"{spec['id']}--{case['id']}--r{repeat}"
        source = directory / "input"
        source.mkdir(parents=True)
        (source / "income.csv").write_text(case["income_csv"], encoding="utf-8")
        save_json(source / "rates.json", suite["rates"])
        error_type = None
        start = time.perf_counter()
        # SDK 错误只记录类型，控制台日志去除所有已加载的密钥。
        from io import StringIO
        log = StringIO()
        try:
            with contextlib.redirect_stdout(log):
                worker(config, task=DEFAULT_TASK, run_dir=directory, source_root=source,
                       max_turns=8, fail_write_once=case.get("fail_write_once", False))
        except Exception as error:
            error_type = type(error).__name__
        elapsed = time.perf_counter() - start
        text = log.getvalue()
        for private in configs:
            text = text.replace(private["api_key"], "[redacted]")
        (directory / "console.log").write_text(text, encoding="utf-8")
        row = {"model_id": spec["id"], "case_id": case["id"], "repeat": repeat,
               **score_trial(case, suite["rates"], directory, elapsed, error_type, spec.get("pricing", {}))}
        rows.append(row)
        with (output / "trials.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        save_reports(output, rows)  # 每项结束后更新，意外中断也保留已完成测量。
        print(f"[{position}/{len(schedule)}] {spec['id']} / {case['id']} / r{repeat}: "
              f"passed={row['passed']}，{elapsed:.2f}s，requests={row['requests']}，error={error_type}")
    print(f"评估完成：{output / 'report.md'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models.example.json")
    parser.add_argument("--model", action="append", help="候选 id，可重复")
    parser.add_argument("--case", action="append", help="样例 id，可重复")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.models, args.repeats, args.model, args.case, args.output, args.seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"评估未完成：{type(error).__name__}；请检查候选配置、密钥环境变量和输出目录") from None
