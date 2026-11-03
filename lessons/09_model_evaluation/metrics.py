"""确定性评分与聚合；不调用另一个 LLM 当裁判。"""

from decimal import Decimal
import json
import math
from pathlib import Path
import statistics

from tools import create_tools


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_trial(case: dict, rates: dict, directory: Path, elapsed: float,
                error_type: str | None, pricing: dict) -> dict:
    path = directory / "trace.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    schemas = {tool.name: tool.args_schema for tool in create_tools(directory)}
    calls, outputs = {}, []
    counts = dict(requests=0, tool_calls=0, invalid_calls=0, unknown_tools=0, invalid_args=0,
                  boundary_requests=0, tool_results=0, tool_errors=0, injected_errors=0)
    read_observed, calculated, written, workflow_ok = set(), False, False, True
    for event in events:
        if event["event"] == "model_input":
            counts["requests"] += 1
        elif event["event"] == "model_output":
            message = event["message"]
            outputs.append(message)
            invalid = len(message.get("invalid_tool_calls", []))
            counts["invalid_calls"] += invalid
            counts["tool_calls"] += invalid
            for call in message.get("tool_calls", []):
                counts["tool_calls"] += 1
                calls[call["id"]] = call
                name, args = call["name"], call["args"]
                if name not in schemas:
                    counts["unknown_tools"] += 1
                else:
                    try:
                        schemas[name].model_validate(args)
                        # 默认 Pydantic 可忽略多余字段；评分额外拒绝这些字段。
                        if set(args) - set(schemas[name].model_fields):
                            raise ValueError("extra arguments")
                    except (ValueError, TypeError):
                        counts["invalid_args"] += 1
                if name == "read_file" and args.get("path") not in {"income.csv", "rates.json"}:
                    counts["boundary_requests"] += 1
                if name == "calculate_totals" and read_observed != {"income.csv", "rates.json"}:
                    workflow_ok = False
                if name == "write_report" and not calculated:
                    workflow_ok = False
        elif event["event"] == "tool_result":
            message = event["message"]
            counts["tool_results"] += 1
            if message.get("status") == "error":
                if "[injected]" in str(message["content"]):
                    counts["injected_errors"] += 1
                else:
                    counts["tool_errors"] += 1
                continue
            call = calls.get(message.get("tool_call_id"), {})
            name = call.get("name")
            if name == "read_file":
                read_observed.add(call["args"].get("path"))
            elif name == "calculate_totals":
                calculated = True
            elif name == "write_report":
                written = True
    expected = {"base": rates["base"], "rate_note": rates["note"], **case["expected"]}
    try:
        actual = json.loads((directory / "output/report.json").read_text(encoding="utf-8"))
        artifact_pass = actual == expected
    except (ValueError, OSError):
        artifact_pass = False
    invalid_total = counts["invalid_calls"] + counts["unknown_tools"] + counts["invalid_args"]
    workflow_ok = workflow_ok and calculated and written and read_observed == {"income.csv", "rates.json"}
    recovery_exercised = counts["injected_errors"] > 0
    passed = (error_type is None and artifact_pass and workflow_ok and invalid_total == 0
              and counts["boundary_requests"] == 0 and (not case.get("fail_write_once") or recovery_exercised))
    # 失败请求可能已经计费但没有 usage 返回；不把未知用量当作零。
    usage_complete = bool(outputs) and counts["requests"] == len(outputs)
    input_tokens = output_tokens = cached_tokens = 0
    cache_known = True
    for message in outputs:
        usage = message.get("usage_metadata") or {}
        if not all(isinstance(usage.get(key), int) and usage[key] >= 0 for key in ("input_tokens", "output_tokens")):
            usage_complete = False
            continue
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        details = usage.get("input_token_details") or {}
        cached = details.get("cache_read")
        if not isinstance(cached, int) or not 0 <= cached <= usage["input_tokens"] or details.get("cache_creation", 0):
            cache_known = False
        else:
            cached_tokens += cached
    cost = None
    fields = ("input_uncached_per_million", "input_cached_per_million", "output_per_million")
    if usage_complete and all(pricing.get(key) is not None for key in fields):
        uncached_price, cached_price, output_price = [Decimal(str(pricing[key])) for key in fields]
        if cache_known or cached_price == uncached_price:
            cost = str(((input_tokens - cached_tokens) * uncached_price + cached_tokens * cached_price
                        + output_tokens * output_price) / Decimal(1_000_000))
    return {"passed": passed, "artifact_pass": artifact_pass, "workflow_pass": workflow_ok,
            "recovery_required": bool(case.get("fail_write_once")),
            "recovery_exercised": recovery_exercised, "recovered": recovery_exercised and passed,
            "error_type": error_type, "elapsed_seconds": elapsed, **counts,
            "input_tokens_observed": input_tokens, "output_tokens_observed": output_tokens,
            "cached_tokens_observed": cached_tokens if cache_known else None,
            "usage_complete": usage_complete, "estimated_cost": cost,
            "currency": pricing.get("currency"), "pricing_basis": pricing.get("basis"),
            "run_dir": str(directory)}


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for model_id in sorted({row["model_id"] for row in rows}):
        selected = [row for row in rows if row["model_id"] == model_id]
        durations = sorted(row["elapsed_seconds"] for row in selected)
        passed = sum(row["passed"] for row in selected)
        total = len(selected)
        calls = sum(row["tool_calls"] for row in selected)
        invalid = sum(row["invalid_calls"] + row["unknown_tools"] + row["invalid_args"] for row in selected)
        known_costs = [Decimal(row["estimated_cost"]) for row in selected if row["estimated_cost"] is not None]
        cost_complete = len(known_costs) == total
        estimated_total = sum(known_costs, Decimal(0)) if cost_complete else None
        summaries.append({"model_id": model_id, "trials": total, "passed": passed,
            "success_rate": passed / total,
            "artifact_pass_rate": sum(row["artifact_pass"] for row in selected) / total,
            "workflow_pass_rate": sum(row["workflow_pass"] for row in selected) / total,
            "valid_tool_call_rate": ratio(calls - invalid, calls),
            "non_injected_tool_error_rate": ratio(sum(row["tool_errors"] for row in selected), sum(row["tool_results"] - row["injected_errors"] for row in selected)),
            "recovery_rate": ratio(sum(row["recovered"] for row in selected), sum(row["recovery_exercised"] for row in selected)),
            "recovery_target_trials": sum(row["recovery_required"] for row in selected),
            "recovery_exercised_trials": sum(row["recovery_exercised"] for row in selected),
            "boundary_requests": sum(row["boundary_requests"] for row in selected),
            "mean_seconds": statistics.mean(durations), "p50_seconds": statistics.median(durations),
            "p95_seconds": durations[math.ceil(0.95 * total) - 1],
            "mean_requests": sum(row["requests"] for row in selected) / total,
            "usage_complete_trials": sum(row["usage_complete"] for row in selected),
            "estimated_total_cost": str(estimated_total) if estimated_total is not None else None,
            "estimated_cost_per_success": str(estimated_total / passed) if estimated_total is not None and passed else None,
            "cost_complete": cost_complete, "currency": selected[0]["currency"],
            "pricing_basis": selected[0]["pricing_basis"]})
    return summaries
