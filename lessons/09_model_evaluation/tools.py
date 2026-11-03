"""教学收入工具：固定样例汇率，Decimal 计算，运行目录内生成报告。"""

import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

from langchain_core.tools import tool

SOURCE_ROOT = Path(__file__).resolve().parent / "demo_inbox" / "01_october"


class VerificationError(ValueError):
    """输入有效，但实际报告未通过验收；外层可以在预算内重试。"""


def calculate_expected(source_root: Path = SOURCE_ROOT) -> dict:
    """按币种先合计，再换算并四舍五入到分；汇率为教学数据。"""
    rates = json.loads((source_root / "rates.json").read_text(encoding="utf-8"))
    if rates["base"] != "CNY":
        raise ValueError("本课只支持换算为 CNY")
    grouped = {}
    with (source_root / "income.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            amount = Decimal(row["amount"])
            if not amount.is_finite() or amount < 0:
                raise ValueError("收入金额必须为非负有限数值")
            currency = row["currency"]
            grouped[currency] = grouped.get(currency, Decimal("0")) + amount
    items = []
    for currency, amount in sorted(grouped.items()):
        rate = Decimal(rates["cny_per_unit"][currency])
        if not rate.is_finite() or rate <= 0:
            raise ValueError("汇率必须为正有限数值")
        converted = (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        items.append({"currency": currency, "amount": str(amount), "rate": str(rate), "cny": str(converted)})
    return {"base": "CNY", "rate_note": rates["note"], "items": items,
            "total_cny": str(sum((Decimal(item["cny"]) for item in items), Decimal("0.00")))}


def create_tools(run_dir: Path, source_root: Path = SOURCE_ROOT, fail_write_once: bool = False) -> list:
    """只暴露三个工具；输出目录与计算结果由程序掌管。"""
    calculated = None

    @tool
    def read_file(path: str) -> str:
        """读取本课输入文件，只允许 income.csv 和 rates.json。"""
        if path not in {"income.csv", "rates.json"}:
            raise ValueError("只能读取 income.csv 或 rates.json")
        return (source_root / path).read_text(encoding="utf-8")

    @tool
    def calculate_totals() -> str:
        """读取本课收入和教学汇率，用 Decimal 按币种汇总并换算 CNY，返回 JSON。"""
        nonlocal calculated
        calculated = calculate_expected(source_root)
        return json.dumps(calculated, ensure_ascii=False)

    @tool
    def write_report() -> str:
        """将 calculate_totals 已得到的结果保存为 output/report.json；先计算，不覆盖文件。"""
        nonlocal fail_write_once
        if calculated is None:
            raise ValueError("请先调用 calculate_totals，看到计算结果后再写入")
        if fail_write_once:
            fail_write_once = False
            raise RuntimeError("[injected] 临时写入故障，请重新调用 write_report")
        target = run_dir / "output" / "report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as file:
            json.dump(calculated, file, ensure_ascii=False, indent=2)
        return f"已创建：{target}"

    return [read_file, calculate_totals, write_report]


def validate_report(run_dir: Path, source_root: Path = SOURCE_ROOT) -> dict:
    """在模型声称结束后检查实际产物；复用确定性计算，不依赖模型最终文本。"""
    path = run_dir / "output" / "report.json"
    if not path.is_file():
        raise VerificationError("验收失败：模型已结束，但报告不存在")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise VerificationError("验收失败：报告不是有效的 UTF-8 JSON") from error
    if actual != calculate_expected(source_root):
        raise VerificationError("验收失败：报告与源数据计算结果不一致")
    return {"status": "passed", "report": str(path), "total_cny": actual["total_cny"]}
