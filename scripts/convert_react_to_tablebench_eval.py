#!/usr/bin/env python3
"""Convert DATAMIND ReAct outputs to TableBench-style eval records.

The converter does not run TableBench's parser because the ReAct runner emits
DATAMIND <answer> tags rather than "Final Answer:" or PoT fenced-code output.
"""

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import string
from typing import Any, Dict, Iterable, List, Tuple


DATA_ANALYSIS_METRICS = {
    "DataAnalysis_CorrelationAnalysis": "EM_with_error_10",
    "DataAnalysis_StatisticalAnalysis": "EM_with_error_10",
    "DataAnalysis_ImpactAnalysis": "EM",
    "DataAnalysis_CausalAnalysis": "ROUGE-L",
    "DataAnalysis_AnomalyDetection": "ROUGE-L",
    "DataAnalysis_DescriptiveAnalysis": "ROUGE-L",
}


FAILURE_COLUMNS = [
    "id",
    "qtype",
    "qsubtype",
    "question",
    "gold_answer",
    "pred_answer",
    "metric_name",
    "metric_score",
    "csv_file",
    "trajectory_file",
    "parse_success",
    "execution_error_count",
    "manual_failure_type",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_em_text(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.whitespace + string.punctuation)


def exact_match(answer: Any, prediction: Any) -> float:
    return 1.0 if normalize_em_text(answer) == normalize_em_text(prediction) else 0.0


def first_number(value: Any) -> float:
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        raise ValueError(f"No numeric value found in: {value}")
    return float(match.group(0))


def em_with_error_10(answer: Any, prediction: Any) -> float:
    try:
        gold = first_number(answer)
        pred = first_number(prediction)
    except ValueError:
        return 0.0
    if gold == 0:
        return 1.0 if abs(pred) <= 1e-12 else 0.0
    return 1.0 if abs(pred - gold) / abs(gold) <= 0.10 else 0.0


def rouge_l(answer: Any, prediction: Any) -> float:
    gold_tokens = normalize_em_text(answer).split()
    pred_tokens = normalize_em_text(prediction).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    dp = [[0] * (len(pred_tokens) + 1) for _ in range(len(gold_tokens) + 1)]
    for i, gold_token in enumerate(gold_tokens, start=1):
        for j, pred_token in enumerate(pred_tokens, start=1):
            if gold_token == pred_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def category_key(qtype: str, qsubtype: str) -> str:
    qtype = str(qtype).strip()
    qsubtype = str(qsubtype).strip()
    known_prefixes = ("DataAnalysis_", "NumericalReasoning_", "FactChecking_", "Visualization_")
    if qsubtype.startswith(known_prefixes):
        return qsubtype
    if qtype and qsubtype:
        return f"{qtype}_{qsubtype}"
    return qsubtype or qtype


def metric_name_for(qtype: str, qsubtype: str) -> str:
    category = category_key(qtype, qsubtype)
    if category == "FactChecking" or category.startswith("FactChecking_"):
        return "EM"
    if category == "NumericalReasoning" or category.startswith("NumericalReasoning_"):
        return "EM"
    if category in DATA_ANALYSIS_METRICS:
        return DATA_ANALYSIS_METRICS[category]
    return "EM"


def evaluate_metric(metric_name: str, answer: Any, prediction: Any) -> float:
    if metric_name == "EM":
        return exact_match(answer, prediction)
    if metric_name == "EM_with_error_10":
        return em_with_error_10(answer, prediction)
    if metric_name == "ROUGE-L":
        return rouge_l(answer, prediction)
    raise ValueError(f"Unsupported metric: {metric_name}")


def evaluate_tablebench_record(record: Dict[str, Any]) -> Tuple[str, float]:
    metric_name = metric_name_for(str(record.get("qtype", "")), str(record.get("qsubtype", "")))
    score = evaluate_metric(metric_name, record.get("answer", ""), record.get("prediction", ""))
    return metric_name, float(score)


def to_tablebench_record(raw: Dict[str, Any], instruction_type: str = "DATAMIND-ReAct") -> Dict[str, Any]:
    model_name = str(raw.get("model_name", "model"))
    if not model_name.endswith("-datamind-react"):
        model_name = f"{model_name}-datamind-react"
    pred = str(raw.get("pred_answer", ""))
    return {
        "id": raw.get("id", ""),
        "model_name": model_name,
        "instruction_type": instruction_type,
        "qtype": raw.get("qtype", ""),
        "qsubtype": raw.get("qsubtype", ""),
        "question": raw.get("question", ""),
        "answer": raw.get("gold_answer", raw.get("answer", "")),
        "prediction": pred,
        "parsed_result": {
            "parsed_prediction": pred,
            "Parse@1": bool(raw.get("parse_success", bool(pred))),
        },
        "llm_judge_score": None,
        "llm_judge_reason": None,
        "llm_judge_model": None,
    }


def is_failure(metric_name: str, metric_score: float, parse_success: bool) -> bool:
    if not parse_success:
        return True
    if metric_name in {"EM", "EM_with_error_10"}:
        return metric_score < 1.0
    if metric_name == "ROUGE-L":
        return metric_score < 1.0
    return metric_score < 1.0


def write_trajectory_file(trajectory_dir: Path, raw: Dict[str, Any]) -> str:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw.get("id", "sample"))) or "sample"
    path = trajectory_dir / f"{sample_id}.json"
    path.write_text(json.dumps(raw.get("traj", []), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def summarize_by_subtype(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = defaultdict(list)
    for record in records:
        buckets[str(record.get("qsubtype", ""))].append(float(record.get("metric_score", 0.0)))
    summary = {}
    for subtype, scores in sorted(buckets.items()):
        summary[subtype] = {
            "count": len(scores),
            "score": sum(scores) / len(scores) if scores else math.nan,
            "llm_judge_count": 0,
            "llm_judge_score": None,
        }
    return summary


def convert_file(
    raw_results_file: Path,
    output_dir: Path,
    instruction_type: str = "DATAMIND-ReAct",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(raw_results_file)

    converted_rows = []
    scored_rows = []
    failure_rows = []
    trajectory_dir = output_dir / "trajectories"

    for raw in raw_rows:
        converted = to_tablebench_record(raw, instruction_type=instruction_type)
        metric_name, metric_score = evaluate_tablebench_record(converted)
        converted["metric_name"] = metric_name
        converted["metric_score"] = metric_score
        converted_rows.append(converted)
        scored_rows.append(
            {
                "id": converted["id"],
                "qtype": converted["qtype"],
                "qsubtype": converted["qsubtype"],
                "metric_name": metric_name,
                "metric_score": metric_score,
            }
        )

        parse_success = bool(converted["parsed_result"]["Parse@1"])
        if is_failure(metric_name, metric_score, parse_success):
            trajectory_file = write_trajectory_file(trajectory_dir, raw)
            failure_rows.append(
                {
                    "id": raw.get("id", ""),
                    "qtype": raw.get("qtype", ""),
                    "qsubtype": raw.get("qsubtype", ""),
                    "question": raw.get("question", ""),
                    "gold_answer": raw.get("gold_answer", ""),
                    "pred_answer": raw.get("pred_answer", ""),
                    "metric_name": metric_name,
                    "metric_score": f"{metric_score:.6f}",
                    "csv_file": raw.get("csv_file", ""),
                    "trajectory_file": trajectory_file,
                    "parse_success": str(parse_success),
                    "execution_error_count": str(raw.get("execution_error_count", 0)),
                    "manual_failure_type": "",
                }
            )

    eval_path = output_dir / "tablebench_eval_results.jsonl"
    score_path = output_dir / "subtype_metrics.json"
    failure_path = output_dir / "failure_cases.csv"
    write_jsonl(eval_path, converted_rows)
    score_path.write_text(
        json.dumps(summarize_by_subtype(scored_rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(failure_rows)

    return {
        "eval_results": eval_path,
        "subtype_metrics": score_path,
        "failure_cases": failure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_results_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instruction_type", default="DATAMIND-ReAct")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = convert_file(
        raw_results_file=Path(args.raw_results_file),
        output_dir=Path(args.output_dir),
        instruction_type=args.instruction_type,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
