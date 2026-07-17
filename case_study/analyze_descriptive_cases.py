#!/usr/bin/env python3
"""Compare DeepSeek and Qwen LLM-Judge results for DescriptiveAnalysis."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "DeepSeek-V4-Pro_tablebench"
    / "tablebench_eval_results.jsonl"
)
QWEN_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "qwen3-8b_tablebench_DataAnalysis+NumericalReasoning_datamind_react"
    / "tablebench_eval_results.jsonl"
)
OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "descriptive_deepseek_correct_qwen_wrong.jsonl"
)

TARGET_SUBTYPE = "DescriptiveAnalysis"
SCORE_FIELD = "llm_judge_score"


def load_subtype_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ordered_rows: List[Dict[str, Any]] = []
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if row.get("qsubtype") != TARGET_SUBTYPE:
                continue
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if sample_id in rows_by_id:
                raise ValueError(f"Duplicate id {sample_id!r} in {path}")
            ordered_rows.append(row)
            rows_by_id[sample_id] = row
    return ordered_rows, rows_by_id


def main() -> int:
    deepseek_ordered, deepseek_by_id = load_subtype_rows(DEEPSEEK_RESULTS)
    _, qwen_by_id = load_subtype_rows(QWEN_RESULTS)

    counts = {
        "both_correct": 0,
        "deepseek_correct_qwen_wrong": 0,
        "deepseek_wrong_qwen_correct": 0,
        "both_wrong": 0,
    }
    selected_cases: List[Dict[str, Any]] = []
    skipped_unjudged = 0

    for deepseek_row in deepseek_ordered:
        sample_id = str(deepseek_row["id"])
        qwen_row = qwen_by_id.get(sample_id)
        if qwen_row is None:
            continue

        deepseek_score = deepseek_row.get(SCORE_FIELD)
        qwen_score = qwen_row.get(SCORE_FIELD)
        if deepseek_score not in (0, 1) or qwen_score not in (0, 1):
            skipped_unjudged += 1
            continue

        if deepseek_score == 1 and qwen_score == 1:
            counts["both_correct"] += 1
        elif deepseek_score == 1 and qwen_score == 0:
            counts["deepseek_correct_qwen_wrong"] += 1
            selected_cases.append(
                {
                    "id": sample_id,
                    "question": deepseek_row.get("question", ""),
                    "answer": deepseek_row.get("answer", ""),
                    "deepseek_result": deepseek_row,
                    "qwen_result": qwen_row,
                }
            )
        elif deepseek_score == 0 and qwen_score == 1:
            counts["deepseek_wrong_qwen_correct"] += 1
        else:
            counts["both_wrong"] += 1

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for case in selected_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    deepseek_ids = set(deepseek_by_id)
    qwen_ids = set(qwen_by_id)
    matched_count = len(deepseek_ids & qwen_ids)

    print(f"Target subtype: {TARGET_SUBTYPE}")
    print(f"Score field: {SCORE_FIELD}")
    print(f"Matched pairs: {matched_count}")
    print(f"Both correct: {counts['both_correct']}")
    print(
        "DeepSeek correct, Qwen wrong: "
        f"{counts['deepseek_correct_qwen_wrong']}"
    )
    print(
        "DeepSeek wrong, Qwen correct: "
        f"{counts['deepseek_wrong_qwen_correct']}"
    )
    print(f"Both wrong: {counts['both_wrong']}")
    print(f"Skipped pairs with missing Judge scores: {skipped_unjudged}")
    print(f"DeepSeek-only ids: {len(deepseek_ids - qwen_ids)}")
    print(f"Qwen-only ids: {len(qwen_ids - deepseek_ids)}")
    print(f"Saved {len(selected_cases)} cases to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
