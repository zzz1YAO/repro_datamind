#!/usr/bin/env python3
"""Prepare all TableBench JSONL records and CSV files for DATAMIND ReAct eval.

The raw TableBench file is expected at data/tablebench_raw/TableBench_data.jsonl
once the dataset is available. The parser is defensive because public TableBench
exports may differ slightly in table-field naming.
"""

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc


def choose(raw: Dict[str, Any], names: List[str], default: Any = "") -> Any:
    for name in names:
        if name in raw and raw[name] not in (None, ""):
            return raw[name]
    return default


def table_to_rows(table: Any) -> Tuple[List[Any], List[List[Any]]]:
    """Return CSV headers and rows without inferring or rewriting cell types."""
    if isinstance(table, dict) and "columns" in table and "data" in table:
        columns = list(table["columns"])
        rows = [list(row) for row in table["data"]]
        validate_table_shape(columns, rows)
        return columns, rows
    if isinstance(table, list):
        if not table:
            return [], []
        if isinstance(table[0], dict):
            columns = list(table[0].keys())
            rows = [[row.get(column) for column in columns] for row in table]
            return columns, rows
        if isinstance(table[0], (list, tuple)):
            columns = list(table[0])
            rows = [list(row) for row in table[1:]]
            validate_table_shape(columns, rows)
            return columns, rows
    if isinstance(table, dict):
        for key in ["rows", "data", "table", "values"]:
            if key in table:
                return table_to_rows(table[key])
    if isinstance(table, str):
        text = table.strip()
        if not text:
            return [], []
        delimiter = "\t" if "\t" in text and "," not in text.splitlines()[0] else ","
        parsed = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        if parsed:
            columns, rows = parsed[0], parsed[1:]
            validate_table_shape(columns, rows)
            return columns, rows
    raise ValueError(f"Cannot convert table to CSV rows: {type(table)}")


def validate_table_shape(columns: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(columns):
            raise ValueError(
                f"Table row {row_number} has {len(row)} cells; expected {len(columns)}"
            )


def write_csv(path: Path, columns: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def extract_table(raw: Dict[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
    table = choose(
        raw,
        [
            "table",
            "table_data",
            "table_array",
            "data",
            "rows",
            "csv",
            "table_text",
        ],
        default=None,
    )
    if table is None:
        raise ValueError(f"No table field found for sample {raw.get('id', '<missing id>')}")
    return table_to_rows(table)


def normalize_row(
    raw: Dict[str, Any], index: int
) -> Tuple[Dict[str, str], List[Any], List[List[Any]]]:
    qsubtype = str(choose(raw, ["qsubtype", "subtype", "question_subtype"], "")).strip()
    qtype = str(choose(raw, ["qtype", "type", "question_type"], "")).strip()
    sample_id = str(choose(raw, ["id", "sample_id", "uid"], f"tablebench_{index:06d}")).strip()
    question = str(choose(raw, ["question", "instruction", "query"], "")).strip()
    answer = str(choose(raw, ["answer", "gold_answer", "label"], "")).strip()

    if not qsubtype:
        raise ValueError(f"Missing qsubtype for sample {sample_id}")
    if not question:
        raise ValueError(f"Missing question for sample {sample_id}")

    csv_file = f"{sample_id}.csv"
    metadata = {
        "id": sample_id,
        "qtype": qtype,
        "qsubtype": qsubtype,
        "question": question,
        "gold_answer": answer,
        "csv_file": csv_file,
    }
    columns, rows = extract_table(raw)
    return metadata, columns, rows


def prepare_tablebench(raw_file: Path, csv_dir: Path, output_jsonl: Path) -> int:
    csv_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for index, raw in enumerate(read_jsonl(raw_file), start=1):
            metadata, columns, rows = normalize_row(raw, index)
            write_csv(csv_dir / metadata["csv_file"], columns, rows)
            out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", default="data/Tablebench/TableBench.jsonl")
    parser.add_argument("--csv_dir", default="data/tablebench_csv_all")
    parser.add_argument("--output_jsonl", default="data/datamind_eval_inputs/tablebench_all.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kept = prepare_tablebench(
        raw_file=Path(args.raw_file),
        csv_dir=Path(args.csv_dir),
        output_jsonl=Path(args.output_jsonl),
    )
    print(f"Prepared {kept} TableBench samples across all question types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
