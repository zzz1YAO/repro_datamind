#!/usr/bin/env python3
"""Dependency-free TableBench JSONL and CSV viewer."""

import argparse
import csv
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


VIEW_DIR = Path(__file__).resolve().parent
REPO_DIR = VIEW_DIR.parent
DEFAULT_JSONL = REPO_DIR / "data" / "datamind_eval_inputs" / "tablebench_all.jsonl"
DEFAULT_CSV_DIR = REPO_DIR / "data" / "tablebench_csv_all"
REQUIRED_FIELDS = (
    "id",
    "qtype",
    "qsubtype",
    "question",
    "gold_answer",
    "csv_file",
)


class DataStore:
    """In-memory JSONL index with on-demand CSV loading."""

    def __init__(self, jsonl_path, csv_dir):
        self.jsonl_path = Path(jsonl_path)
        self.csv_dir = Path(csv_dir)
        self.records = self._load_records()
        self.records_by_id = {}
        for record in self.records:
            self.records_by_id.setdefault(record["id"], record)

        self.qtypes = sorted(
            {record["qtype"] for record in self.records if record["qtype"]}
        )
        self.qsubtypes_by_qtype = {
            qtype: sorted(
                {
                    record["qsubtype"]
                    for record in self.records
                    if record["qtype"] == qtype and record["qsubtype"]
                }
            )
            for qtype in self.qtypes
        }

    def _load_records(self):
        records = []
        with self.jsonl_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error.msg}"
                    ) from error
                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    joined = ", ".join(missing)
                    raise ValueError(
                        f"JSONL line {line_number} is missing required fields: {joined}"
                    )
                normalized = {field: str(record[field]) for field in REQUIRED_FIELDS}
                normalized["record_index"] = len(records)
                records.append(normalized)
        return records

    def search(self, query="", qtype="", qsubtype=""):
        needle = query.strip().casefold()
        matches = []
        for record in self.records:
            if qtype and record["qtype"] != qtype:
                continue
            if qsubtype and record["qsubtype"] != qsubtype:
                continue
            if needle:
                haystack = f'{record["id"]}\n{record["question"]}'.casefold()
                if needle not in haystack:
                    continue
            matches.append(record)
        return matches

    def get_record(self, record_id):
        return self.records_by_id.get(record_id)

    def get_record_by_index(self, record_index):
        if 0 <= record_index < len(self.records):
            return self.records[record_index]
        return None

    def read_table(self, record):
        filename = record["csv_file"]
        filename_path = Path(filename)
        if filename_path.is_absolute() or filename_path.name != filename:
            raise ValueError(f"Unsafe CSV filename: {filename}")

        csv_path = self.csv_dir / filename
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            parsed_rows = list(csv.reader(handle))

        if not parsed_rows:
            return {"headers": [], "rows": []}
        return {"headers": parsed_rows[0], "rows": parsed_rows[1:]}


class ViewerHandler(SimpleHTTPRequestHandler):
    store = None
    static_dir = VIEW_DIR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/records":
            self._serve_record_list(parsed.query)
            return
        if parsed.path.startswith("/api/records/by-index/"):
            raw_index = parsed.path[len("/api/records/by-index/") :]
            self._serve_record_by_index(raw_index)
            return
        if parsed.path.startswith("/api/records/"):
            record_id = unquote(parsed.path[len("/api/records/") :])
            self._serve_record(record_id)
            return
        super().do_GET()

    def _serve_record_list(self, query_string):
        params = parse_qs(query_string)
        query = params.get("query", [""])[0]
        qtype = params.get("qtype", [""])[0]
        qsubtype = params.get("qsubtype", [""])[0]
        matches = self.store.search(query=query, qtype=qtype, qsubtype=qsubtype)
        self._send_json(
            {
                "count": len(matches),
                "records": matches,
                "qtypes": self.store.qtypes,
                "qsubtypes_by_qtype": self.store.qsubtypes_by_qtype,
            }
        )

    def _serve_record(self, record_id):
        if not record_id or "/" in record_id:
            self._send_json({"error": "Invalid record ID"}, status=400)
            return
        record = self.store.get_record(record_id)
        if record is None:
            self._send_json({"error": f"Record not found: {record_id}"}, status=404)
            return

        self._send_record_payload(record)

    def _serve_record_by_index(self, raw_index):
        try:
            record_index = int(raw_index)
        except ValueError:
            self._send_json({"error": "Invalid record index"}, status=400)
            return
        record = self.store.get_record_by_index(record_index)
        if record is None:
            self._send_json(
                {"error": f"Record index not found: {record_index}"}, status=404
            )
            return

        self._send_record_payload(record)

    def _send_record_payload(self, record):
        payload = dict(record)
        try:
            payload["table"] = self.store.read_table(record)
            payload["table_error"] = None
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            payload["table"] = None
            payload["table_error"] = f'Could not read {record["csv_file"]}: {error}'
        self._send_json(payload)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


def build_handler(store, static_dir=VIEW_DIR):
    class BoundViewerHandler(ViewerHandler):
        pass

    BoundViewerHandler.store = store
    BoundViewerHandler.static_dir = Path(static_dir)
    return BoundViewerHandler


def create_server(host, port, store, static_dir=VIEW_DIR):
    return ThreadingHTTPServer((host, port), build_handler(store, static_dir))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    store = DataStore(args.jsonl, args.csv_dir)
    server = create_server(args.host, args.port, store)
    print(
        f"TableBench viewer: http://{args.host}:{server.server_port} "
        f"({len(store.records)} records)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
