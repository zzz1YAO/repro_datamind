import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen


VIEW_DIR = Path(__file__).resolve().parent
SERVER_PATH = VIEW_DIR / "server.py"


def load_server_module():
    if not SERVER_PATH.exists():
        raise AssertionError("server.py is missing")
    spec = importlib.util.spec_from_file_location("tablebench_view_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.csv_dir = root / "csv"
        self.csv_dir.mkdir()
        self.jsonl_path = root / "records.jsonl"
        self.records = [
            {
                "id": "alpha",
                "qtype": "NumericalReasoning",
                "qsubtype": "Aggregation",
                "question": "What is the average number of cyclones?",
                "gold_answer": "10.6",
                "csv_file": "alpha.csv",
            },
            {
                "id": "beta",
                "qtype": "NumericalReasoning",
                "qsubtype": "Arithmetic",
                "question": "What is one plus one?",
                "gold_answer": "2",
                "csv_file": "beta.csv",
            },
            {
                "id": "gamma",
                "qtype": "DataAnalysis",
                "qsubtype": "Lookup",
                "question": "Which row matches the label?",
                "gold_answer": "A",
                "csv_file": "missing.csv",
            },
        ]
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")
        (self.csv_dir / "alpha.csv").write_text(
            "season,tropical cyclones\n1990 - 91,10\n1991 - 92,11\n",
            encoding="utf-8",
        )
        (self.csv_dir / "beta.csv").write_text(
            "expression,result\n1 + 1,2\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_store(self):
        server_module = load_server_module()
        return server_module, server_module.DataStore(self.jsonl_path, self.csv_dir)

    def request_json(self, server_module, store, path):
        handler = server_module.build_handler(store, VIEW_DIR)
        httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}{path}"
            with urlopen(url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                return json.loads(response.read().decode("utf-8"))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_search_filters_qtype_qsubtype_and_keyword(self):
        _, store = self.make_store()

        filtered = store.search(
            query="cyclones",
            qtype="NumericalReasoning",
            qsubtype="Aggregation",
        )

        self.assertEqual([record["id"] for record in filtered], ["alpha"])
        self.assertEqual(
            store.qsubtypes_by_qtype,
            {
                "DataAnalysis": ["Lookup"],
                "NumericalReasoning": ["Aggregation", "Arithmetic"],
            },
        )

    def test_duplicate_ids_remain_separately_addressable(self):
        duplicate = dict(self.records[0])
        duplicate["qsubtype"] = "AlternateAggregation"
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(duplicate) + "\n")
        server_module = load_server_module()
        try:
            store = server_module.DataStore(self.jsonl_path, self.csv_dir)
        except ValueError as error:
            self.fail(f"Duplicate IDs should remain viewable: {error}")

        duplicates = store.search(query="alpha")

        self.assertEqual(len(duplicates), 2)
        self.assertNotEqual(
            duplicates[0]["record_index"], duplicates[1]["record_index"]
        )
        payload = self.request_json(
            server_module,
            store,
            f'/api/records/by-index/{duplicates[1]["record_index"]}',
        )
        self.assertEqual(payload["id"], "alpha")
        self.assertEqual(payload["qsubtype"], "AlternateAggregation")

    def test_read_table_preserves_headers_and_rows(self):
        _, store = self.make_store()

        table = store.read_table(store.get_record("alpha"))

        self.assertEqual(table["headers"], ["season", "tropical cyclones"])
        self.assertEqual(table["rows"][1], ["1991 - 92", "11"])

    def test_record_api_returns_table(self):
        server_module, store = self.make_store()

        payload = self.request_json(server_module, store, "/api/records/alpha")

        self.assertEqual(payload["id"], "alpha")
        self.assertEqual(payload["gold_answer"], "10.6")
        self.assertEqual(payload["table"]["rows"][0], ["1990 - 91", "10"])

    def test_missing_csv_is_reported_without_hiding_metadata(self):
        server_module, store = self.make_store()

        payload = self.request_json(server_module, store, "/api/records/gamma")

        self.assertEqual(payload["id"], "gamma")
        self.assertIsNone(payload["table"])
        self.assertIn("missing.csv", payload["table_error"])

    def test_static_index_contains_filter_and_table_controls(self):
        server_module, store = self.make_store()
        handler = server_module.build_handler(store, VIEW_DIR)
        httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}/"
            with urlopen(url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                html = response.read().decode("utf-8")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        self.assertIn('id="qtype-filter"', html)
        self.assertIn('id="qsubtype-filter"', html)
        self.assertIn('id="table-container"', html)


if __name__ == "__main__":
    unittest.main()
