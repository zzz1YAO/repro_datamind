#!/usr/bin/env python3
"""Run a live HTTP smoke test against the repository's real TableBench data."""

import json
import threading
from urllib.request import urlopen

import server


RECORD_ID = "29ba53ce7ca43a979263ed36798f62a3"


def fetch_json(url):
    with urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise AssertionError(f"Expected HTTP 200, received {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main():
    store = server.DataStore(server.DEFAULT_JSONL, server.DEFAULT_CSV_DIR)
    httpd = server.create_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        listing = fetch_json(f"{base_url}/api/records?query={RECORD_ID}")
        record_index = listing["records"][0]["record_index"]
        detail = fetch_json(f"{base_url}/api/records/by-index/{record_index}")
        with urlopen(f"{base_url}/", timeout=10) as response:
            html = response.read().decode("utf-8")

        assert len(store.records) == 886
        assert listing["count"] == 1
        assert detail["qtype"] == "NumericalReasoning"
        assert detail["qsubtype"] == "Aggregation"
        assert detail["gold_answer"] == "10.6"
        assert "tropical cyclones" in detail["table"]["headers"]
        assert 'id="qsubtype-filter"' in html

        print(
            json.dumps(
                {
                    "records": len(store.records),
                    "matched": listing["count"],
                    "table_rows": len(detail["table"]["rows"]),
                    "http_page": response.status,
                }
            )
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
