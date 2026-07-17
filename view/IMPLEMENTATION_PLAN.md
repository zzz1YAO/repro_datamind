# TableBench Data Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-dependency web viewer for TableBench JSONL records and their referenced CSV tables.

**Architecture:** A Python standard-library HTTP server owns JSONL indexing, filtering, and on-demand CSV parsing. A build-free HTML/CSS/JavaScript frontend requests JSON APIs and renders record metadata, linked qtype/qsubtype filters, navigation controls, and a safe text-only table.

**Tech Stack:** Python 3 standard library, HTML5, CSS, browser JavaScript, `unittest`.

## Global Constraints

- Create every artifact under `/nas-files/ziyi/projects/proj_dsagent/repro_datamind/view`.
- Use no third-party Python or JavaScript dependencies and no build step.
- Default to `../data/datamind_eval_inputs/tablebench_all.jsonl` and `../data/tablebench_csv_all`.
- Treat `qtype` as the first-level filter and `qsubtype` as its dependent second-level filter.
- Insert dataset text with DOM text APIs, never `innerHTML`.
- The remote directory has no `.git`, so this plan does not include commit steps.

---

### Task 1: Data store and API behavior

**Files:**
- Create: `view/test_server.py`
- Create: `view/server.py`

**Interfaces:**
- Produces: `DataStore(jsonl_path, csv_dir)`, `DataStore.search(query="", qtype="", qsubtype="")`, `DataStore.get_record(record_id)`, `DataStore.get_record_by_index(record_index)`, `DataStore.read_table(record)`.
- Produces: `build_handler(store, static_dir)` and `create_server(host, port, store, static_dir)`.

- [ ] **Step 1: Write failing behavior tests**

  Create temporary JSONL/CSV fixtures and assert that search supports qtype plus qsubtype, record lookup parses CSV rows, a missing CSV produces a `table_error`, and `/api/records/<id>` returns JSON. Load `server.py` through a test helper that calls `self.fail("server.py is missing")` while the implementation does not yet exist, so the first run is an intentional assertion failure rather than an import error.

- [ ] **Step 2: Verify the tests fail for the missing feature**

  Run `cd view && python3 -m unittest -v test_server.py`.
  Expected: FAIL containing `server.py is missing`.

- [ ] **Step 3: Implement the minimal server**

  `DataStore` will parse each non-empty JSONL line, validate required IDs, expose sorted qtype/subtype options, filter case-insensitively over ID and question, and parse CSV with `csv.reader`. `ViewerHandler` will handle:

  - `GET /api/records?query=&qtype=&qsubtype=` → `{count, records, qtypes, qsubtypes_by_qtype}`
  - `GET /api/records/by-index/<record_index>` → exact record fields plus `{table: {headers, rows}}` or `table_error`, including when IDs repeat
  - `GET /api/records/<id>` → first matching record, retained as a convenience endpoint
  - other paths → static files rooted at `view/`

  Use `json.dumps(..., ensure_ascii=False)` and explicit JSON error responses. Reject CSV filenames containing directories before reading them.

- [ ] **Step 4: Verify the server tests pass**

  Run `cd view && python3 -m unittest -v test_server.py`.
  Expected: all tests PASS with exit code 0.

### Task 2: Browser interface

**Files:**
- Create: `view/index.html`
- Create: `view/app.js`
- Create: `view/styles.css`
- Create: `view/README.md`

**Interfaces:**
- Consumes: `/api/records` and `/api/records/<id>` from Task 1.
- Produces: browser controls with element IDs `search-input`, `qtype-filter`, `qsubtype-filter`, `previous-button`, `next-button`, `position-input`, `record-detail`, and `table-container`.

- [ ] **Step 1: Add a failing static-file HTTP assertion**

  Extend `test_server.py` to request `/`, assert status 200, and assert the response contains `id="qtype-filter"`, `id="qsubtype-filter"`, and `id="table-container"`.

- [ ] **Step 2: Verify the new test fails**

  Run `cd view && python3 -m unittest -v test_server.py`.
  Expected: FAIL because `index.html` is absent or lacks the required controls.

- [ ] **Step 3: Implement the frontend**

  Build a responsive two-part page: a compact filter/navigation header and a record detail/table workspace. `app.js` debounces keyword search, refetches filtered summaries, rebuilds qsubtype choices from the selected qtype, keeps the current position in range, and fetches the selected record. All dataset values are assigned with `textContent`; table cells are created with `document.createElement`.

  Use a restrained data-inspection visual system: cool gray background, white panels, cobalt controls, a monospace utility face for IDs/table values, sticky CSV headers, visible focus rings, and horizontal table scrolling. Do not add external fonts, animation libraries, or decorative assets.

- [ ] **Step 4: Document the run command**

  `README.md` will document:

  ```bash
  cd /nas-files/ziyi/projects/proj_dsagent/repro_datamind
  python3 view/server.py --host 0.0.0.0 --port 8000
  ```

  It will also list `--jsonl` and `--csv-dir` overrides and explain SSH port forwarding.

- [ ] **Step 5: Verify all tests pass**

  Run `cd view && python3 -m unittest -v test_server.py`.
  Expected: all tests PASS with exit code 0.

### Task 3: Remote smoke verification

**Files:**
- Verify: all files in `view/`

**Interfaces:**
- Consumes: final server and frontend.
- Produces: evidence that the real 886-record dataset is readable and the browser entry point is served.

- [ ] **Step 1: Start an ephemeral server on loopback**

  Run the server on an unused port, wait only until it accepts requests, and ensure the process is stopped after the check.

- [ ] **Step 2: Fetch the real API and page**

  Request `/api/records?query=29ba53ce7ca43a979263ed36798f62a3`, use its `record_index` to request the exact detail, and verify the response contains the expected ID, qtype `NumericalReasoning`, qsubtype `Aggregation`, answer `10.6`, and parsed CSV headers. Request `/` and verify the frontend HTML returns HTTP 200.

- [ ] **Step 3: Run a syntax check and final test suite**

  Run `python3 -m py_compile view/server.py view/test_server.py` and `cd view && python3 -m unittest -v test_server.py`.
  Expected: both commands exit 0 and all tests pass.
