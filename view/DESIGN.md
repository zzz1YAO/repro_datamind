# TableBench Data Viewer Design

## Goal

Provide a small, dependency-free browser for `tablebench_all.jsonl` and the
CSV files referenced by each record. The viewer is for researchers inspecting
evaluation examples, so fast navigation and faithful table rendering matter
more than visual polish or production-scale performance.

## Architecture

- `server.py` uses only the Python standard library. It loads the JSONL index
  at startup, reads a selected CSV on demand, serves JSON APIs, and serves the
  static frontend files from this directory.
- `index.html`, `app.js`, and `styles.css` form a single-page frontend with no
  build step.
- Paths default to the repository's existing
  `data/datamind_eval_inputs/tablebench_all.jsonl` and
  `data/tablebench_csv_all/`, with command-line overrides for reuse.

## User Interface

- A compact header reports the visible result count.
- Filters include keyword search over ID/question, a `qtype` selector, and a
  dependent `qsubtype` selector. Changing `qtype` limits the available
  `qsubtype` values and resets an incompatible subtype selection.
- Previous/next controls and a direct result-position input navigate the
  filtered result set.
- The selected record shows ID, qtype, qsubtype, question, gold answer, and CSV
  filename, followed by the full CSV in a horizontally scrollable table.
- Loading, empty-result, malformed-data, and missing-CSV states show explicit
  messages instead of leaving an empty page.

## Data Flow

1. The browser requests metadata and filter options from `/api/records`.
2. Filtering and result navigation use query parameters; the server returns
   matching record summaries without loading every CSV.
3. Each summary includes a stable JSONL `record_index`. Selecting a record
   requests `/api/records/by-index/<record_index>`, which preserves records
   that share an ID and reads the selected CSV using Python's `csv` module.

All displayed content is inserted with DOM text properties, not raw HTML, so
CSV or JSONL text cannot inject markup.

## Scope and Testing

There is no editing, authentication, database, framework, or deployment layer.
A small standard-library test suite will verify data loading, qtype/qsubtype
filtering, CSV parsing, missing-file handling, and one live HTTP API request.
Manual smoke verification will start the server and fetch the page/API.
