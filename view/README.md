# TableBench Data Viewer

A dependency-free browser for the repository's TableBench evaluation JSONL and
the CSV table referenced by each record.

## Run on the server

```bash
cd /nas-files/ziyi/projects/proj_dsagent/repro_datamind
python3 view/server.py --host 127.0.0.1 --port 8000
```

The defaults are:

- JSONL: `data/datamind_eval_inputs/tablebench_all.jsonl`
- CSV directory: `data/tablebench_csv_all`

Override them when needed:

```bash
python3 view/server.py \
  --jsonl /path/to/records.jsonl \
  --csv-dir /path/to/csv/files \
  --host 127.0.0.1 \
  --port 8000
```

## Open it from a local machine

Keep the server command running, then create an SSH tunnel from local
PowerShell:

```powershell
ssh -F C:\Users\Lenovo\.sshconfig -N -L 8000:127.0.0.1:8000 ustgz-1
```

Open <http://127.0.0.1:8000> in a browser. If local port 8000 is occupied, use
another local port on the left side, for example
`-L 18000:127.0.0.1:8000`, then open <http://127.0.0.1:18000>.

## Test

```bash
cd /nas-files/ziyi/projects/proj_dsagent/repro_datamind/view
python3 -m unittest -v test_server.py
python3 smoke_test.py
```
