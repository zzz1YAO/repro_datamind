# DATAMIND ReAct on TableBench Focus Subtypes

This repro keeps DATAMIND's prompt style, ReAct loop, and Python execution
mechanism, then routes outputs into TableBench-style metrics and failure-case
analysis.

## Main Paths

- `DataMind/datamind/eval/python/eval_tablebench_react.py`: model rollout with
  `<code>` / `<answer>` and DATAMIND `Interpreter`.
- `scripts/prepare_tablebench_focus.py`: converts raw TableBench JSONL to CSV
  files plus DATAMIND eval JSONL when the dataset is available.
- `scripts/convert_react_to_tablebench_eval.py`: converts raw trajectories to
  TableBench-style records, subtype metrics, and `failure_cases.csv`.

## Run

```bash
cd /nas-files/ziyi/projects/proj_dsagent/repro_datamind

python3 scripts/prepare_tablebench_focus.py \
  --raw_file data/tablebench_raw/TableBench_data.jsonl \
  --csv_dir data/tablebench_csv_focus \
  --output_jsonl data/datamind_eval_inputs/tablebench_focus.jsonl

export OPENAI_BASE_URL=http://127.0.0.1:19007/v1
export OPENAI_API_KEY=placeholder_key

python3 DataMind/datamind/eval/python/eval_tablebench_react.py \
  --model qwen7b \
  --input_file data/datamind_eval_inputs/tablebench_focus.jsonl \
  --csv_folder data/tablebench_csv_focus \
  --output_dir outputs/qwen7b_datamind_react \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_turns 9 \
  --workers 5

python3 scripts/convert_react_to_tablebench_eval.py \
  --raw_results_file outputs/qwen7b_datamind_react/raw_react_results.jsonl \
  --output_dir outputs/qwen7b_datamind_react \
  --focus_only


python scripts/judge_tablebench_eval.py \
  --results_file outputs/<model>_tablebench/tablebench_eval_results.jsonl \
  --judge_model <judge-model-name>  
```
