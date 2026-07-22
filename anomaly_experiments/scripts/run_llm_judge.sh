


# ====== environment variables (fill in your values) ======
export OPENAI_BASE_URL="https://aigc-api.hkust-gz.edu.cn/v1"
export OPENAI_API_KEY="${ust_api:?ust_api environment variable is required}"

python scripts/judge_tablebench_eval.py \
  --results_file /nas-files/ziyi/projects/proj_dsagent/repro_datamind/outputs/qwen3-8b_anomaly/tablebench_eval_results.jsonl \
  --judge_model DeepSeek-V4-Flash
