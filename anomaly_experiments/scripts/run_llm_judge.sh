


# ====== environment variables (fill in your values) ======
export OPENAI_BASE_URL="https://aigc-api.hkust-gz.edu.cn/v1"
export OPENAI_API_KEY="${ust_api:?ust_api environment variable is required}"
WORKERS="${WORKERS:-3}"

python scripts/judge_tablebench_eval.py \
  --results_file /nas-files/ziyi/projects/proj_dsagent/repro_datamind/anomaly_experiments/outputs/DeepSeek-V4-Pro_anomaly_vision/tablebench_eval_results.jsonl \
  --judge_model gpt-3.5-turbo \
  --workers "${WORKERS}"   --overwrite


