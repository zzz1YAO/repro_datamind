#!/usr/bin/env bash
set -euo pipefail

# ====== environment variables (fill in your values) ======
export OPENAI_BASE_URL="https://aigc-api.hkust-gz.edu.cn/v1"
export OPENAI_API_KEY="${ust_api:?ust_api environment variable is required}"
export TEMPERATURE="0.7"
export TOP_P="0.95"
export MAX_TURNS="9"
export MAX_RESPONSE_LENGTH="4096"
export OUTPUT_DIR=""
# ====== end environment variables ======

SAMPLE_COUNT="${1:-2}"
MODEL_NAME="${2:-DeepSeek-V4-Pro}"

if [[ ! "${SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLE_COUNT must be a positive integer, got: ${SAMPLE_COUNT}" >&2
  exit 2
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL is required, for example http://127.0.0.1:19007/v1" >&2
  exit 2
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder_key}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALL_INPUT="${REPO_ROOT}/data/datamind_eval_inputs/tablebench_all.jsonl"
CSV_FOLDER="${REPO_ROOT}/data/tablebench_csv_all"
SMOKE_INPUT="${REPO_ROOT}/data/datamind_eval_inputs/tablebench_smoke_${SAMPLE_COUNT}.jsonl"
MODEL_SLUG="${MODEL_NAME//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${MODEL_SLUG}_smoke_${SAMPLE_COUNT}}"

if [[ ! -f "${ALL_INPUT}" ]]; then
  echo "Prepared eval input not found: ${ALL_INPUT}" >&2
  exit 2
fi
if [[ ! -d "${CSV_FOLDER}" ]]; then
  echo "Prepared CSV folder not found: ${CSV_FOLDER}" >&2
  exit 2
fi

AVAILABLE_COUNT="$(wc -l < "${ALL_INPUT}")"
if (( SAMPLE_COUNT > AVAILABLE_COUNT )); then
  echo "Requested ${SAMPLE_COUNT} samples, but only ${AVAILABLE_COUNT} are available." >&2
  exit 2
fi

head -n "${SAMPLE_COUNT}" "${ALL_INPUT}" > "${SMOKE_INPUT}"

echo "Running ${SAMPLE_COUNT} smoke sample(s) with model ${MODEL_NAME}"
echo "Input: ${SMOKE_INPUT}"
echo "Output: ${OUTPUT_DIR}"

python3 "${REPO_ROOT}/DataMind/datamind/eval/python/eval_tablebench_react.py" \
  --model "${MODEL_NAME}" \
  --input_file "${SMOKE_INPUT}" \
  --csv_folder "${CSV_FOLDER}" \
  --output_dir "${OUTPUT_DIR}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.95}" \
  --max_turns "${MAX_TURNS:-9}" \
  --max_response_length "${MAX_RESPONSE_LENGTH:-2048}" \
  --workers 1 \
  --overwrite

RESULT_FILE="${OUTPUT_DIR}/raw_react_results.jsonl"
if [[ ! -f "${RESULT_FILE}" ]]; then
  echo "Smoke test finished without creating ${RESULT_FILE}" >&2
  exit 1
fi

RESULT_COUNT="$(wc -l < "${RESULT_FILE}")"
echo "Smoke test completed: ${RESULT_COUNT}/${SAMPLE_COUNT} result row(s)"
echo "Results: ${RESULT_FILE}"
