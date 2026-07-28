#!/usr/bin/env bash
set -euo pipefail

export OPENAI_BASE_URL="https://aigc-api.hkust-gz.edu.cn/v1"
export OPENAI_API_KEY="${ust_api:?ust_api environment variable is required}"

MODEL_NAME="${MODEL_NAME:-qwen3.5-plus}"
# Leave either array empty to avoid filtering on that level. Examples:
# QTYPES=(DataAnalysis NumericalReasoning)
# QSUBTYPES=(Aggregation AnomalyDetection)
# Evaluate only DataAnalysis / AnomalyDetection:
QTYPES=(DataAnalysis NumericalReasoning)
QSUBTYPES=(AnomalyDetection)

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-4096}"
WORKERS="${WORKERS:-1}"
OVERWRITE="${OVERWRITE:-0}"
USE_VISION="${USE_VISION:-1}"

if [[ -z "${OPENAI_BASE_URL}" ]]; then
  echo "OPENAI_BASE_URL is required, for example http://127.0.0.1:7866/v1" >&2
  exit 2
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1, got: ${OVERWRITE}" >&2
  exit 2
fi
if [[ "${USE_VISION}" != "0" && "${USE_VISION}" != "1" ]]; then
  echo "USE_VISION must be 0 or 1, got: ${USE_VISION}" >&2
  exit 2
fi

for qtype in "${QTYPES[@]}"; do
  case "${qtype}" in
    DataAnalysis|NumericalReasoning|FactChecking|Visualization) ;;
    *)
      echo "Unsupported qtype: ${qtype}" >&2
      echo "Valid qtypes: DataAnalysis NumericalReasoning FactChecking Visualization" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_FILE="${EXPERIMENT_ROOT}/data/anomlay_problem.jsonl"
CSV_FOLDER="${EXPERIMENT_ROOT}/data/csv"
IMAGE_FOLDER="${EXPERIMENT_ROOT}/images"
EVAL_SCRIPT="${EXPERIMENT_ROOT}/eval/eval_tablebench_react_for_qwen35.py"
MODEL_SLUG="${MODEL_NAME//\//_}"

if (( ${#QTYPES[@]} == 0 )); then
  CATALOG_NAME="all"
else
  CATALOG_NAME="$(IFS=+; echo "${QTYPES[*]}")"
fi
if (( ${#QSUBTYPES[@]} > 0 )); then
  QSUBTYPE_NAME="$(IFS=+; echo "${QSUBTYPES[*]}")"
  CATALOG_NAME="${CATALOG_NAME}_${QSUBTYPE_NAME}"
fi
CATALOG_SLUG="${CATALOG_NAME//\//_}"
if [[ "${USE_VISION}" == "1" ]]; then
  VISION_SUFFIX="_native_vision"
else
  VISION_SUFFIX=""
fi
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/${MODEL_SLUG}_anomaly${VISION_SUFFIX}}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Prepared eval input not found: ${INPUT_FILE}" >&2
  exit 2
fi
if [[ ! -d "${CSV_FOLDER}" ]]; then
  echo "Prepared CSV folder not found: ${CSV_FOLDER}" >&2
  exit 2
fi
if [[ "${USE_VISION}" == "1" && ! -d "${IMAGE_FOLDER}" ]]; then
  echo "Image folder not found: ${IMAGE_FOLDER}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Evaluation script not found: ${EVAL_SCRIPT}" >&2
  exit 2
fi

INPUT_COUNT="$(wc -l < "${INPUT_FILE}")"
echo "Starting Qwen3.5 native-multimodal TableBench DATAMIND-ReAct evaluation"
echo "Model: ${MODEL_NAME}"
echo "Catalog: ${CATALOG_NAME}"
echo "Qsubtypes: ${QSUBTYPES[*]:-all}"
echo "Prepared input rows: ${INPUT_COUNT}"
echo "Use native vision: ${USE_VISION}"
if [[ "${USE_VISION}" == "1" ]]; then
  echo "Image folder: ${IMAGE_FOLDER}"
fi
echo "Output: ${OUTPUT_DIR}"
echo "Overwrite: ${OVERWRITE}"

EVAL_ARGS=(
  "${EVAL_SCRIPT}"
  --model "${MODEL_NAME}"
  --input_file "${INPUT_FILE}"
  --csv_folder "${CSV_FOLDER}"
  --output_dir "${OUTPUT_DIR}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_turns "${MAX_TURNS}"
  --max_response_length "${MAX_RESPONSE_LENGTH}"
  --max_obs_length "${MAX_OBS_LENGTH}"
  --workers "${WORKERS}"
)

if [[ "${USE_VISION}" == "1" ]]; then
  EVAL_ARGS+=(--vision_file "${IMAGE_FOLDER}")
fi

if (( ${#QTYPES[@]} > 0 )); then
  EVAL_ARGS+=(--qtypes "${QTYPES[@]}")
fi
if (( ${#QSUBTYPES[@]} > 0 )); then
  EVAL_ARGS+=(--qsubtypes "${QSUBTYPES[@]}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  EVAL_ARGS+=(--overwrite)
fi

python3 "${EVAL_ARGS[@]}"

RESULT_FILE="${OUTPUT_DIR}/raw_react_results.jsonl"
if [[ ! -f "${RESULT_FILE}" ]]; then
  echo "Evaluation finished without creating ${RESULT_FILE}" >&2
  exit 1
fi

RESULT_COUNT="$(wc -l < "${RESULT_FILE}")"
echo "Formal evaluation completed with ${RESULT_COUNT} result row(s)"
echo "Results: ${RESULT_FILE}"
