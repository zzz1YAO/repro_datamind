# VLM Caption Generation Implementation Plan

> **For agentic workers:** Implement inline in the current session. The user explicitly requested no tests and direct deployment.

**Goal:** Add a resumable serial VLM caption generator that preserves full raw API responses and writes validated structured captions as JSONL.

**Architecture:** One Python CLI reads `VLM_prompt.md`, walks `images/*.png` in sorted order, sends one image per OpenAI-compatible request, and persists each response before parsing it. Raw responses are append-only; parsed results are atomically rewritten as one latest record per image ID so downstream evaluation has a stable lookup table.

**Tech Stack:** Python 3 standard library and the existing `openai` Python package.

## Global Constraints

- Work only under `/nas-files/ziyi/projects/proj_dsagent/repro_datamind/anomaly_experiments`.
- Calls must be serial; do not introduce workers or asynchronous requests.
- Read the API key from `ust_api`; never write credentials to disk.
- Do not call the VLM API during deployment.
- Do not add or run tests, per the user's explicit request.
- Preserve every API result or API exception in the raw JSONL before attempting structured parsing.
- Do not use an LLM to repair malformed JSON.

---

### Task 1: Implement the serial VLM caption CLI

**Files:**
- Create: `anomaly_experiments/scripts/generate_vlm_captions.py`

**Interfaces:**
- Consumes: `VLM_prompt.md`, `images/*.png`, `ust_api`, and an explicit `--model` argument.
- Produces: `data/vlm/vlm_raw_responses.jsonl` and `data/vlm/vlm_captions.jsonl`.
- Exposes: `extract_json_object(text)`, `validate_caption(payload)`, and `main()`.

- [ ] Resolve default paths relative to the experiment root rather than the caller's working directory.
- [ ] Convert each local image to a MIME-aware Base64 data URL.
- [ ] Send sorted images one at a time through `OpenAI.chat.completions.create`.
- [ ] Append a raw record containing ID, attempt, model, timestamps, content, full response, and errors.
- [ ] Parse direct JSON, fenced JSON, then balanced JSON objects without changing payload semantics.
- [ ] Validate the required prompt schema and confidence enum.
- [ ] Atomically maintain one latest parsed record per image ID, including explicit failure records.
- [ ] Skip successful IDs by default; retry failed IDs; support `--overwrite`, `--limit`, and `--fail-fast`.
- [ ] Print per-image status and a final summary.

### Task 2: Deploy without API execution

**Files:**
- Create: `anomaly_experiments/docs/superpowers/plans/2026-07-22-vlm-caption-generation.md`
- Create: `anomaly_experiments/scripts/generate_vlm_captions.py`

- [ ] Copy the implementation plan and script to the remote experiment directory.
- [ ] Confirm the deployed files exist and inspect Git status for the target directory.
- [ ] Do not invoke the script with a model or API credential.

