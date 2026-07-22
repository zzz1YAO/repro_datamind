"""Generate serial VLM captions for local PNG chart images via Chat Completions.

The script keeps an append-only raw response history and a compact, atomically
rewritten latest-result file so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


SCHEMA_VERSION = "1.0"
REQUIRED_ROOT_KEYS = {
    "chart_structure",
    "salient_observations",
    "visual_summary",
    "uncertainties",
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for an output record."""
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments using paths relative to this script."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Chat Completions model name.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://aigc-api.hkust-gz.edu.cn/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="ust_api",
        help="Environment variable that contains the API key (default: ust_api).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries after an API exception.")
    parser.add_argument("--limit", type=int, help="Process at most this many sorted PNG files.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess successful images too.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first API exception.")
    parser.add_argument("--prompt", type=Path, default=root / "VLM_prompt.md")
    parser.add_argument("--images", type=Path, default=root / "images")
    parser.add_argument("--raw-output", type=Path, default=root / "data" / "vlm" / "vlm_raw_responses.jsonl")
    parser.add_argument("--parsed-output", type=Path, default=root / "data" / "vlm" / "vlm_captions.jsonl")
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be zero or greater")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or greater")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load valid object records from a JSONL file, if it already exists."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    "warning: ignoring malformed JSONL record in {0}:{1}: {2}".format(
                        path, line_number, exc.msg
                    ),
                    file=sys.stderr,
                )
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                print(
                    "warning: ignoring non-object JSONL record in {0}:{1}".format(path, line_number),
                    file=sys.stderr,
                )
    return records


def latest_by_image_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keep the last record for every string image identifier."""
    latest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        image_id = record.get("id", record.get("image_id"))
        if isinstance(image_id, str):
            latest[image_id] = record
    return latest


def next_attempts(raw_records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Return the next attempt number for each image from raw history."""
    attempts: Dict[str, int] = {}
    for record in raw_records:
        image_id = record.get("id", record.get("image_id"))
        attempt = record.get("attempt_number")
        if isinstance(image_id, str) and isinstance(attempt, int):
            attempts[image_id] = max(attempts.get(image_id, 0), attempt)
    return attempts


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one durable JSONL record, creating its parent only at run time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def rewrite_latest_jsonl(path: Path, latest: Dict[str, Dict[str, Any]]) -> None:
    """Atomically rewrite the one-latest-record-per-image parsed JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for image_id in sorted(latest):
            handle.write(json.dumps(latest[image_id], ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def image_data_url(path: Path) -> str:
    """Encode an image as a MIME-aware Base64 data URL."""
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:{0};base64,{1}".format(mime_type, encoded)


def response_to_jsonable(response: Any) -> Any:
    """Return the SDK response in a form accepted by json.dumps."""
    if hasattr(response, "model_dump"):
        try:
            value = response.model_dump(mode="json")
        except TypeError:
            value = response.model_dump()
    elif hasattr(response, "to_dict"):
        value = response.to_dict()
    else:
        value = response
    # The final conversion protects output persistence if an unusual SDK object remains.
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def response_content(response: Any) -> Optional[str]:
    """Extract the first completion message's textual content, if present."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else None


def balanced_json_objects(text: str) -> Iterable[Any]:
    """Yield decoded balanced JSON objects while correctly tracking strings/escapes."""
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        yield json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    break


def parse_caption(text: Optional[str]) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Parse caption JSON using direct, fenced, then balanced-object strategies."""
    if not isinstance(text, str) or not text.strip():
        return None, None, "Response content was empty or not textual."
    stripped = text.strip()
    try:
        return json.loads(stripped), "direct_json", None
    except json.JSONDecodeError:
        pass

    fence_pattern = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate), "markdown_fence", None
        except json.JSONDecodeError:
            continue

    for value in balanced_json_objects(text):
        return value, "balanced_object", None
    return None, None, "No valid JSON object was found in the response text."


def validate_caption(caption: Any) -> List[str]:
    """Validate the prompt's expected caption schema without changing its payload."""
    errors: List[str] = []
    if not isinstance(caption, dict):
        return ["root must be a JSON object"]
    actual_keys = set(caption)
    missing = REQUIRED_ROOT_KEYS - actual_keys
    unexpected = actual_keys - REQUIRED_ROOT_KEYS
    if missing:
        errors.append("root missing keys: {0}".format(", ".join(sorted(missing))))
    if unexpected:
        errors.append("root has unexpected keys: {0}".format(", ".join(sorted(unexpected))))

    structure = caption.get("chart_structure")
    if not isinstance(structure, dict):
        errors.append("chart_structure must be an object")
    else:
        for field in ("chart_type", "x_axis", "y_axis"):
            if not isinstance(structure.get(field), str):
                errors.append("chart_structure.{0} must be a string".format(field))
        if not isinstance(structure.get("series_or_groups"), list):
            errors.append("chart_structure.series_or_groups must be a list")

    observations = caption.get("salient_observations")
    if not isinstance(observations, list):
        errors.append("salient_observations must be a list")
    else:
        for index, observation in enumerate(observations):
            prefix = "salient_observations[{0}]".format(index)
            if not isinstance(observation, dict):
                errors.append("{0} must be an object".format(prefix))
                continue
            for field in ("pattern_type", "location", "description"):
                if not isinstance(observation.get(field), str):
                    errors.append("{0}.{1} must be a string".format(prefix, field))
            if observation.get("confidence") not in ("high", "medium", "low"):
                errors.append("{0}.confidence must be high, medium, or low".format(prefix))

    if not isinstance(caption.get("visual_summary"), str):
        errors.append("visual_summary must be a string")
    if not isinstance(caption.get("uncertainties"), list):
        errors.append("uncertainties must be a list")
    return errors


def parsed_record(
    image_id: str,
    filename: str,
    model: str,
    attempt_number: int,
    parse_status: str,
    parse_method: Optional[str],
    caption: Optional[Any],
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    schema_errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a parsed-output record with consistent metadata and error details."""
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_timestamp(),
        "id": image_id,
        "image_file": filename,
        "requested_model": model,
        "attempt_number": attempt_number,
        "parse_status": parse_status,
        "parse_method": parse_method,
        "caption": caption,
        "error_type": error_type,
        "error_message": error_message,
        "schema_errors": schema_errors,
    }


def validate_inputs(args: argparse.Namespace) -> Tuple[str, List[Path]]:
    """Check all preconditions before constructing a client or making a request."""
    if not args.prompt.is_file():
        raise SystemExit("Prompt file is missing: {0}".format(args.prompt))
    if not args.images.is_dir():
        raise SystemExit("Images directory is missing: {0}".format(args.images))
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit("API key is unset: set environment variable {0}.".format(args.api_key_env))
    image_paths = sorted(args.images.glob("*.png"), key=lambda item: item.name)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    return args.prompt.read_text(encoding="utf-8"), image_paths


def main() -> int:
    """Run the resumable, serial caption-generation workflow."""
    args = parse_args()
    prompt, image_paths = validate_inputs(args)
    api_key = os.environ[args.api_key_env]
    raw_records = load_jsonl(args.raw_output)
    latest = latest_by_image_id(load_jsonl(args.parsed_output))
    attempts = next_attempts(raw_records)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0)

    counts = {"processed": 0, "skipped": 0, "ok": 0, "schema_error": 0, "json_error": 0, "api_error": 0}
    for image_path in image_paths:
        image_id = image_path.stem
        previous = latest.get(image_id)
        if not args.overwrite and previous and previous.get("parse_status") == "ok":
            counts["skipped"] += 1
            print("{0}: skipped (already ok)".format(image_path.name))
            continue

        counts["processed"] += 1
        data_url = image_data_url(image_path)
        attempt_number = attempts.get(image_id, 0)
        for retry_index in range(args.max_retries + 1):
            attempt_number += 1
            attempts[image_id] = attempt_number
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                raw = {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp": utc_timestamp(),
                    "id": image_id,
                    "image_file": image_path.name,
                    "requested_model": args.model,
                    "attempt_number": attempt_number,
                    "status": "api_error",
                    "raw_content": None,
                    "response": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                append_jsonl(args.raw_output, raw)
                record = parsed_record(
                    image_id, image_path.name, args.model, attempt_number, "api_error", None, None,
                    type(exc).__name__, str(exc),
                )
                latest[image_id] = record
                rewrite_latest_jsonl(args.parsed_output, latest)
                if args.fail_fast:
                    print("{0}: api_error (attempt {1})".format(image_path.name, attempt_number))
                    raise
                if retry_index == args.max_retries:
                    counts["api_error"] += 1
                    print("{0}: api_error (attempt {1})".format(image_path.name, attempt_number))
                continue

            content = response_content(response)
            raw = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": utc_timestamp(),
                "id": image_id,
                "image_file": image_path.name,
                "requested_model": args.model,
                "attempt_number": attempt_number,
                "status": "response",
                "raw_content": content,
                "response": response_to_jsonable(response),
                "error_type": None,
                "error_message": None,
            }
            append_jsonl(args.raw_output, raw)
            caption, method, parse_error = parse_caption(content)
            if parse_error is not None:
                status = "json_error"
                record = parsed_record(
                    image_id, image_path.name, args.model, attempt_number, status, method, caption,
                    "JSONParseError", parse_error,
                )
            else:
                schema_errors = validate_caption(caption)
                if schema_errors:
                    status = "schema_error"
                    record = parsed_record(
                        image_id, image_path.name, args.model, attempt_number, status, method, caption,
                        "SchemaValidationError", "; ".join(schema_errors), schema_errors,
                    )
                else:
                    status = "ok"
                    record = parsed_record(image_id, image_path.name, args.model, attempt_number, status, method, caption)
            latest[image_id] = record
            rewrite_latest_jsonl(args.parsed_output, latest)
            counts[status] += 1
            print("{0}: {1} (attempt {2})".format(image_path.name, status, attempt_number))
            break

    print(
        "summary: processed={processed} skipped={skipped} ok={ok} schema_error={schema_error} "
        "json_error={json_error} api_error={api_error}".format(**counts)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
