#!/usr/bin/env python3
"""Run Qwen3.5 native-vision TableBench tasks with the DATAMIND ReAct loop.

This file intentionally keeps DATAMIND's generation mechanism:
- OpenAI-compatible chat completion API.
- One action per turn, either <code>...</code> or <answer>...</answer>.
- Python execution through DATAMIND's Interpreter with files under data/files.

It intentionally does not use DATAMIND's GPT judge or reward_function. The raw
results are written for a separate TableBench-format evaluation pass.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


DATAMIND_SYSTEM_PROMPT = """You are an expert-level data analyst and statistician who solves data challenges through rigorous logic, systematic planning, and deep investigation. Your task is to answer the user's question by analyzing the provided CSV data source with Python code execution.

# Problem-Solving Protocol
At every assistant turn, follow this process:
1. Reason about the problem and the next step inside <think> and </think> tags.
2. After reasoning, provide exactly one executable action:
   - To inspect or analyze the data, provide Python code inside <code> and </code> tags using this format:
<code>
```python
<your Python code here>
```
</code>
   - When the problem is fully solved, provide the final answer inside <answer> and </answer> tags.

Never return only a <think> block. Every response must contain one complete <code> block or one complete <answer> block after the reasoning. Do not include multiple code or answer actions in the same response.

The system will execute code and return the printed results inside <interpreter> and </interpreter> tags. After receiving an interpreter result, reason about it carefully inside <think> and </think> tags. If the code failed or produced an unexpected result, explain the issue in the reasoning and provide corrected code. If the result is valid but the problem is not yet solved, provide the next code step. If the problem is solved, provide the final answer.

# Auxiliary Visual Evidence
You may inspect the image enclosed in <vision> and </vision> as supporting evidence when forming your judgment. The image itself is supplied as multimodal input. Treat it only as a reference and verify your conclusions against the CSV data.

# CSV Analysis Rules
1. The CSV file is available under data/files, and its exact path is provided in the user message. Use that path directly.
2. In the first code step, load the CSV and use print() to inspect its column names, first three rows, data types, and any other structure needed to understand the data.
3. Use Python and appropriate data-analysis libraries such as pandas, numpy, scipy, statsmodels, or sklearn when available.
4. Use print() for all important intermediate values because only printed output is returned by the interpreter.
5. Proceed one correct step at a time. Each new step should build on the previous successful analysis and its observed result.
6. Check calculations, filtering conditions, units, missing values, and output formatting before finalizing the answer.

# Final Answer Rules
1. Keep the final answer concise, precise, and directly tied to the user's question.
2. The final response must contain the answer inside <answer> and </answer> tags. For questions with a clearly defined short answer, such as numerical, categorical, or ranking questions, output only the answer value itself inside the `<answer>` tags, without any explanatory text or prefixes. for example: <answer>3</answer>
3. Do not use an untagged plain-text final answer.
4. Avoid irrelevant commentary outside <think>, <code>, and <answer> tags.
"""


INVALID_ACTION_OBSERVATION = (
    "Your previous action is invalid. If you want to execute code, put the code "
    "between <code> and </code>. If you want to give the final answer, put the "
    "answer between <answer> and </answer>. Please try again."
)


FINAL_TURN_OBSERVATION = (
    "This is your final turn. Do not call any more tools or provide any more "
    "Python code. Using only the evidence already gathered, provide your best "
    "final response now. Your response must contain exactly one final answer "
    "inside <answer> and </answer> tags. If the available evidence is "
    "insufficient, return <answer>No conclusion can be reached from the "
    "available evidence.</answer>."
)


ACTION_RE = re.compile(r"<(code|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


class ModelAction:
    def __init__(self, kind: Optional[str], content: str) -> None:
        self.kind = kind
        self.content = content


class EvalConfig:
    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_turns: int = 9,
        max_response_length: int = 8192,
        max_obs_length: int = 2048,
        csv_folder: str = "./data/tablebench_csv_focus",
        working_dir: str = "./outputs/workspace",
        vision_image_by_id: Optional[Dict[str, Path]] = None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_turns = max_turns
        self.max_response_length = max_response_length
        self.max_obs_length = max_obs_length
        self.csv_folder = csv_folder
        self.working_dir = working_dir
        self.vision_image_by_id = vision_image_by_id or {}


class OpenAIChatGenerator:
    def __init__(self, model_name: str, max_response_length: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to run model inference. "
                "Install DATAMIND eval requirements or run inside the intended environment."
            ) from exc

        kwargs = {
            "max_retries": 10,
            "timeout": 120.0,
        }
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(**kwargs)
        self.model_name = model_name
        self.max_response_length = max_response_length

    def respond(self, messages: List[Dict[str, Any]], temperature: float, top_p: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_response_length,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""


def encode_image_as_data_url(image_path: Path) -> str:
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_types.get(image_path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_datamind_messages(
    question: str,
    csv_file: str,
    vision_image: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    csv_name = Path(csv_file).name
    user_prompt = (
        "Please answer the question based on the following table.\n\n"
        f"CSV path: data/files/{csv_name}\n\n"
        f"Question: {question}\n\n"
        "For anomaly-detection questions, it is possible that no anomaly exists. "
        "If no anomaly is found, explicitly state that no anomaly is present and "
        "briefly explain why."
    )
    if vision_image is None:
        user_content: Any = f"{user_prompt}\n\nNow begin."
    else:
        user_content = [
            {"type": "text", "text": f"{user_prompt}\n\n<vision>\n"},
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(vision_image)},
            },
            {"type": "text", "text": "\n</vision>\n\nNow begin."},
        ]
    return [
        {"role": "system", "content": DATAMIND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def make_trajectory_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Omit base64 payloads from saved trajectories while preserving message shape."""
    content = message.get("content")
    if not isinstance(content, list):
        return dict(message)

    sanitized_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            sanitized_content.append(
                {"type": "image_url", "image_url": {"url": "<base64 image omitted>"}}
            )
        else:
            sanitized_content.append(item)
    return {**message, "content": sanitized_content}


def postprocess_response(raw_response: str) -> str:
    """Keep the first complete DATAMIND action and normalize tag case."""
    raw_response = raw_response or ""
    match = ACTION_RE.search(raw_response)
    if not match:
        return raw_response.strip()
    kind = match.group(1).lower()
    content = match.group(2).strip()
    if content:
        return f"<{kind}>\n{content}\n</{kind}>"
    return f"<{kind}></{kind}>"


def parse_model_action(response: str) -> ModelAction:
    match = ACTION_RE.search(response or "")
    if not match:
        return ModelAction(None, "")
    return ModelAction(match.group(1).lower(), match.group(2).strip())


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def strip_code_fence(code: str) -> str:
    code = code.strip()
    fence = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", code, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return code


def truncate_observation(observation: str, max_obs_length: int) -> str:
    if len(observation) <= max_obs_length:
        return observation
    suffix = "\n</interpreter>" if observation.endswith("</interpreter>") else ""
    body = observation[: max(0, max_obs_length - len(suffix))]
    return body + suffix


def safe_workspace_name(sample_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id.strip())
    return cleaned or "sample"


def normalize_sample(raw: Dict[str, Any]) -> Dict[str, str]:
    sample_id = str(raw.get("id", "")).strip()
    if not sample_id:
        raise ValueError(f"Missing id in sample: {raw}")

    question = str(raw.get("question", "")).strip()
    csv_file = str(raw.get("csv_file") or raw.get("csv_path") or "").strip()
    if not question:
        raise ValueError(f"Missing question in sample {sample_id}")
    if not csv_file:
        raise ValueError(f"Missing csv_file/csv_path in sample {sample_id}")

    return {
        "id": sample_id,
        "qtype": str(raw.get("qtype", "")).strip(),
        "qsubtype": str(raw.get("qsubtype", "")).strip(),
        "question": question,
        "gold_answer": str(raw.get("gold_answer", raw.get("answer", ""))).strip(),
        "csv_file": Path(csv_file).name,
    }


def make_result_record(
    sample: Dict[str, str],
    model_name: str,
    pred_answer: str,
    trajectory: List[Dict[str, Any]],
    parse_success: bool,
    execution_error_count: int,
) -> Dict[str, Any]:
    return {
        "id": sample["id"],
        "model_name": model_name,
        "qtype": sample.get("qtype", ""),
        "qsubtype": sample.get("qsubtype", ""),
        "question": sample["question"],
        "gold_answer": sample.get("gold_answer", ""),
        "pred_answer": pred_answer,
        "csv_file": sample["csv_file"],
        "traj": trajectory,
        "parse_success": parse_success,
        "execution_error_count": execution_error_count,
    }


class TableBenchReActRunner:
    def __init__(self, model_name: str, config: EvalConfig) -> None:
        self.model_name = model_name
        self.config = config
        self.generator = OpenAIChatGenerator(model_name, config.max_response_length)
        self.interpreter = None

    def _new_interpreter(self) -> Any:
        from interpreter import Interpreter

        return Interpreter(batch_size=1)

    def prepare_workspace(self, sample: Dict[str, str]) -> Path:
        workspace = Path(self.config.working_dir) / safe_workspace_name(sample["id"])
        data_files = workspace / "data" / "files"
        data_files.mkdir(parents=True, exist_ok=True)

        src = Path(self.config.csv_folder) / sample["csv_file"]
        if not src.exists():
            raise FileNotFoundError(f"CSV file not found for {sample['id']}: {src}")
        dst = data_files / sample["csv_file"]
        if not dst.exists():
            shutil.copy2(src, dst)
        return workspace

    def execute_code(self, code: str, workspace: Path) -> str:
        if self.interpreter is None:
            self.interpreter = self._new_interpreter()

        generation_code = strip_code_fence(code)
        origin_workdir = os.getcwd()
        try:
            os.chdir(str(workspace))
            result, report = self.interpreter.apply((0, generation_code))
        finally:
            os.chdir(origin_workdir)

        result = str(result).strip()
        report = str(report).strip()
        if result and report == "Done":
            exec_result = f"The code run successfully:\n{result}"
        elif report != "Done" and not result:
            exec_result = f"The code run failed:\n{report}"
        elif report != "Done" and result:
            exec_result = f"The code run failed:\n{report}\n\nBut we capture part of your code output:\n{result}"
        else:
            exec_result = (
                "We couldn't capture the output from your code. Please rewrite your "
                "last step code and use print() statements to display key values."
            )
        return f"<interpreter>\n{exec_result.strip()}\n</interpreter>"

    def run_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = normalize_sample(raw_sample)
        workspace = self.prepare_workspace(sample)
        self.interpreter = self._new_interpreter()
        vision_image = self.config.vision_image_by_id.get(sample["id"])
        messages = build_datamind_messages(
            sample["question"],
            sample["csv_file"],
            vision_image=vision_image,
        )
        trajectory = [make_trajectory_message(message) for message in messages]
        pred_answer = ""
        parse_success = False
        execution_error_count = 0

        for _step in range(self.config.max_turns):
            if _step == self.config.max_turns - 1:
                final_turn_message = {"role": "user", "content": FINAL_TURN_OBSERVATION}
                messages.append(final_turn_message)
                trajectory.append(final_turn_message)

            raw_response = self.generator.respond(
                messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            response = postprocess_response(raw_response)
            action = parse_model_action(response)
            assistant_message = {"role": "assistant", "content": response}
            messages.append(assistant_message)
            trajectory.append(assistant_message)

            if action.kind == "answer":
                pred_answer = action.content
                parse_success = True
                break

            if action.kind == "code":
                observation = self.execute_code(action.content, workspace)
            else:
                observation = INVALID_ACTION_OBSERVATION

            if "The code run failed:" in observation or observation == INVALID_ACTION_OBSERVATION:
                execution_error_count += 1

            observation = truncate_observation(observation, self.config.max_obs_length)
            observation_message = {"role": "user", "content": observation}
            messages.append(observation_message)
            trajectory.append(observation_message)

        if not pred_answer:
            pred_answer = extract_answer(
                "\n".join(
                    item["content"]
                    for item in trajectory
                    if isinstance(item.get("content"), str)
                )
            )
            parse_success = bool(pred_answer)

        return make_result_record(
            sample=sample,
            model_name=self.model_name,
            pred_answer=pred_answer,
            trajectory=trajectory,
            parse_success=parse_success,
            execution_error_count=execution_error_count,
        )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def load_vision_images(image_folder: Path) -> Dict[str, Path]:
    if not image_folder.is_dir():
        raise ValueError(f"--vision_file must point to an image directory: {image_folder}")

    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    images_by_id: Dict[str, Path] = {}
    for image_path in image_folder.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in supported_suffixes:
            images_by_id[image_path.stem] = image_path
    return images_by_id


def filter_samples_by_qtypes(
    rows: Iterable[Dict[str, Any]],
    qtypes: Optional[Iterable[str]],
    qsubtypes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    if qtypes:
        selected_qtypes = {str(qtype).strip() for qtype in qtypes if str(qtype).strip()}
        rows = [
            row
            for row in rows
            if str(row.get("qtype", "")).strip() in selected_qtypes
        ]
    if qsubtypes:
        selected_qsubtypes = {
            str(qsubtype).strip() for qsubtype in qsubtypes if str(qsubtype).strip()
        }
        rows = [
            row
            for row in rows
            if str(row.get("qsubtype", "")).strip() in selected_qsubtypes
        ]
    return rows


def iter_existing_ids(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row:
                ids.append(str(row["id"]))
    return ids


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_one(raw_sample: Dict[str, Any], model_name: str, config: EvalConfig) -> Dict[str, Any]:
    try:
        runner = TableBenchReActRunner(model_name=model_name, config=config)
        return runner.run_sample(raw_sample)
    except Exception as exc:
        sample_id = str(raw_sample.get("id", "unknown"))
        traceback_text = traceback.format_exc()
        normalized = {
            "id": sample_id,
            "qtype": str(raw_sample.get("qtype", "")),
            "qsubtype": str(raw_sample.get("qsubtype", "")),
            "question": str(raw_sample.get("question", "")),
            "gold_answer": str(raw_sample.get("gold_answer", raw_sample.get("answer", ""))),
            "csv_file": str(raw_sample.get("csv_file", raw_sample.get("csv_path", ""))),
        }
        return make_result_record(
            sample=normalized,
            model_name=model_name,
            pred_answer="",
            trajectory=[
                {"role": "system", "content": "runner exception"},
                {"role": "user", "content": traceback_text},
            ],
            parse_success=False,
            execution_error_count=1,
        )


def run_eval(
    model_name: str,
    input_file: Path,
    csv_folder: Path,
    output_dir: Path,
    temperature: float,
    top_p: float,
    max_turns: int,
    workers: int,
    max_response_length: int,
    max_obs_length: int,
    vision_file: Optional[Path] = None,
    require_vision: bool = False,
    qtypes: Optional[Iterable[str]] = None,
    qsubtypes: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "raw_react_results.jsonl"
    if overwrite and output_file.exists():
        output_file.unlink()

    vision_image_by_id: Dict[str, Path] = {}
    if vision_file is not None:
        vision_image_by_id = load_vision_images(vision_file)
    if require_vision and vision_file is None:
        raise ValueError("--require_vision requires --vision_file")

    rows = filter_samples_by_qtypes(read_jsonl(input_file), qtypes, qsubtypes)
    if require_vision:
        rows = [
            row
            for row in rows
            if str(row.get("id", "")).strip() in vision_image_by_id
        ]
    solved_ids = set(iter_existing_ids(output_file))
    todo = [row for row in rows if str(row.get("id", "")) not in solved_ids]

    config = EvalConfig(
        temperature=temperature,
        top_p=top_p,
        max_turns=max_turns,
        max_response_length=max_response_length,
        max_obs_length=max_obs_length,
        csv_folder=str(csv_folder),
        working_dir=str(output_dir / "workspace"),
        vision_image_by_id=vision_image_by_id,
    )

    completed_count = len(rows) - len(todo)
    with tqdm(
        total=len(rows),
        initial=completed_count,
        desc="TableBench eval",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        if workers <= 1:
            for row in todo:
                write_jsonl_row(output_file, run_one(row, model_name, config))
                progress.update(1)
            return output_file

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, row, model_name, config) for row in todo]
            for future in as_completed(futures):
                write_jsonl_row(output_file, future.result())
                progress.update(1)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--csv_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_turns", type=int, default=9)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max_response_length", type=int, default=8192)
    parser.add_argument("--max_obs_length", type=int, default=2048)
    parser.add_argument(
        "--vision_file",
        type=Path,
        help="Optional image directory containing files named {sample_id}.png.",
    )
    parser.add_argument(
        "--require_vision",
        action="store_true",
        help="Evaluate only samples with a matching image in --vision_file.",
    )
    parser.add_argument(
        "--qtypes",
        nargs="+",
        help="Optional major TableBench qtypes, for example DataAnalysis NumericalReasoning.",
    )
    parser.add_argument(
        "--qsubtypes",
        nargs="+",
        help="Optional TableBench qsubtypes, for example AnomalyDetection Aggregation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = run_eval(
        model_name=args.model,
        input_file=Path(args.input_file),
        csv_folder=Path(args.csv_folder),
        output_dir=Path(args.output_dir),
        temperature=args.temperature,
        top_p=args.top_p,
        max_turns=args.max_turns,
        workers=args.workers,
        max_response_length=args.max_response_length,
        max_obs_length=args.max_obs_length,
        vision_file=args.vision_file,
        require_vision=args.require_vision,
        qtypes=args.qtypes,
        qsubtypes=args.qsubtypes,
        overwrite=args.overwrite,
    )
    print(f"Wrote raw DATAMIND ReAct results to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
