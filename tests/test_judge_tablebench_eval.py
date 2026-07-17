import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "judge_tablebench_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("judge_tablebench_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if not isinstance(outcome, str):
            return outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, outcomes):
        completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


class FakeJudge:
    def __init__(self, model_name="judge-model", error=None):
        self.model_name = model_name
        self.error = error
        self.calls = []

    def evaluate(self, record):
        self.calls.append(record["id"])
        if self.error:
            raise self.error
        return f"Evaluation for {record['id']}.", 1


class JudgeResponseTests(unittest.TestCase):
    def test_parses_one_complete_thought_and_binary_score(self):
        module = load_module()
        reason, score = module.parse_judge_response(
            "<THOUGHT>Semantically equivalent.</THOUGHT>\n<score>1</score>"
        )
        self.assertEqual(reason, "Semantically equivalent.")
        self.assertEqual(score, 1)

        reason, score = module.parse_judge_response(
            "<thought>A material conclusion is wrong.</thought><score>0</score>"
        )
        self.assertEqual(reason, "A material conclusion is wrong.")
        self.assertEqual(score, 0)

    def test_rejects_missing_or_ambiguous_tags(self):
        module = load_module()
        invalid = [
            "<score>1</score>",
            "<thought>Reason.</thought><score>2</score>",
            "<thought>Reason.</thought><score>0</score><score>1</score>",
            "<thought>First.</thought><thought>Second.</thought><score>1</score>",
            "<thought></thought><score>1</score>",
        ]
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    module.parse_judge_response(response)

    def test_prompt_contains_metric_name_but_not_metric_score(self):
        module = load_module()
        record = {
            "qtype": "DataAnalysis",
            "qsubtype": "AnomalyDetection",
            "metric_name": "ROUGE-L",
            "metric_score": 0.123456789,
            "question": "Which row is anomalous?",
            "prediction": "row 2",
            "answer": "row 2",
        }
        messages = module.build_judge_messages(record)
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("ROUGE-L", rendered)
        self.assertIn("Which row is anomalous?", rendered)
        self.assertNotIn("0.123456789", rendered)
        self.assertIn("0.03", rendered)

    def test_openai_judge_retries_invalid_output_then_succeeds(self):
        module = load_module()
        client = FakeClient(
            [
                "invalid response",
                "<thought>The answer matches.</thought><score>1</score>",
            ]
        )
        judge = module.OpenAIChatJudge(
            model_name="judge-model",
            max_retries=2,
            client=client,
        )
        reason, score = judge.evaluate(
            {
                "qtype": "NumericalReasoning",
                "qsubtype": "Counting",
                "metric_name": "EM",
                "question": "How many?",
                "prediction": "3",
                "answer": "3",
            }
        )
        self.assertEqual((reason, score), ("The answer matches.", 1))
        self.assertEqual(len(client.completions.calls), 2)
        request = client.completions.calls[-1]
        self.assertEqual(request["model"], "judge-model")
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 512)

    def test_openai_judge_raises_after_all_attempts_fail(self):
        module = load_module()
        client = FakeClient([RuntimeError("timeout"), "still invalid"])
        judge = module.OpenAIChatJudge(
            model_name="judge-model",
            max_retries=2,
            client=client,
        )
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            judge.evaluate(
                {
                    "qtype": "DataAnalysis",
                    "qsubtype": "ImpactAnalysis",
                    "metric_name": "EM",
                    "question": "What is the impact?",
                    "prediction": "positive",
                    "answer": "positive",
                }
            )

    def test_debug_mode_is_removed_and_null_choices_have_a_clear_error(self):
        module = load_module()
        response = SimpleNamespace(choices=None)
        client = FakeClient([response])
        judge = module.OpenAIChatJudge(
            model_name="judge-model",
            max_retries=1,
            client=client,
        )

        with self.assertRaisesRegex(RuntimeError, "response has no choices"):
            judge.evaluate(
                {
                    "id": "sample-id",
                    "qtype": "NumericalReasoning",
                    "qsubtype": "Aggregation",
                    "metric_name": "EM",
                    "question": "How many?",
                    "prediction": "3",
                    "answer": "3",
                }
            )

        self.assertNotIn(
            "debug", inspect.signature(module.OpenAIChatJudge).parameters
        )
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("--debug", source)
        self.assertNotIn("[judge debug]", source)


class ResultsFileTests(unittest.TestCase):
    def write_rows(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def read_rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def base_row(self, row_id, prediction="answer"):
        return {
            "id": row_id,
            "qtype": "DataAnalysis",
            "qsubtype": "ImpactAnalysis",
            "question": "Question?",
            "answer": "answer",
            "prediction": prediction,
            "parsed_result": {"Parse@1": bool(prediction)},
            "metric_name": "EM",
            "metric_score": 0.0,
            "llm_judge_score": None,
            "llm_judge_reason": None,
            "llm_judge_model": None,
        }

    def test_limit_resume_missing_answer_and_atomic_preservation(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tablebench_eval_results.jsonl"
            metrics_path = path.with_name("subtype_metrics.json")
            rows = [
                self.base_row("already"),
                self.base_row("missing", prediction=""),
                self.base_row("judge-one"),
                self.base_row("judge-two"),
            ]
            rows[0].update(
                llm_judge_score=1,
                llm_judge_reason="existing",
                llm_judge_model="old-judge",
            )
            self.write_rows(path, rows)
            metrics_path.write_text(
                json.dumps(
                    {"ImpactAnalysis": {"count": 4, "score": 0.25}},
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.chmod(path, 0o664)
            expected_mode = path.stat().st_mode & 0o777

            first_judge = FakeJudge()
            first = module.process_results_file(
                path,
                first_judge,
                limit=2,
                show_progress=False,
            )
            updated = self.read_rows(path)
            self.assertEqual(first["processed"], 2)
            self.assertEqual(first_judge.calls, ["judge-one"])
            self.assertEqual(updated[0]["llm_judge_reason"], "existing")
            self.assertEqual(updated[1]["llm_judge_score"], 0)
            self.assertEqual(
                updated[1]["llm_judge_reason"],
                "No valid final answer was captured.",
            )
            self.assertIsNone(updated[1]["llm_judge_model"])
            self.assertEqual(updated[2]["llm_judge_score"], 1)
            self.assertIsNone(updated[3]["llm_judge_score"])
            self.assertEqual([row["id"] for row in updated], [row["id"] for row in rows])
            self.assertEqual(updated[2]["metric_score"], 0.0)
            self.assertEqual(path.stat().st_mode & 0o777, expected_mode)
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
            subtype_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                subtype_metrics["ImpactAnalysis"],
                {
                    "count": 4,
                    "score": 0.25,
                    "llm_judge_count": 3,
                    "llm_judge_score": 2 / 3,
                },
            )
            self.assertFalse(Path(str(metrics_path) + ".tmp").exists())

            second_judge = FakeJudge()
            second = module.process_results_file(
                path,
                second_judge,
                show_progress=False,
            )
            self.assertEqual(second["processed"], 1)
            self.assertEqual(second_judge.calls, ["judge-two"])
            subtype_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                subtype_metrics["ImpactAnalysis"]["llm_judge_count"], 4
            )
            self.assertEqual(
                subtype_metrics["ImpactAnalysis"]["llm_judge_score"], 0.75
            )

            overwrite_judge = FakeJudge(model_name="new-judge")
            overwritten = module.process_results_file(
                path,
                overwrite_judge,
                overwrite=True,
                limit=1,
                show_progress=False,
            )
            self.assertEqual(overwritten["processed"], 1)
            self.assertEqual(overwrite_judge.calls, ["already"])
            self.assertEqual(self.read_rows(path)[0]["llm_judge_model"], "new-judge")

    def test_api_failure_is_saved_as_retryable_null_score(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tablebench_eval_results.jsonl"
            self.write_rows(path, [self.base_row("failed")])
            judge = FakeJudge(error=RuntimeError("service unavailable"))

            result = module.process_results_file(
                path,
                judge,
                show_progress=False,
            )
            row = self.read_rows(path)[0]
            self.assertEqual(result["processed"], 1)
            self.assertIsNone(row["llm_judge_score"])
            self.assertIn("service unavailable", row["llm_judge_reason"])
            self.assertEqual(row["llm_judge_model"], "judge-model")

    def test_parse_failure_with_nonempty_prediction_does_not_call_api(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tablebench_eval_results.jsonl"
            row = self.base_row("parse-failed", prediction="looks valid")
            row["parsed_result"]["Parse@1"] = False
            original_fields = {
                key: value
                for key, value in row.items()
                if not key.startswith("llm_judge_")
            }
            self.write_rows(path, [row])
            judge = FakeJudge()

            module.process_results_file(path, judge, show_progress=False)

            updated = self.read_rows(path)[0]
            self.assertEqual(judge.calls, [])
            self.assertEqual(updated["llm_judge_score"], 0)
            self.assertEqual(
                {
                    key: value
                    for key, value in updated.items()
                    if not key.startswith("llm_judge_")
                },
                original_fields,
            )

    def test_fully_scored_file_refreshes_subtype_metrics_without_api_calls(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tablebench_eval_results.jsonl"
            row = self.base_row("complete")
            row.update(
                llm_judge_score=0,
                llm_judge_reason="incorrect",
                llm_judge_model="old-judge",
            )
            self.write_rows(path, [row])
            judge = FakeJudge()

            result = module.process_results_file(path, judge, show_progress=False)

            metrics = json.loads(
                path.with_name("subtype_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["processed"], 0)
            self.assertEqual(judge.calls, [])
            self.assertEqual(
                metrics["ImpactAnalysis"],
                {
                    "count": 1,
                    "score": 0.0,
                    "llm_judge_count": 1,
                    "llm_judge_score": 0.0,
                },
            )


if __name__ == "__main__":
    unittest.main()
