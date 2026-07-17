import csv
import importlib.util
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "convert_react_to_tablebench_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_react_to_tablebench_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_record_matches_tablebench_eval_shape():
    module = load_module()
    raw = {
        "id": "tb_1",
        "model_name": "qwen7b",
        "qtype": "NumericalReasoning",
        "qsubtype": "NumericalReasoning_Counting",
        "question": "How many?",
        "gold_answer": "10",
        "pred_answer": "10",
        "csv_file": "tb_1.csv",
        "traj": [],
        "parse_success": True,
        "execution_error_count": 0,
    }

    converted = module.to_tablebench_record(raw, instruction_type="DATAMIND-ReAct")

    assert converted["id"] == "tb_1"
    assert converted["model_name"] == "qwen7b-datamind-react"
    assert converted["instruction_type"] == "DATAMIND-ReAct"
    assert converted["answer"] == "10"
    assert converted["prediction"] == "10"
    assert converted["parsed_result"] == {
        "parsed_prediction": "10",
        "Parse@1": True,
    }
    assert converted["llm_judge_score"] is None
    assert converted["llm_judge_reason"] is None
    assert converted["llm_judge_model"] is None


def test_metric_selection_and_scores_follow_tablebench_guidance():
    module = load_module()

    em_record = {
        "qtype": "FactChecking",
        "qsubtype": "FactChecking_Multi-hop FactChecking",
        "answer": "Yes",
        "prediction": " yes ",
    }
    assert module.evaluate_tablebench_record(em_record) == ("EM", 1.0)

    tol_record = {
        "qtype": "DataAnalysis",
        "qsubtype": "DataAnalysis_CorrelationAnalysis",
        "answer": "100",
        "prediction": "109.9",
    }
    assert module.evaluate_tablebench_record(tol_record) == ("EM_with_error_10", 1.0)

    rouge_record = {
        "qtype": "DataAnalysis",
        "qsubtype": "DataAnalysis_CausalAnalysis",
        "answer": "sales increased after the campaign",
        "prediction": "sales increased after campaign",
    }
    metric_name, score = module.evaluate_tablebench_record(rouge_record)
    assert metric_name == "ROUGE-L"
    assert 0.8 < score < 1.0


def test_metric_selection_supports_separate_qtype_and_plain_qsubtype():
    module = load_module()

    anomaly_record = {
        "qtype": "DataAnalysis",
        "qsubtype": "AnomalyDetection",
        "answer": "no anomalies are detected",
        "prediction": "no anomalies detected",
    }
    metric_name, score = module.evaluate_tablebench_record(anomaly_record)
    assert metric_name == "ROUGE-L"
    assert 0.0 < score < 1.0

    correlation_record = {
        "qtype": "DataAnalysis",
        "qsubtype": "CorrelationAnalysis",
        "answer": "100",
        "prediction": "109.9",
    }
    assert module.evaluate_tablebench_record(correlation_record) == (
        "EM_with_error_10",
        1.0,
    )


def test_focus_only_is_removed_from_converter_api():
    module = load_module()

    assert "focus_only" not in inspect.signature(module.convert_file).parameters
    assert not hasattr(module, "FOCUS_QSUBTYPES")


def test_convert_file_writes_failure_cases(tmp_path):
    module = load_module()
    raw_path = tmp_path / "raw_react_results.jsonl"
    out_dir = tmp_path / "converted"
    raw_rows = [
        {
            "id": "ok",
            "model_name": "qwen7b",
            "qtype": "NumericalReasoning",
            "qsubtype": "NumericalReasoning_Counting",
            "question": "How many?",
            "gold_answer": "2",
            "pred_answer": "2",
            "csv_file": "ok.csv",
            "traj": [{"role": "assistant", "content": "<answer>2</answer>"}],
            "parse_success": True,
            "execution_error_count": 0,
        },
        {
            "id": "bad",
            "model_name": "qwen7b",
            "qtype": "NumericalReasoning",
            "qsubtype": "NumericalReasoning_Counting",
            "question": "How many?",
            "gold_answer": "2",
            "pred_answer": "3",
            "csv_file": "bad.csv",
            "traj": [{"role": "assistant", "content": "<answer>3</answer>"}],
            "parse_success": True,
            "execution_error_count": 1,
        },
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8")

    module.convert_file(raw_path, out_dir)

    failures = list(csv.DictReader((out_dir / "failure_cases.csv").open(encoding="utf-8")))
    assert len(failures) == 1
    assert failures[0]["id"] == "bad"
    assert failures[0]["metric_name"] == "EM"
    assert failures[0]["manual_failure_type"] == ""

    subtype_metrics = json.loads(
        (out_dir / "subtype_metrics.json").read_text(encoding="utf-8")
    )
    assert subtype_metrics["NumericalReasoning_Counting"] == {
        "count": 2,
        "score": 0.5,
        "llm_judge_count": 0,
        "llm_judge_score": None,
    }
