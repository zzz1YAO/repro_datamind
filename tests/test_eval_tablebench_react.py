import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "DataMind" / "datamind" / "eval" / "python" / "eval_tablebench_react.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_tablebench_react", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_datamind_prompt_uses_lowercase_react_protocol():
    module = load_module()

    messages = module.build_datamind_messages(
        question="What is the total sales?",
        csv_file="case_001.csv",
    )

    assert messages[0]["role"] == "system"
    assert "<code>" in messages[0]["content"]
    assert "</code>" in messages[0]["content"]
    assert "<answer>" in messages[0]["content"]
    assert "</answer>" in messages[0]["content"]
    assert "Final Answer:" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "data/files/case_001.csv" in messages[1]["content"]
    assert "What is the total sales?" in messages[1]["content"]


def test_postprocess_response_keeps_first_complete_datamind_action():
    module = load_module()

    raw = "thinking\n<code>\nprint(1)\n</code>\nextra text <answer>wrong</answer>"
    clipped = module.postprocess_response(raw)
    action = module.parse_model_action(clipped)

    assert clipped == "<code>\nprint(1)\n</code>"
    assert action.kind == "code"
    assert action.content == "print(1)"


def test_make_result_preserves_tablebench_metadata_and_execution_counts():
    module = load_module()
    sample = {
        "id": "tb_1",
        "qtype": "NumericalReasoning",
        "qsubtype": "NumericalReasoning_Counting",
        "question": "How many rows?",
        "gold_answer": "5",
        "csv_file": "tb_1.csv",
    }

    result = module.make_result_record(
        sample=sample,
        model_name="qwen7b",
        pred_answer="5",
        trajectory=[{"role": "assistant", "content": "<answer>5</answer>"}],
        parse_success=True,
        execution_error_count=2,
    )

    assert result["id"] == "tb_1"
    assert result["model_name"] == "qwen7b"
    assert result["gold_answer"] == "5"
    assert result["pred_answer"] == "5"
    assert result["csv_file"] == "tb_1.csv"
    assert result["parse_success"] is True
    assert result["execution_error_count"] == 2
