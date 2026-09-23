from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cubineer/python"))
from workload import Workload, compare_outputs


@pytest.mark.parametrize(
    "ref,opt,correct,message",
    [
        ([1.0, 2.0], [1.0, 2.0], True, None),
        ([1.0, 2.0], [1.0, 4.0], False, "Numerical error"),
        ([1.0, 2.0], [1.0], False, "Shape mismatch"),
        ([float("nan")], [0.0], False, "Numerical error"),
    ],
)
def test_tensor_comparison(ref, opt, correct, message):
    ref, opt = torch.tensor(ref), torch.tensor(opt)
    actual = compare_outputs(torch, ref, opt, 0.01, 0.1)
    assert actual["correct"] is correct
    if message:
        assert message in actual["message"]
        if message == "Numerical error":
            assert actual["details"] is not None
    else:
        assert actual["message"] is None
        assert actual["details"] is None


@pytest.mark.parametrize(
    "expression", ["x + 1", "x + 2", "(x + 1, x * 2)", "x[:1]", "(x + 1, 2)"]
)
def test_workload_loading_seed_and_validation(tmp_path, expression):
    inputs = "import torch\ndef get_inputs(): return [torch.randn(4)]\ndef get_init_inputs(): return []\n"
    reference_code = (
        "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x + 1\n"
        + inputs
    )
    source = tmp_path / "reference.py"
    source.write_text(reference_code)
    candidate = tmp_path / "solution.py"
    candidate.write_text(
        f"import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self, x): return {expression}\n"
    )
    settings = {"seed": 42, "rtol": 0.001, "atol": 0.01}
    new = Workload(
        torch,
        {
            "canonical_solution": reference_code,
            "input_generator": inputs,
            "file_path": str(source),
        },
        settings,
    )
    result = new.compare(str(candidate), "cpu")
    assert result["correct"] is (expression == "x + 1")
    assert (result["message"] is None) is result["correct"]


def test_stateless_function_workloads(tmp_path):
    source = tmp_path / "reference.py"
    source.write_text("def run(x): return x + 1\n")
    candidate = tmp_path / "solution.py"
    candidate.write_text(
        "def run(x): return x + 2\ndef run_optimized(x): return x + 1\n"
    )
    inputs = "import torch\ndef get_inputs(): return [torch.ones(4)]\ndef get_init_inputs(): return []\n"
    new = Workload(
        torch,
        {
            "canonical_solution": source.read_text(),
            "input_generator": inputs,
            "file_path": str(source),
        },
        {"seed": 42, "rtol": 0.001, "atol": 0.01},
    )
    assert new.compare(str(candidate), "cpu") == {
        "correct": True,
        "message": None,
        "details": None,
    }
