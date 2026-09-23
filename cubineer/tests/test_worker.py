import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

WORKER = Path(__file__).resolve().parents[1] / "python/worker.py"


@pytest.mark.parametrize("version,operation", [(5, "metadata"), (1, "invalid")])
def test_protocol_rejects_invalid_request(tmp_path, version, operation):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"version": version, "id": "test", "operation": operation, "payload": {}}
        )
    )
    result = subprocess.run(
        [sys.executable, str(WORKER), str(request)],
        capture_output=True,
        text=True,
        check=True,
    )
    response = json.loads(result.stdout)
    assert response["id"] == "test"
    assert response["error"]["type"] == "ValueError"


def test_gpu_workload_never_falls_back_to_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("requires a CPU-only host")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "cpu",
                "operation": "workload",
                "payload": {"output_dir": str(tmp_path)},
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(WORKER), str(request)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "version": 1,
        "id": "cpu",
        "error": {
            "type": "RuntimeError",
            "message": "GPU evaluation requires CUDA; no CPU fallback",
        },
    }


def test_metadata_uses_selected_gpu(monkeypatch):
    monkeypatch.syspath_prepend(str(WORKER.parent))
    import worker

    capability = Mock(
        side_effect=lambda: (
            (9, 0) if os.environ["CUDA_VISIBLE_DEVICES"] == "5" else (8, 0)
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=capability,
                get_device_name=lambda: "synthetic GPU",
            ),
            __version__="synthetic",
            version=SimpleNamespace(cuda="synthetic"),
        ),
    )
    with patch.dict(os.environ):
        result = worker.dispatch(
            {"version": 1, "operation": "metadata", "payload": {"gpu_id": 5}}
        )
    assert result["capability"] == [9, 0]
    capability.assert_called_once_with()
