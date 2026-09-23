import json
import os
import subprocess
import sys

import pytest

from scripted_endpoint import Endpoint
from test_pipeline import BINARY, ROOT, configured_case

pytestmark = [
    pytest.mark.skipif(not BINARY, reason="set CUBINEER_TEST_BINARY"),
    pytest.mark.skipif(
        os.name != "posix", reason="synthetic POSIX toolkit executables"
    ),
]


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "numerical",
        "ncu_failure",
        "integer_metrics",
        "workload_timeout",
        "library",
        "rules_failure",
        "cublas_only",
    ],
)
def test_native_gpu_orchestration_with_synthetic_tools(tmp_path, hardware, mode):
    fixture = {
        **hardware,
        "ncu_failure": mode == "ncu_failure",
        "mode": mode,
        "gpu_id": 5,
    }
    if mode == "integer_metrics":
        fixture["ncu_csv"] = fixture["ncu_csv"].replace(".0", "")
    if mode == "numerical":
        fixture["comparison"] = {
            "correct": False,
            "message": "Numerical error: synthetic mismatch",
            "details": {"max_diff": 0.5},
        }
    fixture_path = tmp_path / "hardware.json"
    fixture_path.write_text(json.dumps(fixture))
    tools = tmp_path / "tools"
    tools.mkdir()
    source = (ROOT / "cubineer/tests/fake_cuda.py").read_text()
    for name in ["worker.py", "ncu", "cuobjdump", "nm", "c++filt"]:
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
    with Endpoint() as endpoint:
        command, env, output = configured_case(
            tmp_path, hardware, endpoint, extra=["--ncu-full", "--gpus", "5"]
        )
        command[command.index("--evaluator") + 1] = "gpu"
        command[command.index("--worker") + 1] = str(tools / "worker.py")
        if mode == "workload_timeout":
            command.extend(["--timeout-seconds", "1"])
        env.update(
            PATH=f"{tools}:{os.environ['PATH']}", CUBINEER_FAKE_CUDA=str(fixture_path)
        )
        process = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=180
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert endpoint.errors == []
    evidence = json.loads(
        next((output / "candidates").glob("*/evidence.json")).read_text()
    )
    evaluation = evidence["evaluation"]
    if mode == "workload_timeout":
        assert evaluation["outcome"]["status"] == "failed"
        assert evaluation["outcome"]["speedup"] is None
        assert evaluation["outcome"]["error_context"]["error_type"] == "TimeoutError"
        assert evaluation["profile"] is None
        return
    assert evaluation["outcome"]["status"] == (
        "failed" if mode == "numerical" else "success"
    )
    assert evaluation["outcome"]["speedup"] == 2
    assert len(evaluation["build_identity"]["artifacts"]) == (
        1 if mode == "library" else 2
    )
    assert evaluation["build_identity"]["torch_version"] == "synthetic"
    if mode == "numerical":
        assert evaluation["profile"] is None
        assert evaluation["outcome"]["error_context"]["max_diff"] == 0.5
    else:
        report = evaluation["profile"]
        profile_failed = mode == "ncu_failure"
        assert report["success"] != profile_failed
        if mode == "library":
            assert report["sass"] is None
        else:
            assert report["sass"]["uses_cublas"]
            assert report["sass"]["wgmma_count"] == 0
        if not profile_failed:
            assert report["library_fallback"] == (mode in {"library", "cublas_only"})
            assert [
                rule["est_speedup_pct"] for rule in report["rule_recommendations"]
            ] == ([] if mode == "rules_failure" else [70, 20])
            commands = [
                json.loads(line)
                for path in output.rglob("commands.jsonl")
                for line in path.read_text().splitlines()
            ]
            ncu = [command["args"] for command in commands if command["tool"] == "ncu"]
            assert sum("--set" in args for args in ncu) == 1
            assert any("--launch-count=5" in args for args in ncu)
            assert any("--kernel-name=regex:(vector_add)" in args for args in ncu) == (
                mode not in {"library", "cublas_only"}
            )
