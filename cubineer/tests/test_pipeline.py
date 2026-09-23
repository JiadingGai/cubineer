import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

import pytest

from scripted_endpoint import Endpoint

ROOT = Path(__file__).resolve().parents[2]
BINARY = os.environ.get("CUBINEER_TEST_BINARY")
MAIN_BINARY = os.environ.get("CUBINEER_TEST_MAIN_BINARY")
pytestmark = pytest.mark.skipif(
    not BINARY, reason="set CUBINEER_TEST_BINARY to a built codex-kernel executable"
)


@pytest.mark.parametrize(
    "strategy,mode,dataset",
    [
        ("mcts", "proactive", "kernelbench"),
        ("mcts", "reactive", "kernelbench_veomni"),
        ("greedy", "proactive", "kernelbench"),
    ],
)
def test_real_codex_sessions_with_synthetic_gpu(
    tmp_path, hardware, strategy, mode, dataset
):
    home = tmp_path / "home"
    home.mkdir()
    if dataset == "kernelbench":
        dataset_root = tmp_path / "dataset"
        (dataset_root / "level1").mkdir(parents=True)
        task = "1_Vector_add"
        (dataset_root / "level1" / f"{task}.py").write_text(
            "import torch\nclass Model(torch.nn.Module):\n    def forward(self, a, b):\n        return a + b\n"
            "def get_inputs():\n    return [torch.randn(1024), torch.randn(1024)]\ndef get_init_inputs():\n    return []\n"
        )
    else:
        dataset_root = ROOT / "cubineer/benchmarks/veomni"
        task = "103_VeOmni_MoE_dW1_GroupGEMM_mn_triton"
    scenario = tmp_path / "scenario.json"
    failed = {
        **hardware,
        "candidate_seconds": None,
        "error_context": {
            "error_type": "execution_error",
            "error_message": "synthetic compilation failure",
        },
    }
    scenario.write_text(
        json.dumps(
            {
                "reference": hardware,
                "batches": [
                    [failed, failed],
                    [hardware, hardware],
                    [hardware, hardware],
                ],
            }
        )
    )
    output = tmp_path / "run"
    with Endpoint() as endpoint:
        (home / "config.toml").write_text(
            'approval_policy = "never"\nsandbox_mode = "workspace-write"\n'
            '[model_providers.scripted]\nname = "Scripted Responses"\n'
            f'base_url = "{endpoint.url}"\nwire_api = "responses"\nrequires_openai_auth = false\n'
        )
        command = [
            BINARY,
            "optimize",
            "--provider",
            "scripted",
            "--model",
            "gpt-5.5",
            "--dataset",
            dataset,
            "--dataset-root",
            str(dataset_root),
            "--task",
            task,
            "--strategy",
            strategy,
            "--iterations",
            "3",
            "--candidates",
            "2",
            "--evaluator",
            "simulated",
            "--scenario",
            str(scenario),
            "--output",
            str(output),
            "--profiling",
            mode,
            "--ncu-full",
            "--timeout-seconds",
            "120",
        ]
        result = subprocess.run(
            command,
            env={**os.environ, "CODEX_HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        (tmp_path / "process.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert endpoint.errors == []
        assert {request["model"] for request in endpoint.requests} == {"gpt-5.5"}
        request_count = len(endpoint.requests)
        candidate_requests = [
            request
            for request in endpoint.requests
            if "code" in request["text"]["format"]["schema"]["properties"]
        ]
        assert len(candidate_requests) == 6
        for request in candidate_requests:
            assert not any(item.get("role") == "assistant" for item in request["input"])
    tree = json.loads((output / "tree.json").read_text())
    assert tree["current_iteration"] == 3
    assert not (output / "requests").exists()
    assert len(tree["nodes"]) == 7
    assert tree["nodes"][1]["outcome"]["status"] == "failed"
    assert tree["nodes"][3]["outcome"]["speedup"] == 2
    assert json.loads((output / "memory.json").read_text())["version"] == 3
    assert (
        json.loads((output / "classification.json").read_text())["analysis"][
            "final_bottleneck"
        ]
        == "memory_bound"
    )
    evidence = [
        json.loads(path.read_text())
        for path in (output / "candidates").glob("*/evidence.json")
    ]
    assert len({item["session"]["thread_id"] for item in evidence}) == 6
    assert all(item["evaluation"]["validation"] == "simulated" for item in evidence)
    usage = json.loads((output / "completion.json").read_text())["usage"]
    assert (
        sum(item["total"]["totalTokens"] for item in usage.values())
        == request_count * 125
    )


def configured_case(tmp_path, hardware, endpoint, provider="vllm", extra=()):
    home = tmp_path / "home"
    home.mkdir()
    data = tmp_path / "dataset" / "level1"
    data.mkdir(parents=True)
    (data / "1_Add.py").write_text(
        "class Model:\n    pass\ndef get_inputs():\n    return []\ndef get_init_inputs():\n    return []\n"
    )
    config = 'approval_policy = "never"\nsandbox_mode = "workspace-write"\n'
    env = {**os.environ, "CODEX_HOME": str(home), "AWS_EC2_METADATA_DISABLED": "true"}
    model = "gpt-5.6-sol"
    if provider == "openai":
        config += f'openai_base_url = "{endpoint.url}"\n'
        env["CODEX_API_KEY"] = "sk-local-fixture-not-a-real-key"
    elif provider == "amazon-bedrock":
        config += f'[model_providers.amazon-bedrock]\nbase_url = "{endpoint.url}"\n'
        config += '[model_providers.amazon-bedrock.aws]\nregion = "us-east-1"\n'
        env.update(
            AWS_ACCESS_KEY_ID="TESTACCESSKEY",
            AWS_SECRET_ACCESS_KEY="test-secret-not-real",
            AWS_REGION="us-east-1",
        )
        model = "openai.gpt-5.6-sol"
    else:
        config += f'[model_providers.vllm]\nname = "Local fixture"\nbase_url = "{endpoint.url}"\nwire_api = "responses"\nrequest_max_retries = 0\nstream_max_retries = 0\n'
        model = "Qwen/Qwen3.5-2B"
    (home / "config.toml").write_text(config)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps({"reference": hardware, "batches": [[hardware] for _ in range(4)]})
    )
    output = tmp_path / "run"
    command = [
        BINARY,
        "optimize",
        "--provider",
        provider,
        "--model",
        model,
        "--python",
        sys.executable,
        "--worker",
        str(ROOT / "cubineer/python/worker.py"),
        "--dataset-root",
        str(data.parent),
        "--task",
        "1_Add",
        "--strategy",
        "mcts",
        "--iterations",
        "1",
        "--candidates",
        "1",
        "--evaluator",
        "simulated",
        "--scenario",
        str(scenario),
        "--output",
        str(output),
        *extra,
    ]
    return command, env, output


@pytest.mark.parametrize("provider", ["openai", "vllm", "amazon-bedrock"])
def test_native_provider_transport_and_authentication(tmp_path, hardware, provider):
    with Endpoint() as endpoint:
        command, env, output = configured_case(tmp_path, hardware, endpoint, provider)
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert endpoint.errors == []
        headers = [
            {key.lower(): value for key, value in item.items()}
            for item in endpoint.headers
        ]
        if provider == "openai":
            assert all(
                item["authorization"] == "Bearer sk-local-fixture-not-a-real-key"
                for item in headers
            )
        elif provider == "amazon-bedrock":
            assert all(
                item["authorization"].startswith("AWS4-HMAC-SHA256 ")
                for item in headers
            )
        run = json.loads((output / "run.json").read_text())
        assert run["provider"] == provider
        assert {request["model"] for request in endpoint.requests} == {run["model"]}


@pytest.mark.parametrize("failure", ["access", "timeout"])
def test_explicit_model_failures_stop_without_a_fake_winner(
    tmp_path, hardware, failure
):
    with Endpoint() as endpoint:
        if failure == "access":
            endpoint.failure_status = 403
        elif failure == "timeout":
            endpoint.delay = 3
        extra = (
            ["--timeout-seconds", "1"]
            if failure == "timeout"
            else []
        )
        command, env, output = configured_case(
            tmp_path, hardware, endpoint, extra=extra
        )
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode != 0
        completion = json.loads((output / "completion.json").read_text())
        assert completion["status"] == "failed"
        assert not (output / "winner.json").exists()


def test_candidate_failure_keeps_successful_sibling(tmp_path, hardware):
    with Endpoint() as endpoint:
        endpoint.candidate_failure = 1
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        command[command.index("--candidates") + 1] = "2"
        scenario = Path(command[command.index("--scenario") + 1])
        fixture = json.loads(scenario.read_text())
        fixture["batches"][0] = [hardware]
        scenario.write_text(json.dumps(fixture))
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert endpoint.errors == []

    tree = json.loads((output / "tree.json").read_text())
    assert len(tree["nodes"]) == 2
    assert tree["nodes"][1]["outcome"]["status"] == "success"
    errors = list((output / "generation-errors").glob("*.json"))
    assert len(errors) == 1
    assert json.loads(errors[0].read_text())["error"]
    assert json.loads((output / "winner.json").read_text())["node"] == 1


def test_candidate_timeout_keeps_completed_sibling(tmp_path, hardware):
    with Endpoint() as endpoint:
        endpoint.candidate_delay = 1
        endpoint.candidate_delay_seconds = 3
        command, env, output = configured_case(
            tmp_path, hardware, endpoint, extra=["--timeout-seconds", "1"]
        )
        command[command.index("--candidates") + 1] = "2"
        scenario = Path(command[command.index("--scenario") + 1])
        fixture = json.loads(scenario.read_text())
        fixture["batches"][0] = [hardware]
        scenario.write_text(json.dumps(fixture))
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr

    tree = json.loads((output / "tree.json").read_text())
    assert len(tree["nodes"]) == 2
    assert tree["nodes"][1]["outcome"]["status"] == "success"
    errors = list((output / "generation-errors").glob("*.json"))
    assert len(errors) == 1
    assert "wall-clock budget" in json.loads(errors[0].read_text())["error"]


def test_candidate_evaluation_error_keeps_search_batch(tmp_path, hardware):
    invalid = {**hardware, "candidate_seconds": 0}
    with Endpoint() as endpoint:
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        command[command.index("--candidates") + 1] = "2"
        scenario = Path(command[command.index("--scenario") + 1])
        scenario.write_text(
            json.dumps({"reference": hardware, "batches": [[invalid, hardware]]})
        )
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert endpoint.errors == []

    tree = json.loads((output / "tree.json").read_text())
    assert [node["outcome"]["status"] for node in tree["nodes"][1:]] == [
        "failed",
        "success",
    ]
    assert tree["nodes"][1]["outcome"]["error_context"]["error_type"] == "WorkerCrash"
    assert (output / "candidates/1/evaluation-error.json").exists()
    assert json.loads((output / "winner.json").read_text())["node"] == 2


def test_success_without_profile_uses_fresh_parent_fallback(tmp_path, hardware):
    without_profile = {**hardware, "kernel_names": ["not_in_report"]}
    with Endpoint() as endpoint:
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        command[command.index("--iterations") + 1] = "2"
        scenario = Path(command[command.index("--scenario") + 1])
        scenario.write_text(
            json.dumps(
                {
                    "reference": hardware,
                    "batches": [[without_profile], [hardware]],
                }
            )
        )
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert endpoint.errors == []

    initial = (
        output / "workspaces/candidate-0-0/domain-instructions.md"
    ).read_text()
    fallback = (
        output / "workspaces/candidate-1-0/domain-instructions.md"
    ).read_text()
    context = (output / "workspaces/candidate-1-0/task-context.txt").read_text()
    assert fallback == initial
    assert "needs a fresh approach" in context
    tree = json.loads((output / "tree.json").read_text())
    assert tree["nodes"][2]["parent"] == 1


def test_memory_failure_keeps_previous_snapshot(tmp_path, hardware):
    with Endpoint() as endpoint:
        endpoint.memory_failure = 2
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        command[command.index("--iterations") + 1] = "3"
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert (output / "memory-error-1.json").exists()
        third = json.loads((output / "candidates/3/evidence.json").read_text())
        assert third["memory_version"] == 1
        assert json.loads((output / "memory.json").read_text())["version"] == 2


def test_interrupt_records_failure_and_shuts_down(tmp_path, hardware):
    with Endpoint() as endpoint:
        endpoint.delay = 5
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as process:
            deadline = time.monotonic() + 30
            while (
                not endpoint.requests
                and time.monotonic() < deadline
                and process.poll() is None
            ):
                time.sleep(0.05)
            assert endpoint.requests
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=15)
            assert process.returncode != 0, stderr
        assert (
            "cancelled" in json.loads((output / "completion.json").read_text())["error"]
        )


def test_role_overrides_and_memory_off(tmp_path, hardware):
    with Endpoint() as endpoint:
        command, env, output = configured_case(
            tmp_path,
            hardware,
            endpoint,
            extra=[
                "--reference-model",
                "gpt-5.6-sol",
                "--profile-model",
                "gpt-6-astra",
                "--memory-off",
            ],
        )
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert {request["model"] for request in endpoint.requests} == {
            "Qwen/Qwen3.5-2B",
            "gpt-5.6-sol",
            "gpt-6-astra",
        }
        assert endpoint.memory_calls == 0
        assert not (output / "memory.json").exists()
        evidence = json.loads((output / "candidates/1/evidence.json").read_text())
        assert evidence["evaluation"]["profiling_options"] == {
            "tool_mode": "proactive",
            "ncu_full": False,
            "nsys": False,
        }


@pytest.mark.skipif(
    not MAIN_BINARY, reason="set CUBINEER_TEST_MAIN_BINARY to built codex"
)
def test_main_cli_kernel_entrypoint(tmp_path, hardware):
    with Endpoint() as endpoint:
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        command[:1] = [MAIN_BINARY, "kernel"]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert (
            json.loads((output / "completion.json").read_text())["status"]
            == "completed"
        )


@pytest.mark.skipif(
    sys.platform == "win32", reason="native sandbox probe validated on macOS/Linux"
)
def test_native_tools_cannot_write_outside_candidate_workspace(tmp_path, hardware):
    with Endpoint() as endpoint:
        endpoint.workspace_probe = True
        command, env, output = configured_case(
            tmp_path, hardware, endpoint, provider="openai"
        )
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert endpoint.errors == []
        assert result.returncode == 0, result.stderr
        assert (
            output / "workspaces/candidate-0-0/local-proof.txt"
        ).read_text() == "inside"
        assert not (output / "workspaces/escape-proof.txt").exists()


def test_invalid_worker_protocol_fails_without_scoring(tmp_path, hardware):
    with Endpoint() as endpoint:
        command, env, output = configured_case(tmp_path, hardware, endpoint)
        invalid = tmp_path / "invalid_worker.py"
        invalid.write_text("print('not a protocol response')\n")
        command[command.index("--worker") + 1] = str(invalid)
        command[command.index("--evaluator") + 1] = "gpu"
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode != 0
        assert (
            "worker response"
            in json.loads((output / "completion.json").read_text())["error"]
        )
        assert not (output / "tree.json").exists()
