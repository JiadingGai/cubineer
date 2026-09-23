"""Synthetic command-line GPU observations, never a CUDA correctness test."""

import json
import os
from pathlib import Path
import sys
import time


def main():
    fixture = json.loads(Path(os.environ["CUBINEER_FAKE_CUDA"]).read_text())
    tool = Path(sys.argv[0]).name
    runtime = {
        "name": "NVIDIA H200 (synthetic)",
        "capability": [9, 0],
        "uid": 0,
        "torch_version": "synthetic",
        "cuda_version": "synthetic",
        "cuda_arch_list": "9.0",
    }
    if tool == "worker.py":
        request = json.loads(Path(sys.argv[1]).read_text())
        assert request["payload"]["gpu_id"] == fixture["gpu_id"]
        if request["operation"] == "metadata":
            result = runtime
        else:
            assert request["operation"] == "workload"
            payload = request["payload"]
            candidate = payload["operation"] == "candidate"
            assert payload["settings"] == {
                "seed": 42,
                "rtol": 0.001,
                "atol": 0.01,
                "warmup": 5,
                "trials": 5,
                "set_arch": candidate,
                "time_numerical_mismatch": True,
            }
            if candidate:
                if fixture.get("mode") == "workload_timeout":
                    time.sleep(2)
                build = Path(payload["output_dir"]) / "build" / "extension"
                build.mkdir(parents=True)
                if fixture.get("mode") != "library":
                    (build / "kernel.so").write_bytes(b"synthetic binary")
                (build / "build.ninja").write_text("synthetic recipe")
            result = {
                "runtime": runtime,
                "seconds": fixture[
                    "candidate_seconds" if candidate else "reference_seconds"
                ],
                "exception": None,
                "comparison": fixture.get(
                    "comparison", {"correct": True, "message": None, "details": None}
                )
                if candidate
                else None,
            }
        print(json.dumps({"version": 1, "id": request["id"], "result": result}))
        return
    with Path("commands.jsonl").open("a") as log:
        log.write(json.dumps({"tool": tool, "args": sys.argv[1:]}) + "\n")
    if tool == "ncu":
        if fixture.get("ncu_failure"):
            print("synthetic NCU permission error", file=sys.stderr)
            sys.exit(1)
        if "--csv" in sys.argv:
            path = next(
                arg.split("=", 1)[1]
                for arg in sys.argv
                if arg.startswith("--log-file=")
            )
            Path(path).write_text(fixture["ncu_csv"])
        elif "--set" in sys.argv:
            if fixture.get("mode") == "rules_failure":
                sys.exit(1)
            Path(sys.argv[sys.argv.index("-o") + 1] + ".ncu-rep").touch()
        elif "--import" in sys.argv:
            print(fixture["ncu_rules"])
    elif tool == "cuobjdump":
        if fixture.get("mode") == "cublas_only":
            sys.exit(1)
        if "-symbols" in sys.argv:
            print("STT_FUNC STB_GLOBAL STO_ENTRY _Z10vector_addv")
        elif "--dump-sass" in sys.argv:
            print(fixture["sass"])
    elif tool == "c++filt":
        print("vector_add()")
    elif tool == "nm":
        print("U cublasGemmEx")
    else:
        raise AssertionError(tool)


if __name__ == "__main__":
    main()
