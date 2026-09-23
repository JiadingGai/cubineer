"""Model-free workload protocol. Invoke with the configured venv interpreter."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


def dispatch(request):
    if request.get("version") != 1:
        raise ValueError("unsupported worker protocol version")
    if request["operation"] == "metadata":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(request["payload"].get("gpu_id", 0))
        import torch
        from workload import runtime_info

        return runtime_info(torch)
    if request["operation"] == "workload":
        from workload import execute

        return execute(request["payload"])
    raise ValueError("unsupported worker operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("worker requires a venv Python")
    request = json.loads(args.request.read_text())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(request)
        serialized = json.dumps(
            {"version": 1, "id": request["id"], "result": result}, allow_nan=False
        )
    except Exception as error:
        serialized = json.dumps(
            {
                "version": 1,
                "id": request.get("id"),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
            allow_nan=False,
        )
    print(serialized)


if __name__ == "__main__":
    main()
