"""Emit synthetic hardware feedback, independent of generated candidate code."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--candidates", type=int, default=2)
args = parser.parse_args()
report = {
    "reference_seconds": 0.002,
    "candidate_seconds": 0.001,
    "kernel_names": ["vector_add"],
    "ncu_csv": "Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum\n,%,%,ns\nvector_add,12.0,80.0,1000000.0\n",
    "sass": "Function : vector_add\n/*0000*/ FFMA R1, R2, R3, R4;\n",
    "ncu_rules": "OPT Est. Speedup: 20%\n  Improve coalescing.\n",
}
failed = {
    **report,
    "candidate_seconds": None,
    "error_context": {
        "error_type": "execution_error",
        "error_message": "Synthetic compilation failure",
    },
}
batches = [
    [failed if iteration == 0 else report for _ in range(args.candidates)]
    for iteration in range(max(1, args.iterations))
]
args.output.write_text(json.dumps({"reference": report, "batches": batches}, indent=2))
