# Cubineer

## Install

From the repository root, install Rustup, Python 3.12, and a C/C++ toolchain.

```bash
python3.12 -m venv venv
venv/bin/python -m pip install -r cubineer/python/requirements.txt
cd codex-rs
cargo build --locked -p codex-cli -p codex-kernel -p codex-kernel-search
cd ..
codex-rs/target/debug/codex login
```

GPU evaluation also requires CUDA-compatible PyTorch, Triton, and Ninja in
`venv`, plus `nvcc`, `cuobjdump`, `nm`, and Nsight Compute on the host.

## Run VeOmni

Set `CUTLASS_ROOT` to a CUTLASS checkout. This example uses simulated profiling;
replace `EVALUATOR` with the commented GPU assignment on a Hopper GPU.

```bash
venv/bin/python cubineer/examples/make_scenario.py \
  /tmp/veomni-scenario.json --iterations 2 --candidates 1
EVALUATOR=(--evaluator simulated --scenario /tmp/veomni-scenario.json)
# EVALUATOR=(--evaluator gpu --gpus 0 --python "$PWD/venv/bin/python" \
#   --worker "$PWD/cubineer/python/worker.py")

RUN="$PWD/runs/veomni-$(date +%Y%m%d-%H%M%S)"
codex-rs/target/debug/codex kernel optimize \
  --provider openai --model gpt-6-astra \
  --dataset kernelbench_veomni \
  --dataset-root "$PWD/cubineer/benchmarks/veomni" \
  --task 103_VeOmni_MoE_dW1_GroupGEMM_mn_triton \
  --cutlass-root "${CUTLASS_ROOT:?Set CUTLASS_ROOT}" \
  --strategy mcts --iterations 2 --candidates 1 --parallel-sessions 1 \
  --profiling proactive --ncu-full \
  "${EVALUATOR[@]}" --output "$RUN"
```

Tasks `101`, `102`, and `104` are in the same directory. Results are written to
`$RUN`, including `winner.json`, `tree.json`, `memory.json`, and `candidates/`.

## Run KernelBench L3 Or Your Kernel

Point `KERNELBENCH_ROOT` at the directory containing `level3/` and set `TASK` to
a Python filename without `.py`. A custom task uses the same layout and must
define a reference `Model`, `get_inputs()`, and `get_init_inputs()`.

```bash
KERNELBENCH_ROOT=/path/to/KernelBench/KernelBench
TASK=your_level3_task
venv/bin/python cubineer/examples/make_scenario.py \
  /tmp/l3-scenario.json --iterations 10 --candidates 3
EVALUATOR=(--evaluator simulated --scenario /tmp/l3-scenario.json)
# EVALUATOR=(--evaluator gpu --gpus 0 --python "$PWD/venv/bin/python" \
#   --worker "$PWD/cubineer/python/worker.py")

RUN="$PWD/runs/l3-$(date +%Y%m%d-%H%M%S)"
codex-rs/target/debug/codex kernel optimize \
  --provider openai --model gpt-6-astra \
  --dataset kernelbench --dataset-root "$KERNELBENCH_ROOT" \
  --level 3 --task "$TASK" \
  --cutlass-root "${CUTLASS_ROOT:?Set CUTLASS_ROOT}" \
  --strategy mcts --iterations 10 --candidates 3 --parallel-sessions 3 \
  --profiling proactive --ncu-full \
  "${EVALUATOR[@]}" --output "$RUN"
```
