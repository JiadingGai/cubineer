"""PyTorch/Triton execution boundary; evaluation policy is supplied by Rust."""

import importlib.util
import os
from pathlib import Path
import traceback


def runtime_info(torch):
    available = torch.cuda.is_available()
    return {
        "name": torch.cuda.get_device_name() if available else None,
        "capability": list(torch.cuda.get_device_capability()) if available else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
    }


class FunctionModel:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)

    def to(self, device):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}


class Workload:
    def __init__(self, torch, problem, settings):
        self.torch = torch
        self.problem = problem
        self.settings = settings
        self.triton = False
        self.seed()
        try:
            self.reference = self.load(problem["file_path"], optimized=False)
        except ValueError as error:
            if "does not define 'Model'" not in str(
                error
            ) and "does not define 'run'" not in str(error):
                raise
            namespace = {"torch": torch, "nn": torch.nn}
            exec(problem["canonical_solution"], namespace)
            if "Model" not in namespace:
                raise ValueError(
                    "[ReferenceModel] Code does not define a 'Model' class"
                )
            exec(problem["input_generator"], namespace)
            self.reference = namespace["Model"](
                *namespace.get("get_init_inputs", lambda: [])()
            )

    def seed(self):
        self.torch.manual_seed(self.settings["seed"])
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed(self.settings["seed"])

    def load(self, filename, optimized):
        spec = importlib.util.spec_from_file_location("loaded_module", filename)
        if spec is None:
            raise ValueError(f"Cannot load module from {filename} (file may not exist)")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "Model") or hasattr(module, "ModelNew"):
            name = "ModelNew" if optimized else "Model"
            if not hasattr(module, name):
                if optimized and hasattr(module, "Model"):
                    name = "Model"
                else:
                    raise ValueError(f"File {filename} does not define '{name}' class")
            namespace = {"torch": self.torch, "nn": self.torch.nn, **module.__dict__}
            exec(self.problem["input_generator"], namespace)
            return getattr(module, name)(
                *namespace.get("get_init_inputs", lambda: [])()
            )
        if hasattr(module, "run"):
            self.triton = True
            return FunctionModel(
                module.run_optimized
                if optimized and hasattr(module, "run_optimized")
                else module.run
            )
        raise ValueError(
            f"File {filename} does not define 'Model' class or 'run' function. Expected either KernelBench Model class or Triton run() function."
        )

    def inputs(self, device):
        namespace = {"torch": self.torch}
        exec(self.problem["input_generator"], namespace)
        if "get_inputs" not in namespace:
            raise ValueError("Input generator code does not define 'get_inputs'")
        return [
            value.to(device) if self.torch.is_tensor(value) else value
            for value in namespace["get_inputs"]()
        ]

    def output(self, model, inputs, device):
        torch = self.torch
        self.seed()
        value = model(*inputs)
        if device == "cuda":
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if torch.is_tensor(item) and item.is_cuda:
                    item.sum()
                    break
            torch.cuda.synchronize()
        result = (
            tuple(item.cpu() if torch.is_tensor(item) else item for item in value)
            if isinstance(value, (tuple, list))
            else value.cpu()
        )
        del value
        if device == "cuda":
            torch.cuda.empty_cache()
        return result

    def compare(self, filename, device):
        torch = self.torch
        try:
            self.seed()
            candidate = self.load(filename, optimized=True).to(device).eval()
            self.reference = self.reference.to(device).eval()
            if not self.triton:
                ref_keys = set(self.reference.state_dict())
                opt_keys = set(candidate.state_dict())
                assert ref_keys == opt_keys, (
                    "Model architectures differ - cannot verify weight sync"
                )
                for key in ref_keys:
                    assert torch.equal(
                        self.reference.state_dict()[key], candidate.state_dict()[key]
                    ), (
                        f"Weight mismatch for '{key}' - seed={self.settings['seed']} failed"
                    )
            inputs = self.inputs(device)
            with torch.no_grad():
                reference = self.output(self.reference, inputs, device)
                optimized = self.output(candidate, inputs, device)
            del inputs
            if device == "cuda":
                torch.cuda.empty_cache()
            return compare_outputs(
                torch,
                reference,
                optimized,
                self.settings["rtol"],
                self.settings["atol"],
            )
        except Exception as error:
            return {
                "correct": False,
                "message": f"Execution error: {error}",
                "details": None,
            }

    def timing(self, model):
        torch = self.torch
        model = model.to("cuda").eval()
        inputs = self.inputs("cuda")
        for _ in range(self.settings["warmup"]):
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.settings["trials"]):
            with torch.no_grad():
                model(*inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / self.settings["trials"] / 1000.0


def compare_outputs(torch, reference, optimized, rtol, atol):
    refs = tuple(reference) if isinstance(reference, (tuple, list)) else (reference,)
    opts = tuple(optimized) if isinstance(optimized, (tuple, list)) else (optimized,)
    result = {"correct": False, "message": None, "details": None}
    if len(refs) != len(opts):
        result["message"] = (
            f"Output count mismatch: optimized returns {len(opts)} vs reference {len(refs)}"
        )
        return result
    for i, (ref, opt) in enumerate(zip(refs, opts)):
        if not torch.is_tensor(ref) or not torch.is_tensor(opt):
            continue
        prefix = f"[output {i}] " if len(refs) > 1 else ""
        if ref.shape != opt.shape:
            result["message"] = f"{prefix}Shape mismatch: {opt.shape} vs {ref.shape}"
            return result
        if not torch.allclose(ref, opt, rtol=rtol, atol=atol):
            diff = torch.abs(ref - opt)
            maximum = torch.max(diff).item()
            index = torch.argmax(diff)
            ref_value = ref.flatten()[index].item()
            opt_value = opt.flatten()[index].item()
            result["message"] = (
                f"{prefix}Numerical error: max diff = {maximum:.2e} (ref={ref_value:.6e}, opt={opt_value:.6e}) at index {index} (rtol = {rtol:.2e}, atol = {atol:.2e})"
            )
            result["details"] = {
                "max_diff": maximum,
                "max_idx": int(index),
                "ref_val": ref_value,
                "opt_val": opt_value,
            }
            return result
    return {"correct": True, "message": None, "details": None}


def execute(payload):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload.get("gpu_id", 0))
    output = Path(payload["output_dir"])
    os.environ["TORCH_EXTENSIONS_DIR"] = str(output / "build")
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requires CUDA; no CPU fallback")
    settings = payload["settings"]
    if settings["set_arch"] and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    result = {"seconds": None, "comparison": None, "exception": None}
    try:
        workload = Workload(torch, payload["problem"], settings)
        if payload["operation"] == "reference":
            result["seconds"] = workload.timing(workload.reference)
        else:
            comparison = workload.compare(payload["source"], "cuda")
            result["comparison"] = comparison
            if comparison["correct"] or (
                settings["time_numerical_mismatch"]
                and "Numerical error" in (comparison["message"] or "")
            ):
                try:
                    result["seconds"] = workload.timing(
                        workload.load(payload["source"], optimized=True)
                    )
                except Exception:
                    if comparison["correct"]:
                        raise
    except Exception as error:
        result["exception"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
    result["runtime"] = runtime_info(torch)
    return result
