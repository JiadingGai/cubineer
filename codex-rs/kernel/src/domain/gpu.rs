use super::gpu_binary;
use super::gpu_profile::Profiler;
use super::gpu_profile::ncu_binary;
use super::scripts;
use super::simulated::validate_task;
use crate::worker::Worker;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use codex_app_server_client::InProcessAppServerClient;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;

pub(crate) async fn evaluate(
    worker: &Worker,
    client: &InProcessAppServerClient,
    operation: &str,
    mut payload: Value,
) -> Result<Value> {
    let output = Path::new(payload["output_dir"].as_str().context("GPU output")?).to_path_buf();
    tokio::fs::create_dir_all(&output).await?;
    let problem = payload["problem"].clone();
    validate_task(&problem).await?;
    let candidate = operation == "evaluate";
    let digest = if candidate {
        let digest = format!(
            "{:x}",
            Sha256::digest(
                tokio::fs::read(payload["source"].as_str().context("candidate source")?).await?
            )
        );
        ensure!(
            payload["source_sha256"] == digest,
            "candidate source identity mismatch"
        );
        ensure!(
            payload["reference"]["task_sha256"] == problem["sha256"],
            "reference task identity mismatch"
        );
        Some(digest)
    } else {
        None
    };
    payload["operation"] = json!(if candidate { "candidate" } else { "reference" });
    payload["settings"] = json!({"seed": 42, "rtol": 1e-3, "atol": 1e-2, "warmup": 5, "trials": 5,
        "set_arch": candidate, "time_numerical_mismatch": true});
    let observation = worker.external(client, "workload", payload.clone()).await?;
    let runtime = &observation["runtime"];
    let mut env = json!({"CUDA_VISIBLE_DEVICES": payload["gpu_id"].as_u64().unwrap_or_default().to_string(),
        "TORCH_EXTENSIONS_DIR": output.join("build"), "TRITON_CACHE_DIR": output.join("triton")});
    if let Some(arch) = runtime["cuda_arch_list"].as_str() {
        env["TORCH_CUDA_ARCH_LIST"] = json!(arch);
    }
    let profiler = Profiler {
        worker,
        client,
        output: &output,
        env,
        runtime,
    };
    if !candidate {
        ensure!(
            observation["exception"].is_null(),
            "reference workload: {}",
            observation["exception"]
        );
        let seconds = positive_timing(&observation["seconds"], "reference")?
            .context("reference timing must be finite and positive")?;
        let profile = if ncu_binary().is_none() {
            json!({"error": "NCU not available on this system", "sm_throughput": 0.0, "dram_throughput": 0.0, "latency_ms": 0.0, "occupancy": 0.0})
        } else {
            let script = scripts::write("reference", &payload).await?;
            summarize_reference(&profiler.collect(&script, &[], /*full*/ false).await?)?
        };
        validate_task(&problem).await?;
        return Ok(
            json!({"reference_seconds": seconds, "profile": profile, "task_sha256": problem["sha256"], "validation": "gpu"}),
        );
    }
    let error = candidate_error(&observation)?;
    let failed = !error.is_null();
    let seconds = positive_timing(&observation["seconds"], "candidate")?;
    ensure!(
        failed || seconds.is_some(),
        "successful candidate has no timing"
    );
    let report = if failed {
        Value::Null
    } else {
        let _ = scripts::write("benchmark", &payload).await;
        let report: Result<Value> = async {
            let names = gpu_binary::kernel_names(&profiler)
                .await
                .unwrap_or_default();
            let script = scripts::write("candidate", &payload).await?;
            let mut report = profiler
                .collect(&script, &names, payload["ncu_full"] == true)
                .await?;
            report["sass"] = gpu_binary::sass(&profiler).await.unwrap_or(Value::Null);
            Ok(report)
        }
        .await;
        match report {
            Ok(report) => report,
            Err(error) => json!({"success": false, "error": error.to_string()}),
        }
    };
    let after = format!(
        "{:x}",
        Sha256::digest(
            tokio::fs::read(payload["source"].as_str().context("candidate source")?).await?
        )
    );
    ensure!(
        digest.as_deref() == Some(&after),
        "candidate source changed during evaluation"
    );
    validate_task(&problem).await?;
    let speedup = seconds
        .map(|seconds| {
            payload["reference"]["reference_seconds"]
                .as_f64()
                .context("reference timing")
                .map(|reference| reference / seconds)
        })
        .transpose()?;
    ensure!(speedup.is_none_or(f64::is_finite), "speedup must be finite");
    let mut paths = gpu_binary::artifacts(&output.join("build"));
    paths.retain(|path| {
        path.extension().is_some_and(|extension| extension == "so")
            || path.file_name().is_some_and(|name| name == "build.ninja")
    });
    paths.sort();
    let mut artifacts = Vec::new();
    for path in paths {
        artifacts.push(json!({"path": path.strip_prefix(&output)?, "sha256": format!("{:x}", Sha256::digest(tokio::fs::read(&path).await?))}));
    }
    Ok(
        json!({"outcome": {"status": if failed { "failed" } else { "success" }, "speedup": speedup, "error_context": error},
        "candidate_seconds": seconds, "source_sha256": digest, "task_sha256": problem["sha256"], "profile": report,
        "build_identity": {"torch_version": runtime["torch_version"], "cuda_version": runtime["cuda_version"],
            "cuda_arch_list": runtime["cuda_arch_list"], "artifacts": artifacts}, "validation": "gpu",
        "profiling_options": {"tool_mode": payload["tool_mode"].as_str().unwrap_or("proactive"), "ncu_full": payload["ncu_full"] == true, "nsys": false}}),
    )
}

fn positive_timing(value: &Value, label: &str) -> Result<Option<f64>> {
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_f64()
        .context(format!("{label} timing must be finite and positive"))?;
    ensure!(
        value.is_finite() && value > 0.0,
        "{label} timing must be finite and positive"
    );
    Ok(Some(value))
}

fn candidate_error(observation: &Value) -> Result<Value> {
    if !observation["exception"].is_null() {
        return Ok(observation["exception"].clone());
    }
    let comparison = &observation["comparison"];
    if comparison["correct"] == true {
        return Ok(Value::Null);
    }
    let message = comparison["message"]
        .as_str()
        .context("workload comparison result")?;
    let mut error = json!({"error_type": if message.contains("Numerical error") { "correctness_error" } else { "execution_error" }, "error_message": message});
    if comparison["details"]
        .as_object()
        .is_some_and(|details| !details.is_empty())
    {
        error["max_diff"] = comparison["details"]
            .get("max_diff")
            .cloned()
            .unwrap_or(json!(0.0));
    }
    Ok(error)
}

fn summarize_reference(report: &Value) -> Result<Value> {
    if report["success"] != true {
        return Ok(json!({"error": report["error"]}));
    }
    let metrics = report["metrics"].as_object().context("NCU metrics")?;
    let Some((name, values)) = metrics.iter().next() else {
        return Ok(json!({"error": "No metrics collected by NCU"}));
    };
    let value = |key: &str| -> Result<f64> {
        let value = &values[key];
        if value.is_null() {
            return Ok(0.0);
        }
        value
            .as_f64()
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            .context("numeric reference metric")
    };
    Ok(
        json!({"success": true, "metrics": metrics, "kernel_name": name,
        "sm_throughput": value("sm__throughput.avg.pct_of_peak_sustained_elapsed")?,
        "dram_throughput": value("dram__throughput.avg.pct_of_peak_sustained_elapsed")?,
        "latency_ms": value("gpu__time_duration.sum")? / 1_000_000.0,
        "occupancy": value("sm__warps_active.avg.pct_of_peak_sustained_active")?}),
    )
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
