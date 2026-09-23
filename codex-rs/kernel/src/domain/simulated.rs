use super::profilers;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;

pub(crate) async fn validate_task(problem: &Value) -> Result<()> {
    if let Some(expected) = problem.get("file_sha256") {
        let bytes =
            tokio::fs::read(problem["file_path"].as_str().context("reference file")?).await?;
        ensure!(
            *expected == format!("{:x}", Sha256::digest(bytes)),
            "reference file changed after task preparation"
        );
    }
    Ok(())
}

async fn report(payload: &Value, output: &Path) -> Result<Value> {
    let fixture = &payload["fixture"];
    let csv = fixture["ncu_csv"].as_str().context("scenario NCU CSV")?;
    let csv_path = output.join("profile.csv");
    tokio::fs::write(&csv_path, csv).await?;
    let names: Vec<String> =
        serde_json::from_value(fixture.get("kernel_names").cloned().unwrap_or(json!([])))?;
    let metrics = profilers::ncu_csv(csv, &names)?;
    let names: Vec<_> = metrics
        .as_object()
        .context("metrics")?
        .keys()
        .cloned()
        .collect();
    let sass_text = fixture["sass"].as_str().unwrap_or_default();
    tokio::fs::write(output.join("kernel.sass"), sass_text).await?;
    let sass = if sass_text.is_empty() {
        Value::Null
    } else {
        profilers::sass(sass_text, /*kernel*/ None)?
    };
    let rules = if payload["ncu_full"] == true {
        profilers::rules(fixture["ncu_rules"].as_str().unwrap_or_default())?
    } else {
        json!([])
    };
    Ok(
        json!({"success": !names.is_empty(), "metrics": metrics, "kernel_names": names, "sass": sass, "rule_recommendations": rules, "csv_path": csv_path, "error": if names.is_empty() { Some("No NCU metrics") } else { None }}),
    )
}

pub(crate) async fn reference(payload: &Value) -> Result<Value> {
    let output = Path::new(payload["output_dir"].as_str().context("reference output")?);
    tokio::fs::create_dir_all(output).await?;
    let output = output.canonicalize()?;
    let problem = &payload["problem"];
    validate_task(problem).await?;
    let seconds = payload["fixture"]["reference_seconds"]
        .as_f64()
        .context("reference timing must be finite and positive")?;
    let report = report(payload, &output).await?;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "reference timing must be finite and positive"
    );
    let name = report["kernel_names"][0].as_str().unwrap_or_default();
    let metrics = &report["metrics"][name];
    let value = |key: &str| metrics.get(key).cloned().unwrap_or(json!(0));
    let profile = json!({"success": report["success"], "metrics": report["metrics"], "kernel_name": name,
        "sm_throughput": value("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_throughput": value("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "latency_ms": value("gpu__time_duration.sum").as_f64().context("NCU duration")? / 1_000_000.0,
        "occupancy": value("sm__warps_active.avg.pct_of_peak_sustained_active")});
    validate_task(problem).await?;
    Ok(
        json!({"reference_seconds": seconds, "profile": profile, "task_sha256": problem["sha256"], "validation": "simulated"}),
    )
}

pub(crate) async fn evaluate(payload: &Value) -> Result<Value> {
    let output = Path::new(payload["output_dir"].as_str().context("candidate output")?);
    tokio::fs::create_dir_all(output).await?;
    let output = output.canonicalize()?;
    let source = Path::new(payload["source"].as_str().context("candidate source")?);
    let digest = format!("{:x}", Sha256::digest(tokio::fs::read(source).await?));
    ensure!(
        payload["source_sha256"] == digest,
        "candidate source identity mismatch"
    );
    let problem = &payload["problem"];
    validate_task(problem).await?;
    ensure!(
        payload["reference"]["task_sha256"] == problem["sha256"],
        "reference task identity mismatch"
    );
    let error = &payload["fixture"]["error_context"];
    let failed = error.as_object().is_some_and(|error| !error.is_empty());
    let seconds = payload["fixture"]["candidate_seconds"].as_f64();
    let report = if failed {
        Value::Null
    } else {
        report(payload, &output).await?
    };
    ensure!(
        digest == format!("{:x}", Sha256::digest(tokio::fs::read(source).await?)),
        "candidate source changed during evaluation"
    );
    validate_task(problem).await?;
    if !payload["fixture"]["candidate_seconds"].is_null() {
        ensure!(
            seconds.is_some_and(|seconds| seconds.is_finite() && seconds > 0.0),
            "candidate timing must be finite and positive"
        );
    }
    ensure!(
        failed || seconds.is_some(),
        "successful candidate has no timing"
    );
    let speedup = seconds.map(|seconds| {
        payload["reference"]["reference_seconds"]
            .as_f64()
            .unwrap_or_default()
            / seconds
    });
    ensure!(speedup.is_none_or(f64::is_finite), "speedup must be finite");
    Ok(
        json!({"outcome": {"status": if failed { "failed" } else { "success" }, "speedup": speedup, "error_context": error},
        "candidate_seconds": seconds, "source_sha256": digest, "task_sha256": problem["sha256"],
        "build_identity": null, "profile": report, "validation": "simulated", "profiling_options": {
            "tool_mode": payload["tool_mode"].as_str().unwrap_or("proactive"), "ncu_full": payload["ncu_full"].as_bool().unwrap_or_default(), "nsys": false}}),
    )
}

#[cfg(test)]
#[path = "simulated_tests.rs"]
mod tests;
