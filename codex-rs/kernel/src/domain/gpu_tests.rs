use super::*;
use pretty_assertions::assert_eq;

#[cfg(unix)]
#[test]
fn build_artifacts_include_symlinked_files() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let binary = directory.path().join("original.so");
    let link = directory.path().join("linked.so");
    std::fs::write(&binary, b"binary")?;
    std::os::unix::fs::symlink(&binary, &link)?;
    let mut paths = gpu_binary::artifacts(directory.path());
    paths.sort();
    assert_eq!(paths, vec![link, binary]);
    Ok(())
}

#[test]
fn workload_observations_preserve_failure_classification() -> Result<()> {
    let observation = |correct, message: &str, details: Value| json!({"comparison": {"correct": correct, "message": message, "details": details}});
    assert_eq!(
        candidate_error(&observation(true, "", Value::Null))?,
        Value::Null
    );
    for (message, category) in [
        ("Execution error: launch failure", "execution_error"),
        ("Shape mismatch", "execution_error"),
        ("Numerical error: mismatch", "correctness_error"),
    ] {
        assert_eq!(
            candidate_error(&observation(false, message, json!({"max_diff": 2.0})))?,
            json!({"error_type": category, "error_message": message, "max_diff": 2.0})
        );
    }
    let exception = json!({"error_type": "RuntimeError", "error_message": "compile failed", "traceback": "trace"});
    assert_eq!(
        candidate_error(&json!({"exception": exception}))?,
        exception
    );
    assert!(candidate_error(&json!({})).is_err());
    for invalid in [json!(0), json!(-1), json!("nan")] {
        assert!(positive_timing(&invalid, "candidate").is_err());
    }
    assert_eq!(positive_timing(&Value::Null, "candidate")?, None);
    assert_eq!(positive_timing(&json!(0.25), "candidate")?, Some(0.25));
    Ok(())
}

#[test]
fn reference_summary_preserves_first_kernel_and_failure() -> Result<()> {
    assert_eq!(
        summarize_reference(&json!({"success": false, "error": "no ncu"}))?,
        json!({"error": "no ncu"})
    );
    assert_eq!(
        summarize_reference(&json!({"success": true, "metrics": {}}))?,
        json!({"error": "No metrics collected by NCU"})
    );
    let metrics = json!({"first": {"gpu__time_duration.sum": 2000000.0, "sm__throughput.avg.pct_of_peak_sustained_elapsed": "25.0"}, "second": {"gpu__time_duration.sum": 1000000.0}});
    assert_eq!(
        summarize_reference(&json!({"success": true, "metrics": metrics}))?,
        json!({"success": true, "metrics": metrics, "kernel_name": "first", "sm_throughput": 25.0, "dram_throughput": 0.0, "latency_ms": 2.0, "occupancy": 0.0})
    );
    Ok(())
}
