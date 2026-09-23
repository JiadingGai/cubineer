use super::*;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn simulation_checks_identity_and_never_executes_candidate() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("solution.py");
    let code = "raise RuntimeError('must not execute')";
    tokio::fs::write(&source, code).await?;
    let hash = format!("{:x}", Sha256::digest(code.as_bytes()));
    let mut payload = json!({"output_dir": directory.path(), "source": source, "source_sha256": hash, "problem": {"sha256": "task"}, "reference": {"task_sha256": "task", "reference_seconds": 0.002}, "fixture": {"candidate_seconds": 0.001, "ncu_csv": "Kernel Name,gpu__time_duration.sum\n,ns\nprobe,1000.0\n", "sass": "HGMMA FFMA"}});
    let result = evaluate(&payload).await?;
    assert_eq!(
        result["outcome"],
        json!({"status": "success", "speedup": 2.0, "error_context": null})
    );
    assert_eq!(result["profile"]["sass"]["wgmma_count"], 0);
    payload["fixture"]["error_context"] = json!({"error_type": "correctness_error"});
    assert_eq!(
        evaluate(&payload).await?["outcome"],
        json!({"status": "failed", "speedup": 2.0, "error_context": {"error_type": "correctness_error"}})
    );
    payload["source_sha256"] = "changed".into();
    assert!(evaluate(&payload).await.is_err());
    payload["source_sha256"] = hash.into();
    payload["fixture"]["candidate_seconds"] = 0.into();
    assert!(evaluate(&payload).await.is_err());
    payload["problem"]["file_path"] = source.to_string_lossy().into_owned().into();
    payload["problem"]["file_sha256"] = "changed".into();
    assert!(validate_task(&payload["problem"]).await.is_err());
    Ok(())
}
