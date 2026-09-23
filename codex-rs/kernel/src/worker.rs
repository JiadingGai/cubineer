use crate::host::rpc;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use codex_app_server_client::InProcessAppServerClient;
use serde_json::Value;
use serde_json::json;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

pub(crate) struct Worker {
    pub python: Option<PathBuf>,
    pub script: Option<PathBuf>,
    pub requests: PathBuf,
    pub timeout_ms: u64,
    sequence: AtomicU64,
}

impl Worker {
    pub fn new(
        python: Option<PathBuf>,
        script: Option<PathBuf>,
        requests: PathBuf,
        timeout_ms: u64,
    ) -> Self {
        Self {
            python,
            script,
            requests,
            timeout_ms,
            sequence: AtomicU64::new(0),
        }
    }

    pub async fn call(
        &self,
        client: &InProcessAppServerClient,
        operation: &str,
        payload: Value,
    ) -> Result<Value> {
        if operation == "analyze" {
            return crate::domain::analysis::analyze(&payload);
        }
        if payload["backend"] == "gpu" && matches!(operation, "reference" | "evaluate") {
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(self.timeout_ms),
                crate::domain::gpu::evaluate(self, client, operation, payload.clone()),
            )
            .await;
            return match result {
                Ok(result) => result,
                Err(_) if operation == "evaluate" => Ok(json!({
                    "outcome": {"status": "failed", "speedup": null, "error_context": {
                        "error_type": "TimeoutError",
                        "error_message": format!("Evaluation exceeded {}s timeout", self.timeout_ms / 1000),
                    }},
                    "candidate_seconds": null,
                    "source_sha256": payload["source_sha256"],
                    "task_sha256": payload["problem"]["sha256"],
                    "build_identity": null,
                    "profile": null,
                    "validation": "gpu",
                    "profiling_options": {
                        "tool_mode": payload["tool_mode"],
                        "ncu_full": payload["ncu_full"],
                        "nsys": false,
                    },
                })),
                Err(error) => Err(error).context("GPU evaluation timed out"),
            };
        }
        if payload["backend"] == "simulated" {
            match operation {
                "reference" => return crate::domain::simulated::reference(&payload).await,
                "evaluate" => return crate::domain::simulated::evaluate(&payload).await,
                _ => {}
            }
        }
        if operation == "prepare" {
            let mut problem = crate::domain::dataset::prepare(&payload).await?;
            let metadata = if payload["backend"] == "gpu" {
                self.external(
                    client,
                    "metadata",
                    json!({"backend": "gpu", "gpu_id": payload["gpu_id"]}),
                )
                .await?
            } else {
                Value::Null
            };
            crate::domain::prompts::populate(&mut problem, &payload, &metadata)?;
            return Ok(problem);
        }
        if operation == "feedback" {
            return crate::domain::feedback::feedback(&payload);
        }
        if operation == "tool" {
            return crate::domain::analyzers::invoke(&payload);
        }
        if operation == "validate" {
            return crate::domain::models::validate(
                payload["role"].as_str().context("missing model role")?,
                &payload["value"],
            );
        }
        self.external(client, operation, payload).await
    }

    pub(crate) async fn execute(
        &self,
        client: &InProcessAppServerClient,
        command: Vec<String>,
        directory: &Path,
        mut env: Value,
        timeout_ms: u64,
    ) -> Result<Value> {
        let temporary = directory.join("tmp");
        tokio::fs::create_dir_all(&temporary).await?;
        env["TMPDIR"] = json!(temporary);
        env["PYTHONDONTWRITEBYTECODE"] = json!("1");
        rpc(
            client,
            "command/exec",
            json!({
                "command": command, "cwd": directory,
                "timeoutMs": timeout_ms.min(self.timeout_ms), "outputBytesCap": 4 * 1024 * 1024,
                "env": env,
                "sandboxPolicy": {"type": "workspaceWrite", "writableRoots": [directory],
                    "networkAccess": false, "excludeTmpdirEnvVar": true, "excludeSlashTmp": true},
            }),
        )
        .await
    }

    pub(crate) async fn external(
        &self,
        client: &InProcessAppServerClient,
        operation: &str,
        payload: Value,
    ) -> Result<Value> {
        let id = self.sequence.fetch_add(1, Ordering::Relaxed);
        let python = self
            .python
            .as_ref()
            .context("GPU workloads require --python")?;
        let script = self
            .script
            .as_ref()
            .context("GPU workloads require --worker")?;
        tokio::fs::create_dir_all(&self.requests).await?;
        let request = self.requests.join(format!("{id:06}-{operation}.json"));
        let directory = payload["output_dir"]
            .as_str()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.requests.join(format!("worker-{id}")));
        tokio::fs::create_dir_all(&directory).await?;
        tokio::fs::write(
            &request,
            serde_json::to_vec(
                &json!({"version": 1, "id": id, "operation": operation, "payload": payload}),
            )?,
        )
        .await?;
        let output = self
            .execute(
                client,
                [python, script, &request]
                    .map(|path| path.to_string_lossy().into_owned())
                    .to_vec(),
                &directory,
                json!({}),
                self.timeout_ms,
            )
            .await?;
        tokio::fs::write(
            request.with_extension("log"),
            output["stderr"].as_str().unwrap_or_default(),
        )
        .await?;
        ensure!(
            output["exitCode"] == 0,
            "worker {operation} failed: {}",
            output["stderr"]
        );
        let response: Value =
            serde_json::from_str(output["stdout"].as_str().context("missing worker stdout")?)
                .context("invalid or truncated worker response")?;
        ensure!(
            response["version"] == 1 && response["id"] == id,
            "worker response identity mismatch"
        );
        ensure!(
            response.get("error").is_none(),
            "worker {operation}: {}",
            response["error"]
        );
        let result = response
            .get("result")
            .cloned()
            .context("missing worker result")?;
        Ok(result)
    }
}
