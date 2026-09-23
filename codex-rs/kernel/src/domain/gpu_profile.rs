use super::profilers;
use crate::worker::Worker;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use codex_app_server_client::InProcessAppServerClient;
use serde_json::Value;
use serde_json::json;
use std::path::Path;
use std::path::PathBuf;

pub(crate) struct Profiler<'a> {
    pub worker: &'a Worker,
    pub client: &'a InProcessAppServerClient,
    pub output: &'a Path,
    pub env: Value,
    pub runtime: &'a Value,
}

impl Profiler<'_> {
    pub(crate) async fn run(&self, command: Vec<String>, timeout_ms: u64) -> Result<Value> {
        self.worker
            .execute(
                self.client,
                command,
                self.output,
                self.env.clone(),
                timeout_ms,
            )
            .await
    }

    fn privileged(&self, command: Vec<String>) -> Vec<String> {
        if self.runtime["uid"] != 0
            && let Ok(sudo) = which::which("sudo")
        {
            [
                vec![
                    sudo.to_string_lossy().into_owned(),
                    "-E".into(),
                    "--preserve-env=PATH,LD_LIBRARY_PATH,CUDA_VISIBLE_DEVICES,TORCH_EXTENSIONS_DIR"
                        .into(),
                ],
                command,
            ]
            .concat()
        } else {
            command
        }
    }

    pub(crate) async fn collect(
        &self,
        script: &Path,
        names: &[String],
        full: bool,
    ) -> Result<Value> {
        let Some(ncu) = ncu_binary() else {
            return Ok(json!({"success": false, "error": "NCU not available", "metrics": {}}));
        };
        match self.collect_inner(&ncu, script, names, full).await {
            Ok(report) => Ok(report),
            Err(error) => Ok(
                json!({"success": false, "error": format!("NCU profiling error: {error}"), "metrics": {}}),
            ),
        }
    }

    async fn collect_inner(
        &self,
        ncu: &Path,
        script: &Path,
        names: &[String],
        full: bool,
    ) -> Result<Value> {
        ensure!(
            script
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .starts_with("profile_"),
            "expected profile script"
        );
        let python = self
            .worker
            .python
            .as_ref()
            .context("GPU workloads require --python")?;
        let csv = self.output.join(format!(
            "ncu_profile_{}.csv",
            script.file_stem().unwrap_or_default().to_string_lossy()
        ));
        let command = metrics_command(ncu, python, script, &csv, names)?;
        let output = self.run(self.privileged(command), 300_000).await?;
        if output["exitCode"] != 0 {
            return Ok(
                json!({"success": false, "error": format!("NCU failed: {}", output["stderr"].as_str().unwrap_or_default()), "metrics": {}}),
            );
        }
        let metrics = if csv.exists() {
            profilers::ncu_csv(&tokio::fs::read_to_string(&csv).await?, names)?
        } else {
            json!({})
        };
        let discovered = if names.is_empty() {
            metrics
                .as_object()
                .context("NCU metrics")?
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            names.to_vec()
        };
        let rules = if full {
            self.rules(ncu, python, script, names)
                .await
                .unwrap_or_else(|_| json!([]))
        } else {
            json!([])
        };
        Ok(
            json!({"success": true, "kernel_names": discovered, "library_fallback": names.is_empty(),
            "metrics": metrics, "csv_path": csv, "rule_recommendations": rules}),
        )
    }

    async fn rules(
        &self,
        ncu: &Path,
        python: &Path,
        script: &Path,
        names: &[String],
    ) -> Result<Value> {
        let base = self.output.join("ncu_rules");
        let mut command = vec![
            ncu.to_string_lossy().into_owned(),
            "--set".into(),
            "full".into(),
            "--launch-skip".into(),
            "0".into(),
            "--launch-count".into(),
            "1".into(),
            "-f".into(),
            "-o".into(),
            base.to_string_lossy().into_owned(),
        ];
        if !names.is_empty() {
            command.push(kernel_filter(names));
        }
        command.extend([
            python.to_string_lossy().into_owned(),
            script.to_string_lossy().into_owned(),
        ]);
        let output = self.run(self.privileged(command), 600_000).await?;
        let report = base.with_extension("ncu-rep");
        if output["exitCode"] != 0 || !report.exists() {
            return Ok(json!([]));
        }
        let output = self
            .run(
                vec![
                    ncu.to_string_lossy().into_owned(),
                    "--import".into(),
                    report.to_string_lossy().into_owned(),
                    "--page".into(),
                    "details".into(),
                ],
                120_000,
            )
            .await?;
        if output["exitCode"] != 0 {
            return Ok(json!([]));
        }
        profilers::rules(output["stdout"].as_str().unwrap_or_default())
    }
}

pub(crate) fn ncu_binary() -> Option<PathBuf> {
    which::which("ncu").ok().or_else(|| {
        let path = PathBuf::from("/usr/local/cuda/bin/ncu");
        path.exists().then_some(path)
    })
}

fn kernel_filter(names: &[String]) -> String {
    let pattern = names
        .iter()
        .map(|name| regex::escape(name))
        .collect::<Vec<_>>()
        .join("|");
    format!("--kernel-name=regex:({pattern})")
}

fn metrics_command(
    ncu: &Path,
    python: &Path,
    script: &Path,
    csv: &Path,
    names: &[String],
) -> Result<Vec<String>> {
    let metrics = [
        "compute",
        "memory",
        "occupancy",
        "stall",
        "timing",
        "branch",
    ]
    .into_iter()
    .map(|group| {
        profilers::ASSETS["metrics"][group]
            .as_array()
            .context("metric group")
    })
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
    .map(|name| name.as_str().context("metric name"))
    .collect::<Result<Vec<_>>>()?
    .join(",");
    let mut command = vec![
        ncu.to_string_lossy().into_owned(),
        "--csv".into(),
        "--page=raw".into(),
        "--kernel-name-base=demangled".into(),
        "--target-processes=application-only".into(),
        "--replay-mode=kernel".into(),
        "--profile-from-start=on".into(),
        format!("--log-file={}", csv.display()),
    ];
    if !names.is_empty() {
        command.push(kernel_filter(names));
    }
    command.extend([
        format!("--metrics={metrics}"),
        "--launch-skip=5".into(),
        format!("--launch-count={}", if names.is_empty() { 5 } else { 1 }),
        python.to_string_lossy().into_owned(),
        script.to_string_lossy().into_owned(),
    ]);
    Ok(command)
}

#[cfg(test)]
#[path = "gpu_profile_tests.rs"]
mod tests;
