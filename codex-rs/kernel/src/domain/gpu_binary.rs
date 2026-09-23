use super::gpu_profile::Profiler;
use super::profilers;
use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use serde_json::json;
use std::path::Path;
use std::path::PathBuf;
use walkdir::WalkDir;

pub(crate) fn artifacts(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .map(walkdir::DirEntry::into_path)
        .collect()
}

pub(crate) async fn kernel_names(profiler: &Profiler<'_>) -> Result<Vec<String>> {
    let (Ok(cuobjdump), Ok(cxxfilt)) = (which::which("cuobjdump"), which::which("c++filt")) else {
        return Ok(Vec::new());
    };
    let entry = Regex::new(r"(?m)^STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+(\S+)")?;
    let template = Regex::new(r"<.*>")?;
    let mut names = Vec::new();
    for path in artifacts(&profiler.output.join("build")) {
        if path.extension().is_none_or(|extension| extension != "so") {
            continue;
        }
        let extracted: Result<Vec<String>> = async {
            let output = profiler
                .run(
                    vec![
                        cuobjdump.to_string_lossy().into_owned(),
                        "-symbols".into(),
                        path.to_string_lossy().into_owned(),
                    ],
                    30_000,
                )
                .await?;
            if output["exitCode"] != 0 {
                return Ok(Vec::new());
            }
            let mut command = vec![cxxfilt.to_string_lossy().into_owned()];
            command.extend(
                entry
                    .captures_iter(output["stdout"].as_str().unwrap_or_default())
                    .map(|capture| capture[1].to_owned()),
            );
            if command.len() == 1 {
                return Ok(Vec::new());
            }
            let demangled = profiler.run(command, 10_000).await?;
            let _ = profiler
                .run(
                    vec![
                        cuobjdump.to_string_lossy().into_owned(),
                        "--dump-resource-usage".into(),
                        path.to_string_lossy().into_owned(),
                    ],
                    30_000,
                )
                .await?;
            Ok(demangled["stdout"]
                .as_str()
                .unwrap_or_default()
                .trim()
                .lines()
                .map(|line| {
                    let base = line.split('(').next().unwrap_or_default().trim();
                    let base = template.replace_all(base, "");
                    base.split_whitespace()
                        .last()
                        .unwrap_or_default()
                        .to_owned()
                })
                .collect())
        }
        .await;
        for name in extracted.unwrap_or_default() {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    Ok(names)
}

pub(crate) async fn sass(profiler: &Profiler<'_>) -> Result<Value> {
    let mut newest = None;
    for path in artifacts(&profiler.output.join("build")) {
        if path.extension().is_none_or(|extension| extension != "so") {
            continue;
        }
        let modified = path.metadata()?.modified()?;
        if newest
            .as_ref()
            .is_none_or(|(_, previous)| modified > *previous)
        {
            newest = Some((path, modified));
        }
    }
    let Some((path, _)) = newest else {
        return Ok(Value::Null);
    };
    let cuobjdump = which::which("cuobjdump")
        .ok()
        .or_else(|| {
            [
                "/usr/local/cuda/bin/cuobjdump",
                "/usr/local/cuda-12/bin/cuobjdump",
                "/usr/local/cuda-11/bin/cuobjdump",
            ]
            .into_iter()
            .map(PathBuf::from)
            .find(|path| path.exists())
        })
        .unwrap_or_else(|| PathBuf::from("cuobjdump"));
    let cublas =
        Regex::new(r"(?i)cublas[A-Z]gemm|cublasGemmEx|cublasLt|cublas.*Batched|cublasGemmStrided")?;
    let mut symbols = Vec::new();
    if let Ok(output) = profiler
        .run(
            vec![
                "nm".into(),
                "-D".into(),
                path.to_string_lossy().into_owned(),
            ],
            10_000,
        )
        .await
        && output["exitCode"] == 0
    {
        for line in output["stdout"].as_str().unwrap_or_default().lines() {
            if cublas.is_match(line)
                && let Some(symbol) = line.split_whitespace().last()
                && !symbols.iter().any(|value| value == symbol)
            {
                symbols.push(symbol.to_owned());
            }
        }
    }
    let result: Result<Option<String>> = async {
        let command = |argument: &str| {
            vec![
                cuobjdump.to_string_lossy().into_owned(),
                argument.into(),
                path.to_string_lossy().into_owned(),
            ]
        };
        let listed = profiler.run(command("--list-ptx"), 30_000).await?;
        let dumped = profiler.run(command("--dump-sass"), 30_000).await?;
        let text = dumped["stdout"].as_str().unwrap_or_default();
        Ok(
            if dumped["exitCode"] == 0 && (listed["exitCode"] != 0 || !text.trim().is_empty()) {
                Some(text.to_owned())
            } else if listed["exitCode"] != 0 && !symbols.is_empty() {
                Some(String::new())
            } else {
                None
            },
        )
    }
    .await;
    let text = match result {
        Ok(text) => text,
        Err(_) if !symbols.is_empty() => Some(String::new()),
        Err(_) => None,
    };
    let Some(text) = text else {
        return Ok(Value::Null);
    };
    let mut result = profilers::sass(&text, /*kernel*/ None)?;
    result["uses_cublas"] = json!(!symbols.is_empty());
    result["cublas_symbols"] = json!(symbols);
    Ok(result)
}
