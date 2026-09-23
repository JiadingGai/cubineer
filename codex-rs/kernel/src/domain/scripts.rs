use anyhow::Context;
use anyhow::Result;
use codex_utils_template::Template;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

static ASSETS: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("fixtures/scripts.json"))
        .unwrap_or_else(|error| panic!("bundled workload templates must parse: {error}"))
});

fn render(name: &str, values: &BTreeMap<String, String>) -> Result<String> {
    let template = Template::parse(ASSETS[name].as_str().context("workload template")?)?;
    let arguments = template
        .placeholders()
        .map(|key| {
            Ok((
                key,
                values
                    .get(key)
                    .context(format!("missing script field {key}"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(template.render(arguments)?)
}

pub(crate) async fn write(name: &str, payload: &Value) -> Result<PathBuf> {
    let output = Path::new(payload["output_dir"].as_str().context("workload output")?);
    let problem = &payload["problem"];
    let source = payload["source"].as_str().unwrap_or_default();
    let filename = match name {
        "reference" => "profile_reference.py",
        "candidate" => "profile_solution.py",
        "benchmark" => "benchmark_solution.py",
        _ => anyhow::bail!("unknown workload template"),
    };
    let inputs = problem["input_generator"]
        .as_str()
        .context("input generator")?;
    let clean = Regex::new(r#"(?s)\"\"\".*?\"\"\""#)?
        .replace_all(inputs, "# [docstring removed]")
        .into_owned();
    let clean = Regex::new("(?s)'''.*?'''")?
        .replace_all(&clean, "# [docstring removed]")
        .into_owned();
    let values = BTreeMap::from([
        (
            "reference".into(),
            problem["canonical_solution"]
                .as_str()
                .context("reference source")?
                .into(),
        ),
        ("inputs".into(), inputs.into()),
        ("seed".into(), "42".into()),
        ("solution_file".into(), source.into()),
        ("solution_path".into(), source.into()),
        ("num_warmup".into(), "5".into()),
        ("num_trials".into(), "5".into()),
        (
            "num_iterations".into(),
            if name == "reference" { "10" } else { "1" }.into(),
        ),
        ("output_name".into(), filename.into()),
        (
            "reference_file_path_str".into(),
            problem["file_path"]
                .as_str()
                .context("reference path")?
                .into(),
        ),
        ("clean_input_gen".into(), clean),
        (
            "solution_name".into(),
            Path::new(source)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        ),
    ]);
    let path = output.join(filename);
    tokio::fs::write(&path, render(name, &values)?).await?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).await?;
    }
    Ok(path)
}

#[cfg(test)]
#[path = "scripts_tests.rs"]
mod tests;
