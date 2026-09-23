use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use codex_utils_template::Template;
use serde_json::Value;
use serde_json::json;
use std::sync::LazyLock;

static ASSETS: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("fixtures/prompts.json"))
        .unwrap_or_else(|error| panic!("bundled prompt assets must parse: {error}"))
});

fn asset(name: &str) -> Result<&'static str> {
    ASSETS[name]
        .as_str()
        .context(format!("missing prompt {name}"))
}

fn candidate(
    mode: &str,
    baseline: &str,
    cutlass: Option<&str>,
    capability: &[u64],
) -> Result<String> {
    ensure!(
        matches!(mode, "initial" | "repair" | "optimize"),
        "invalid candidate mode"
    );
    ensure!(matches!(baseline, "pytorch" | "triton"), "invalid baseline");
    let mut sections = vec![asset(&format!("mode_instructions/{mode}.txt"))?];
    if mode == "initial" {
        sections.push(asset(&format!(
            "dataset_formats/kernelbench_{baseline}.txt"
        ))?);
        sections.push(asset("mode_instructions/initial_requirements.txt")?);
    }
    if mode == "optimize" {
        sections.push(asset("base/optimization_strategies.txt")?);
    }
    sections.push(asset("base/cutlass_setup.txt")?);
    if mode == "initial" || mode == "repair" {
        sections.push(asset("base/common_pitfalls.txt")?);
    }
    sections.push(asset("base/pytorch_integration.txt")?);
    if mode == "initial"
        && let Some(example) =
            ASSETS[format!("examples/kernelbench_{baseline}_example.txt")].as_str()
    {
        sections.push(example);
    }
    sections.push(asset("base/output_format.txt")?);
    let gpu = match capability {
        [9, _] => asset("hopper")?,
        _ => "",
    };
    Ok(sections.join("\n\n")
        .replace("{{CUTLASS_ROOT}}", cutlass.filter(|s| !s.is_empty()).unwrap_or("/path/to/cutlass"))
        .replace("{{GPU_SPECIFIC_CUTLASS_INFO}}", gpu)
        .replace(asset("base/output_format.txt")?, "Return exactly one JSON CodeCompletion object with code, approach, and confidence. No completions wrapper or markdown fences."))
}

fn reference(gpu: &str, ridge: f64) -> Result<String> {
    Ok(Template::parse(asset("reference")?)?.render([
        ("gpu_name", gpu.to_owned()),
        ("ridge", format!("{ridge:.0}")),
        ("compute", format!("{:.0}", ridge * 4.0)),
        ("memory", format!("{:.0}", ridge / 4.0)),
    ])?)
}

pub(crate) fn populate(problem: &mut Value, payload: &Value, metadata: &Value) -> Result<()> {
    let baseline = if problem["task_id"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase()
        .contains("_triton")
    {
        "triton"
    } else {
        "pytorch"
    };
    let capability: Vec<_> = metadata["capability"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    let cutlass = payload["cutlass_root"].as_str();
    problem["instructions"] = json!({
        "initial": candidate("initial", baseline, cutlass, &capability)?,
        "repair": candidate("repair", baseline, cutlass, &capability)?,
        "optimize": candidate("optimize", baseline, cutlass, &capability)?,
    });
    let (gpu, ridge) = if payload["backend"] == "simulated" {
        ("NVIDIA H200 (simulated)", 412.0)
    } else {
        match metadata["name"].as_str().unwrap_or_default() {
            "NVIDIA B200" => ("NVIDIA B200", 4500.0 / 8.0),
            "NVIDIA H100 80GB HBM3" => ("NVIDIA H100 80GB HBM3", 1979.0 / 3.35),
            _ => ("NVIDIA H200", 1979.0 / 4.8),
        }
    };
    problem["reference_instructions"] = reference(gpu, ridge)?.into();
    problem["profile_instructions"] = asset("profile")?.into();
    problem["schemas"] = super::models::schemas();
    Ok(())
}

#[cfg(test)]
#[path = "prompts_tests.rs"]
mod tests;
