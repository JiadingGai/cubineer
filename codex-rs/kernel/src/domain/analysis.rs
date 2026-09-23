use super::analyzers;
use super::analyzers::display;
use super::profilers;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

pub(crate) fn analyze(payload: &Value) -> Result<Value> {
    let report = &payload["report"];
    let mode = payload["tool_mode"].as_str().unwrap_or("proactive");
    ensure!(
        matches!(mode, "proactive" | "reactive"),
        "unknown profiling mode"
    );
    let bottleneck = payload["bottleneck"].as_str().unwrap_or_default();
    let selected = analyzers::selected(bottleneck)?;
    let selected_names: Vec<_> = selected
        .clone()
        .unwrap_or_else(analyzers::registered)
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    let tools = selected.unwrap_or_else(analyzers::prioritized);
    let all_tools = analyzers::prioritized();
    let mut lines = vec!["# GPU Profiling Metrics\n".to_owned()];
    for name in report["kernel_names"].as_array().context("kernel names")? {
        let name = name.as_str().context("kernel name")?;
        let Some(metrics) = report["metrics"].get(name) else {
            continue;
        };
        lines.push(format!("## Kernel: {name}\n"));
        for group in [
            "compute",
            "memory",
            "occupancy",
            "stall",
            "timing",
            "branch",
        ] {
            let label = format!("{}{}", group[..1].to_uppercase(), &group[1..]);
            lines.push(format!(
                "{}### {label} Metrics:",
                if group == "compute" { "" } else { "\n" }
            ));
            for key in profilers::ASSETS["metrics"][group]
                .as_array()
                .context("metric group")?
            {
                let key = key.as_str().context("metric name")?;
                if let Some(value) = metrics.get(key) {
                    lines.push(format!("- `{key}`: {}", display(value)));
                }
            }
        }
        let sass = &report["sass"];
        if sass.as_object().is_some_and(|value| !value.is_empty())
            && ["wgmma", "hmma", "imma", "dmma", "bmma"]
                .iter()
                .all(|kind| sass[format!("{kind}_count")].as_u64().unwrap_or_default() == 0)
            && sass["uses_cublas"] != true
        {
            lines.extend([
                "\n### SASS Binary Analysis (Bottleneck Detected)".into(),
                "- WGMMA instructions (Hopper TC): 0".into(),
                "- HMMA instructions (Ampere TC): 0".into(),
                "- cuBLAS detected: No".into(),
                format!("- FP32 FMA (FFMA): {}", sass["ffma_count"].as_u64().unwrap_or_default()),
                "\n**SASS Analysis: 0 tensor core instructions detected.**".into(),
                "If this kernel performs matrix operations, consider leveraging tensor core instructions (cuBLAS, CUTLASS, or inline PTX) to improve performance.".into(),
            ]);
        }
        if mode == "proactive" {
            let mut guidance = Vec::new();
            for tool in &tools {
                if tool.should_trigger(metrics)
                    && let Some(output) = tool.analyze(metrics, sass)?
                {
                    guidance.push(output.text(name));
                }
            }
            if !guidance.is_empty() {
                lines.push("\n## Automated Bottleneck Analysis".into());
                lines.push(guidance.join("\n"));
            }
        } else {
            lines.push("\n## Available Analysis Tools".into());
            let classification = if bottleneck.is_empty() {
                "None".into()
            } else {
                format!("BottleneckType.{}", bottleneck.to_uppercase())
            };
            lines.push(format!(
                "Stage 1 bottleneck classification: **{classification}**"
            ));
            lines.push("You may call any of the following tools to get detailed analysis:".into());
            lines.extend(
                all_tools
                    .iter()
                    .map(|tool| format!("- **{}**: {}", tool.name, tool.description)),
            );
        }
        lines.push("\n".into());
    }
    lines.push(
        "\nProvide your analysis following the guidelines in the system instructions.".into(),
    );
    let mut text = lines.join("\n");
    if let Some(rules) = report["rule_recommendations"]
        .as_array()
        .filter(|rules| !rules.is_empty())
    {
        let assets: Value = serde_json::from_str(include_str!("fixtures/analysis.json"))?;
        let header = assets["rules_header"]
            .as_str()
            .context("rule engine header")?;
        let ranked = rules
            .iter()
            .enumerate()
            .map(|(index, rule)| {
                format!(
                    "{}. [est. {:.0}% speedup] {}",
                    index + 1,
                    rule["est_speedup_pct"].as_f64().unwrap_or_default(),
                    rule["text"].as_str().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        text = format!("{header}\n{ranked}\n\n{text}");
    }
    Ok(
        json!({"text": text, "selected_tools": if mode == "proactive" { selected_names } else { Vec::new() }, "tools": if mode == "reactive" { all_tools.iter().map(|tool| json!({"name": tool.name, "description": tool.description})).collect::<Vec<_>>() } else { Vec::new() }}),
    )
}

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;
