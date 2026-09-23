mod numeric;
mod sass;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde::Deserialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::sync::LazyLock;

pub(crate) static ASSETS: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../fixtures/analyzers.json"))
        .unwrap_or_else(|error| panic!("bundled analyzer assets must parse: {error}"))
});

#[derive(Clone, Deserialize)]
pub(crate) struct Tool {
    pub name: String,
    pub description: String,
    pub priority: u64,
    required: Vec<String>,
}

pub(crate) fn registered() -> Vec<Tool> {
    serde_json::from_value(ASSETS["registry"].clone())
        .unwrap_or_else(|error| panic!("bundled analyzer registry must parse: {error}"))
}

pub(crate) fn prioritized() -> Vec<Tool> {
    let mut tools = registered();
    tools.sort_by_key(|tool| std::cmp::Reverse(tool.priority));
    tools
}

pub(crate) fn selected(bottleneck: &str) -> Result<Option<Vec<Tool>>> {
    match bottleneck {
        "compute_bound" | "memory_bound" => Ok(Some(
            registered()
                .into_iter()
                .filter(|tool| {
                    let affinity = match tool.name.as_str() {
                        "tensor_core_underutilization" | "wgmma_instruction_detector" => {
                            "compute_bound"
                        }
                        "memory_coalescing" | "high_dram_throughput" => "memory_bound",
                        _ => return true,
                    };
                    affinity == bottleneck
                })
                .collect(),
        )),
        "" | "mixed" | "latency_bound" | "occupancy_limited" | "launch_overhead" => Ok(None),
        _ => bail!("unknown bottleneck: {bottleneck}"),
    }
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(crate) struct Output {
    severity: String,
    title: String,
    summary: String,
    metrics_observed: Map<String, Value>,
    metrics_thresholds: Map<String, Value>,
    root_cause: String,
    recommendations: Vec<String>,
    expected_improvement: String,
    code_example: Option<String>,
    references: Vec<String>,
}

impl Output {
    fn base(key: &str) -> Result<Self> {
        Ok(serde_json::from_value(ASSETS["outputs"][key].clone())?)
    }

    fn observed(&mut self, values: Value) {
        let Value::Object(values) = values else {
            unreachable!("observed metrics are constructed as JSON objects");
        };
        self.metrics_observed = values;
    }

    pub(crate) fn text(&self, kernel: &str) -> String {
        let mut lines = vec![format!(
            "\n### [{}] {}",
            self.severity.to_uppercase(),
            self.title
        )];
        if !self.summary.is_empty() {
            lines.push(format!("\n**Kernel `{kernel}`**: {}", self.summary));
        }
        if !self.metrics_observed.is_empty() {
            lines.push("\n**Observed Metrics:**".into());
            for (key, value) in &self.metrics_observed {
                let suffix = self
                    .metrics_thresholds
                    .get(key)
                    .filter(|value| !value.is_null())
                    .map(|value| format!(" (threshold: {})", display(value)))
                    .unwrap_or_default();
                lines.push(format!("- `{key}`: {}{suffix}", display(value)));
            }
        }
        if !self.root_cause.is_empty() {
            lines.push(format!("\n**Root Cause:**\n{}", self.root_cause));
        }
        if !self.recommendations.is_empty() {
            lines.push("\n**Recommendations:**".into());
            lines.extend(
                self.recommendations
                    .iter()
                    .enumerate()
                    .map(|(index, value)| format!("{}. {value}", index + 1)),
            );
        }
        if !self.expected_improvement.is_empty() {
            lines.push(format!(
                "\n**Expected Improvement:** {}",
                self.expected_improvement
            ));
        }
        if let Some(example) = &self.code_example {
            lines.push(format!("\n**Example Fix:**\n```cuda\n{example}\n```"));
        }
        if !self.references.is_empty() {
            lines.push("\n**References:**".into());
            lines.extend(self.references.iter().map(|value| format!("- {value}")));
        }
        lines.join("\n")
    }
}

pub(crate) fn display(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        _ => value.to_string(),
    }
}

fn metric(metrics: &Value, key: &str, default: Value) -> Value {
    match metrics.get(key) {
        Some(Value::Number(value)) => value.clone().into(),
        Some(Value::Bool(value)) => (*value).into(),
        Some(Value::String(value)) => value
            .replace([',', '%'], "")
            .trim()
            .parse::<f64>()
            .ok()
            .map(Value::from)
            .unwrap_or(default),
        _ => default,
    }
}

fn number(metrics: &Value, key: &str, default: f64) -> f64 {
    match metric(metrics, key, default.into()) {
        Value::Bool(value) => f64::from(u8::from(value)),
        value => value.as_f64().unwrap_or(default),
    }
}

fn title(value: &str) -> String {
    value
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Tool {
    pub(crate) fn should_trigger(&self, metrics: &Value) -> bool {
        self.required.iter().all(|key| metrics.get(key).is_some())
    }

    pub(crate) fn analyze(&self, metrics: &Value, sass: &Value) -> Result<Option<Output>> {
        match self.name.as_str() {
            "wgmma_instruction_detector" => sass::tensor(sass),
            "register_spill_detector" => sass::spills(metrics, sass),
            _ => numeric::analyze(self, metrics),
        }
    }
}

pub(crate) fn invoke(payload: &Value) -> Result<Value> {
    let tool = registered()
        .into_iter()
        .find(|tool| payload["name"] == tool.name)
        .context("unsupported or deferred analyzer")?;
    let mut results = Vec::new();
    for (name, metrics) in payload["report"]["metrics"]
        .as_object()
        .context("report metrics")?
    {
        if !tool.should_trigger(metrics) {
            results.push(format!(
                "[{}] Kernel `{name}`: not triggered (thresholds not met)",
                tool.name
            ));
        } else if let Some(result) = tool.analyze(metrics, &payload["report"]["sass"])? {
            results.push(result.text(name));
        } else {
            results.push(format!(
                "[{}] Kernel `{name}`: analyzed but no issue detected",
                tool.name
            ));
        }
    }
    Ok(
        json!({"text": if results.is_empty() { format!("[{}] No kernels to analyze", tool.name) } else { results.join("\n") }}),
    )
}

#[cfg(test)]
#[path = "analyzers_tests.rs"]
mod tests;
