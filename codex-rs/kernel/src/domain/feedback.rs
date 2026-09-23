use anyhow::Context;
use anyhow::Result;
use codex_utils_template::Template;
use regex::Regex;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::LazyLock;

static ASSETS: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("fixtures/feedback.json"))
        .unwrap_or_else(|error| panic!("bundled feedback assets must parse: {error}"))
});

fn render(name: &str, values: &BTreeMap<&str, String>) -> Result<String> {
    let template = Template::parse(
        ASSETS["templates"][name]
            .as_str()
            .context("feedback template")?,
    )?;
    let arguments = template
        .placeholders()
        .map(|name| {
            Ok((
                name,
                values
                    .get(name)
                    .context(format!("missing template value {name}"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(template.render(arguments)?)
}

fn text(value: &Value, key: &str, default: &str) -> String {
    match value.get(key) {
        None => default.to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(value) => value.to_string(),
    }
}

fn error_feedback(error: &Value, code: &str) -> Result<String> {
    let message = text(error, "error_message", "No details");
    let kind = text(error, "error_type", "Unknown");
    let lower = message.to_lowercase();
    let guidance = if kind == "correctness_error" {
        0
    } else if lower.contains("cuda") {
        if lower.contains("misaligned") {
            1
        } else if lower.contains("launch") {
            2
        } else {
            3
        }
    } else if kind == "SyntaxError" {
        4
    } else if kind == "ImportError" {
        5
    } else {
        6
    };
    let traceback = error["traceback"].as_str().unwrap_or_default();
    let mut values = BTreeMap::from([
        ("error_type", kind),
        ("error_msg", message.clone()),
        ("stage", text(error, "stage", "unknown")),
        ("previous_code", code.to_owned()),
        (
            "guidance",
            ASSETS["guidance"][guidance]
                .as_str()
                .context("error guidance")?
                .to_owned(),
        ),
    ]);
    let mut output = render("error_header", &values)?;
    if let Some(summary) = error["error_summary"]
        .as_str()
        .filter(|value| !value.is_empty())
    {
        values.insert("error_summary", summary.to_owned());
        values.insert(
            "error_abbrev",
            message
                .chars()
                .skip(message.chars().count().saturating_sub(1500))
                .collect(),
        );
        output.push_str(&render("error_summary", &values)?);
    } else {
        output.push_str(&render("error_message", &values)?);
    }
    if !traceback.is_empty() {
        if traceback.contains("static assertion failed") || traceback.contains("copy_traits") {
            let assertions = Regex::new("static assertion failed with \"([^\"]+)\"")?;
            let errors = Regex::new("error:.*")?;
            let assertions: BTreeSet<_> = assertions
                .captures_iter(traceback)
                .map(|capture| capture[1].to_owned())
                .collect();
            let mut summary = String::new();
            if !assertions.is_empty() {
                summary.push_str("**Root cause (static_assert):**\n");
                for assertion in assertions {
                    summary.push_str(&format!("- {assertion}\n"));
                }
            }
            let errors: Vec<_> = errors
                .find_iter(traceback)
                .take(5)
                .map(|error| error.as_str())
                .collect();
            if !errors.is_empty() {
                summary.push_str("**Compiler errors:**\n```\n");
                summary.push_str(&errors.join("\n"));
                summary.push_str("\n```\n");
            }
            values.insert("cute_summary", summary);
            values.insert(
                "traceback_tail",
                traceback
                    .chars()
                    .skip(traceback.chars().count().saturating_sub(2000))
                    .collect(),
            );
            output.push_str(&render("error_cute", &values)?);
        } else {
            values.insert("traceback_head", traceback.chars().take(2000).collect());
            output.push_str(&render("error_traceback", &values)?);
        }
    }
    output.push_str(&render("error_tail", &values)?);
    Ok(output)
}

fn profile_feedback(profile: &Value, code: &str) -> Result<String> {
    let recommendations = profile["optimization_recommendations"]
        .as_array()
        .filter(|values| !values.is_empty());
    let mut values = BTreeMap::from([
        ("bottleneck", text(profile, "bottleneck_type", "Unknown")),
        ("severity", text(profile, "severity", "Unknown")),
        (
            "priority",
            format!(
                "{:.1}",
                profile["priority_score"].as_f64().unwrap_or_default()
            ),
        ),
        (
            "improvement",
            format!(
                "{:.1}",
                profile["estimated_improvement"]
                    .as_f64()
                    .unwrap_or_default()
            ),
        ),
        ("previous_code", code.to_owned()),
        (
            "recs_text",
            recommendations
                .map(|items| {
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            format!("{}. {}", index + 1, value.as_str().unwrap_or_default())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_else(|| "No specific recommendations available.".to_owned()),
        ),
    ]);
    let rules = profile["rule_recommendations"]
        .as_array()
        .filter(|values| !values.is_empty());
    let rule_block = if let Some(rules) = rules {
        values.insert(
            "ranked",
            rules
                .iter()
                .enumerate()
                .map(|(index, rule)| {
                    format!(
                        "{}. [est. {:.0}% speedup] {}",
                        index + 1,
                        rule["est_speedup_pct"].as_f64().unwrap_or_default(),
                        rule["text"].as_str().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        render("rules", &values)?
    } else {
        String::new()
    };
    values.insert("rule_block", rule_block);
    let mut output = render("profile_header", &values)?;
    if let Some(details) = profile["detailed_analysis"]
        .as_str()
        .filter(|value| !value.is_empty())
    {
        values.insert("detailed", details.to_owned());
        output.push_str(&render("profile_details", &values)?);
    }
    output.push_str(&render("profile_tail", &values)?);
    Ok(output)
}

pub(crate) fn feedback(payload: &Value) -> Result<Value> {
    let parent = &payload["parent"];
    let code = parent["submission"]["code"].as_str().unwrap_or_default();
    let mut output = if parent.is_null() {
        String::new()
    } else if parent["outcome"]["status"] == "failed" {
        error_feedback(&parent["outcome"]["error_context"], code)?
    } else if parent["analysis"]
        .as_object()
        .is_some_and(|value| !value.is_empty())
    {
        profile_feedback(&parent["analysis"], code)?
    } else if payload["mode"] == "initial" {
        render(
            "fresh",
            &BTreeMap::from([("previous_code", code.to_owned())]),
        )?
    } else {
        code.to_owned()
    };
    if let Some(hint) = payload["hint"].as_str().filter(|value| !value.is_empty()) {
        output.push_str("\n\n");
        output.push_str(&render(
            "hint",
            &BTreeMap::from([("hint", hint.to_owned())]),
        )?);
    }
    Ok(json!({"text": output}))
}

#[cfg(test)]
#[path = "feedback_tests.rs"]
mod tests;
