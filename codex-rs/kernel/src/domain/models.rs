use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Map;
use serde_json::Number;
use serde_json::Value;
use std::sync::LazyLock;

static MODELS: LazyLock<Map<String, Value>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("fixtures/models.json"))
        .unwrap_or_else(|error| panic!("bundled model contracts must parse: {error}"))
});

static SCHEMAS: LazyLock<Value> = LazyLock::new(|| {
    Value::Object(
        MODELS
            .iter()
            .map(|(role, model)| {
                let validation = &model["validation"];
                let schema = strict_schema(validation, validation).unwrap_or_else(|error| {
                    panic!("bundled {role} schema must be strict: {error}")
                });
                (role.clone(), schema)
            })
            .collect(),
    )
});

pub(crate) fn schemas() -> Value {
    SCHEMAS.clone()
}

pub(crate) fn validate(role: &str, value: &Value) -> Result<Value> {
    let schema = MODELS.get(role).context("unknown model role")?;
    normalize(value, &schema["validation"], &schema["validation"])
}

fn normalize(value: &Value, schema: &Value, root: &Value) -> Result<Value> {
    if let Some(reference) = schema["$ref"].as_str() {
        return normalize(
            value,
            root.pointer(reference.trim_start_matches('#'))
                .context("schema reference")?,
            root,
        );
    }
    if let Some(choices) = schema["anyOf"].as_array() {
        for choice in choices {
            if let Ok(value) = normalize(value, choice, root) {
                return Ok(value);
            }
        }
        bail!("value does not match any schema alternative");
    }
    let result = match schema["type"].as_str().context("schema type")? {
        "object" => {
            let input = value.as_object().context("expected object")?;
            let properties = schema["properties"]
                .as_object()
                .context("schema properties")?;
            if schema["additionalProperties"] == false {
                ensure!(
                    input.keys().all(|key| properties.contains_key(key)),
                    "unexpected field"
                );
            }
            let mut result = Map::new();
            for (name, field) in properties {
                let value = match input.get(name) {
                    Some(value) => normalize(value, field, root).with_context(|| name.clone())?,
                    None => field
                        .get("default")
                        .context(format!("missing field {name}"))?
                        .clone(),
                };
                result.insert(name.clone(), value);
            }
            Value::Object(result)
        }
        "array" => Value::Array(
            value
                .as_array()
                .context("expected array")?
                .iter()
                .map(|value| normalize(value, &schema["items"], root))
                .collect::<Result<_>>()?,
        ),
        "string" => Value::String(value.as_str().context("expected string")?.to_owned()),
        "boolean" => {
            let boolean = match value {
                Value::Bool(value) => *value,
                Value::Number(value) if value.as_f64() == Some(0.0) => false,
                Value::Number(value) if value.as_f64() == Some(1.0) => true,
                Value::String(value) => match value.to_ascii_lowercase().as_str() {
                    "0" | "off" | "f" | "false" | "n" | "no" => false,
                    "1" | "on" | "t" | "true" | "y" | "yes" => true,
                    _ => bail!("expected boolean"),
                },
                _ => bail!("expected boolean"),
            };
            Value::Bool(boolean)
        }
        "integer" => {
            if let Some(text) = value.as_str() {
                ensure!(!text.contains(['e', 'E']), "expected integer string");
            }
            if value.is_i64() || value.is_u64() {
                value.clone()
            } else {
                let number = numeric(value)?;
                ensure!(number.fract() == 0.0, "expected integer");
                ensure!(
                    number >= i64::MIN as f64 && number < 18_446_744_073_709_551_616.0,
                    "integer out of range"
                );
                if number < 0.0 {
                    Value::from(number as i64)
                } else {
                    Value::from(number as u64)
                }
            }
        }
        "number" => Value::Number(Number::from_f64(numeric(value)?).context("nonfinite number")?),
        "null" => {
            ensure!(value.is_null(), "expected null");
            Value::Null
        }
        kind => bail!("unsupported schema type {kind}"),
    };
    if let Some(allowed) = schema["enum"].as_array() {
        ensure!(allowed.contains(&result), "unknown enum value");
    }
    if let Some(number) = result.as_f64() {
        if let Some(minimum) = schema["minimum"].as_f64() {
            ensure!(number >= minimum, "number below minimum");
        }
        if let Some(maximum) = schema["maximum"].as_f64() {
            ensure!(number <= maximum, "number above maximum");
        }
    }
    Ok(result)
}

fn strict_schema(schema: &Value, root: &Value) -> Result<Value> {
    let mut schema = schema
        .as_object()
        .context("schema must be an object")?
        .clone();
    if schema.is_empty() {
        return Ok(serde_json::json!({
            "additionalProperties": false,
            "type": "object",
            "properties": {},
            "required": [],
        }));
    }
    for name in ["$defs", "definitions"] {
        if let Some(definitions) = schema.get(name).and_then(Value::as_object).cloned() {
            schema.insert(
                name.into(),
                Value::Object(
                    definitions
                        .iter()
                        .map(|(key, value)| Ok((key.clone(), strict_schema(value, root)?)))
                        .collect::<Result<_>>()?,
                ),
            );
        }
    }
    if schema.get("type").and_then(Value::as_str) == Some("object") {
        match schema.get("additionalProperties") {
            None => {
                schema.insert("additionalProperties".into(), Value::Bool(false));
            }
            Some(Value::Bool(false)) => {}
            Some(_) => bail!("additionalProperties must be false for object schemas"),
        }
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object).cloned() {
        schema.insert(
            "required".into(),
            Value::Array(properties.keys().cloned().map(Value::String).collect()),
        );
        schema.insert(
            "properties".into(),
            Value::Object(
                properties
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), strict_schema(value, root)?)))
                    .collect::<Result<_>>()?,
            ),
        );
    }
    if let Some(items) = schema.get("items").cloned() {
        schema.insert("items".into(), strict_schema(&items, root)?);
    }
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array).cloned() {
        schema.insert(
            "anyOf".into(),
            Value::Array(
                any_of
                    .iter()
                    .map(|value| strict_schema(value, root))
                    .collect::<Result<_>>()?,
            ),
        );
    }
    if let Some(one_of) = schema.remove("oneOf") {
        let mut any_of = schema
            .remove("anyOf")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        any_of.extend(
            one_of
                .as_array()
                .context("oneOf must be an array")?
                .iter()
                .map(|value| strict_schema(value, root))
                .collect::<Result<Vec<_>>>()?,
        );
        schema.insert("anyOf".into(), Value::Array(any_of));
    }
    if let Some(all_of) = schema.remove("allOf") {
        let all_of = all_of.as_array().context("allOf must be an array")?;
        if all_of.len() == 1 {
            schema.extend(
                strict_schema(&all_of[0], root)?
                    .as_object()
                    .context("allOf entry must be an object")?
                    .clone(),
            );
        } else {
            schema.insert(
                "allOf".into(),
                Value::Array(
                    all_of
                        .iter()
                        .map(|value| strict_schema(value, root))
                        .collect::<Result<_>>()?,
                ),
            );
        }
    }
    schema.remove("default");
    if schema.len() > 1
        && let Some(reference) = schema.remove("$ref")
    {
        let reference = reference
            .as_str()
            .context("schema reference must be a string")?;
        ensure!(reference.starts_with("#/"), "unsupported schema reference");
        let resolved = root
            .pointer(reference.trim_start_matches('#'))
            .context("schema reference")?
            .as_object()
            .context("schema reference must resolve to an object")?;
        let mut merged = resolved.clone();
        merged.extend(schema);
        return strict_schema(&Value::Object(merged), root);
    }
    Ok(Value::Object(schema))
}

fn numeric(value: &Value) -> Result<f64> {
    let number = match value {
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Number(value) => value.as_f64().context("invalid number")?,
        Value::String(value) => {
            let text = value.trim();
            let bytes = text.as_bytes();
            for (index, byte) in bytes.iter().enumerate() {
                if *byte == b'_' {
                    ensure!(
                        index > 0
                            && index + 1 < bytes.len()
                            && bytes[index - 1].is_ascii_digit()
                            && bytes[index + 1].is_ascii_digit(),
                        "invalid numeric separator"
                    );
                }
            }
            text.replace('_', "")
                .parse()
                .context("invalid numeric string")?
        }
        _ => bail!("expected number"),
    };
    ensure!(number.is_finite(), "nonfinite number");
    Ok(number)
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
