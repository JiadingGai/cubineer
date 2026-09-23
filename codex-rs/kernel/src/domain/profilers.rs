use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use regex::Regex;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::sync::LazyLock;

pub(crate) static ASSETS: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("fixtures/profilers.json"))
        .unwrap_or_else(|error| panic!("bundled profiler assets must parse: {error}"))
});

pub(crate) fn sass(source: &str, kernel: Option<&str>) -> Result<Value> {
    let mut text = source;
    let mut name = "";
    if let Some(kernel) = kernel.filter(|name| !name.is_empty())
        && let Some(start) = source.find(&format!(".text.{kernel}"))
    {
        let tail = &source[start + 6 + kernel.len()..];
        let end = tail
            .find(".text.")
            .map_or(source.len(), |offset| start + 6 + kernel.len() + offset);
        text = &source[start..end];
        name = kernel;
    }
    let mut output = json!({"sass_text": source, "kernel_name": name, "uses_cublas": false, "cublas_symbols": []});
    for (name, pattern) in ASSETS["sass_patterns"]
        .as_object()
        .context("SASS patterns")?
    {
        output[format!("{name}_count")] = Regex::new(&format!(
            "(?i){}",
            pattern.as_str().context("SASS pattern")?
        ))?
        .find_iter(text)
        .count()
        .into();
    }
    Ok(output)
}

pub(crate) fn rules(source: &str) -> Result<Value> {
    let pattern = Regex::new(r"Est\.\s*(?:Local\s*)?Speedup:\s*([0-9]+(?:\.[0-9]+)?)\s*%")?;
    let lines: Vec<_> = source.lines().collect();
    let mut output = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(capture) = pattern.captures(lines[index]) else {
            index += 1;
            continue;
        };
        let percentage: f64 = capture[1].parse()?;
        index += 1;
        let mut body = Vec::new();
        while index < lines.len() {
            let line = lines[index].trim();
            if line.is_empty()
                || line.starts_with("OPT")
                || line.starts_with("Section:")
                || line.starts_with("----")
                || pattern.is_match(line)
            {
                break;
            }
            body.push(line);
            index += 1;
        }
        if !body.is_empty() {
            output.push((percentage, body.join(" ")));
        }
    }
    output.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(output
        .into_iter()
        .map(|(percentage, text)| json!({"est_speedup_pct": percentage, "text": text}))
        .collect())
}

fn is_missing(value: &str) -> bool {
    matches!(
        value,
        "" | "NaN"
            | "nan"
            | "NA"
            | "N/A"
            | "NULL"
            | "null"
            | "None"
            | "#N/A"
            | "#NA"
            | "<NA>"
            | "-NaN"
            | "-nan"
            | "#N/A N/A"
            | "n/a"
            | "-1.#IND"
            | "1.#IND"
            | "-1.#QNAN"
            | "1.#QNAN"
    )
}

pub(crate) fn ncu_csv(source: &str, names: &[String]) -> Result<Value> {
    // Preserve pandas' physical skiprows=[1], including status-line interactions.
    let mut cleaned = String::new();
    let mut quoted = false;
    for (index, line) in source.lines().enumerate() {
        if index == 1 {
            continue;
        }
        for character in line.chars() {
            if character == '"' {
                quoted = !quoted;
            }
            if character == '=' && !quoted {
                break;
            }
            cleaned.push(character);
        }
        cleaned.push('\n');
    }
    let mut reader = csv::ReaderBuilder::new()
        .flexible(/*yes*/ true)
        .from_reader(cleaned.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Ok(json!({})),
    };
    let Some(kernel_column) = headers.iter().position(|column| column == "Kernel Name") else {
        return Ok(json!({}));
    };
    let rows = match reader.records().collect::<std::result::Result<Vec<_>, _>>() {
        Ok(rows) => rows,
        Err(_) => return Ok(json!({})),
    };
    if rows.iter().any(|row| row.len() > headers.len()) {
        return Ok(json!({}));
    }
    let mut last_rows = Map::new();
    for row in &rows {
        let raw_name = row.get(kernel_column).unwrap_or_default();
        let name = if is_missing(raw_name) {
            "nan"
        } else {
            raw_name
        };
        last_rows.insert(name.to_owned(), Value::Null);
    }
    let mut selected = Vec::new();
    let requested: Vec<_> = if names.is_empty() {
        last_rows
            .keys()
            .filter(|name| name.as_str() != "nan" && !name.trim().is_empty())
            .cloned()
            .collect()
    } else {
        names.to_vec()
    };
    for name in requested {
        let row = rows.iter().rfind(|row| {
            let value = row.get(kernel_column).unwrap_or_default();
            let value = if is_missing(value) { "nan" } else { value };
            if names.is_empty() {
                value == name
            } else {
                value.contains(&name)
            }
        });
        let Some(row) = row else {
            continue;
        };
        let mut metrics = Map::new();
        for (column, header) in headers.iter().enumerate() {
            if column == kernel_column {
                continue;
            }
            let value = row.get(column).unwrap_or_default();
            ensure!(!is_missing(value), "nonfinite NCU metric {header}");
            let cleaned = value.replace([',', '%'], "");
            let cleaned = cleaned.trim();
            let value = if let Ok(number) = cleaned.parse::<i64>() {
                Value::from(number)
            } else if let Ok(number) = cleaned.parse::<u64>() {
                Value::from(number)
            } else {
                match cleaned.parse::<f64>() {
                    Ok(number) => {
                        ensure!(number.is_finite(), "nonfinite NCU metric {header}");
                        Value::from(number)
                    }
                    Err(_) => Value::from(cleaned),
                }
            };
            metrics.insert(header.to_owned(), value);
        }
        selected.push((name, Value::Object(metrics)));
    }
    if names.is_empty() {
        selected.sort_by(|a, b| {
            b.1["gpu__time_duration.sum"]
                .as_f64()
                .unwrap_or_default()
                .total_cmp(&a.1["gpu__time_duration.sum"].as_f64().unwrap_or_default())
        });
        selected.truncate(5);
    }
    Ok(Value::Object(selected.into_iter().collect()))
}

#[cfg(test)]
#[path = "profilers_tests.rs"]
mod tests;
