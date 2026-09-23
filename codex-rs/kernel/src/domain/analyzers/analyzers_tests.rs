use super::*;
use pretty_assertions::assert_eq;
use sha2::Digest;
use sha2::Sha256;

#[test]
fn empty_sass_report_is_absent_evidence() -> Result<()> {
    for name in ["wgmma_instruction_detector", "register_spill_detector"] {
        let report =
            |sass| json!({"name": name, "report": {"metrics": {"kernel": {}}, "sass": sass}});
        assert_eq!(invoke(&report(json!({})))?, invoke(&report(Value::Null))?);
    }
    Ok(())
}

#[test]
fn tool_outputs_match_python_thresholds_and_wording() -> Result<()> {
    let cases: Vec<Value> = serde_json::from_str(include_str!("../fixtures/analyzer_cases.json"))?;
    for case in cases {
        let output = invoke(&case["input"])?;
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(output["text"].as_str().unwrap().as_bytes())
            ),
            case["sha256"].as_str().unwrap(),
            "input={}\noutput={}",
            case["input"],
            output["text"]
        );
    }
    Ok(())
}
