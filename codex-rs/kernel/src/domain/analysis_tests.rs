use super::*;
use pretty_assertions::assert_eq;
use sha2::Digest;
use sha2::Sha256;

#[test]
fn analysis_matches_python_in_both_modes() -> Result<()> {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/analysis_cases.json"))?;
    for case in cases {
        let mut result = analyze(&case["input"])?;
        let text = result.as_object_mut().unwrap().remove("text").unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(text.as_str().unwrap().as_bytes())),
            case["sha256"].as_str().unwrap(),
            "{}: {}",
            case["input"],
            text
        );
        assert_eq!(result, case["result"]);
    }
    Ok(())
}
