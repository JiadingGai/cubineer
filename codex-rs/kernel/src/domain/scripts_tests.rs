use super::*;
use pretty_assertions::assert_eq;
use sha2::Digest;
use sha2::Sha256;

#[test]
fn workload_script_snapshots_are_current() -> Result<()> {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/script_cases.json"))?;
    for case in cases {
        let output = render(
            case["name"].as_str().unwrap(),
            &serde_json::from_value(case["values"].clone())?,
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(output.as_bytes())),
            case["sha256"].as_str().unwrap()
        );
    }
    Ok(())
}
