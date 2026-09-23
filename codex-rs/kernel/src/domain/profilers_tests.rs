use super::*;
use pretty_assertions::assert_eq;

#[test]
fn profiler_parsers_match_python() -> Result<()> {
    let cases: Value = serde_json::from_str(include_str!("fixtures/profiler_cases.json"))?;
    for case in cases["csv"].as_array().unwrap() {
        let names = serde_json::from_value::<Vec<String>>(case["names"].clone())?;
        let result = ncu_csv(case["csv"].as_str().unwrap(), &names);
        if case["invalid"] == true {
            assert!(result.is_err(), "{case}");
        } else {
            let result = result?;
            assert_eq!(result, case["result"], "{case}");
            assert_eq!(
                json!(result.as_object().unwrap().keys().collect::<Vec<_>>()),
                case["order"]
            );
        }
    }
    for case in cases["sass"].as_array().unwrap() {
        assert_eq!(
            sass(case["source"].as_str().unwrap(), case["kernel"].as_str())?,
            case["result"]
        );
    }
    for case in cases["rules"].as_array().unwrap() {
        assert_eq!(rules(case["source"].as_str().unwrap())?, case["result"]);
    }
    Ok(())
}
