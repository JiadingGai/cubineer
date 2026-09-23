use super::*;
use pretty_assertions::assert_eq;

#[test]
fn feedback_matches_original_formatters() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/feedback_cases.json")).unwrap();
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            feedback(&case["input"]).unwrap(),
            case["result"],
            "case {index}"
        );
    }
}

#[test]
fn successful_parent_without_profile_uses_fresh_approach_feedback() {
    let result = feedback(&json!({
        "parent": {
            "submission": {"code": "candidate"},
            "outcome": {"status": "success"},
            "analysis": null
        },
        "mode": "initial",
        "hint": null
    }))
    .unwrap();

    assert_eq!(
        result["text"],
        "\n**Previous Solution Context:**\nThe following solution was generated but needs a fresh approach:\n\n```python\ncandidate\n```\n\n**Task:** Generate a new solution with a different optimization strategy.\n"
    );
}
