use super::*;
use pretty_assertions::assert_eq;

#[test]
fn validation_matches_python_contract() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/model_cases.json")).unwrap();
    for (index, case) in cases.iter().enumerate() {
        let actual = validate(case["role"].as_str().unwrap(), &case["value"]);
        if case["invalid"] == true {
            assert!(actual.is_err(), "case {index}: {case}");
        } else {
            assert_eq!(actual.unwrap(), case["result"], "case {index}: {case}");
        }
    }
}

#[test]
fn output_schemas_match_agents_sdk_strict_contract() {
    let schemas = schemas();
    let validation: Value = serde_json::from_str(include_str!("fixtures/models.json")).unwrap();
    let bottleneck = &validation["reference"]["validation"]["$defs"]["BottleneckType"];
    let reference = &schemas["reference"];
    let mut expected = bottleneck.as_object().unwrap().clone();
    expected.insert(
        "description".into(),
        validation["reference"]["validation"]["properties"]["final_bottleneck"]["description"]
            .clone(),
    );

    assert_eq!(
        reference["properties"]["final_bottleneck"],
        Value::Object(expected)
    );
    assert!(
        reference["properties"]["invalid_reference_code"]
            .get("default")
            .is_none()
    );
    assert_eq!(
        schemas["candidate"]["example"],
        validation["candidate"]["validation"]["example"]
    );
    assert_wire_schema(&schemas);
    assert!(strict_schema(&serde_json::json!({}), &serde_json::json!({})).is_ok());
}

fn assert_wire_schema(value: &Value) {
    match value {
        Value::Object(object) => {
            assert!(!object.contains_key("default"));
            if object.get("type").and_then(Value::as_str) == Some("object") {
                let properties = object["properties"].as_object().unwrap();
                let required = object["required"].as_array().unwrap();
                assert_eq!(object["additionalProperties"], false);
                assert_eq!(required.len(), properties.len());
                for name in properties.keys() {
                    assert!(
                        required
                            .iter()
                            .any(|required| required.as_str() == Some(name))
                    );
                }
            }
            for child in object.values() {
                assert_wire_schema(child);
            }
        }
        Value::Array(array) => {
            for child in array {
                assert_wire_schema(child);
            }
        }
        _ => {}
    }
}
