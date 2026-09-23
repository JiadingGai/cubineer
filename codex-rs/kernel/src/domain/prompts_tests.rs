use super::*;
use pretty_assertions::assert_eq;
use sha2::Digest;
use sha2::Sha256;

#[test]
fn matches_pinned_python_composition() -> Result<()> {
    let cases: Value = serde_json::from_str(include_str!("fixtures/prompt_cases.json"))?;
    for case in cases["candidates"].as_array().unwrap() {
        let capability: Vec<_> = case["capability"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let output = candidate(
            case["mode"].as_str().unwrap(),
            case["baseline"].as_str().unwrap(),
            case["cutlass"].as_str(),
            &capability,
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(output.as_bytes())),
            case["sha256"].as_str().unwrap(),
            "{case}"
        );
    }
    for case in cases["references"].as_array().unwrap() {
        assert_eq!(
            reference(
                case["gpu"].as_str().unwrap(),
                case["ridge"].as_f64().unwrap()
            )?,
            case["result"].as_str().unwrap()
        );
    }
    Ok(())
}

#[test]
fn gpu_prompts_use_target_hardware() -> Result<()> {
    let payload = json!({"backend": "gpu"});
    for (name, capability, ridge) in [
        ("NVIDIA H100 80GB HBM3", [9, 0], 1979.0 / 3.35),
        ("NVIDIA H200", [9, 0], 1979.0 / 4.8),
        ("NVIDIA B200", [10, 0], 4500.0 / 8.0),
    ] {
        let mut problem = json!({"task_id": "example_triton"});
        populate(
            &mut problem,
            &payload,
            &json!({"name": name, "capability": capability}),
        )?;
        assert_eq!(problem["reference_instructions"], reference(name, ridge)?);
        assert_eq!(
            problem["instructions"]["initial"],
            candidate("initial", "triton", /*cutlass*/ None, &capability)?
        );
    }
    Ok(())
}

#[test]
fn simulated_prompts_do_not_require_a_gpu() -> Result<()> {
    let mut problem = json!({"task_id": "example"});
    populate(&mut problem, &json!({"backend": "simulated"}), &Value::Null)?;
    assert_eq!(
        problem["reference_instructions"],
        reference("NVIDIA H200 (simulated)", /*ridge*/ 412.0)?
    );
    Ok(())
}
