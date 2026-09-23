use super::*;
use pretty_assertions::assert_eq;

#[test]
fn matches_python_dataset_extraction() -> Result<()> {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/dataset_cases.json"))?;
    for case in cases {
        let result = parse(
            case["source"].as_str().unwrap().as_bytes(),
            Path::new(case["filename"].as_str().unwrap()),
            1,
        );
        if case["invalid"] == true {
            assert!(result.is_err(), "{case}");
        } else {
            assert_eq!(result?, case["result"], "{}", case["filename"]);
        }
    }
    Ok(())
}

#[tokio::test]
async fn selection_preserves_skip_and_fail_behavior() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let level = temporary.path().join("level1");
    tokio::fs::create_dir(&level).await?;
    let code = "def get_inputs(): return []\ndef get_init_inputs(): return []\n";
    tokio::fs::write(level.join("good.py"), code).await?;
    tokio::fs::write(level.join("bad.py"), "invalid = (").await?;
    let payload =
        json!({"dataset_root": temporary.path(), "dataset": "kernelbench", "task": "good"});
    assert_eq!(
        prepare(&payload).await?,
        parse(code.as_bytes(), &level.join("good.py").canonicalize()?, 1)?
    );
    let payload = json!({"dataset_root": level, "dataset": "kernelbench_veomni", "task": "good"});
    assert!(prepare(&payload).await.is_err());
    Ok(())
}
