use super::*;
use pretty_assertions::assert_eq;

#[test]
fn runtime_defaults_match_original_entrypoint_and_preserve_overrides() {
    let mut config = SearchConfig {
        max_iterations: Some(4),
        use_recent_failed_only: Some(true),
        ..SearchConfig::default()
    };
    apply_runtime_defaults(&mut config);

    assert_eq!(
        config,
        SearchConfig {
            max_iterations: Some(4),
            num_candidates: Some(3),
            max_consecutive_failures: Some(3),
            speedup_threshold: Some(-100.0),
            use_recent_failed_only: Some(true),
            ..SearchConfig::default()
        }
    );
}

#[test]
fn candidate_mode_matches_original_parent_feedback_fallbacks() {
    let mut tree = SearchState::new(Strategy::Greedy, SearchConfig::default()).unwrap();
    let failed = tree
        .record_batch(
            None,
            vec![Outcome {
                status: Status::Failed,
                speedup: None,
                error_context: None,
            }],
        )
        .unwrap()[0];
    let successful = tree
        .record_batch(
            None,
            vec![Outcome {
                status: Status::Success,
                speedup: Some(2.0),
                error_context: None,
            }],
        )
        .unwrap()[0];
    let discarded = tree
        .record_batch(
            None,
            vec![Outcome {
                status: Status::Discarded,
                speedup: None,
                error_context: None,
            }],
        )
        .unwrap()[0];
    let mut candidates = BTreeMap::from([
        (failed, json!({"analysis": null})),
        (successful, json!({"analysis": null})),
        (discarded, json!({"analysis": {"severity": "stale"}})),
    ]);

    assert_eq!(candidate_mode(None, &tree, &candidates), "initial");
    assert_eq!(candidate_mode(Some(failed), &tree, &candidates), "repair");
    assert_eq!(
        candidate_mode(Some(successful), &tree, &candidates),
        "initial"
    );
    assert_eq!(
        candidate_mode(Some(discarded), &tree, &candidates),
        "initial"
    );
    candidates.insert(successful, json!({"analysis": {"severity": "high"}}));
    assert_eq!(
        candidate_mode(Some(successful), &tree, &candidates),
        "optimize"
    );
}

#[test]
fn candidate_evaluation_errors_become_failed_search_outcomes() {
    let evaluated = failed_evaluation("gpu", &anyhow::anyhow!("worker exited"));
    let outcome: Outcome = serde_json::from_value(evaluated["outcome"].clone()).unwrap();

    assert_eq!(outcome.status, Status::Failed);
    assert_eq!(outcome.speedup, None);
    assert_eq!(outcome.error_context.unwrap()["error_type"], "WorkerCrash");
    assert_eq!(evaluated["profile"], Value::Null);
    assert_eq!(evaluated["validation"], "gpu");
}
