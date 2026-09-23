use super::*;
use pretty_assertions::assert_eq;

fn success(speedup: f64) -> Outcome {
    Outcome {
        status: Status::Success,
        speedup: Some(speedup),
        error_context: None,
    }
}

fn failure() -> Outcome {
    Outcome {
        status: Status::Failed,
        speedup: None,
        error_context: None,
    }
}

#[test]
fn failed_repairs_prune_only_after_the_whole_batch_is_attached() {
    let mut tree = SearchState::new(Strategy::Mcts, SearchConfig::default()).unwrap();
    let parent = tree.record_batch(None, vec![failure()]).unwrap()[0];
    tree.record_batch(Some(parent), vec![failure(), failure()])
        .unwrap();
    assert!(!tree.nodes[parent].terminal);
    tree.record_batch(Some(parent), vec![failure(), success(1.2)])
        .unwrap();
    assert!(!tree.nodes[parent].terminal);
    assert_eq!((tree.nodes[ROOT].visits, tree.nodes[parent].visits), (5, 5));
}

#[test]
fn three_failed_repairs_prune_and_allow_a_root_restart() {
    let mut tree = SearchState::new(Strategy::Mcts, SearchConfig::default()).unwrap();
    let parent = tree.record_batch(None, vec![failure()]).unwrap()[0];
    tree.record_batch(Some(parent), vec![failure(), failure(), failure()])
        .unwrap();
    assert!(tree.nodes[parent].terminal);
    assert_eq!(tree.select(), None);
    assert!(tree.nodes[ROOT].terminal);
    tree.record_batch(None, vec![success(2.0)]).unwrap();
    assert_eq!(tree.winner(), Some(5));
    assert!(tree.nodes[ROOT].terminal);
}

#[test]
fn mcts_current_best_and_final_winner_have_different_clamping() {
    let mut tree = SearchState::new(Strategy::Mcts, SearchConfig::default()).unwrap();
    tree.record_batch(None, vec![success(0.02), success(0.08)])
        .unwrap();
    assert_eq!((tree.current_best, tree.winner()), (Some(1), Some(2)));
}

#[test]
fn greedy_convergence_uses_batch_best_not_global_best() {
    let mut tree = SearchState::new(Strategy::Greedy, SearchConfig::default()).unwrap();
    tree.record_batch(None, vec![success(2.0)]).unwrap();
    tree.record_batch(Some(1), vec![success(1.5)]).unwrap();
    assert_eq!(
        (tree.should_continue(), tree.winner(), tree.speedup_history),
        (false, Some(1), vec![2.0, 1.5])
    );
}

#[test]
fn invalid_batch_is_atomic() {
    let mut tree = SearchState::new(Strategy::Mcts, SearchConfig::default()).unwrap();
    let before = tree.clone();
    assert!(
        tree.record_batch(None, vec![success(2.0), success(f64::NAN)])
            .is_err()
    );
    assert_eq!(tree, before);
}

#[test]
fn serialized_state_preserves_selection() {
    let mut tree = SearchState::new(Strategy::Mcts, SearchConfig::default()).unwrap();
    tree.record_batch(None, vec![failure(), success(1.5)])
        .unwrap();
    let mut restored: SearchState =
        serde_json::from_str(&serde_json::to_string(&tree).unwrap()).unwrap();
    assert_eq!(tree.select(), restored.select());
    assert_eq!(tree, restored);
}

#[test]
fn accepts_original_progressive_widening_config_names() {
    let config: SearchConfig = serde_json::from_value(serde_json::json!({
        "pw_C_root": 4.0,
        "pw_C_repair": 5.0,
        "pw_C_optimize": 6.0
    }))
    .unwrap();

    assert_eq!(config.pw_c_root, Some(4.0));
    assert_eq!(config.pw_c_repair, Some(5.0));
    assert_eq!(config.pw_c_optimize, Some(6.0));
}
