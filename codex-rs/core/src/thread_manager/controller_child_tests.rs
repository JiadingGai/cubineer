use super::*;
use crate::config::RolloutBudgetConfig;
use crate::config::test_config;
use codex_protocol::protocol::TokenUsage;
use core_test_support::PathBufExt;
use core_test_support::PathExt;
use pretty_assertions::assert_eq;
use std::time::Duration;

#[tokio::test]
async fn controller_children_share_budget_but_not_history_or_execution_capacity()
-> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let mut config = test_config().await;
    config.codex_home = directory.path().join("home").abs();
    config.cwd = config.codex_home.abs();
    config.agent_max_threads = Some(1);
    config.rollout_budget = Some(RolloutBudgetConfig {
        limit_tokens: 100,
        reminder_at_remaining_tokens: vec![75],
        sampling_token_weight: 1.0,
        prefill_token_weight: 1.0,
    });
    std::fs::create_dir_all(&config.codex_home)?;
    let manager = ThreadManager::with_models_provider_and_home_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    );
    let parent = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await?;
    let other = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await?;
    config.cwd = directory.path().join("candidate").abs();
    std::fs::create_dir_all(&config.cwd)?;
    let child = manager
        .start_controller_child(parent.thread_id, StartThreadOptions::new(config.clone()))
        .await?;
    let snapshot = child.thread.config_snapshot().await;
    assert_eq!(snapshot.parent_thread_id, Some(parent.thread_id));
    assert_eq!(snapshot.forked_from_thread_id, None);
    assert_eq!(snapshot.cwd(), &config.cwd);
    assert_eq!(
        child.thread.multi_agent_version(),
        Some(MultiAgentVersion::Disabled)
    );
    assert!(
        manager
            .start_controller_child(parent.thread_id, StartThreadOptions::new(config.clone()))
            .await
            .is_err()
    );
    assert!(
        manager
            .start_controller_child(child.thread_id, StartThreadOptions::new(config.clone()))
            .await
            .is_err()
    );
    assert!(
        manager
            .close_controller_child(other.thread_id, child.thread_id)
            .await
            .is_err()
    );
    child
        .thread
        .session
        .services
        .agent_control
        .record_rollout_budget_usage(&TokenUsage {
            output_tokens: 25,
            ..Default::default()
        })?;
    let reminder = parent
        .thread
        .session
        .services
        .agent_control
        .pending_budget_reminder(parent.thread_id, "window")
        .expect("shared budget");
    assert_eq!(reminder.remaining_tokens, 75);
    manager
        .close_controller_child(parent.thread_id, child.thread_id)
        .await?;
    assert!(manager.get_thread(child.thread_id).await.is_err());
    let next = manager
        .start_controller_child(parent.thread_id, StartThreadOptions::new(config))
        .await?;
    assert_ne!(next.thread_id, child.thread_id);
    let shutdown = manager
        .shutdown_all_threads_bounded(Duration::from_secs(10))
        .await;
    assert!(shutdown.timed_out.is_empty());
    Ok(())
}
