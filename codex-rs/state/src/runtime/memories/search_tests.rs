use crate::StateRuntime;
use crate::runtime::test_support::unique_temp_dir;
use codex_utils_absolute_path::AbsolutePathBuf;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn search_snapshots_are_scoped_and_compare_and_swap() -> anyhow::Result<()> {
    let directory = unique_temp_dir();
    let home = AbsolutePathBuf::try_from(directory)?;
    let runtime =
        StateRuntime::init(crate::SqliteConfig::new_for_testing(home), "test".into()).await?;
    let store = runtime.memories();
    store.begin_search_extraction("run-a", 0).await?;
    store.commit_search_snapshot("run-a", 0, 0, "first").await?;
    assert_eq!(
        store.load_search_snapshot("run-a").await?,
        Some("first".into())
    );
    assert_eq!(store.load_search_snapshot("run-b").await?, None);
    store.begin_search_extraction("run-a", 1).await?;
    assert!(
        store
            .commit_search_snapshot("run-a", 1, 0, "stale")
            .await
            .is_err()
    );
    store
        .fail_search_extraction("run-a", 1, "invalid extraction")
        .await?;
    assert!(
        store
            .commit_search_snapshot("run-a", 1, 1, "failed job")
            .await
            .is_err()
    );
    assert_eq!(
        store.load_search_snapshot("run-a").await?,
        Some("first".into())
    );
    store.begin_search_extraction("run-b", 0).await?;
    store
        .commit_search_snapshot("run-b", 0, 0, "other run")
        .await?;
    store.begin_search_extraction("run-a", 2).await?;
    store
        .commit_search_snapshot("run-a", 2, 1, "second")
        .await?;
    assert_eq!(
        store.load_search_snapshot("run-a").await?,
        Some("second".into())
    );
    assert_eq!(
        store.load_search_snapshot("run-b").await?,
        Some("other run".into())
    );
    store.clear_memory_data().await?;
    assert_eq!(store.load_search_snapshot("run-a").await?, None);
    assert_eq!(store.load_search_snapshot("run-b").await?, None);
    Ok(())
}

#[tokio::test]
async fn search_snapshot_rejects_unclaimed_and_oversized_updates() -> anyhow::Result<()> {
    let directory = unique_temp_dir();
    let home = AbsolutePathBuf::try_from(directory)?;
    let runtime =
        StateRuntime::init(crate::SqliteConfig::new_for_testing(home), "test".into()).await?;
    let store = runtime.memories();
    assert!(
        store
            .commit_search_snapshot("run", 0, 0, "unclaimed")
            .await
            .is_err()
    );
    store.begin_search_extraction("run", 0).await?;
    assert!(
        store
            .commit_search_snapshot("run", 0, 0, &"x".repeat(65537))
            .await
            .is_err()
    );
    assert_eq!(store.load_search_snapshot("run").await?, None);
    Ok(())
}
