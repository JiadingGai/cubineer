use super::MemoryStore;
use anyhow::Result;
use anyhow::ensure;

impl MemoryStore {
    pub async fn load_search_snapshot(&self, run_id: &str) -> Result<Option<String>> {
        Ok(sqlx::query_scalar("SELECT snapshot FROM search_memory_snapshots WHERE run_id = ? ORDER BY version DESC LIMIT 1")
            .bind(run_id).fetch_optional(self.pool.as_ref()).await?)
    }

    pub async fn begin_search_extraction(&self, run_id: &str, batch: i64) -> Result<()> {
        ensure!(
            !run_id.is_empty() && batch >= 0,
            "invalid search memory scope"
        );
        sqlx::query("INSERT INTO jobs(kind, job_key, status, started_at, retry_remaining) VALUES('kernel_search_extract', ?, 'running', unixepoch(), 0)")
            .bind(format!("{run_id}:{batch}")).execute(self.pool.as_ref()).await?;
        Ok(())
    }

    pub async fn fail_search_extraction(
        &self,
        run_id: &str,
        batch: i64,
        error: &str,
    ) -> Result<()> {
        sqlx::query("UPDATE jobs SET status = 'failed', last_error = ?, finished_at = unixepoch() WHERE kind = 'kernel_search_extract' AND job_key = ? AND status = 'running'")
            .bind(error).bind(format!("{run_id}:{batch}")).execute(self.pool.as_ref()).await?;
        Ok(())
    }

    pub async fn commit_search_snapshot(
        &self,
        run_id: &str,
        batch: i64,
        previous_version: i64,
        snapshot: &str,
    ) -> Result<()> {
        ensure!(
            snapshot.len() <= 64 * 1024,
            "search memory snapshot is too large"
        );
        ensure!(previous_version >= 0, "invalid search memory version");
        let mut tx = self.pool.begin().await?;
        let version: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(version), 0) FROM search_memory_snapshots WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_one(&mut *tx)
        .await?;
        ensure!(version == previous_version, "stale search memory update");
        let updated = sqlx::query("UPDATE jobs SET status = 'done', finished_at = unixepoch() WHERE kind = 'kernel_search_extract' AND job_key = ? AND status = 'running'")
            .bind(format!("{run_id}:{batch}")).execute(&mut *tx).await?.rows_affected();
        ensure!(updated == 1, "search memory extraction is not active");
        sqlx::query("INSERT INTO search_memory_snapshots(run_id, version, snapshot, created_at) VALUES(?, ?, ?, unixepoch())")
            .bind(run_id).bind(version + 1).bind(snapshot).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
