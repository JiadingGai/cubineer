use super::*;

impl ThreadManager {
    /// Close an owned child through the native agent lifecycle and release its slot.
    pub async fn close_controller_child(
        &self,
        parent_id: ThreadId,
        child_id: ThreadId,
    ) -> CodexResult<()> {
        let parent = self.get_thread(parent_id).await?;
        let child = self.get_thread(child_id).await?;
        if child.session_source.parent_thread_id() != Some(parent_id) {
            return Err(CodexErr::InvalidRequest(
                "child is not owned by this parent".into(),
            ));
        }
        parent
            .session
            .services
            .agent_control
            .close_agent(child_id)
            .await?;
        Ok(())
    }

    /// Start an idle, fresh-history child owned by a host-controlled root session.
    pub async fn start_controller_child(
        &self,
        parent_thread_id: ThreadId,
        mut options: StartThreadOptions,
    ) -> CodexResult<NewThread> {
        let parent = self.get_thread(parent_thread_id).await?;
        if !parent.is_running()
            || parent.session_source.parent_thread_id().is_some()
            || !matches!(options.initial_history, InitialHistory::New)
            || options.allow_provider_model_fallback
        {
            return Err(CodexErr::InvalidRequest(
                "controller children require a live root, fresh history, and no model fallback"
                    .into(),
            ));
        }
        options
            .config
            .features
            .disable(codex_features::Feature::MultiAgentV2)
            .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;
        options.config.agents_enabled = false;
        parent
            .session
            .services
            .agent_control
            .spawn_controller_agent(options, parent_thread_id)
            .await
    }
}

#[cfg(test)]
#[path = "controller_child_tests.rs"]
mod tests;
