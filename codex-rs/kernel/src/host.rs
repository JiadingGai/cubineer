mod startup;

use crate::cli::OptimizeArgs;
use crate::worker::Worker;
use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use codex_app_server_client::EnvironmentManager;
use codex_app_server_client::ExecServerRuntimePaths;
use codex_app_server_client::InProcessAppServerClient;
use codex_app_server_client::InProcessClientStartArgs;
use codex_app_server_client::InProcessServerEvent;
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::ServerNotification;
use codex_app_server_protocol::ServerRequest;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::TurnStatus;
use codex_arg0::Arg0DispatchPaths;
use codex_config::CloudConfigBundleLoader;
use codex_config::LoaderOverrides;
use codex_core::config::ConfigBuilder;
use codex_core::config::ConfigOverrides;
use codex_core::context::ContextualUserFragment;
use codex_core::context::KernelContextFragment;
use codex_feedback::CodexFeedback;
use codex_protocol::protocol::SessionSource;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

pub(crate) async fn rpc(
    client: &InProcessAppServerClient,
    method: &str,
    params: Value,
) -> Result<Value> {
    let request: ClientRequest = serde_json::from_value(json!({
        "id": NEXT_REQUEST.fetch_add(1, Ordering::Relaxed), "method": method, "params": params,
    }))?;
    Ok(client.request_typed(request).await?)
}

pub(crate) struct Host {
    pub client: InProcessAppServerClient,
    pub state: Option<Arc<codex_state::StateRuntime>>,
    pub root: String,
    pub provider: String,
    pub model: String,
    pub timeout: Duration,
    pub usage: BTreeMap<String, Value>,
    parallel_sessions: usize,
}

#[derive(Clone)]
pub(crate) struct SessionSpec {
    pub role: String,
    pub model: String,
    pub cwd: PathBuf,
    pub instructions: String,
    pub input: String,
    pub schema: Value,
    pub tools: Vec<Value>,
    pub evidence: Value,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SessionResult {
    pub thread_id: String,
    pub turn_id: String,
    pub role: String,
    pub model: String,
    pub value: Option<Value>,
    pub error: Option<String>,
    pub events: Vec<Value>,
}

impl Host {
    pub async fn batch(
        &mut self,
        specs: Vec<SessionSpec>,
        worker: &Worker,
    ) -> Result<Vec<SessionResult>> {
        let mut results = Vec::with_capacity(specs.len());
        for chunk in specs.chunks(self.parallel_sessions) {
            results.extend(self.active_batch(chunk, worker).await?);
        }
        Ok(results)
    }

    async fn active_batch(
        &mut self,
        specs: &[SessionSpec],
        worker: &Worker,
    ) -> Result<Vec<SessionResult>> {
        let mut results = Vec::new();
        let mut active = BTreeMap::new();
        let execution = async {
        for (index, spec) in specs.iter().enumerate() {
            tokio::fs::create_dir_all(&spec.cwd).await?;
            tokio::fs::write(spec.cwd.join("domain-instructions.md"), &spec.instructions).await?;
            tokio::fs::write(spec.cwd.join("task-context.txt"), &spec.input).await?;
            let instructions = KernelContextFragment { text: spec.instructions.clone(), file: "domain-instructions.md".into() }.body();
            let input = KernelContextFragment { text: spec.input.clone(), file: "task-context.txt".into() }.body();
            let child = rpc(&self.client, "thread/startChild", json!({
                "parentThreadId": self.root,
                "startup": {"model": spec.model, "modelProvider": self.provider,
                    "cwd": spec.cwd, "runtimeWorkspaceRoots": [spec.cwd],
                    "developerInstructions": instructions, "dynamicTools": spec.tools,
                    "sandbox": "workspace-write",
                    "config": {"agents.enabled": false, "features.multi_agent_v2": false,
                               "features.memories": false, "sandbox_workspace_write.writable_roots": [],
                               "sandbox_workspace_write.exclude_tmpdir_env_var": true,
                               "sandbox_workspace_write.exclude_slash_tmp": true}},
            })).await?;
            ensure!(child["model"] == spec.model && child["modelProvider"] == self.provider,
                "resolved child provider/model differs from requested configuration");
            let thread = child["thread"]["id"].as_str().context("missing child ID")?.to_owned();
            results.push(SessionResult { thread_id: thread.clone(), turn_id: String::new(), role: spec.role.clone(),
                model: spec.model.clone(), value: None, error: None, events: Vec::new() });
            let turn = rpc(&self.client, "turn/start", json!({
                "threadId": thread, "input": [{"type": "text", "text": input, "textElements": []}],
                "outputSchema": spec.schema,
            })).await?;
            let turn_id = turn["turn"]["id"].as_str().context("missing turn ID")?.to_owned();
            active.insert(thread.clone(), index);
            results[index].turn_id = turn_id;
        }
        let deadline = tokio::time::sleep(self.timeout);
        tokio::pin!(deadline);
        while !active.is_empty() {
            let event = tokio::select! {
                event = self.client.next_event() => Some(event.context("app-server disconnected")?),
                _ = &mut deadline => {
                    let timed_out = active.clone();
                    for (thread, index) in timed_out {
                        let interrupt = rpc(&self.client, "turn/interrupt", json!({
                            "threadId": thread, "turnId": results[index].turn_id,
                        })).await;
                        results[index].error = Some(match interrupt {
                            Ok(_) => format!("session exceeded its {}s wall-clock budget", self.timeout.as_secs()),
                            Err(error) => format!("session exceeded its {}s wall-clock budget; interrupt failed: {error}", self.timeout.as_secs()),
                        });
                    }
                    active.clear();
                    None
                }
                _ = tokio::signal::ctrl_c() => bail!("kernel run cancelled"),
            };
            let Some(event) = event else {
                continue;
            };
            match event {
                InProcessServerEvent::Lagged { skipped } => bail!("lost {skipped} session events; refusing incomplete evidence"),
                InProcessServerEvent::ServerRequest(request) => match *request {
                    ServerRequest::DynamicToolCall { request_id, params } => {
                        let index = *active.get(&params.thread_id).context("tool call outside active batch")?;
                        let spec = &specs[index];
                        ensure!(spec.tools.iter().any(|tool| tool["name"] == params.tool), "unregistered domain tool");
                        ensure!(params.arguments == json!({}), "domain tools take no arguments");
                        let value = if params.tool == "profile_reference_with_ncu" {
                            spec.evidence.clone()
                        } else {
                            worker.call(&self.client, "tool", json!({"name": params.tool, "report": spec.evidence})).await?
                        };
                        results[index].events.push(json!({"tool": params.tool, "evidence": value}));
                        let evidence_file = format!("tool-{}.json", params.tool);
                        let text = serde_json::to_string(&value)?;
                        tokio::fs::write(spec.cwd.join(&evidence_file), &text).await?;
                        let text = KernelContextFragment { text, file: evidence_file }.body();
                        self.client.resolve_server_request(request_id, json!({
                            "success": true, "contentItems": [{"type": "inputText", "text": text}],
                        })).await?;
                    }
                    request => {
                        self.client.reject_server_request(request.id().clone(), codex_app_server_protocol::JSONRPCErrorError {
                            code: -32000, message: "This headless run cannot answer an interactive approval; use an appropriate existing Codex policy.".into(), data: None,
                        }).await?;
                    }
                },
                InProcessServerEvent::ServerNotification(notification) => match *notification {
                    ServerNotification::ItemCompleted(item) => {
                        if let Some(index) = active.get(&item.thread_id) {
                            results[*index].events.push(serde_json::to_value(&item.item)?);
                            if let ThreadItem::AgentMessage { text, phase, .. } = item.item
                                && phase != Some(codex_protocol::models::MessagePhase::Commentary)
                                && let Ok(value) = serde_json::from_str(&text)
                            {
                                results[*index].value = Some(value);
                            }
                        }
                    }
                    ServerNotification::TurnCompleted(turn) => {
                        if let Some(index) = active.remove(&turn.thread_id) {
                            if turn.turn.status != TurnStatus::Completed {
                                results[index].error = Some(serde_json::to_string(&turn.turn)?);
                            } else if results[index].value.is_none() {
                                results[index].error = Some("missing structured submission".into());
                            }
                        }
                    }
                    ServerNotification::ThreadTokenUsageUpdated(usage) => {
                        self.usage.insert(usage.thread_id, serde_json::to_value(usage.token_usage)?);
                    }
                    _ => {}
                },
            }
        }
        Ok::<_, anyhow::Error>(())
        }.await;
        let mut cleanup_error = None;
        for result in &results {
            for (method, params) in [
                (
                    "thread/closeChild",
                    json!({"parentThreadId": self.root, "threadId": result.thread_id}),
                ),
                ("thread/archive", json!({"threadId": result.thread_id})),
            ] {
                if let Err(error) = rpc(&self.client, method, params).await {
                    cleanup_error = Some(error);
                }
            }
        }
        execution?;
        if let Some(error) = cleanup_error {
            return Err(error);
        }
        if let Some(result) = results.iter().find(|result| {
            result.role != "candidate" && result.error.is_some() && result.value.is_none()
        }) {
            bail!(
                "{} session {} failed: {}",
                result.role,
                result.thread_id,
                result.error.as_deref().unwrap_or_default()
            );
        }
        Ok(results)
    }
}

pub(crate) fn tool(name: &str, description: &str) -> Value {
    json!({"type": "function", "name": name, "description": description,
        "inputSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}})
}
