use crate::cli::OptimizeArgs;
use crate::host::Host;
use crate::host::SessionSpec;
use crate::memory::Snapshot;
use crate::roles::classify;
use crate::roles::evaluate;
use crate::roles::profile;
use crate::roles::update_memory;
use crate::worker::Worker;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use codex_arg0::Arg0DispatchPaths;
use codex_kernel_search::Outcome;
use codex_kernel_search::SearchConfig;
use codex_kernel_search::SearchState;
use codex_kernel_search::Status;
use codex_kernel_search::Strategy;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) async fn save(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let temporary = path.with_extension("pending");
    tokio::fs::write(&temporary, serde_json::to_vec_pretty(value)?).await?;
    tokio::fs::rename(temporary, path).await?;
    Ok(())
}

pub(crate) async fn run(
    mut args: OptimizeArgs,
    paths: Arg0DispatchPaths,
    overrides: codex_utils_cli::CliConfigOverrides,
) -> Result<()> {
    ensure!(
        !args.output.exists(),
        "output already exists; select a new run directory"
    );
    ensure!(
        args.timeout_seconds > 0 && !args.gpus.is_empty(),
        "invalid worker budget"
    );
    ensure!(
        args.backend() != "simulated" || args.scenario.is_some(),
        "simulated evaluation requires --scenario"
    );
    if args.backend() == "gpu" {
        let python = args
            .python
            .as_ref()
            .context("GPU workloads require --python")?;
        ensure!(python.is_file(), "configured venv Python not found");
        args.python = Some(std::path::absolute(python)?);
        args.worker = Some(
            args.worker
                .as_ref()
                .context("GPU workloads require --worker")?
                .canonicalize()?,
        );
    }
    args.dataset_root = args.dataset_root.canonicalize()?;
    args.cutlass_root = args
        .cutlass_root
        .map(|path| path.canonicalize())
        .transpose()?;
    tokio::fs::create_dir_all(&args.output).await?;
    args.output = args.output.canonicalize()?;
    let worker = Worker::new(
        args.python.clone(),
        args.worker.clone(),
        args.output.join("requests"),
        args.timeout_seconds * 1000,
    );
    let mut host = match Host::start(&args, paths, overrides).await {
        Ok(host) => host,
        Err(error) => {
            save(
                &args.output.join("completion.json"),
                &json!({"status": "failed", "phase": "startup", "error": error.to_string()}),
            )
            .await?;
            return Err(error);
        }
    };
    let result = tokio::select! {
        result = optimize(&args, &worker, &mut host) => result,
        _ = tokio::signal::ctrl_c() => Err(anyhow::anyhow!("kernel run cancelled")),
    };
    let completion = save(
        &args.output.join("completion.json"),
        &json!({
            "status": if result.is_ok() {"completed"} else {"failed"},
            "error": result.as_ref().err().map(ToString::to_string), "usage": host.usage,
            "root_thread_id": host.root, "validation": args.backend(),
        }),
    )
    .await;
    let shutdown = host.client.shutdown().await;
    result?;
    completion?;
    shutdown?;
    println!("Kernel search artifacts: {}", args.output.display());
    Ok(())
}

async fn optimize(args: &OptimizeArgs, worker: &Worker, host: &mut Host) -> Result<()> {
    let strategy = match args.strategy.as_str() {
        "mcts" => Strategy::Mcts,
        _ => Strategy::Greedy,
    };
    let mut config: SearchConfig = match &args.search_config {
        Some(path) => serde_json::from_slice(&tokio::fs::read(path).await?)?,
        None => SearchConfig::default(),
    };
    apply_runtime_defaults(&mut config);
    if let Some(iterations) = args.iterations {
        config.max_iterations = Some(iterations);
    }
    if let Some(candidates) = args.candidates {
        config.num_candidates = Some(candidates);
    }
    ensure!(
        config.num_candidates() > 0,
        "at least one candidate is required"
    );
    let mut tree = SearchState::new(strategy, config)?;
    let scenario: Value = match &args.scenario {
        Some(path) => serde_json::from_slice(&tokio::fs::read(path).await?)?,
        None => Value::Null,
    };
    let problem = worker
        .call(
            &host.client,
            "prepare",
            json!({
                "dataset_root": args.dataset_root, "dataset": args.dataset,
                "task": args.task, "level": args.level, "backend": args.backend(),
                "cutlass_root": args.cutlass_root, "gpu_id": args.gpus[0],
            }),
        )
        .await?;
    save(&args.output.join("task.json"), &problem).await?;
    save(
        &args.output.join("run.json"),
        &json!({
            "root_thread_id": host.root, "provider": host.provider, "model": host.model,
            "reference_model": args.reference_model.as_ref().unwrap_or(&host.model),
            "profile_model": args.profile_model.as_ref().unwrap_or(&host.model),
            "memory_model": args.memory_model.as_ref().unwrap_or(&host.model),
            "strategy": strategy, "search_config": tree.config, "evaluator": args.backend(),
            "profiling": args.profiling, "ncu_full": args.ncu_full, "memory_off": args.memory_off,
            "task_sha256": problem["sha256"],
        }),
    )
    .await?;
    let reference = worker.call(&host.client, "reference", json!({
        "backend": args.backend(), "problem": problem, "output_dir": args.output.join("reference"),
        "fixture": scenario["reference"], "ncu_full": args.ncu_full, "gpu_id": args.gpus[0],
    })).await?;
    save(&args.output.join("reference.json"), &reference).await?;
    let classification = classify(args, worker, host, &problem, &reference).await;
    let bottleneck = classification
        .as_ref()
        .ok()
        .and_then(|value| value.get("final_bottleneck"))
        .cloned();
    save(&args.output.join("classification.json"), &json!({
        "analysis": classification.as_ref().ok(), "error": classification.as_ref().err().map(ToString::to_string),
    })).await?;
    let mut candidates: BTreeMap<usize, Value> = BTreeMap::new();
    let mut snapshot = Snapshot::default();
    loop {
        let iteration = tree.current_iteration;
        let parent = if iteration == 0 { None } else { tree.select() };
        let mode = candidate_mode(parent, &tree, &candidates);
        let parent_context = parent.and_then(|id| candidates.get(&id)).map(|value| {
            json!({
                "submission": value["submission"], "outcome": value["evaluation"]["outcome"],
                "analysis": value["analysis"], "validation": value["evaluation"]["validation"],
            })
        });
        let feedback = worker
            .call(
                &host.client,
                "feedback",
                json!({"parent": parent_context, "mode": mode, "hint": args.hint}),
            )
            .await?;
        let specs = (0..tree.config.num_candidates()).map(|index| SessionSpec {
            role: "candidate".into(), model: host.model.clone(),
            cwd: args.output.join(format!("workspaces/candidate-{iteration}-{index}")),
            instructions: problem["instructions"][mode].as_str().unwrap_or_default().to_owned(),
            input: serde_json::to_string(&json!({
                "reference": problem["canonical_solution"], "task": problem["description"],
                "parent": parent_context, "reference_analysis": classification.as_ref().ok(),
                "memory": if args.memory_off { String::new() } else { snapshot.context(&host.root) },
                "feedback": feedback["text"],
                "submission": "Return one complete CodeCompletion. Evaluation is controller-owned; do not run full candidate evaluation or modify evaluator rules. Use native file/search tools for source research. No delegation or sibling communication.",
            })).unwrap_or_default(),
            schema: problem["schemas"]["candidate"].clone(), tools: Vec::new(), evidence: Value::Null,
        }).collect();
        let sessions = host.batch(specs, worker).await?;
        let mut batch = Vec::new();
        for (index, session) in sessions.into_iter().enumerate() {
            if session.error.is_some() || session.value.is_none() {
                let directory = args.output.join("generation-errors");
                tokio::fs::create_dir_all(&directory).await?;
                save(
                    &directory.join(format!("{iteration}-{index}.json")),
                    &session,
                )
                .await?;
                continue;
            }
            let id = tree.nodes.len() + batch.len();
            let candidate_index = batch.len();
            let directory = args.output.join(format!("candidates/{id}"));
            tokio::fs::create_dir_all(&directory).await?;
            save(&directory.join("session.json"), &session).await?;
            let evaluated = match evaluate(
                args,
                worker,
                host,
                &problem,
                &reference,
                &scenario,
                &session,
                iteration,
                candidate_index,
                &directory,
            )
            .await
            {
                Ok(value) => value,
                Err(error) => {
                    save(&directory.join("evaluation-error.json"), &error.to_string()).await?;
                    failed_evaluation(args.backend(), &error)
                }
            };
            let mut analysis = Value::Null;
            if evaluated["profile"]["success"] == true {
                match profile(
                    args,
                    worker,
                    host,
                    &problem,
                    &evaluated["profile"],
                    bottleneck.clone(),
                    id,
                )
                .await
                {
                    Ok(value) => analysis = value,
                    Err(error) => {
                        save(&directory.join("profiling-error.json"), &error.to_string()).await?
                    }
                }
            }
            let evidence = json!({"id": format!("candidate:{id}"), "evaluation": evaluated,
                "analysis": analysis, "submission": session.value, "session": session,
                "memory_version": snapshot.version});
            save(&directory.join("evidence.json"), &evidence).await?;
            batch.push(evidence);
        }
        ensure!(
            !batch.is_empty(),
            "all candidate sessions failed in iteration {iteration}"
        );
        let outcomes: Vec<Outcome> = batch
            .iter()
            .map(|value| serde_json::from_value(value["evaluation"]["outcome"].clone()))
            .collect::<std::result::Result<_, _>>()?;
        let ids = tree.record_batch(parent, outcomes)?;
        for (id, evidence) in ids.iter().zip(&batch) {
            candidates.insert(*id, evidence.clone());
        }
        save(&args.output.join("tree.json"), &tree).await?;
        if !args.memory_off {
            match update_memory(args, worker, host, iteration, &batch, &snapshot).await {
                Ok(next) => snapshot = next,
                Err(error) => {
                    save(
                        &args.output.join(format!("memory-error-{iteration}.json")),
                        &error.to_string(),
                    )
                    .await?
                }
            }
            if let Some(store) = host.state.as_ref().map(|state| state.memories())
                && let Some(stored) = store.load_search_snapshot(&host.root).await?
            {
                snapshot = serde_json::from_str(&stored)?;
                save(&args.output.join("memory.json"), &snapshot).await?;
            }
        }
        save(&args.output.join("winner.json"), &json!({"node": tree.winner(),
            "evidence": tree.winner().and_then(|id| candidates.get(&id)), "validation": args.backend()})).await?;
        if !tree.should_continue() {
            break;
        }
    }
    Ok(())
}

fn apply_runtime_defaults(config: &mut SearchConfig) {
    config.max_iterations.get_or_insert(10);
    config.num_candidates.get_or_insert(3);
    config.max_consecutive_failures.get_or_insert(3);
    config.speedup_threshold.get_or_insert(-100.0);
    config.use_recent_failed_only.get_or_insert(false);
}

fn candidate_mode(
    parent: Option<usize>,
    tree: &SearchState,
    candidates: &BTreeMap<usize, Value>,
) -> &'static str {
    match parent {
        None => "initial",
        Some(id) if tree.nodes[id].outcome.status == Status::Failed => "repair",
        Some(id)
            if tree.nodes[id].outcome.status == Status::Success
                && candidates
                    .get(&id)
                    .and_then(|value| value["analysis"].as_object())
                    .is_some_and(|analysis| !analysis.is_empty()) =>
        {
            "optimize"
        }
        Some(_) => "initial",
    }
}

fn failed_evaluation(validation: &str, error: &anyhow::Error) -> Value {
    json!({
        "outcome": {
            "status": "failed",
            "speedup": null,
            "error_context": {
                "error_type": "WorkerCrash",
                "error_message": error.to_string(),
            },
        },
        "candidate_seconds": null,
        "profile": null,
        "validation": validation,
    })
}

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
