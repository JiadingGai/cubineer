use std::io::BufRead;

use anyhow::Result;
use codex_kernel_search::Outcome;
use codex_kernel_search::SearchConfig;
use codex_kernel_search::SearchState;
use codex_kernel_search::Strategy;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Trace {
    strategy: Strategy,
    #[serde(default)]
    config: SearchConfig,
    batches: Vec<Vec<Outcome>>,
}

fn main() -> Result<()> {
    for line in std::io::stdin().lock().lines() {
        let trace: Trace = serde_json::from_str(&line?)?;
        let mut state = SearchState::new(trace.strategy, trace.config)?;
        let mut steps = Vec::new();
        for (iteration, outcomes) in trace.batches.into_iter().enumerate() {
            if iteration > 0 && !state.should_continue() {
                break;
            }
            let parent = if iteration == 0 { None } else { state.select() };
            state.record_batch(parent, outcomes)?;
            steps.push(json!({
                "parent": parent,
                "nodes": state.nodes,
                "current_iteration": state.current_iteration,
                "current_best": state.current_best,
                "consecutive_failures": state.consecutive_failures,
                "speedup_history": state.speedup_history,
                "root_expansions": state.root_expansions,
                "pw_gate_closures": state.pw_gate_closures,
                "virtual_child_wins": state.virtual_child_wins,
                "continue": state.should_continue(),
                "winner": state.winner(),
            }));
        }
        println!("{}", serde_json::to_string(&steps)?);
    }
    Ok(())
}
