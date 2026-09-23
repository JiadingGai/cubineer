mod config;
mod greedy;
mod mcts;
mod node;

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;

pub use config::SearchConfig;
pub use config::Strategy;
pub use node::Node;
pub use node::NodeId;
pub use node::Outcome;
pub use node::ROOT;
pub use node::Status;

use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SearchState {
    pub strategy: Strategy,
    pub config: SearchConfig,
    pub nodes: Vec<Node>,
    pub current_iteration: usize,
    pub current_best: Option<NodeId>,
    pub consecutive_failures: usize,
    pub speedup_history: Vec<f64>,
    pub root_expansions: u64,
    pub pw_gate_closures: u64,
    pub virtual_child_wins: u64,
}

impl SearchState {
    pub fn new(strategy: Strategy, config: SearchConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            strategy,
            config,
            nodes: vec![Node {
                id: ROOT,
                name: "ROOT".into(),
                iteration: -1,
                parent: None,
                children: Vec::new(),
                outcome: Outcome {
                    status: Status::Pending,
                    speedup: None,
                    error_context: None,
                },
                visits: 0,
                total_reward: 0.0,
                terminal: false,
            }],
            current_iteration: 0,
            current_best: None,
            consecutive_failures: 0,
            speedup_history: Vec::new(),
            root_expansions: 0,
            pw_gate_closures: 0,
            virtual_child_wins: 0,
        })
    }

    pub fn should_continue(&self) -> bool {
        if self.current_iteration >= self.config.max_iterations(self.strategy) {
            return false;
        }
        if self.strategy == Strategy::Greedy && self.speedup_history.len() >= 2 {
            let n = self.speedup_history.len();
            let improvement = self.speedup_history[n - 1] - self.speedup_history[n - 2];
            if improvement < self.config.speedup_threshold.unwrap_or(0.1) {
                return false;
            }
        }
        true
    }

    pub fn select(&mut self) -> Option<NodeId> {
        match self.strategy {
            Strategy::Greedy => self.select_greedy(),
            Strategy::Mcts => self.select_mcts(),
        }
    }

    pub fn record_batch(
        &mut self,
        parent: Option<NodeId>,
        outcomes: Vec<Outcome>,
    ) -> Result<Vec<NodeId>> {
        if let Some(parent) = parent {
            ensure!(parent > ROOT && parent < self.nodes.len(), "invalid parent");
        }
        for outcome in &outcomes {
            outcome.validate()?;
        }
        let mut added = Vec::new();
        for (index, outcome) in outcomes.into_iter().enumerate() {
            let id = self.nodes.len();
            let iteration = self.current_iteration;
            let name = match parent {
                Some(parent) => format!("iter{iteration}_p{}_c{index}", self.nodes[parent].name),
                None => format!("iter{iteration}_sol{index}"),
            };
            self.nodes.push(Node {
                id,
                name,
                iteration: iteration as i64,
                parent,
                children: Vec::new(),
                outcome,
                visits: 0,
                total_reward: 0.0,
                terminal: false,
            });
            if let Some(parent) = parent {
                self.nodes[parent].children.push(id);
            }
            added.push(id);
        }
        if self.strategy == Strategy::Mcts {
            self.update_mcts(&added);
        }
        let mut iteration_best = None;
        for id in &added {
            if self.nodes[*id].successful()
                && iteration_best.is_none_or(|best| self.score(*id) > self.score(best))
            {
                iteration_best = Some(*id);
            }
        }
        if let Some(best) = iteration_best {
            self.consecutive_failures = 0;
            if self
                .current_best
                .is_none_or(|old| self.score(best) > self.score(old))
            {
                self.current_best = Some(best);
            }
            if self.strategy == Strategy::Greedy {
                self.speedup_history.push(self.nodes[best].speedup());
            }
        } else {
            self.consecutive_failures += 1;
        }
        self.current_iteration += 1;
        Ok(added)
    }

    pub fn winner(&self) -> Option<NodeId> {
        let mut best = None;
        for node in self.nodes.iter().skip(1).filter(|node| node.successful()) {
            if best.is_none_or(|id: NodeId| node.speedup() > self.nodes[id].speedup()) {
                best = Some(node.id);
            }
        }
        best
    }

    fn score(&self, id: NodeId) -> f64 {
        match self.strategy {
            Strategy::Greedy => self.nodes[id].speedup(),
            Strategy::Mcts => self.nodes[id].mcts_score(),
        }
    }
}
