use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    #[default]
    Greedy,
    Mcts,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    pub max_iterations: Option<usize>,
    pub num_candidates: Option<usize>,
    pub max_consecutive_failures: Option<usize>,
    pub speedup_threshold: Option<f64>,
    pub use_recent_failed_only: Option<bool>,
    pub exploration_constant: Option<f64>,
    pub pw_alpha: Option<f64>,
    #[serde(alias = "pw_C_root")]
    pub pw_c_root: Option<f64>,
    #[serde(alias = "pw_C_repair")]
    pub pw_c_repair: Option<f64>,
    #[serde(alias = "pw_C_optimize")]
    pub pw_c_optimize: Option<f64>,
    pub max_children_per_node: Option<usize>,
}

impl SearchConfig {
    pub fn max_iterations(&self, strategy: Strategy) -> usize {
        self.max_iterations.unwrap_or(match strategy {
            Strategy::Greedy => 3,
            Strategy::Mcts => 10,
        })
    }

    pub fn num_candidates(&self) -> usize {
        self.num_candidates.unwrap_or(2)
    }

    pub fn validate(&self) -> Result<()> {
        for value in [
            self.speedup_threshold,
            self.exploration_constant,
            self.pw_alpha,
            self.pw_c_root,
            self.pw_c_repair,
            self.pw_c_optimize,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(value.is_finite(), "search parameters must be finite");
        }
        Ok(())
    }
}
