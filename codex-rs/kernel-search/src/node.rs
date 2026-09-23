use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

pub type NodeId = usize;
pub const ROOT: NodeId = 0;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pending,
    Success,
    Failed,
    Discarded,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub status: Status,
    pub speedup: Option<f64>,
    #[serde(default)]
    pub error_context: Option<Value>,
}

impl Outcome {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.status != Status::Pending,
            "cannot commit a pending outcome"
        );
        if let Some(speedup) = self.speedup {
            ensure!(speedup.is_finite(), "speedup must be finite");
        }
        if self.status == Status::Success {
            ensure!(
                self.speedup.is_some(),
                "success requires a measured speedup"
            );
        }
        if let Some(error) = &self.error_context {
            ensure!(error.is_object(), "error_context must be an object");
            if let Some(diff) = error.get("max_diff") {
                ensure!(diff.as_f64().is_some(), "max_diff must be numeric");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub iteration: i64,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub outcome: Outcome,
    pub visits: u64,
    pub total_reward: f64,
    pub terminal: bool,
}

impl Node {
    pub fn speedup(&self) -> f64 {
        self.outcome.speedup.unwrap_or(0.0)
    }

    pub fn successful(&self) -> bool {
        self.outcome.status == Status::Success
    }

    pub fn mcts_reward(&self) -> f64 {
        if self.successful() {
            self.speedup().max(0.1).ln()
        } else if self.outcome.speedup.is_some() {
            -2.0
        } else {
            -3.0
        }
    }

    pub fn mcts_score(&self) -> f64 {
        if !self.successful() || self.speedup() == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.speedup().max(0.1).ln()
        }
    }
}
