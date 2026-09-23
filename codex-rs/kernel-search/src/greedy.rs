use crate::Node;
use crate::NodeId;
use crate::SearchState;
use crate::Status;

impl SearchState {
    pub(crate) fn select_greedy(&self) -> Option<NodeId> {
        if self.current_iteration == 0
            || self.consecutive_failures >= self.config.max_consecutive_failures.unwrap_or(2)
        {
            return None;
        }
        if let Some(best) = self.current_best
            && self.nodes[best].successful()
        {
            return Some(best);
        }
        let most_recent = self.nodes.iter().skip(1).map(|n| n.iteration).max();
        let mut best = None;
        for node in self.nodes.iter().skip(1) {
            if node.outcome.status != Status::Failed
                || (self.config.use_recent_failed_only.unwrap_or(true)
                    && Some(node.iteration) != most_recent)
            {
                continue;
            }
            if best.is_none_or(|id: NodeId| priority(node) > priority(&self.nodes[id])) {
                best = Some(node.id);
            }
        }
        best
    }
}

fn priority(node: &Node) -> (i32, f64, f64) {
    let Some(error) = node.outcome.error_context.as_ref() else {
        return (0, 0.0, 0.0);
    };
    if error.as_object().is_some_and(serde_json::Map::is_empty) {
        return (0, 0.0, 0.0);
    }
    let stage = match error.get("error_type").and_then(serde_json::Value::as_str) {
        Some("correctness_error") => 3,
        Some("RuntimeError" | "CudaError" | "execution_error" | "cuda_compile_runtime_error") => 2,
        Some("SyntaxError" | "ImportError") => 1,
        _ => 0,
    };
    let diff = error
        .get("max_diff")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    (stage, -diff, node.speedup().min(2.0))
}
