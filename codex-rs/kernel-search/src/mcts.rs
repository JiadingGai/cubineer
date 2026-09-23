use crate::NodeId;
use crate::ROOT;
use crate::SearchState;
use crate::Status;

impl SearchState {
    pub(crate) fn select_mcts(&mut self) -> Option<NodeId> {
        let mut id = ROOT;
        loop {
            if self.nodes[id].children.is_empty() {
                break;
            }
            let mut best: Option<(NodeId, f64)> = None;
            for child in &self.nodes[id].children {
                if self.nodes[*child].terminal {
                    continue;
                }
                let score = self.uct(*child);
                if best.is_none_or(|(_, old)| score > old) {
                    best = Some((*child, score));
                }
            }
            let Some((best_child, best_score)) = best else {
                self.nodes[id].terminal = true;
                return None;
            };
            if self.can_add_child(id) {
                if self.virtual_uct(id) > best_score {
                    self.virtual_child_wins += 1;
                    break;
                }
            } else {
                self.pw_gate_closures += 1;
            }
            id = best_child;
        }
        if id == ROOT {
            self.root_expansions += 1;
            None
        } else {
            Some(id)
        }
    }

    fn can_add_child(&self, id: NodeId) -> bool {
        let node = &self.nodes[id];
        let c = if id == ROOT {
            self.config.pw_c_root.unwrap_or(2.0)
        } else if node.outcome.status == Status::Failed {
            self.config.pw_c_repair.unwrap_or(2.0)
        } else {
            self.config.pw_c_optimize.unwrap_or(3.0)
        };
        let mut limit = (c * (node.visits as f64).powf(self.config.pw_alpha.unwrap_or(0.5)))
            .trunc()
            .max(1.0);
        if let Some(cap) = self.config.max_children_per_node
            && cap > 0
        {
            limit = limit.min(cap as f64);
        }
        (node.children.len() as f64) < limit
    }

    fn uct(&self, id: NodeId) -> f64 {
        let node = &self.nodes[id];
        if node.visits == 0 {
            return f64::INFINITY;
        }
        let mut score = node.total_reward / node.visits as f64;
        if let Some(parent) = node.parent
            && self.nodes[parent].visits > 0
        {
            score += self.config.exploration_constant.unwrap_or(1.414)
                * ((self.nodes[parent].visits as f64).ln() / node.visits as f64).sqrt();
        }
        score
    }

    fn virtual_uct(&self, id: NodeId) -> f64 {
        let node = &self.nodes[id];
        if node.visits == 0 {
            return f64::INFINITY;
        }
        let visits = node.visits as f64;
        let fair = (visits / (node.children.len() + 1) as f64).max(1.0);
        node.total_reward / visits
            + self.config.exploration_constant.unwrap_or(1.414) * (visits.ln() / fair).sqrt()
    }

    pub(crate) fn update_mcts(&mut self, added: &[NodeId]) {
        for id in added {
            if self.nodes[*id].parent.is_none() {
                self.nodes[*id].parent = Some(ROOT);
                self.nodes[ROOT].children.push(*id);
            }
            let reward = self.nodes[*id].mcts_reward();
            let mut ancestor = Some(*id);
            while let Some(id) = ancestor {
                self.nodes[id].visits += 1;
                self.nodes[id].total_reward += reward;
                ancestor = self.nodes[id].parent;
            }
            if let Some(parent) = self.nodes[*id].parent
                && parent != ROOT
                && self.nodes[parent].outcome.status == Status::Failed
                && self.nodes[parent].children.len() >= 3
                && self.nodes[parent]
                    .children
                    .iter()
                    .all(|child| self.nodes[*child].outcome.status == Status::Failed)
            {
                self.nodes[parent].terminal = true;
            }
        }
    }
}
