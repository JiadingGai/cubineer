use anyhow::Result;
use anyhow::ensure;
use codex_core::context::ContextualUserFragment;
use codex_core::context::MemoryContextFragment;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Category {
    SearchFindings,
    ErrorsAndCorrections,
    SuccessfulPatterns,
    KeyFiles,
    Learnings,
}

const CATEGORIES: [Category; 5] = [
    Category::SearchFindings,
    Category::ErrorsAndCorrections,
    Category::SuccessfulPatterns,
    Category::KeyFiles,
    Category::Learnings,
];
const MAX_SECTION_CHARS: usize = 8_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Finding {
    pub category: Category,
    pub text: String,
    pub evidence: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Entry {
    pub finding: Finding,
    pub batch: usize,
    pub evaluator_fact: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub version: i64,
    pub entries: Vec<Entry>,
}

impl Snapshot {
    pub fn merge(
        &self,
        batch: usize,
        findings: Vec<Finding>,
        facts: Vec<Finding>,
        evidence: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut next = self.clone();
        for (finding, evaluator_fact) in findings
            .into_iter()
            .map(|f| (f, false))
            .chain(facts.into_iter().map(|f| (f, true)))
        {
            ensure!(
                !finding.text.trim().is_empty()
                    && finding.text.chars().count() <= 150
                    && !finding.text.contains(['\n', '\r']),
                "memory entries must be one line, at most 150 characters"
            );
            ensure!(
                evidence.contains(&finding.evidence),
                "memory entry references unknown evidence"
            );
            if let Some(entry) = next.entries.iter_mut().find(|entry| {
                entry.finding.category == finding.category && entry.finding.text == finding.text
            }) {
                if evaluator_fact {
                    *entry = Entry {
                        finding,
                        batch,
                        evaluator_fact,
                    };
                }
                continue;
            }
            next.entries.push(Entry {
                finding,
                batch,
                evaluator_fact,
            });
        }
        for category in CATEGORIES {
            while next
                .entries
                .iter()
                .filter(|entry| entry.finding.category == category)
                .map(|entry| {
                    entry.finding.text.chars().count() + entry.finding.evidence.chars().count() + 64
                })
                .sum::<usize>()
                > MAX_SECTION_CHARS
            {
                if let Some(index) = next
                    .entries
                    .iter()
                    .position(|entry| entry.finding.category == category)
                {
                    next.entries.remove(index);
                }
            }
        }
        next.version += 1;
        Ok(next)
    }

    pub fn context(&self, run_id: &str) -> String {
        MemoryContextFragment::ReadInstructions(format!(
            "Run-scoped search memory ({run_id}, version {}). Entries marked evaluator_fact are measurements; all others are unverified inferences.\n{}",
            self.version, serde_json::to_string(&self.entries).unwrap_or_default(),
        )).body()
    }
}

pub(crate) fn extraction_input(
    evidence: &Value,
    snapshot: &Snapshot,
    run_id: &str,
    workspace: &std::path::Path,
) -> Result<String> {
    let text = format!(
        "Distill this batch's candidate events and evaluator evidence. Cite only the supplied evidence IDs. Return new, concise inferences, not claims of verified correctness. Existing memory:\n{}\nBatch:\n{}",
        snapshot.context(run_id),
        evidence,
    );
    Ok(
        MemoryContextFragment::ExtractionEvidence(
            codex_memories_write::build_scoped_input_message(
                &workspace.join("task-context.txt"),
                workspace,
                &text,
            )?,
        )
        .body(),
    )
}

pub(crate) fn schema() -> Value {
    json!({"type": "object", "additionalProperties": false, "required": ["findings"],
    "properties": {"findings": {"type": "array", "maxItems": 30, "items": {
        "type": "object", "additionalProperties": false, "required": ["category", "text", "evidence"],
        "properties": {"category": {"type": "string", "enum": CATEGORIES},
            "text": {"type": "string", "maxLength": 150}, "evidence": {"type": "string"}}
    }}}})
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
