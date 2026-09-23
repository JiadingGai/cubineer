use super::ContextualUserFragment;
use codex_protocol::models::ContentItemKind;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::truncate_text;

/// Bounded controller context with complete evidence available in workspace files.
pub struct KernelContextFragment {
    pub text: String,
    pub file: String,
}

impl ContextualUserFragment for KernelContextFragment {
    fn role(&self) -> &'static str {
        "user"
    }
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("kernel.evidence".into())
    }
    fn requires_separate_message(&self) -> bool {
        true
    }
    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }
    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }
    fn body(&self) -> String {
        let excerpt = truncate_text(&self.text, TruncationPolicy::Bytes(8_000));
        let file = truncate_text(&self.file, TruncationPolicy::Bytes(256));
        format!(
            "Complete controller-provided content: {file}. Read it when this excerpt is truncated.\n{excerpt}"
        )
    }
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
