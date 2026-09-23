use super::*;

#[test]
fn caps_both_evidence_and_filename_for_multibyte_input() {
    let fragment = KernelContextFragment {
        text: "\u{754c}".repeat(20_000),
        file: "f".repeat(30_000),
    };
    let rendered = fragment.body();
    assert!(rendered.len() < 8_600);
    assert!(rendered.contains("Read it when this excerpt is truncated"));
}
