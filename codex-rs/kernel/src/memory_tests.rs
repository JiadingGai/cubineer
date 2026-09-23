use super::*;
use pretty_assertions::assert_eq;

fn finding(text: &str, evidence: &str) -> Finding {
    Finding {
        category: Category::Learnings,
        text: text.into(),
        evidence: evidence.into(),
    }
}

#[test]
fn deduplicates_in_order_and_preserves_evidence() {
    let evidence = BTreeSet::from(["candidate:1".into(), "candidate:2".into()]);
    let initial = Snapshot::default();
    let next = initial
        .merge(
            0,
            vec![finding("Use contiguous loads", "candidate:1")],
            Vec::new(),
            &evidence,
        )
        .unwrap();
    let final_snapshot = next
        .merge(
            1,
            vec![finding("Use contiguous loads", "candidate:2")],
            Vec::new(),
            &evidence,
        )
        .unwrap();
    assert_eq!(
        final_snapshot,
        Snapshot {
            version: 2,
            entries: next.entries
        }
    );
    assert_eq!(initial, Snapshot::default());
}

#[test]
fn rejects_unknown_evidence_without_mutating_previous_snapshot() {
    let initial = Snapshot::default();
    assert!(
        initial
            .merge(
                0,
                vec![finding("Unverified claim", "other-run:1")],
                Vec::new(),
                &BTreeSet::new()
            )
            .is_err()
    );
    assert_eq!(initial, Snapshot::default());
}

#[test]
fn evicts_oldest_entries_and_bounds_multibyte_context() {
    let evidence = BTreeSet::from(["candidate:1".into()]);
    let findings = (0..100)
        .map(|index| finding(&format!("{index}: {}", "x".repeat(140)), "candidate:1"))
        .collect();
    let snapshot = Snapshot::default()
        .merge(0, findings, Vec::new(), &evidence)
        .unwrap();
    assert!(snapshot.entries.len() > 25);
    assert!(snapshot.entries.len() < 45);
    assert!(
        snapshot
            .entries
            .last()
            .unwrap()
            .finding
            .text
            .starts_with("99:")
    );
    assert!(snapshot.context("run-a").len() < 12_000);
}

#[test]
fn retains_the_original_per_section_memory_budget() {
    let evidence = BTreeSet::from(["candidate:1".into()]);
    let findings = (0..20)
        .map(|index| finding(&format!("{index}: {}", "x".repeat(120)), "candidate:1"))
        .collect();
    let snapshot = Snapshot::default()
        .merge(0, findings, Vec::new(), &evidence)
        .unwrap();

    assert_eq!(snapshot.entries.len(), 20);
}

#[test]
fn inference_and_evaluator_facts_remain_distinct() {
    let evidence = BTreeSet::from(["candidate:1".into()]);
    let snapshot = Snapshot::default()
        .merge(
            0,
            vec![finding("Likely register pressure", "candidate:1")],
            vec![finding("Simulated timing: 1 ms", "candidate:1")],
            &evidence,
        )
        .unwrap();
    assert_eq!(
        snapshot
            .entries
            .iter()
            .map(|entry| entry.evaluator_fact)
            .collect::<Vec<_>>(),
        vec![false, true]
    );
}

#[test]
fn inference_cannot_shadow_a_matching_evaluator_fact() {
    let evidence = BTreeSet::from(["candidate:1".into(), "candidate:2".into()]);
    let snapshot = Snapshot::default()
        .merge(
            0,
            vec![finding("Measured", "candidate:1")],
            vec![finding("Measured", "candidate:2")],
            &evidence,
        )
        .unwrap();
    assert_eq!(
        snapshot.entries,
        vec![Entry {
            finding: finding("Measured", "candidate:2"),
            batch: 0,
            evaluator_fact: true
        }]
    );
}
