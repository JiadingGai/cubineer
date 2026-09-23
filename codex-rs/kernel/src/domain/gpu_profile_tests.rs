use super::*;
use pretty_assertions::assert_eq;

#[test]
fn metric_collection_switches_between_extracted_and_library_kernels() -> Result<()> {
    let args = |names: &[String]| {
        metrics_command(
            Path::new("ncu"),
            Path::new("venv/python"),
            Path::new("profile_solution.py"),
            Path::new("report.csv"),
            names,
        )
    };
    let library = args(&[])?;
    let named = args(&["foo<int>".into(), "a.b".into()])?;
    assert!(library.contains(&"--launch-count=5".into()));
    assert!(named.contains(&"--launch-count=1".into()));
    let mut normalized = named;
    let filter = normalized.remove(8);
    assert_eq!(filter, "--kernel-name=regex:(foo<int>|a\\.b)");
    normalized[10] = "--launch-count=5".into();
    assert_eq!(normalized, library);
    Ok(())
}
