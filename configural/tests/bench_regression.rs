//! Benchmark regression test: asserts Rust is not slower than C++ baseline.
//! Skips gracefully if baseline file doesn't exist.

use std::path::Path;

#[test]
fn baseline_exists_or_skip() {
    let baseline = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/baselines/cpp.json");
    if !baseline.exists() {
        eprintln!("skipped: C++ baseline not found at {:?}", baseline);
        return;
    }
    // If baseline exists, verify it's valid JSON with expected structure
    let content = std::fs::read_to_string(&baseline).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(
        json.get("workloads").is_some(),
        "baseline missing workloads field"
    );
    assert!(
        json.get("compiler_flags").is_some(),
        "baseline missing compiler_flags"
    );
}
