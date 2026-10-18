//! Host lowering resolves a name by its terminal segment at the cost of the
//! names that share that segment, not of the whole program (chelis#3854).
//!
//! A name not declared exactly, such as a builtin's, resolves to the one
//! authored signature or type-environment key whose last `__` or `.` segment
//! matches it. Both lookups compared the name against every entry of their
//! map, and host lowering asks for every call it lowers, so `n` definitions
//! that call builtins did about `n^2` comparisons. They now read per-session
//! tables keyed by terminal segment.
//!
//! The oracle is host lowering's own count of the names it compares or looks
//! up by terminal segment, kept per thread while the test counts, which is
//! deterministic for a fixed program. Doubling the definitions must grow the
//! count by at most 2.5 times: linear growth gives about 2, quadratic about
//! 4.

use std::fmt::Write as _;

use chelis_types::CheckedProgram;

const SIZES: [usize; 4] = [64, 128, 256, 512];

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

fn checked(source: &str) -> CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let checked = chelis_types::check_typed_program(&deep).expect("the probe checks");
    let checked = chelis_effects::check_program(&checked).expect("the probe has no effect errors");
    chelis_types::check_linearity(&checked).expect("the probe is linear")
}

/// `k` one-line definitions calling builtins, and one root calling each.
/// Each definition's typed parameters and result are an authored
/// signature, so the signature table holds `k` entries.
fn builtin_calls(k: usize) -> CheckedProgram {
    let mut source = String::new();
    for index in 0..k {
        writeln!(
            source,
            "def f{index}(x: i64) -> i64 = add(mul(x, x), {index}i64)"
        )
        .unwrap();
    }
    let calls = (0..k)
        .map(|index| format!("f{index}(1i64)"))
        .collect::<Vec<_>>();
    writeln!(source, "r = [{}]", calls.join(", ")).unwrap();
    checked(&source)
}

fn comparisons(program: &CheckedProgram) -> u64 {
    chelis_ir::host::counting_terminal_name_comparisons(|| {
        drop(chelis_ir::host::try_lower_compiled_program(program).expect("the probe lowers"));
    })
    .1
}

fn sweep(probe: fn(usize) -> CheckedProgram) -> [u64; SIZES.len()] {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || SIZES.map(|k| comparisons(&probe(k))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn assert_linear(what: &str, counts: [u64; SIZES.len()]) {
    eprintln!("{what} at n = {SIZES:?}: {counts:?}");
    assert!(counts[0] > 0, "{what}: the lookups ran on the probe");
    for pair in counts.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "{what} grew superlinearly in the definitions: {counts:?}"
        );
    }
}

#[test]
fn builtin_calls_resolve_with_linearly_many_name_comparisons() {
    assert_linear(
        "terminal-name comparisons, builtin calls",
        sweep(builtin_calls),
    );
}

#[test]
fn the_detector_rejects_a_scan_of_the_program_per_call() {
    // Negative control: what the removed lookups did, every call compared
    // against every definition.
    fn scan_per_call(k: u64) -> u64 {
        2 * k * k
    }
    assert!(
        !grows_linearly(scan_per_call(256), scan_per_call(512)),
        "the detector accepted a scan of the program per call"
    );
}
