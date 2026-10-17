//! Realizability inference pays for each `let` binding once, not once per
//! enclosing binding (chelis#3797). A Surf block of `k` bindings is a chain of
//! `k` nested `let` layers, and the walk that finds each definition's
//! top-level dependencies kept one table of bound names per layer, so every
//! variable read consulted every enclosing table.
//!
//! The oracle is the walk's own count of the binder tables it consults, kept
//! per thread while the test counts, which is deterministic for a fixed
//! program. Doubling the chain must grow the count by at most 2.5 times:
//! linear growth gives about 2, quadratic about 4.

use std::fmt::Write as _;

use chelis_types::CheckedProgram;

const SIZES: [usize; 4] = [125, 250, 500, 1000];

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

/// chelis#3793's probe: one definition whose body chains `k` bindings, each
/// reading the parameter, and returns them as a list, and one eager use.
fn chain(k: usize) -> CheckedProgram {
    let mut source = String::from("def f(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        writeln!(source, "  t{index} = add(x, {index}i64)").unwrap();
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    writeln!(source, "  [{}]\n}}", items.join(", ")).unwrap();
    source.push_str("r = index(f(1i64), 0i64)\n");
    let declarations = chelis_surf::parser::parse_str(&source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    chelis_types::check_typed_program(&deep).expect("the probe checks")
}

/// Binder tables the dependency walk consulted while inferring `program`'s
/// realizability.
fn binder_lookups(program: &CheckedProgram) -> u64 {
    chelis_effects::realizability::counting_dependency_binder_lookups(|| {
        drop(chelis_effects::realizability::infer_realizability(
            program,
            &[],
        ))
    })
    .1
}

fn sweep() -> [u64; SIZES.len()] {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| SIZES.map(|k| binder_lookups(&chain(k))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn assert_linear(what: &str, counts: [u64; SIZES.len()]) {
    eprintln!("{what} at k = {SIZES:?}: {counts:?}");
    assert!(counts[0] > 0, "{what}: the walk ran on the probe");
    for pair in counts.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "{what} grew superlinearly in the chain: {counts:?}"
        );
    }
}

#[test]
fn the_dependency_walk_consults_linearly_many_binder_tables_in_a_let_chain() {
    assert_linear("binder tables consulted", sweep());
}

#[test]
fn the_detector_rejects_work_per_enclosing_layer() {
    // Negative control: what the removed walk did, one table per enclosing
    // layer for every read.
    fn per_enclosing_layer(k: u64) -> u64 {
        (0..k).map(|layer| layer + 1).sum()
    }
    assert!(
        !grows_linearly(per_enclosing_layer(500), per_enclosing_layer(1000)),
        "the detector accepted per-layer work over the enclosing layers"
    );
}
