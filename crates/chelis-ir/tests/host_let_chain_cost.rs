//! Host lowering of a body pays for each `let` binding once, not once per
//! enclosing binding. A Surf block of `k` bindings is a chain of `k` nested
//! `let` layers, and two host walks paid per layer for every layer around it
//! (chelis#3797):
//!
//! - the free-variable walks behind the top-level read qualification kept
//!   one binder table per layer, so each variable read consulted every
//!   enclosing table, and the renaming copied the rest of the chain at each
//!   layer before replacing it with its renamed form;
//! - host type refinement copied the whole type scope at each layer.
//!
//! Each oracle is the walk's own count of that work, kept per thread while
//! the test counts, which is deterministic for a fixed program. Doubling the
//! chain must grow the count by at most 2.5 times: linear growth gives about
//! 2, quadratic about 4.

use std::fmt::Write as _;

use chelis_types::CheckedProgram;

const SIZES: [usize; 4] = [125, 250, 500, 1000];

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

fn checked(source: &str) -> CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let expanded = chelis_macros::expand_program(
        &deep,
        &chelis_macros::ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("the probe expands");
    let checked = chelis_types::check_typed_program(expanded.exprs()).expect("the probe checks");
    let checked = chelis_effects::check_program(&checked).expect("the probe has no effect errors");
    chelis_types::check_linearity(&checked).expect("the probe is linear")
}

/// One host definition whose body chains `k` bindings, each reading `read`,
/// and returns them as a list, and one eager use. `prelude` declares what
/// `read` names when it is not the parameter.
fn chain(k: usize, prelude: &str, read: &str) -> CheckedProgram {
    let mut source = String::from(prelude);
    source.push_str("def f(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        writeln!(source, "  t{index} = add({read}, {index}i64)").unwrap();
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    writeln!(source, "  [{}]\n}}", items.join(", ")).unwrap();
    source.push_str("r = index(f(1i64), 0i64)\n");
    checked(&source)
}

/// chelis#3793's probe: every binding reads the parameter.
fn parameter_chain(k: usize) -> CheckedProgram {
    chain(k, "", "x")
}

/// Every binding reads a top-level value, so host lowering renames the
/// body's free reads to the value's declaration.
fn top_level_chain(k: usize) -> CheckedProgram {
    chain(k, "c = 7i64\n", "c")
}

fn lower(program: &CheckedProgram) {
    drop(chelis_ir::host::try_lower_compiled_program(program).expect("the probe lowers"));
}

/// `count` of host lowering at each chain length in [`SIZES`], on a thread
/// whose stack holds the deepest chain.
fn sweep(
    probe: fn(usize) -> CheckedProgram,
    count: fn(&CheckedProgram) -> u64,
) -> [u64; SIZES.len()] {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || SIZES.map(|k| count(&probe(k))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn free_variable_walk_work(program: &CheckedProgram) -> u64 {
    chelis_types::linearity::counting_free_variable_walk_work(|| lower(program)).1
}

fn refine_scope_entries(program: &CheckedProgram) -> u64 {
    chelis_ir::host::counting_refine_scope_entries(|| lower(program)).1
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
fn the_free_variable_walks_do_linear_work_over_a_parameter_chain() {
    assert_linear(
        "free-variable walk work, parameter reads",
        sweep(parameter_chain, free_variable_walk_work),
    );
}

#[test]
fn the_free_variable_walks_do_linear_work_over_a_top_level_read_chain() {
    assert_linear(
        "free-variable walk work, top-level reads",
        sweep(top_level_chain, free_variable_walk_work),
    );
}

#[test]
fn type_refinement_touches_linearly_many_scope_entries() {
    assert_linear(
        "refinement scope entries",
        sweep(parameter_chain, refine_scope_entries),
    );
}

#[test]
fn the_detector_rejects_work_per_enclosing_layer() {
    // Negative control: what the removed walks did, work at each layer for
    // every layer around it.
    fn per_enclosing_layer(k: u64) -> u64 {
        (0..k).map(|layer| layer + 1).sum()
    }
    assert!(
        !grows_linearly(per_enclosing_layer(500), per_enclosing_layer(1000)),
        "the detector accepted per-layer work over the enclosing layers"
    );
}
