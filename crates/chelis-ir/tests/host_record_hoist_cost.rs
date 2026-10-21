//! Host lowering of a body pays for the body once, not once per level of it.
//! The record-projection hoist rewrites every host body, and at every node it
//! copied the node, children included, before replacing those children with
//! their rewritten forms, so a chain of `k` `let` bindings copied `k^2 / 2`
//! nodes even when nothing was hoisted (chelis#3793).
//!
//! The oracle is the hoist's own count of the nodes it builds or copies, kept
//! per thread while the test counts, which is deterministic for a fixed
//! program. Doubling the chain
//! must grow the count by at most 2.5 times: linear growth gives about 2,
//! quadratic about 4.

use std::fmt::Write as _;

use chelis_types::CheckedProgram;

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

fn checked(source: &str) -> CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("the probe parses");
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .expect("the probe desugars");
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

/// chelis#3793's probe: one host definition whose body chains `k` bindings
/// and returns them as a list, and one eager use.
fn chain(k: usize) -> CheckedProgram {
    let mut source = String::from("def f(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        writeln!(source, "  t{index} = add(x, {index}i64)").unwrap();
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    writeln!(source, "  [{}]\n}}", items.join(", ")).unwrap();
    source.push_str("r = index(f(1i64), 0i64)\n");
    checked(&source)
}

/// Nodes the hoist built or copied while host lowering `program`.
fn hoist_nodes(program: &CheckedProgram) -> u64 {
    let (lowered, nodes) = chelis_ir::host::counting_record_hoist_nodes(|| {
        chelis_ir::host::try_lower_compiled_program(program).expect("the probe lowers")
    });
    drop(lowered);
    nodes
}

#[test]
fn the_record_hoist_builds_linearly_many_nodes_in_a_let_chain() {
    let sizes = [125, 250, 500, 1000];
    let nodes = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || sizes.map(|k| hoist_nodes(&chain(k))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    eprintln!("record hoist nodes at k = {sizes:?}: {nodes:?}");
    assert!(nodes[0] > 0, "the hoist ran on the probe");
    for pair in nodes.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "the record hoist built superlinearly many nodes in the chain: {nodes:?}"
        );
    }
}

#[test]
fn the_detector_rejects_a_subtree_copied_per_level() {
    // Negative control: what the removed copy did, the rest of the chain
    // copied at every level of it.
    fn copy_per_level(k: u64) -> u64 {
        (0..k).map(|level| k - level).sum()
    }
    assert!(
        !grows_linearly(copy_per_level(500), copy_per_level(1000)),
        "the detector accepted a per-level subtree copy"
    );
}
