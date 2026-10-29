//! The checker's declaration walks pay for each `let` binding once, not once
//! per enclosing binding (chelis#3797). A Surf block of `k` bindings is a
//! chain of `k` nested `let` layers, and the walks that find a declaration's
//! top-level references and its parameters' consuming uses kept one table of
//! bound names per layer, so every variable read consulted every enclosing
//! table.
//!
//! The oracle is the walks' own count of the binder tables they consult.
//! Doubling the chain must grow it by at most 2.5 times: linear growth gives
//! about 2, quadratic about 4.

use super::*;

/// chelis#3793's probe: one definition whose body chains `k` bindings, each
/// reading the parameter, and returns them as a list, and one eager use.
fn chain_program(k: usize) -> Vec<deep::Expr> {
    let mut source = String::from("def f(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        source.push_str(&format!("  t{index} = add(x, {index}i64)\n"));
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    source.push_str(&format!(
        "  [{}]\n}}\nr = index(f(1i64), 0i64)\n",
        items.join(", ")
    ));
    let declarations = chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
        .expect("generated Surf parses");
    chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .expect("generated Surf desugars")
}

fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

#[test]
fn issue_3797_declaration_walks_consult_linearly_many_binder_tables() {
    let sizes = [125, 250, 500, 1000];
    let lookups = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            crate::test_support::run(|operation| {
                sizes.map(|k| {
                    let program = chain_program(k);
                    let _ = take_binder_lookups();
                    crate::test_support::check_typed_program(&program, operation)
                        .expect("the probe checks");
                    take_binder_lookups()
                })
            })
        })
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes");
    eprintln!("binder tables consulted at k = {sizes:?}: {lookups:?}");
    assert!(lookups[0] > 0, "the walks ran on the probe");
    for pair in lookups.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "the declaration walks consulted superlinearly many binder tables: {lookups:?}"
        );
    }
}

#[test]
fn issue_3797_detector_rejects_work_per_enclosing_layer() {
    // Negative control: what the removed walks did, one table per enclosing
    // layer for every read.
    fn per_enclosing_layer(k: usize) -> usize {
        (0..k).map(|layer| layer + 1).sum()
    }
    assert!(
        !grows_linearly(per_enclosing_layer(500), per_enclosing_layer(1000)),
        "the detector accepted per-layer work over the enclosing layers"
    );
}
