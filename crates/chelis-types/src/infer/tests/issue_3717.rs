//! Signature-metadata inference visits each function declaration once, and
//! a callee's inferred read-only parameters still reach every caller.

use super::*;

/// `count` helpers in a call chain, each passing its parameter to the
/// previous one and reading it again, so every caller's read-only inference
/// depends on its callee's. `reversed` writes callers before callees.
fn chain_program(count: usize, reversed: bool) -> Vec<deep::Expr> {
    let mut defs = vec!["def c0(x, y: tensor[4, f32]) = add(x, y)".to_string()];
    for index in 1..count {
        let previous = index - 1;
        defs.push(format!(
            "def c{index}(x, y: tensor[4, f32]) = {{\n  z: tensor[4, f32] = c{previous}(x, y)\n  add(x, z)\n}}"
        ));
    }
    if reversed {
        defs.reverse();
    }
    let source = format!("{}\n", defs.join("\n"));
    let declarations = chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
        .expect("generated Surf parses");
    chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .expect("generated Surf desugars")
}

#[test]
fn issue_3717_signature_sweep_visits_each_declaration_once() {
    for count in [4, 16, 64] {
        for reversed in [false, true] {
            let program = chain_program(count, reversed);
            let _ = take_signature_sweep_visits();
            let checked = check_typed_program(&program).expect("generated chain checks");
            assert_eq!(
                take_signature_sweep_visits(),
                count,
                "a {count}-declaration program must cost one sweep visit per declaration"
            );
            let functions = &checked.signature_inference().functions;
            assert_eq!(functions.len(), count);
            for index in 0..count {
                let function = &functions[&format!("c{index}")];
                assert!(!function.recursive_cycle);
                assert!(
                    function.params[0].inferred_read_only,
                    "c{index}'s `x` reaches its callee through the callee's inferred \
                     signature and must stay read-only (reversed: {reversed})"
                );
            }
        }
    }
}

#[test]
fn issue_3717_consuming_callee_keeps_its_callers_owned() {
    // The negative counterpart: when the chain's root consumes `x`, nothing
    // above it may infer `x` read-only, because each caller hands `x` to a
    // callee that owns it and then reads `x` again.
    let mut source = String::from("def c0(x, y: tensor[4, f32]) = realize(x)\n");
    for index in 1..8 {
        let previous = index - 1;
        source.push_str(&format!(
            "def c{index}(x, y: tensor[4, f32]) = {{\n  z: tensor[4, f32] = c{previous}(x, y)\n  add(x, z)\n}}\n"
        ));
    }
    let declarations = chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
        .expect("generated Surf parses");
    let program =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .expect("generated Surf desugars");
    let checked = check_typed_program(&program).expect("generated chain checks");
    for index in 0..8 {
        let function = &checked.signature_inference().functions[&format!("c{index}")];
        assert!(
            !function.params[0].inferred_read_only,
            "c{index}'s `x` is consumed below it and must stay owned"
        );
    }
}
