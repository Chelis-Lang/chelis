//! chelis#3230: DAG record construction stores its slots in the
//! constructor's declared field order, which positional patterns read
//! ([04-PAT-3]), whatever order the record literal writes its fields in.
//!
//! The subexpression lowering `chelis eval` uses for `grad` and the C lane
//! uses for tensor helpers must bind `P(first, second)` to the declared
//! fields `w` and `b`. A lowering without the program's declarations cannot
//! know that order and must reject the pattern rather than bind written
//! order.

use chelis_deep::ast::Expr;
use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::lower::{
    SubexprLoweringContext, try_lower_subexpr_program, try_lower_subexpr_program_with_context,
};
use chelis_unord::UnordMap;

const TYPES: &str = "type P =\n  | P { w: f32, b: f32 }\nprobe = 0.0f32\n";

fn checked_surf(source: &str) -> chelis_types::CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let exprs =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let checked = chelis_types::check_ir_program(&exprs).unwrap_or_else(|report| {
        panic!(
            "type check failed: {:?}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        )
    });
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

/// `match P { <written> } with { | P(first, second) => <taken> }` over the
/// free inputs `in_w` and `in_b`.
fn positional_match(written: [&str; 2], taken: &str) -> Expr {
    let fields = written
        .map(|field| format!("(kv {{}} {field} (var {{}} in_{field}))"))
        .join(" ");
    let source = format!(
        "(match {{}} (record {{}} P {fields}) \
         (arm {{}} (pat-ctor {{}} P (pat-var {{}} first) (pat-var {{}} second)) () \
         (var {{}} {taken})))"
    );
    chelis_deep::parser::parse_str(&source)
        .expect("Deep parse")
        .remove(0)
}

/// The free inputs the lowered graph's roots read, in root order.
fn root_inputs(dag: &Dag) -> Vec<String> {
    dag.roots()
        .iter()
        .map(|root| {
            let mut node = dag.get(*root).expect("root node");
            while let RiscOp::Store { .. } = node.op {
                node = dag.get(node.inputs[0]).expect("stored node");
            }
            match &node.op {
                RiscOp::Load { name } => name.as_str().to_string(),
                other => panic!("a root reads no input: {other:?}"),
            }
        })
        .collect()
}

#[test]
fn checked_subexpression_lowering_binds_positional_patterns_by_declared_order() {
    let checked = checked_surf(TYPES);
    let context =
        SubexprLoweringContext::from_checked_program(&checked, UnordMap::new(), UnordMap::new());
    for written in [["w", "b"], ["b", "w"]] {
        for (binder, declared) in [("first", "in_w"), ("second", "in_b")] {
            let dag = try_lower_subexpr_program_with_context(
                &positional_match(written, binder),
                UnordMap::new(),
                &context,
            )
            .unwrap_or_else(|diagnostic| panic!("{written:?}: {diagnostic}"));
            assert_eq!(
                root_inputs(&dag),
                [declared],
                "written order {written:?}: `{binder}` is declared field `{declared}`"
            );
        }
    }
}

#[test]
fn a_lowering_without_declarations_rejects_a_positional_pattern_over_a_record() {
    for written in [["w", "b"], ["b", "w"]] {
        let diagnostic = try_lower_subexpr_program(
            &positional_match(written, "first"),
            UnordMap::new(),
            UnordMap::new(),
            UnordMap::new(),
        )
        .expect_err("an unknown record layout must not answer a positional pattern");
        assert!(
            diagnostic
                .to_string()
                .contains("declared field order is unknown"),
            "{written:?}: {diagnostic}"
        );
    }
}

#[test]
fn a_registry_supplied_to_a_bare_context_restores_declared_order() {
    // `chelis eval` without a host session builds its transform context from
    // type and definition tables, then supplies the checked registry.
    let checked = checked_surf(TYPES);
    let context = SubexprLoweringContext::new(UnordMap::new(), UnordMap::new(), UnordMap::new())
        .with_adt_registry(checked.adt_registry());
    let dag = try_lower_subexpr_program_with_context(
        &positional_match(["b", "w"], "first"),
        UnordMap::new(),
        &context,
    )
    .unwrap_or_else(|diagnostic| panic!("{diagnostic}"));
    assert_eq!(root_inputs(&dag), ["in_w"]);
}
