use chelis_dag_core::axis_sources::{
    derive_dim_witnesses, derive_runtime_dim_classes, entry_extent_guards, entry_validation_plan,
    witness_entry_obligations,
};
use chelis_dag_core::dag::RiscOp;

#[test]
fn source_lowering_preserves_shared_extent_entry_derivations() {
    for inputs in [4, 8, 16, 32] {
        let library = shared_extent_library(inputs);
        let dag = library.dag();
        let witness = dag
            .nodes()
            .iter()
            .rev()
            .find(|node| matches!(node.op, RiscOp::ExtentWitness { .. }))
            .expect("the probe has caller witnesses")
            .id;
        assert_eq!(entry_extent_guards(dag).len(), inputs - 1);
        assert_eq!(derive_runtime_dim_classes(dag).len(), 1);
        entry_validation_plan(dag);
        derive_dim_witnesses(dag);
        witness_entry_obligations(dag, witness);
    }
}

/// The library of chelis#3726's probe: `f` reads `inputs` parameters that
/// all claim the runtime extent `n`, and adds them in a chain.
fn shared_extent_library(inputs: usize) -> chelis_ir::lower::LoweredLibrary {
    let params = (1..=inputs)
        .map(|i| format!("x{i}: tensor[n, f32]"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = (2..=inputs).fold("x1".to_string(), |body, i| format!("add({body}, x{i})"));
    let source = format!("def f[n]({params}) -> tensor[n, f32] = {body}\n");
    let declarations =
        chelis_surf::parser::parse_str(&source, &chelis_syntax_host::NativeStack).unwrap();
    let checked = chelis_types::check_typed_program(
        &chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap(),
    )
    .unwrap();
    let checked = chelis_effects::check_program(&checked).unwrap();
    let checked = chelis_types::check_linearity(&checked).unwrap();
    chelis_ir::lower::try_lower_program_to_library(&checked).unwrap()
}
