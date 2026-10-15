//! OP33 public transform path: runtime-sized scans, their adjoints and batching.
mod ownership_support;
use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;

#[test]
fn runtime_scan_grad_and_vmap_execute_in_eval_and_c() {
    let source = include_str!("../../../examples/grad_cumsum.ch");
    let actual = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    let displayed = actual
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().unwrap(),
                root.display.as_deref().unwrap()
            )
        })
        .collect::<String>();
    assert!(displayed.contains("data=[3.0, 2.0, 1.0]"), "{displayed}");
    assert!(displayed.contains("6.0, 4.0, 9.0, 15.0]"), "{displayed}");
    let generated = ownership_support::emit(source, "scan_lanes");
    let (ledger, compiled) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    assert_eq!(compiled, displayed);
}

#[test]
fn scan_bool_and_integer_gradient_are_rejected_before_execution() {
    for source in [
        "module Demo.Main\nout = cumsum(to_tensor([true, false]), 0i32)\n",
        "module Demo.Main\ndef loss(x: tensor[3, i32]) -> tensor[i32] = sum(cumsum(x, 0i32), 0i32)\nout = grad(loss)(to_tensor([1i32, 2i32, 3i32]))\n",
    ] {
        assert!(
            eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                bindings: BTreeMap::new()
            })
            .is_err()
        );
        assert!(
            compile(CompileRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                target: CompileTarget::C,
                entry_name: Some("scan_negative".into())
            })
            .is_err()
        );
    }
}
