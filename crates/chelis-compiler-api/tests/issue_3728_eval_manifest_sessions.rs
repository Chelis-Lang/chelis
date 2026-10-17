//! chelis#3728 and chelis#1921: an evaluation that selects parameterized host
//! declarations asks each one's kernel input demand through one session over
//! the prepared host program. It opened a fresh session per selected
//! declaration, each re-deriving every program-wide fact. The count of fresh
//! sessions is process-wide, and nextest runs each test in its own process.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_ir::host::fresh_host_lowering_sessions;
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

/// `declarations` effectful tensor declarations, each a host call whose
/// input demand evaluation asks of the kernel decision.
fn source(declarations: usize) -> String {
    (0..declarations)
        .map(|index| {
            format!(
                "def e{index}(x: &tensor[2, f32]) -> tensor[2, f32] = {{\n  _ = print(\"e{index}\")\n  add(x, x)\n}}\n"
            )
        })
        .collect()
}

fn evaluate(declarations: usize, roots: &[String]) -> (EvalResult, u64) {
    let bindings = BTreeMap::from([(
        "x".to_string(),
        TensorValue {
            shape: vec![2],
            data: wire_values::storage_f32(vec![1.0, 2.0]),
        },
    )]);
    let before = fresh_host_lowering_sessions();
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source(declarations),
            bindings,
        },
        roots,
    )
    .expect("evaluate");
    (result, fresh_host_lowering_sessions() - before)
}

fn root_value(result: &EvalResult, name: &str) -> serde_json::Value {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("root {name} in {result:?}"));
    serde_json::to_value(&root.value).expect("serialize")
}

/// Evidentiary status: REGRESSION TEST, watched opening one more session per
/// selected declaration on the per-candidate session it replaces.
#[test]
fn selected_host_declarations_share_one_demand_session() {
    const DECLARATIONS: usize = 6;
    let all = (0..DECLARATIONS)
        .map(|index| format!("e{index}"))
        .collect::<Vec<_>>();
    let (one, one_sessions) = evaluate(DECLARATIONS, &all[..1]);
    let (every, every_sessions) = evaluate(DECLARATIONS, &all);
    assert_eq!(
        every_sessions, one_sessions,
        "selecting {DECLARATIONS} declarations opened {every_sessions} sessions, one opened {one_sessions}"
    );
    assert_eq!(root_value(&every, "e0"), root_value(&one, "e0"));
    for name in &all {
        let (alone, _) = evaluate(DECLARATIONS, std::slice::from_ref(name));
        assert_eq!(root_value(&every, name), root_value(&alone, name), "{name}");
        assert_eq!(
            alone.transcript,
            vec![name.clone()],
            "{name} prints only itself"
        );
    }
    assert_eq!(
        every.transcript, all,
        "every selected declaration runs once"
    );
}
