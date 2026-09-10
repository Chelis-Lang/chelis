//! [05-OBS-7,11]: function-valued bindings remain owed, selected calls run.
use chelis_compiler_api::compiler::{check, eval, eval_selected};
use chelis_compiler_api::schema::{CheckRequest, EvalRequest, SourceKind};

fn source(module: bool, annotated: bool, chain: bool) -> String {
    format!(
        "{}\ndef anchor(x: int32) -> int32 = x\n{} = anchor\n{}\ndef user() -> int32 = {}(1)\n",
        if module { "module AliasUser" } else { "" },
        if annotated {
            "alias: (int32) -> int32"
        } else {
            "alias"
        },
        if chain { "second = alias" } else { "" },
        if chain { "second" } else { "alias" },
    )
}

fn request(source: String) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: Default::default(),
    }
}

#[test]
fn selected_call_through_function_alias_has_its_own_root() {
    for module in [false, true] {
        for annotated in [false, true] {
            for chain in [false, true] {
                let source = source(module, annotated, chain);
                let checked = check(CheckRequest {
                    source_kind: SourceKind::Surf,
                    source: source.clone(),
                })
                .unwrap();
                assert!(checked.errors.is_empty(), "{:?}", checked.errors);
                let result = eval_selected(request(source), &["user".into()]).unwrap();
                assert_eq!(result.roots.len(), 1);
                assert_eq!(result.roots[0].name.as_deref(), Some("user"));
                assert_eq!(result.roots[0].display.as_deref(), Some("1"));
            }
        }
    }
}

#[test]
fn observing_function_alias_rejects_without_dropping_the_root() {
    for module in [false, true] {
        for annotated in [false, true] {
            let source = source(module, annotated, false);
            for selected in [false, true] {
                let result = if selected {
                    eval_selected(request(source.clone()), &["alias".into()])
                } else {
                    eval(request(source.clone()))
                };
                let error = result.expect_err("a function value has no observation representation");
                let message = error
                    .errors
                    .iter()
                    .map(|item| item.message.as_str())
                    .collect::<Vec<_>>()
                    .join("; ");
                for required in ["[05-UNS-1]", "alias", "Host", "function"] {
                    assert!(message.contains(required), "{message}");
                }
                assert!(!message.contains("root count mismatch"), "{message}");
            }
        }
    }
}

#[test]
fn ordinary_aliases_and_callable_declarations_remain_observable() {
    let result = eval(request("def anchor(x: int32) -> int32 = x\nvalue = 1\nother = value\ndef user() -> int32 = anchor(other)".into())).unwrap();
    let names = result
        .roots
        .iter()
        .map(|root| root.name.as_deref().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(names, ["value", "other", "user"]);
    assert!(
        result
            .roots
            .iter()
            .all(|root| root.display.as_deref() == Some("1"))
    );
}

#[test]
fn deep_function_alias_preserves_selection_and_rejection() {
    let deep =
        chelis_compiler_api::compiler::desugar(chelis_compiler_api::schema::DesugarRequest {
            source: source(true, true, true),
        })
        .unwrap()
        .deep_text;
    let mut req = request(deep);
    req.source_kind = SourceKind::Deep;
    let result = eval_selected(req.clone(), &["user".into()]).unwrap();
    assert_eq!(result.roots[0].display.as_deref(), Some("1"));
    let error = eval(req).unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message.contains("unavailable root `alias`"))
    );
}

#[test]
fn nullary_alias_is_a_value_not_an_implicitly_applied_declaration() {
    let source = "def anchor() -> int32 = 7\nalias = anchor\ndef user() -> int32 = alias()";
    let result = eval_selected(request(source.into()), &["user".into()]).unwrap();
    assert_eq!(result.roots[0].display.as_deref(), Some("7"));
    let error = eval_selected(request(source.into()), &["alias".into()]).unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message.contains("unavailable root `alias`"))
    );
}

#[test]
fn tensor_function_alias_preserves_concrete_call_result() {
    let source = "def anchor(x: tensor[2, int32]) -> tensor[2, int32] = x\nalias = anchor\ndef user() -> tensor[2, int32] = alias(to_tensor([1, 2]))";
    let result = eval_selected(request(source.into()), &["user".into()]).unwrap();
    assert_eq!(
        result.roots[0].display.as_deref(),
        Some("tensor(shape=[2], data=[1, 2])")
    );
    let error = eval_selected(request(source.into()), &["alias".into()]).unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message.contains("unavailable root `alias`"))
    );
}
