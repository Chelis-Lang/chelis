//! #1493: aliases remain callable; concrete values keep distinct observation roots.
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
fn function_aliases_remain_callable_entries_without_display_roots() {
    for module in [false, true] {
        for annotated in [false, true] {
            let source = source(module, annotated, true);
            let result = eval(request(source.clone())).unwrap();
            assert_eq!(result.roots.len(), 1);
            assert_eq!(result.roots[0].name.as_deref(), Some("user"));
            assert_eq!(result.roots[0].display.as_deref(), Some("1"));
            // Selecting a callable without supplying a concrete call does
            // not invent a display value for its closure.
            let selected = eval_selected(request(source), &["alias".into()]).unwrap();
            assert!(selected.roots.is_empty(), "{selected:?}");
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
fn deep_function_alias_preserves_only_concrete_observations() {
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
    let all = eval(req).unwrap();
    assert_eq!(all.roots.len(), 1);
    assert_eq!(all.roots[0].name.as_deref(), Some("user"));
    assert_eq!(all.roots[0].display.as_deref(), Some("1"));
}

#[test]
fn nullary_alias_is_a_value_not_an_implicitly_applied_declaration() {
    let source = "def anchor() -> int32 = 7\nalias = anchor\ndef user() -> int32 = alias()";
    let result = eval_selected(request(source.into()), &["user".into()]).unwrap();
    assert_eq!(result.roots[0].display.as_deref(), Some("7"));
    let selected = eval_selected(request(source.into()), &["alias".into()]).unwrap();
    assert!(selected.roots.is_empty(), "{selected:?}");
}

#[test]
fn tensor_function_alias_preserves_concrete_call_result() {
    let source = "def anchor(x: tensor[2, int32]) -> tensor[2, int32] = x\nalias = anchor\ndef user() -> tensor[2, int32] = alias(to_tensor([1, 2]))";
    let result = eval_selected(request(source.into()), &["user".into()]).unwrap();
    assert_eq!(
        result.roots[0].display.as_deref(),
        Some("tensor(shape=[2], data=[1, 2])")
    );
    let selected = eval_selected(request(source.into()), &["alias".into()]).unwrap();
    assert!(selected.roots.is_empty(), "{selected:?}");
}

#[test]
fn nullary_alias_remains_callable_as_an_argument_or_local_value() {
    for body in ["invoke(alias)", "{\nlocal = alias\nlocal()\n}"] {
        let source = format!(
            "def anchor() -> int32 = 7\nalias = anchor\ndef invoke(f) -> int32 = f()\ndef user() -> int32 = {body}"
        );
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
        })
        .unwrap();
        assert!(checked.errors.is_empty(), "{:?}", checked.errors);
        let result = eval_selected(request(source.clone()), &["user".into()]).unwrap();
        assert_eq!(result.roots[0].display.as_deref(), Some("7"));
        let selected = eval_selected(request(source), &["alias".into()]).unwrap();
        assert!(selected.roots.is_empty(), "{selected:?}");
    }
}
