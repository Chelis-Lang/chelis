extern crate alloc;
use alloc::vec::Vec;
use chelis_deep::Expr;
use chelis_effects_core::{analyze_effects, infer_effect_rows, EffectErrorKind};
use chelis_type_core::Effect;

struct CallerStack;
impl chelis_stack_services::StackServices for CallerStack {
    fn grow<R>(&self, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn maybe_grow<R>(&self, _: usize, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn remaining_stack(&self) -> Option<usize> { None }
}
fn data(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source, &CallerStack).unwrap()
}
pub fn exercise() {
    let library = data(r#"
(def {} emit (fn {} (params {} x) (app {} (var {} debug) (var {} x))))
(def {} identity (fn {} (params {} x) (var {} x)))
"#);
    let original = library.clone();
    let extension = data(r#"
(def {} wrapper (fn {} (params {} x) (app {} (var {} emit) (var {} x))))
(def {} transitive (fn {} (params {} x) (app {} (var {} wrapper) (var {} x))))
(def {} pure (fn {} (params {} x) (app {} (var {} identity) (var {} x))))
"#);
    let rows = infer_effect_rows(&extension, Some(&library));
    assert_eq!(rows.len(), 3);
    assert!(rows["wrapper"].contains(&Effect::Io));
    assert!(rows["transitive"].contains(&Effect::Io));
    assert!(rows["pure"].is_empty());
    assert!(!rows.contains_key("emit"));
    let annotated = analyze_effects(&extension, Some(&library)).unwrap();
    assert_ne!(annotated, extension);
    assert_eq!(infer_effect_rows(&annotated, Some(&library)), rows);
    assert_eq!(library, original);
    assert_eq!(analyze_effects(&extension, Some(&library)).unwrap(), annotated);
    let shadow = data(r#"
(def {} emit (fn {} (params {} x) (var {} x)))
(def {} caller (fn {} (params {} x) (app {} (var {} emit) (var {} x))))
"#);
    assert!(infer_effect_rows(&shadow, Some(&library))["caller"].is_empty());
    assert!(infer_effect_rows(&extension, Some(&library))["wrapper"].contains(&Effect::Io));
    let invalid = data(r#"
(defsig {} bad (t-fn {eff: (effects {})} (t-prim {} i64) (t-prim {} i64)))
(def {} bad (fn {} (params {} x) (app {} (var {} emit) (var {} x))))
"#);
    let errors = analyze_effects(&invalid, Some(&library)).unwrap_err();
    assert!(errors.iter().any(|error| error.kind == EffectErrorKind::UnhandledEffect
        && !error.message.is_empty()));
    let invalid_handler = data(r#"
(def {} bad (handle-effect {effect: resource} (var {} computed) (lit {} 1)))
"#);
    let errors = analyze_effects(&invalid_handler, None).unwrap_err();
    assert!(errors.iter().any(|error| error.kind == EffectErrorKind::InvalidHandler));
}
