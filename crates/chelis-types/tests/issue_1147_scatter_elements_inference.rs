//! `scatter_elements` enforces tensor, index, axis, shape, and precision constraints.

use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep = chelis_deep::parser::parse_str(source).expect("Deep fixture");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

#[test]
fn valid_scatter_elements_is_accepted() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:117..160"}
      (var {span: "surf:117..133"} scatter_elements)
      (var {span: "surf:134..138"} data)
      (var {span: "surf:140..147"} indices)
      (var {span: "surf:149..156"} updates)
      (lit {span: "surf:158..159", type: (t-prim {} i32)} 1))))
"#,
    );
    assert!(errors.is_empty(), "valid scatter rejected: {errors:#?}");
}

#[test]
fn string_axis_is_rejected() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:117..165"}
      (var {span: "surf:117..133"} scatter_elements)
      (var {span: "surf:134..138"} data)
      (var {span: "surf:140..147"} indices)
      (var {span: "surf:149..156"} updates)
      (lit {span: "surf:158..164", type: (t-prim {} string)} "zero"))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn out_of_bounds_axis_is_rejected() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:117..161"}
      (var {span: "surf:117..133"} scatter_elements)
      (var {span: "surf:134..138"} data)
      (var {span: "surf:140..147"} indices)
      (var {span: "surf:149..156"} updates)
      (lit {span: "surf:158..160", type: (t-prim {} i32)} 99))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn floating_point_indices_are_rejected() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:117..160"}
      (var {span: "surf:117..133"} scatter_elements)
      (var {span: "surf:134..138"} data)
      (var {span: "surf:140..147"} indices)
      (var {span: "surf:149..156"} updates)
      (lit {span: "surf:158..159", type: (t-prim {} i32)} 1))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn string_data_is_rejected() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-var {} _)))

(def {}
  apply
  (fn {}
    (params {} (indices {type: (t-var {} _)}) (updates {type: (t-var {} _)}))
    (app {span: "surf:71..122"}
      (var {span: "surf:71..87"} scatter_elements)
      (lit {span: "surf:88..100", type: (t-prim {} string)} "notatensor")
      (var {span: "surf:102..109"} indices)
      (var {span: "surf:111..118"} updates)
      (lit {span: "surf:120..121", type: (t-prim {} i32)} 0))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn updates_must_match_indices_shape_and_data_precision() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f64))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:117..160"}
      (var {span: "surf:117..133"} scatter_elements)
      (var {span: "surf:134..138"} data)
      (var {span: "surf:140..147"} indices)
      (var {span: "surf:149..156"} updates)
      (lit {span: "surf:158..159", type: (t-prim {} i32)} 1))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn unprovable_symbolic_containment_is_rejected() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (n k)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-var {} k) (d-lit {} 3) (t-prim {} i32))
    (t-tensor {} (d-var {} k) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (d-lit {} 3) (t-prim {} f32))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:123..166"}
      (var {span: "surf:123..139"} scatter_elements)
      (var {span: "surf:140..144"} data)
      (var {span: "surf:146..153"} indices)
      (var {span: "surf:155..162"} updates)
      (lit {span: "surf:164..165", type: (t-prim {} i32)} 1))))
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "{errors:#?}"
    );
}

#[test]
fn precision_variable_is_unified_by_the_same_rule_as_other_tensor_ops() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (p)
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-var {} p))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-var {} p))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:116..159"}
      (var {span: "surf:116..132"} scatter_elements)
      (var {span: "surf:133..137"} data)
      (var {span: "surf:139..146"} indices)
      (var {span: "surf:148..155"} updates)
      (lit {span: "surf:157..158", type: (t-prim {} i32)} 1))))
"#,
    );
    assert_eq!(errors.len(), 1, "expected one rejection: {errors:#?}");
    assert!(
        matches!(errors[0].kind, CheckErrorKind::TypeMismatch),
        "{errors:#?}"
    );
}

#[test]
fn a_precision_binder_shared_by_data_and_updates_stays_accepted() {
    let errors = diagnostics(
        r#"(defsig {}
  apply
  (p)
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-var {} p))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} i32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-var {} p))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-var {} p))))

(def {}
  apply
  (fn {}
    (params {}
      (data {type: (t-var {} _)})
      (indices {type: (t-var {} _)})
      (updates {type: (t-var {} _)}))
    (app {span: "surf:114..157"}
      (var {span: "surf:114..130"} scatter_elements)
      (var {span: "surf:131..135"} data)
      (var {span: "surf:137..144"} indices)
      (var {span: "surf:146..153"} updates)
      (lit {span: "surf:155..156", type: (t-prim {} i32)} 1))))
"#,
    );
    assert!(errors.is_empty(), "valid scatter rejected: {errors:#?}");
}
