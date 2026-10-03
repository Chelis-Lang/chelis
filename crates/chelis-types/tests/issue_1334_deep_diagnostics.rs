//! A checker rejection uses the Deep producer's coordinate and identity,
//! without inventing a length or changing the accepted/rejected boundary.

use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

#[test]
fn native_deep_bad_copy_points_to_the_actual_copy_node() {
    let source = "(def {} x (copy {} (lit {type: (t-prim {} i32)} 1)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("copying a scalar must reject");
    let at = source.find("(copy").unwrap();
    let error = report
        .errors
        .iter()
        .find(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch) && error.span_offset == Some(at)
        })
        .unwrap_or_else(|| panic!("copy's own rejection: {:?}", report.errors));
    assert!(
        error.message.contains("copy") && error.message.contains("argument 1"),
        "actionable operand: {error:?}"
    );
    assert_eq!(error.span_offset, Some(at));
    assert_eq!(
        error.span_id, None,
        "native Deep does not invent an external identity"
    );
    assert_eq!(error.expected.as_deref(), Some("tensor"));
    assert_eq!(error.got.as_deref(), Some("i32"));
}

#[test]
fn external_deep_identity_is_preserved_not_derived_from_its_coordinate() {
    let source = "(def {} x (copy {span: \"octant:line:7\"} (lit {type: (t-prim {} i32)} 1)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("copying a scalar must reject");
    let at = source.find("(copy").unwrap();
    let error = report
        .errors
        .iter()
        .find(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch) && error.span_offset == Some(at)
        })
        .unwrap_or_else(|| panic!("copy's own rejection: {:?}", report.errors));
    assert!(
        error.message.contains("copy") && error.message.contains("argument 1"),
        "actionable operand: {error:?}"
    );
    assert_eq!(error.span_id.as_deref(), Some("octant:line:7"));
    assert_eq!(error.span_offset, Some(at));
}

#[test]
fn opaque_application_identity_survives_alongside_the_native_coordinate() {
    let source = "(def {} bad
      (fn {} (params {}
        (x {type: (t-tensor {} (d-lit {} 1) (d-lit {} 2) (t-prim {} f32))})
        (y {type: (t-tensor {} (d-lit {} 3) (d-lit {} 1) (t-prim {} f32))}))
        (app {span: \"octant:line:7\"} (var {} matmul) (var {} x) (var {} y))))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("incompatible contraction axes must reject");
    let error = report
        .errors
        .iter()
        .find(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("matmul argument 2, axis 0")
        })
        .unwrap_or_else(|| panic!("the matmul mismatch: {:#?}", report.errors));
    assert_eq!(error.span_id.as_deref(), Some("octant:line:7"));
    assert_eq!(
        error.span_offset,
        Some(source.find("(app {span:").unwrap()),
        "native Deep coordinate is independent of the external identity"
    );
    assert_eq!(error.expected.as_deref(), Some("2"));
    assert_eq!(error.got.as_deref(), Some("3"));
}

#[test]
fn native_deep_application_with_no_external_id_keeps_its_measured_coordinate() {
    let source = "(def {} bad
      (fn {} (params {}
        (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}))
        (app {} (var {} to_list) (var {} x))))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("rank-two conversion rejects");
    let at = source.find("(app {} (var {} to_list)").unwrap();
    let error = report
        .errors
        .iter()
        .find(|error| error.message.contains("to_list argument 1"))
        .unwrap_or_else(|| panic!("missing conversion rejection: {:#?}", report.errors));
    assert_eq!(error.span_offset, Some(at));
    assert_eq!(error.span_id, None);
    assert_eq!(error.expected.as_deref(), Some("rank-1 tensor"));
}

#[test]
fn native_deep_tensor_to_scalar_reports_its_measured_call_without_an_external_id() {
    for (operand, expected, got) in [
        (
            "(t-tensor {} (d-lit {} 2) (t-prim {} f32))",
            "rank-0 tensor",
            "rank-1 tensor",
        ),
        ("(t-prim {} i32)", "tensor", "i32"),
    ] {
        let source = format!(
            "(def {{}} bad (fn {{}} (params {{}} (x {{type: {operand}}})) (app {{}} (var {{}} tensor_to_scalar) (var {{}} x))))"
        );
        let exprs = chelis_deep::parser::parse_str(&source).expect("native Deep fixture parses");
        let report = check_ir_program(&exprs).expect_err("invalid conversion rejects");
        let error = report
            .errors
            .iter()
            .find(|error| {
                matches!(error.kind, CheckErrorKind::TypeMismatch)
                    && error.message.contains("tensor_to_scalar argument 1")
            })
            .unwrap_or_else(|| panic!("missing conversion rejection: {:?}", report.errors));
        assert_eq!(
            error.span_offset,
            source.find("(app {} (var {} tensor_to_scalar)")
        );
        assert_eq!(error.span_id, None);
        assert_eq!(error.expected.as_deref(), Some(expected));
        assert_eq!(error.got.as_deref(), Some(got));
    }
    let accepted = "(def {} good (fn {} (params {} (x {type: (t-tensor {} (t-prim {} f32))})) (app {} (var {} tensor_to_scalar) (var {} x))))";
    let exprs = chelis_deep::parser::parse_str(accepted).expect("rank-zero Deep fixture parses");
    assert!(
        check_ir_program(&exprs).is_ok(),
        "rank-zero tensor is convertible"
    );
}

#[test]
fn opaque_numeric_range_id_does_not_override_measured_native_coordinate() {
    let source = "(def {} x (copy {span: \"octant:line:42..69\"} (lit {type: (t-prim {} i32)} 1)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("scalar copy rejects");
    let error = report
        .errors
        .iter()
        .find(|error| error.message.contains("copy argument 1"))
        .unwrap_or_else(|| panic!("missing copy rejection: {:#?}", report.errors));
    assert_eq!(error.span_id.as_deref(), Some("octant:line:42..69"));
    assert_eq!(error.span_offset, Some(source.find("(copy").unwrap()));
}

#[test]
fn valid_tensor_conversion_does_not_acquire_a_spurious_rejection() {
    let source = "(def {} xs (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.0) (var {} Nil))))\n(def {} out (app {} (var {} to_list) (var {} xs)))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let outcome = check_ir_program(&exprs);
    assert!(
        outcome.is_ok(),
        "rank-one tensor converts cleanly: {:?}",
        outcome.err().map(|report| report.errors)
    );
}

#[test]
fn native_deep_axis_dtype_rejection_uses_the_application_coordinate() {
    let source = "(def {} bad (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})) (app {} (var {} expand) (var {} x) (lit {type: (t-prim {} i64)} 0) (lit {type: (t-prim {} i64)} 3))))";
    let exprs = chelis_deep::parser::parse_str(source).expect("valid Deep source");
    let report = check_ir_program(&exprs).expect_err("i64 is not an axis dtype");
    let error = report
        .errors
        .iter()
        .find(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch)
                && error.message.contains("expand argument 2 (axis)")
        })
        .unwrap_or_else(|| panic!("missing expand axis rejection: {:#?}", report.errors));
    assert_eq!(
        error.span_offset,
        Some(source.find("(app {} (var {} expand)").unwrap())
    );
    assert_eq!(error.span_id, None);
    assert_eq!(error.expected.as_deref(), Some("i32"));
    assert_eq!(error.got.as_deref(), Some("i64"));
}
