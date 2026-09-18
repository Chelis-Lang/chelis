use chelis_ir::host::{find_direct_builtin_call, program_has_top_level_value_bindings};
use chelis_types::{CheckedProgram, check_ir_program, check_typed_program};

fn checked_ingresses(source: &str) -> (CheckedProgram, CheckedProgram) {
    let stamped = chelis_deep::parse_and_stamp_file(source).expect("fixture must parse and stamp");
    let typed = check_typed_program(&stamped)
        .unwrap_or_else(|errors| panic!("typed ingress must check: {:?}", errors.errors));
    let ir = check_ir_program(&stamped)
        .unwrap_or_else(|errors| panic!("IR ingress must check: {:?}", errors.errors));
    (typed, ir)
}

#[test]
fn top_level_value_detection_has_positive_and_negative_carrier_parity() {
    let (typed_value, ir_value) = checked_ingresses("(def {} global_value (lit {} 1))");
    assert!(program_has_top_level_value_bindings(&typed_value));
    assert!(program_has_top_level_value_bindings(&ir_value));

    let function_only = "
        (defsig {} identity (t-fn {} (t-prim {} i32) (t-prim {} i32)))
        (def {} identity
          (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))
    ";
    let (typed_function, ir_function) = checked_ingresses(function_only);
    assert!(!program_has_top_level_value_bindings(&typed_function));
    assert!(!program_has_top_level_value_bindings(&ir_function));
}

#[test]
fn top_level_value_detection_flattens_modules_on_both_ingresses() {
    let module_with_value = "
        (module {} host.parity
          (def {} global_value (lit {} 1)))
    ";
    let (typed_value, ir_value) = checked_ingresses(module_with_value);
    assert!(
        program_has_top_level_value_bindings(&typed_value),
        "typed ingress: {:#?}",
        typed_value.exprs()
    );
    assert!(
        program_has_top_level_value_bindings(&ir_value),
        "IR ingress: {:#?}",
        ir_value.exprs()
    );
}

#[test]
fn direct_builtin_detection_has_positive_and_negative_carrier_parity() {
    let with_abs = "
        (defsig {} absolute (t-fn {} (t-prim {} i32) (t-prim {} i32)))
        (def {} absolute
          (fn {} (params {} (x {type: (t-prim {} i32)}))
            (app {} (var {} abs) (var {} x))))
    ";
    let (typed_abs, ir_abs) = checked_ingresses(with_abs);
    assert_eq!(
        find_direct_builtin_call(&typed_abs, &["abs"]),
        Some("abs".to_string())
    );
    assert_eq!(
        find_direct_builtin_call(&ir_abs, &["abs"]),
        Some("abs".to_string())
    );

    let without_abs = "
        (defsig {} identity (t-fn {} (t-prim {} i32) (t-prim {} i32)))
        (def {} identity
          (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))
    ";
    let (typed_identity, ir_identity) = checked_ingresses(without_abs);
    assert_eq!(find_direct_builtin_call(&typed_identity, &["abs"]), None);
    assert_eq!(find_direct_builtin_call(&ir_identity, &["abs"]), None);
}

#[test]
fn direct_builtin_detection_traverses_decoded_node_metadata() {
    let with_metadata_call = "
        (def {property_seed: (app {} (var {} abs) (lit {} 1))}
          value
          (lit {} 0))
    ";
    let (typed, ir) = checked_ingresses(with_metadata_call);
    assert_eq!(
        find_direct_builtin_call(&typed, &["abs"]),
        Some("abs".to_string())
    );
    assert_eq!(
        find_direct_builtin_call(&ir, &["abs"]),
        Some("abs".to_string())
    );
}
