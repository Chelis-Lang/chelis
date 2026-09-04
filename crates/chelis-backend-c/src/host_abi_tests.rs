//! chelis#730 Phase 2 contract for the C-host Table-B type boundary.
//!
//! The backend owns an opaque, crate-private `HostAbiType`. General values
//! have only the fallible conversion from a fully resolved
//! `ConcreteHostType`; typed callback declarators cross a separate,
//! position-restricted constructor and have no standalone/opaque C spelling.

use crate::host_abi::HostAbiType;
use chelis_ir::{ConcreteHostType, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

#[test]
fn abi_conversion_accepts_only_resolved_logical_types() {
    let conversion: fn(&ConcreteHostType) -> Result<HostAbiType, Unsupported> =
        HostAbiType::try_from_concrete;
    let _ = conversion;
}

#[test]
fn supported_concrete_types_map_to_exact_c_host_abis() {
    let i8_abi = HostAbiType::try_from_concrete(&ConcreteHostType::Scalar(Prim::Int8))
        .expect("int8 has an exact C-host representation");
    assert_eq!(i8_abi.c_type_name(), Some("int8_t"));

    let i16_abi = HostAbiType::try_from_concrete(&ConcreteHostType::Scalar(Prim::Int16))
        .expect("int16 has an exact C-host representation");
    assert_eq!(i16_abi.c_type_name(), Some("int16_t"));

    let f32_abi = HostAbiType::try_from_concrete(&ConcreteHostType::Scalar(Prim::F32))
        .expect("f32 has a C-host representation");
    assert_eq!(f32_abi.c_type_name(), Some("float"));

    for precision in [Prim::F16, Prim::Bf16] {
        let abi = HostAbiType::try_from_concrete(&ConcreteHostType::Scalar(precision))
            .expect("reduced floats have exact uint16 C-host storage in Phase 3");
        assert_eq!(abi.c_type_name(), Some("uint16_t"));
    }

    let i32_abi = HostAbiType::try_from_concrete(&ConcreteHostType::Scalar(Prim::Int32))
        .expect("int32 has a C-host representation");
    assert_eq!(i32_abi.c_type_name(), Some("int32_t"));

    let tensor_abi = HostAbiType::try_from_concrete(&ConcreteHostType::Tensor(TensorType {
        dims: Vec::new(),
        precision: Prim::F16,
    }))
    .expect("f16 tensors use the typed tensor runtime and have a pointer ABI");
    assert_eq!(tensor_abi.c_type_name(), Some("chelis_tensor*"));
}

#[test]
fn options_of_active_scalars_share_the_exact_tagged_scalar_carrier() {
    for precision in [
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
        Prim::Bool,
    ] {
        let option = ConcreteHostType::Option(Box::new(ConcreteHostType::Scalar(precision)));
        let abi = HostAbiType::try_from_concrete(&option)
            .expect("every active runtime scalar has an exact Option ABI");
        assert_eq!(
            abi.c_type_name(),
            Some("chelis_option*"),
            "Option<{precision:?}> must not retain a per-dtype compatibility struct"
        );
    }

    let string_option = ConcreteHostType::Option(Box::new(ConcreteHostType::Scalar(Prim::String)));
    let abi = HostAbiType::try_from_concrete(&string_option)
        .expect("strings use the generic exact value carrier");
    assert_eq!(abi.c_type_name(), Some("chelis_option*"));
}

/// Once Phase 3 supplies exact scalar storage, every container recursively
/// preserves the same reduced-float ABI; no boxed-only exception remains.
#[test]
fn nested_values_preserve_reduced_float_abi() {
    let list = ConcreteHostType::List(Box::new(ConcreteHostType::Scalar(Prim::F16)));
    let abi = HostAbiType::try_from_concrete(&list).expect("f16 list has an exact ABI");
    assert_eq!(
        abi,
        HostAbiType::List(Box::new(HostAbiType::Float16)),
        "the element state must stay exact, never erased to f32/f64/void*"
    );

    let tuple = ConcreteHostType::Tuple(vec![ConcreteHostType::Scalar(Prim::Bf16)]);
    assert_eq!(
        HostAbiType::try_from_concrete(&tuple).expect("bf16 tuple has an exact ABI"),
        HostAbiType::Tuple(vec![HostAbiType::BFloat16])
    );

    let dict = ConcreteHostType::Dict(
        Box::new(ConcreteHostType::Scalar(Prim::Int64)),
        Box::new(ConcreteHostType::Scalar(Prim::F16)),
    );
    assert_eq!(
        HostAbiType::try_from_concrete(&dict).expect("f16 dict value has an exact ABI"),
        HostAbiType::Dict(Box::new(HostAbiType::Int64), Box::new(HostAbiType::Float16))
    );
}

#[test]
fn first_class_function_type_has_no_general_c_host_value_abi() {
    let function = ConcreteHostType::Function(
        vec![ConcreteHostType::Scalar(Prim::Int8)],
        Box::new(ConcreteHostType::Scalar(Prim::Int8)),
    );
    let error = HostAbiType::try_from_concrete(&function)
        .expect_err("function values must not acquire an opaque C representation");
    assert_eq!(error.stage, Stage::Codegen("c"));
    // The type RESOLVED; the C target has no ABI for it. That is the
    // `HostAbi` state, never the unresolved `HostType` state (chelis#730
    // Phase 2 red team, finding F5).
    assert!(matches!(error.what, UnsupportedKind::HostAbi(_)));
    assert!(error.to_string().contains("function value"));
    assert!(!error.to_string().contains("unresolved"));
}

/// chelis#841 review, finding 5: the marker guards in `project_expr` are
/// defense in depth with no CLI-reachable trigger; lock them directly so
/// a marker can never emit as a C symbol.
#[test]
fn unresolved_callee_markers_are_rejected_at_projection_in_both_positions() {
    use chelis_ir::host::{
        ConcreteHostProgram, HOST_UNRESOLVED_CALLABLE_MARKER, HOST_UNRESOLVED_TRANSFORM_MARKER,
        HostBinding, HostExpr, HostExprKind,
    };

    let program_with = |kind: chelis_ir::host::HostExprKind<ConcreteHostType>| {
        let mut program = ConcreteHostProgram::default();
        program.globals.push(HostBinding {
            name: "probe".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: ConcreteHostType::Scalar(Prim::Int32),
            value: HostExpr::new(kind),
        });
        program
    };

    let call_marker = program_with(HostExprKind::Call {
        function: HOST_UNRESOLVED_CALLABLE_MARKER.into(),
        args: Vec::new(),
        arg_tys: Vec::new(),
        ty: ConcreteHostType::Scalar(Prim::Int32),
    });
    let err = crate::host_abi::project_program(&call_marker)
        .expect_err("a callable marker in Call position must never project");
    let rendered = err.to_string();
    assert!(rendered.contains("unresolved function value"), "{rendered}");
    assert!(!rendered.contains("#chelis-unresolved"), "{rendered}");

    let builtin_marker = program_with(HostExprKind::Builtin {
        name: HOST_UNRESOLVED_TRANSFORM_MARKER.into(),
        args: Vec::new(),
        ty: ConcreteHostType::Scalar(Prim::Int32),
    });
    let err = crate::host_abi::project_program(&builtin_marker)
        .expect_err("a transform marker in Builtin position must never project");
    let rendered = err.to_string();
    assert!(
        rendered.contains("transform application"),
        "the transform marker carries its own semantic payload: {rendered}"
    );
    assert!(!rendered.contains("#chelis-unresolved"), "{rendered}");
}
