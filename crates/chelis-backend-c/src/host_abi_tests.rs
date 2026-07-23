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
fn unsupported_scalar_abis_return_the_shared_typed_diagnostic() {
    for precision in [Prim::F16, Prim::Bf16] {
        let logical = ConcreteHostType::Scalar(precision);
        let err = HostAbiType::try_from_concrete(&logical)
            .expect_err("this scalar C-host representation is unimplemented");
        assert_eq!(
            err.what,
            UnsupportedKind::Dtype(precision.name().to_string())
        );
        assert_eq!(err.stage, Stage::Codegen("c"));
        assert!(err.to_string().starts_with("unsupported: "));
    }
}

#[test]
fn nested_values_cannot_hide_an_unsupported_scalar_abi() {
    let list = ConcreteHostType::List(Box::new(ConcreteHostType::Scalar(Prim::F16)));
    let err = HostAbiType::try_from_concrete(&list)
        .expect_err("a pointer container must not erase an unsupported element ABI");
    assert_eq!(err.what, UnsupportedKind::Dtype("f16".into()));
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
