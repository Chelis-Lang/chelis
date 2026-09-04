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
    let err = crate::host_abi::project_binding(
        call_marker.globals.into_iter().next().unwrap(),
        &chelis_unord::UnordSet::new(),
    )
    .expect_err("a callable marker in Call position must never project");
    let rendered = err.to_string();
    assert!(rendered.contains("unresolved function value"), "{rendered}");
    assert!(!rendered.contains("#chelis-unresolved"), "{rendered}");

    let builtin_marker = program_with(HostExprKind::Builtin {
        name: HOST_UNRESOLVED_TRANSFORM_MARKER.into(),
        args: Vec::new(),
        ty: ConcreteHostType::Scalar(Prim::Int32),
    });
    let err = crate::host_abi::project_binding(
        builtin_marker.globals.into_iter().next().unwrap(),
        &chelis_unord::UnordSet::new(),
    )
    .expect_err("a transform marker in Builtin position must never project");
    let rendered = err.to_string();
    assert!(
        rendered.contains("transform application"),
        "the transform marker carries its own semantic payload: {rendered}"
    );
    assert!(!rendered.contains("#chelis-unresolved"), "{rendered}");
}

/// Phase 2A boundary ratchet: a compiled backend may accept raw payloads only
/// by value in its explicit pre-verification selection step. Every borrowed
/// public emission edge must require a sealed verified specialization.
#[test]
fn public_backend_emission_edges_cannot_borrow_raw_payloads() {
    let sources = [
        ("c", include_str!("lib.rs")),
        ("hip", include_str!("../../chelis-backend-hip/src/lib.rs")),
        (
            "metal",
            include_str!("../../chelis-backend-metal/src/lib.rs"),
        ),
    ];

    for (backend, source) in sources {
        for signature in source.match_indices("pub fn ").map(|(start, _)| {
            let rest = &source[start..];
            &rest[..rest.find('{').unwrap_or(rest.len())]
        }) {
            assert!(
                !signature.contains("&Dag")
                    && !signature.contains("&chelis_ir::dag::Dag")
                    && !signature.contains("&ConcreteHostProgram")
                    && !signature.contains("&chelis_ir::host::ConcreteHostProgram"),
                "{backend} exposes a public borrowed raw emission payload:\n{signature}"
            );
        }
    }

    let c = sources[0].1;
    assert!(c.contains("dag: &chelis_ir::ownership::VerifiedDagProgram"));
    assert!(c.contains("program: &chelis_ir::ownership::VerifiedHostProgram"));
    assert!(c.contains("mod emit;"));
    assert!(c.contains("mod host_emit;"));

    let hip = sources[1].1;
    assert!(hip.contains("dag: &chelis_ir::ownership::VerifiedDagProgram"));

    let metal = sources[2].1;
    assert!(metal.contains("dag: &chelis_ir::ownership::VerifiedDagProgram"));
}

fn verified_host_from_source(source: &str) -> chelis_ir::ownership::VerifiedHostProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("parse host source");
    let deep = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check host source: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects host source");
    let checked = chelis_types::check_linearity(&checked).expect("linearity host source");
    let realizability =
        chelis_effects::realizability::infer_realizability(&checked, crate::TENSOR_CAPABLE_PRIMS);
    let manifest = chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
    let lowered = chelis_ir::host::try_lower_compiled_program_with_manifest(&checked, &manifest)
        .expect("lower host source");
    let host =
        crate::prepare_host_program_for_codegen(lowered.host.expect("source uses the host lane"))
            .expect("select C host payload");
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        manifest,
        chelis_types::types::Target::C,
    );
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_host_ownership(&manifested, host)
            .expect("lower host ownership"),
    )
    .expect("verify host ownership")
}

#[test]
fn recursive_calls_target_the_consuming_body_not_the_external_clone_adapter() {
    let source = include_str!(
        "../../chelis-cli/tests/fixtures/compiled_value_ownership/issue_1206_depth_1.ch"
    );
    let verified = verified_host_from_source(source);
    let emitted = crate::codegen_host_program(&verified, "recursive_depth_1")
        .unwrap()
        .c_source;
    let body = "step__chelis_owned_body";

    assert!(emitted.contains("chelis_tensor* step(chelis_tensor* state"));
    assert!(
        emitted.matches(&format!("{body}(")).count() >= 3,
        "declaration, definition, wrapper entry, and recursive body call must retain the internal symbol:\n{emitted}"
    );
    assert_eq!(
        emitted.matches("= step(").count(),
        0,
        "internal recursion must never re-enter the external cloning adapter:\n{emitted}"
    );
}

#[test]
fn backend_local_host_ownership_inference_cannot_return() {
    let source = include_str!("host_emit.rs");
    assert!(
        !source.contains(".names()"),
        "generic verified owners must not expose binder spelling to C ownership emission"
    );
    assert!(source.contains("VerifiedHostAction::Operation"));
    assert!(source.contains("VerifiedHostTerminator::Return"));
    assert!(source.contains("emit_expression_site"));
    assert!(source.contains("binding_name"));
}

#[test]
fn abi_projection_preserves_exact_verified_site_and_nested_dag_cursors() {
    let source = include_str!(
        "../../chelis-cli/tests/fixtures/compiled_value_ownership/issue_543_adt_tensor.ch"
    );
    let declarations = chelis_surf::parser::parse_str(source).expect("parse ownership fixture");
    let deep = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check fixture: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects fixture");
    let checked = chelis_types::check_linearity(&checked).expect("linearity fixture");
    let realizability =
        chelis_effects::realizability::infer_realizability(&checked, crate::TENSOR_CAPABLE_PRIMS);
    let manifest = chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
    let lowered = chelis_ir::host::try_lower_compiled_program_with_manifest(&checked, &manifest)
        .expect("lower fixture");
    let mut host = lowered
        .host
        .expect("aggregate tensor fixture uses host lane");
    let helper_ty = TensorType {
        dims: vec![chelis_ir::DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let mut helper_dag = chelis_ir::Dag::new();
    let helper_root = helper_dag.add_node(
        chelis_ir::RiscOp::Load { name: "x".into() },
        Vec::new(),
        helper_ty.clone(),
        None,
    );
    helper_dag.add_root(helper_root);
    host.global_tensor_helpers
        .push(chelis_ir::host::HostTensorHelper {
            name: "__verified_cursor_probe".into(),
            dag: helper_dag,
            inputs: vec![chelis_ir::host::HostTensorInput {
                name: "x".into(),
                ty: helper_ty.clone(),
            }],
            output: helper_ty,
            specialization: None,
            summary_rejection: None,
        });
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        manifest,
        chelis_types::types::Target::C,
    );
    let selected = crate::prepare_host_program_for_codegen(host).expect("select C host payload");
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_host_ownership(&manifested, selected)
            .expect("lower host ownership"),
    )
    .expect("verify host ownership");
    let emission = verified.emission();
    let expected_sites = emission
        .sites()
        .map(|site| {
            (
                site.id(),
                site.kind(),
                site.action_kinds().collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let projected = crate::host_abi::project_program(emission).expect("project verified host");

    assert_eq!(projected.sites().len(), expected_sites.len());
    for (projected, expected) in projected.sites().iter().zip(expected_sites) {
        assert_eq!((projected.id, projected.kind), (expected.0, expected.1));
        assert_eq!(projected.actions, expected.2);
    }
    let helper = projected
        .global_tensor_helper(0)
        .or_else(|| {
            (0..projected.program().functions.len())
                .find_map(|function| projected.function_tensor_helper(function, 0))
        })
        .expect("aggregate tensor fixture retains its verified nested helper");
    assert!(!helper.dag().is_empty());
}
