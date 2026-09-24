//! chelis#2110: cache-backed checked-library composition must retain authored
//! local tensor-ascription obligations through lowering.

use chelis_compiler_api::{
    LibraryContext, StdLibContext, build_library_context, build_stdlib_context, pipeline,
};
use chelis_ir::dag::{ExtentWitnessSite, RiscOp};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{CheckedProgram, check_ir_with_context};

fn dependency_source() -> &'static str {
    "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
     y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
     y\n\
     }\n"
}

fn assert_one_checked_obligation(program: &CheckedProgram) {
    let [ascription] = program.local_tensor_ascriptions() else {
        panic!(
            "cached/composed checked program must retain one local obligation: {:?}",
            program.local_tensor_ascriptions()
        );
    };
    assert_eq!(ascription.binding_name(), "y");
}

fn assert_one_lowered_site(program: &CheckedProgram) {
    assert_one_checked_obligation(program);
    let session = chelis_ir::host::HostLoweringSession::new(program);
    let kernel = chelis_ir::host::host_def_kernel(&session, "entry")
        .expect("cached composition lowers")
        .expect("entry is a tensor kernel");
    let dag = &kernel.dag;
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(
                node.op,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::LocalAscriptionClaim { .. },
                    ..
                }
            ))
            .count(),
        1,
        "the cached checker carrier must lower to one explicit site"
    );
}

#[test]
fn stdlib_and_dependency_cache_roundtrips_preserve_composed_local_obligations() {
    let declarations = parse_str(dependency_source()).expect("dependency parses");
    let stdlib = build_stdlib_context(&declarations).expect("stdlib context");
    let stdlib_decoded: StdLibContext =
        bincode::deserialize(&bincode::serialize(&stdlib).expect("stdlib encode"))
            .expect("stdlib decode");

    let empty = build_stdlib_context(&[]).expect("empty stdlib");
    let dependency = build_library_context(&empty, &declarations)
        .expect("dependency context")
        .expect("dependency composes");
    let dependency_decoded: LibraryContext =
        bincode::deserialize(&bincode::serialize(&dependency).expect("dependency encode"))
            .expect("dependency decode");

    assert_one_checked_obligation(stdlib_decoded.library_checked());
    assert_one_checked_obligation(dependency_decoded.library_checked());

    let entry = desugar_program(
        &parse_str("def entry(x: tensor[*, f32]) -> tensor[*, f32] = helper(x)\n")
            .expect("entry parses"),
    )
    .expect("Surf fixture must desugar");
    let checked_entry =
        check_ir_with_context(dependency_decoded.type_env(), &entry).expect("entry checks");
    let composed = CheckedProgram::compose(dependency_decoded.library_checked(), &checked_entry)
        .expect("checked library and entry compose");
    assert_one_lowered_site(&composed);
}

#[test]
fn contextual_lowering_carries_dependency_local_obligations_into_inlined_helpers() {
    let declarations = parse_str(
        "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         y: tensor[2, f32] = exp(x)\n  \
         y\n\
         }\n",
    )
    .expect("dependency parses");
    let empty = build_stdlib_context(&[]).expect("empty stdlib");
    let dependency = build_library_context(&empty, &declarations)
        .expect("dependency context")
        .expect("dependency composes");
    let lowered = pipeline::lower_library(dependency.checked_library())
        .expect("dependency library lowers through the production carrier");
    let [dependency_ascription] = dependency.library_checked().local_tensor_ascriptions() else {
        panic!(
            "dependency must retain its authored local obligation: {:?}",
            dependency.library_checked().local_tensor_ascriptions()
        );
    };
    assert_eq!(
        lowered.raw().local_tensor_ascriptions(),
        dependency.library_checked().local_tensor_ascriptions(),
        "the production lowered-library carrier must retain the checked obligations"
    );

    let entry = desugar_program(
        &parse_str("def entry(x: tensor[*, f32]) -> tensor[2, f32] = helper(x)\n")
            .expect("entry parses"),
    )
    .expect("Surf fixture must desugar");
    let checked_entry =
        check_ir_with_context(dependency.type_env(), &entry).expect("entry checks in context");
    let lowering_map = chelis_ir::lower::top_level_lowering_map_with_context(
        lowered.raw(),
        checked_entry.exprs(),
        checked_entry.type_env(),
    );
    assert_eq!(
        lowering_map.get("entry"),
        Some(&true),
        "the contextual acceptance witness must actually enter IR lowering: {lowering_map:?}"
    );
    let dag = chelis_ir::lower::lower_program_with_context(lowered.raw(), &checked_entry);
    let claims = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            RiscOp::ExtentWitness {
                site:
                    ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id,
                        binding,
                        ..
                    },
                ..
            } => Some((*ascription_id, binding.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        claims,
        [(dependency_ascription.id().get(), "y")],
        "contextual helper inlining must use the checked dependency obligation: {dag:?}"
    );
}

#[test]
fn composed_sources_with_equal_local_names_and_offsets_lower_their_own_identities() {
    let dependency_declarations = parse_str(
        "def same(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         y: tensor[2, f32] = exp(x)\n  \
         y\n\
         }\n",
    )
    .expect("dependency parses");
    let empty = build_stdlib_context(&[]).expect("empty stdlib");
    let dependency = build_library_context(&empty, &dependency_declarations)
        .expect("dependency context")
        .expect("dependency composes");

    let application_declarations = parse_str(
        "def same(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         y: tensor[3, f32] = exp(x)\n  \
         y\n\
         }\n\
         def entry(x: tensor[*, f32]) -> tensor[*, f32] = same(x)\n",
    )
    .expect("application parses");
    let checked_application = check_ir_with_context(
        dependency.type_env(),
        &desugar_program(&application_declarations).expect("Surf fixture must desugar"),
    )
    .expect("application checks");
    let lowered_dependency =
        pipeline::lower_library(dependency.checked_library()).expect("dependency library lowers");
    let lowering_map = chelis_ir::lower::top_level_lowering_map_with_context(
        lowered_dependency.raw(),
        checked_application.exprs(),
        checked_application.type_env(),
    );
    let contextual = chelis_ir::lower::lower_program_with_context(
        lowered_dependency.raw(),
        &checked_application,
    );
    let contextual_claims = contextual
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            RiscOp::ExtentWitness {
                site:
                    ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id,
                        binding,
                        ..
                    },
                ..
            } => Some((*ascription_id, binding.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        contextual_claims,
        [(0, "y")],
        "the replacement body must not select its shadowed dependency's colliding record; \
         lowering map: {lowering_map:?}; DAG: {contextual:#?}"
    );

    let composed = CheckedProgram::compose(dependency.library_checked(), &checked_application)
        .expect("checked library and application compose");
    let [application_ascription] = composed.local_tensor_ascriptions() else {
        panic!(
            "the replacement definition must own the only reachable obligation: {:?}",
            composed.local_tensor_ascriptions()
        );
    };
    assert_eq!(application_ascription.binding_name(), "y");
    assert_eq!(application_ascription.declaration_name(), Some("same"));

    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&composed, "entry")
        .expect("the composed application lowers");
    let claims = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            RiscOp::ExtentWitness {
                site:
                    ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id,
                        binding,
                        ..
                    },
                ..
            } => Some((node.id, *ascription_id, binding.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(token, ascription_id, binding)] = claims.as_slice() else {
        panic!("only the selected application declaration should lower: {dag:?}");
    };
    assert_eq!(binding, &"y");
    assert_eq!(*ascription_id, application_ascription.id().get());
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| node.shape_deps.contains(token))
            .count(),
        1,
        "the selected application token has one initializer owner"
    );
}
