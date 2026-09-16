//! chelis#2110: cache-backed checked-library composition must retain authored
//! local tensor-ascription obligations through lowering.

use chelis_compiler_api::{
    LibraryContext, StdLibContext, build_library_context, build_stdlib_context,
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
    let execution =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 0,
        });
    let plan = chelis_ir::host::host_def_evaluation_plan(&session, "entry", &execution)
        .expect("cached composition lowers")
        .expect("entry is a tensor kernel");
    let dag = &plan.kernel_for_inspection().dag;
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
    );
    let checked_entry =
        check_ir_with_context(dependency_decoded.type_env(), &entry).expect("entry checks");
    let composed = CheckedProgram::compose(dependency_decoded.library_checked(), &checked_entry)
        .expect("checked library and entry compose");
    assert_one_lowered_site(&composed);
}
