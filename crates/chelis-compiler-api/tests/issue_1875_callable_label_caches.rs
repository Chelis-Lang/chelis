//! #1875 trusted owner-cache checker transport. Compiled-context lowering
//! diagnostics are retained separately; this test does not claim that repair.
use chelis_compiler_api::{
    LibraryContext, StdLibContext, build_library_context, build_stdlib_context,
};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_ir_with_context;

fn helper(reverse: bool) -> String {
    let body = if reverse {
        "mul(gain,x)"
    } else {
        "mul(x,gain)"
    };
    format!("def aligned[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = {body}")
}

#[test]
fn stdlib_and_dependency_direct_snapshots_keep_callable_labels() {
    for reverse in [false, true] {
        let declarations = parse_str(&helper(reverse)).unwrap();
        let stdlib = build_stdlib_context(&declarations).unwrap();
        let stdlib_decoded: StdLibContext =
            bincode::deserialize(&bincode::serialize(&stdlib).unwrap()).unwrap();
        let empty = build_stdlib_context(&[]).unwrap();
        let dependency = build_library_context(&empty, &declarations)
            .unwrap()
            .unwrap();
        let dependency_decoded: LibraryContext =
            bincode::deserialize(&bincode::serialize(&dependency).unwrap()).unwrap();
        for env in [
            stdlib.type_env(),
            stdlib_decoded.type_env(),
            dependency.type_env(),
            dependency_decoded.type_env(),
        ] {
            for (source, accepts) in [
                (
                    "def use[d](x: tensor[d,f32], gain: tensor[fixed,f32]) = { f = aligned\n sum(f(x,gain),fixed) }",
                    true,
                ),
                (
                    "def wrong[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[17,f32] = { f = aligned\n f(x,gain) }",
                    false,
                ),
            ] {
                let verdict = check_ir_with_context(
                    env,
                    &desugar_program(&parse_str(source).unwrap())
                        .expect("Surf fixture must desugar"),
                );
                assert_eq!(verdict.is_ok(), accepts, "{source}: {verdict:?}");
            }
        }
    }
}
