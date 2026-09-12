//! Persistent semantic-label transport, separate from the immutable prototype oracle.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{
    TypeEnv, build_compiled_library_context, check_ir_program, check_ir_with_context,
};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_str(source).expect("fixture parses"))
}

#[test]
fn concrete_instantiations_and_value_aliases_keep_names_without_coupling_calls() {
    for reverse in [false, true] {
        let operation = if reverse {
            "mul(gain,x)"
        } else {
            "mul(x,gain)"
        };
        let library = format!(
            "def aligned[d](x: tensor[d,f32], gain: tensor[fixed,f32]) -> tensor[d,f32] = {operation}"
        );
        let consumer = "def use() = { f = aligned\n g = f\n a = g(to_tensor([1.0f32,2.0f32]),to_tensor([1.0f32,2.0f32]))\n b = g(to_tensor([1.0f32,2.0f32,3.0f32]),to_tensor([1.0f32,2.0f32,3.0f32]))\n (sum(a,fixed),sum(b,fixed)) }";
        check_ir_program(&parse(&format!("{library}\n{consumer}")))
            .expect("whole: independent calls and aliases");
        let (live, _) = build_compiled_library_context(&parse(&library)).unwrap();
        let bytes = bincode::serialize(&live).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bytes).unwrap();
        for context in [&live, &decoded] {
            check_ir_with_context(context, &parse(consumer))
                .expect("context: independent calls and aliases");
        }
    }
}
