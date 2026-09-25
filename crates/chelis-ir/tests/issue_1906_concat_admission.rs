//! [05-OP-62], spec/04 §4.5, spec/05 §4.2: actual concat capability,
//! not result-annotation spelling, decides whether a helper has a DAG.
use chelis_ir::host::{HostLoweringSession, host_def_kernel};

fn checked(src: &str) -> chelis_types::CheckedProgram {
    let parsed = chelis_surf::parser::parse_str(src).expect("parse");
    let expanded = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&parsed).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand")
    .into_exprs();
    let typed = chelis_types::check_ir_program(&expanded).expect("type check");
    let effected = chelis_effects::check_program(&typed).expect("effect check");
    chelis_types::check_linearity(&effected).expect("linearity")
}

fn decision(source: &str, name: &str, kernel: bool) {
    let program = checked(source);
    let result = host_def_kernel(&HostLoweringSession::new(&program), name)
        .map(|value| value.is_some())
        .map_err(|error| error.to_string());
    assert_eq!(result, Ok(kernel), "{source}");
}

const JOIN: &str =
    "def join_columns[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([x, y], 1i32)\n";

#[test]
fn annotated_dynamic_helper_is_host_before_lowering() {
    decision(
        &format!(
            "{JOIN}def probabilities[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = softmax(join_columns(x, y), -1)\n"
        ),
        "probabilities",
        false,
    );
}

#[test]
fn annotated_dynamic_direct_is_host_before_lowering() {
    decision(
        "def probabilities[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = softmax(concat([x, y], 1i32), -1)\n",
        "probabilities",
        false,
    );
}

#[test]
fn static_direct_concat_remains_kernel() {
    decision(
        "def probabilities[s](x: tensor[s, 2, f32], y: tensor[s, 2, f32]) -> tensor[s, 4, f32] = softmax(concat([x, y], 1i32), -1)\n",
        "probabilities",
        true,
    );
}

#[test]
fn static_actuals_through_wildcard_helper_remain_kernel() {
    decision(
        &format!(
            "{JOIN}def probabilities[s](x: tensor[s, 2, f32], y: tensor[s, 2, f32]) -> tensor[s, 4, f32] = softmax(join_columns(x, y), -1)\n"
        ),
        "probabilities",
        true,
    );
}

#[test]
fn static_lexical_alias_and_list_remain_kernel() {
    decision(
        "def probabilities[s](x: tensor[s, 2, f32], y: tensor[s, 2, f32]) -> tensor[s, 4, f32] = {\n a = x\n rows = [a, y]\n softmax(concat(rows, 1i32), -1)\n}\n",
        "probabilities",
        true,
    );
}

#[test]
fn dynamic_lexical_alias_is_host() {
    decision(
        "def probabilities[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = {\n a = x\n rows = [a, y]\n softmax(concat(rows, 1i32), -1)\n}\n",
        "probabilities",
        false,
    );
}

#[test]
fn distinct_calls_do_not_reuse_first_shape_decision() {
    let source = format!(
        "{JOIN}def fixed[s](x: tensor[s, 2, f32]) -> tensor[s, 4, f32] = softmax(join_columns(x, x), -1)\ndef dynamic[s](x: tensor[s, *, f32]) -> tensor[s, *, f32] = softmax(join_columns(x, x), -1)\n"
    );
    decision(&source, "fixed", true);
    decision(&source, "dynamic", false);
    decision(&source, "fixed", true);
    decision(&source, "dynamic", false);
}

#[test]
fn same_activation_static_then_dynamic_is_host() {
    decision(
        &format!(
            "{JOIN}def run[s](x: tensor[s, 2, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = {{\n a = join_columns(x, x)\n _ = a\n softmax(join_columns(y, y), -1)\n}}\n"
        ),
        "run",
        false,
    );
}

#[test]
fn same_activation_dynamic_then_static_is_host() {
    decision(
        &format!(
            "{JOIN}def run[s](x: tensor[s, *, f32], y: tensor[s, 2, f32]) -> tensor[s, *, f32] = {{\n a = join_columns(x, x)\n _ = a\n softmax(join_columns(y, y), -1)\n}}\n"
        ),
        "run",
        false,
    );
}

#[test]
fn literal_shadow_does_not_inherit_runtime_input_fact() {
    decision(
        "def run(x: tensor[2, *, f32]) -> tensor[2, 4, f32] = {\n x = to_tensor([[1.0, 2.0], [3.0, 4.0]])\n softmax(concat([x, x], 1i32), -1)\n}\n",
        "run",
        true,
    );
}

#[test]
fn swapped_actuals_with_fresh_formals_preserve_static_kernel() {
    let helper =
        "def second[s](a: tensor[s, *, f32], b: tensor[s, *, f32]) = concat([b, b], 1i32)\n";
    for (first_width, second_width, kernel) in [("*", "2", false), ("2", "*", true)] {
        decision(
            &format!(
                "{helper}def run[s](x: tensor[s, {first_width}, f32], y: tensor[s, {second_width}, f32]) -> tensor[s, *, f32] = softmax(second(y, x), -1)\n"
            ),
            "run",
            kernel,
        );
    }
}

#[test]
fn shape_identity_arithmetic_preserves_dynamic_concat_geometry() {
    decision(
        "def run[s](x: tensor[s, *, f32]) -> tensor[s, *, f32] = {\n z = mul(x, x)\n softmax(concat([z, z], 1i32), -1)\n}\n",
        "run",
        false,
    );
}

#[test]
fn checked_matmul_producer_selects_host_for_runtime_concat_geometry() {
    decision(
        "def run[s](x: tensor[s, s, f32]) -> tensor[s, *, f32] = {\n z = matmul(x, x)\n softmax(concat([z, z], 1i32), -1)\n}\n",
        "run",
        false,
    );
}

#[test]
fn forced_dynamic_concat_rejects_instead_of_fabricating_a_load() {
    use chelis_ir::dag::{DimInfo, TensorType};
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;
    let expr = chelis_deep::parser::parse_str("(app {} (var {} concat) (app {} (var {} Cons) (var {} x) (app {} (var {} Cons) (var {} x) (var {} Nil))) (lit {} 1))").unwrap().remove(0);
    let scope = UnordMap::from([(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("*".to_string(), None)],
            precision: Prim::F32,
        },
    )]);
    let error =
        chelis_ir::lower::try_lower_subexpr_program(&expr, scope, UnordMap::new(), UnordMap::new())
            .expect_err("no fabricated input");
    assert!(
        error
            .to_string()
            .contains("tensor concat cannot be represented by the static tensor DAG")
    );
}
