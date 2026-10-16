//! chelis#1277 Slice B2h: the shared kernel decision, `host_def_kernel`,
//! answers "is this def a kernel" before any lowering runs, and its answer
//! is the one the C host program acts on.
//!
//! Each `None` below names a class the byte-identity corpus found the kernel
//! lowering rejecting AFTER a kernel decision (the C lane then fell through
//! to host code, chelis#1515; the eval lane errored). The decision now names
//! the class itself, so both lanes choose host code for the same reason and
//! no lowering error is caught to decide a lane.
//!
//! EVIDENTIARY STATUS: `a_pure_tensor_def_is_a_kernel`,
//! `an_io_effect_row_is_host` and `a_match_on_a_constructor_literal_stays_a_kernel`
//! are disposition locks (they hold on the tree without the walker too); the
//! four others are regression tests, watched returning `Err` (a failed kernel
//! lowering) on the tree without the walker and `Ok(false)` with it.

use chelis_ir::host::{HostLoweringSession, host_def_kernel};

fn checked_surf(src: &str) -> chelis_types::CheckedProgram {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs).unwrap_or_else(|report| {
        panic!(
            "type check failed: {:?}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        )
    });
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn decision(src: &str, name: &str) -> Result<bool, String> {
    let program = checked_surf(src);
    host_def_kernel(&HostLoweringSession::new(&program), name)
        .map(|kernel| kernel.is_some())
        .map_err(|diagnostic| diagnostic.to_string())
}

#[test]
fn a_pure_tensor_def_is_a_kernel() {
    let src = "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = mul(x, x)\n";
    assert_eq!(decision(src, "f"), Ok(true));
}

#[test]
fn an_io_effect_row_is_host() {
    let src = "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = {\n  _ = print(x)\n  x\n}\n";
    assert_eq!(decision(src, "f"), Ok(false));
}

/// chelis#520 D1: only a compile-time-known constructor scrutinee lowers. The
/// match sits inside a `let` block (the issue_1200 shape): a `match` at the
/// body ROOT is already on the keep-in-host list, so only the nested form
/// reaches the walker.
#[test]
fn a_match_on_a_runtime_scrutinee_is_host_before_lowering() {
    let src = "type Flag =\n  | On\n  | Off\n\
               def run(c: Flag, x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               y = add(x, x)\n  match c with {\n    | On => y\n    | Off => x\n  }\n}\n";
    assert_eq!(decision(src, "run"), Ok(false));
}

/// The positive twin: a constructor-literal scrutinee inside the same block
/// stays a kernel, so the class does not swallow the static matches the
/// lowering does select.
#[test]
fn a_match_on_a_constructor_literal_stays_a_kernel() {
    let src = "type Flag =\n  | On\n  | Off\n\
               def run(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               y = add(x, x)\n  match On with {\n    | On => y\n    | Off => x\n  }\n}\n";
    assert_eq!(decision(src, "run"), Ok(true));
}

/// chelis#1058: the compiled lowering carries literal window and stride
/// lists only; a runtime list keeps the def in host code on both lanes.
#[test]
fn a_runtime_window_list_is_host_before_lowering() {
    let src = "def f(x: tensor[6, f32], w: i64, s: i64) -> tensor[5, f32] = \
               reduce_window_max(x, [w], [s])\n";
    assert_eq!(decision(src, "f"), Ok(false));
}

/// chelis#776: a `pad` fill that does not resolve statically keeps the def in
/// host code on both lanes.
#[test]
fn a_runtime_pad_fill_is_host_before_lowering() {
    let src = "def f(x: tensor[4, f32], r: f32) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], r)\n";
    assert_eq!(decision(src, "f"), Ok(false));
}

/// A host-only builtin reached through a `let` inside the body (the
/// issue_1222 shape the corpus found).
#[test]
fn a_host_only_builtin_inside_the_body_is_host_before_lowering() {
    let src = "def f(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
               z: List[i64] = [7i64]\n  m = len(z)\n  if (m > 0i64) then x else x\n}\n";
    assert_eq!(decision(src, "f"), Ok(false));
}

/// Section 4.7 orders interface checks by the declaring signature even when
/// the implementation extracts a tensor helper. Shape-only inputs remain ABI
/// witnesses; an unrelated dead input must not change their relative order.
#[test]
fn tensor_helper_retains_declaring_parameter_order() {
    use chelis_ir::host::lower_named_tensor_entry_dag;
    let program = checked_surf(
        "def f(b: tensor[f32], unused: tensor[f32], z: tensor[rows, f32], a: tensor[cols, f32]) -> tensor[4, 3, f32] = insert(insert(b, 0i32, shape(a, 0i32)), 0i32, shape(z, 0i32))\n",
    );
    let kernel = host_def_kernel(&HostLoweringSession::new(&program), "f")
        .unwrap()
        .unwrap();
    assert_eq!(
        kernel
            .inputs
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["b", "z", "a"]
    );
    assert_load_order_after_rebuilding(&kernel.dag, &["b", "z", "a"]);
    let dag = lower_named_tensor_entry_dag(&program, "f").unwrap();
    assert_load_order_after_rebuilding(&dag, &["b", "z", "a"]);
}

fn assert_load_order_after_rebuilding(dag: &chelis_ir::Dag, expected: &[&str]) {
    let mut dag = chelis_ir::optimize::dead_code_eliminate(dag);
    chelis_ir::optimize::constant_fold(&mut dag);
    let dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    let names = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            chelis_ir::RiscOp::Load { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(names, expected);
}

/// A signatureless subexpression has an assigned deterministic ABI order.
/// Reversing insertion into its unordered scope must not reverse that ABI.
#[test]
fn signatureless_helper_retains_assigned_abi_order() {
    use chelis_ir::dag::{DimInfo, TensorType};
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;
    let expr = chelis_deep::parser::parse_str("(app {} (var {} add) (var {} z) (var {} a))")
        .unwrap()
        .remove(0);
    for names in [["z", "a"], ["a", "z"]] {
        let mut scope = UnordMap::new();
        for name in names {
            scope.insert(
                name.to_owned(),
                TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::F32,
                },
            );
        }
        let dag = chelis_ir::lower::try_lower_subexpr_program(
            &expr,
            scope,
            UnordMap::new(),
            UnordMap::new(),
        )
        .unwrap();
        assert_load_order_after_rebuilding(&dag, &["a", "z"]);
    }
}

/// [04-INF-9]/[05-OP-20..22]: a lexical runtime callable has no DAG
/// representation. Select host execution before visiting either value arm.
#[test]
fn runtime_callable_values_are_host_before_lowering() {
    for choice in [
        "if flag then is_nan else is_finite",
        "if flag then is_finite else is_nan",
        "if flag then known else is_finite",
    ] {
        for alias in ["f(x)", "{ g = f\n g(x) }", "{ g = copy(f)\n g(x) }"] {
            let source = format!(
                "def selected(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = {{\n known = is_nan\n f = {choice}\n {alias}\n}}\n"
            );
            assert_eq!(decision(&source, "selected"), Ok(false), "{source}");
        }
    }
}

/// The host disposition must not consume ordinary tensor conditionals or
/// statically resolved builtin aliases, which already have a DAG identity.
#[test]
fn static_callable_and_tensor_conditionals_keep_the_kernel_route() {
    for body in [
        "{ f = is_finite\n f(x) }",
        "{ y = if flag then is_finite(x) else is_nan(x)\n y }",
    ] {
        let source =
            format!("def selected(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = {body}\n");
        assert_eq!(decision(&source, "selected"), Ok(true), "{source}");
    }
}

/// Deep sequential let / [04-INF-9]: sparse checked lexical arm nodes retain
/// their result evidence before a numeric graph can encounter control flow.
#[test]
fn lexical_block_callable_choices_select_host_before_numeric_lowering() {
    for explicit in [false, true] {
        for nested_copy in [false, true] {
            for reverse in [false, true] {
                let signature = if explicit {
                    ": &tensor[2,f64] -> tensor[2,bool]"
                } else {
                    ""
                };
                let tail = if nested_copy {
                    "copy(copy(local))"
                } else {
                    "local"
                };
                let (left, right) = if reverse {
                    ("is_finite", "is_nan")
                } else {
                    ("is_nan", "is_finite")
                };
                let source = format!(
                    "def run(flag: bool, x: tensor[2,f64]) -> tensor[2,bool] = {{\n f{signature} = if flag then {{ local = {left}\n {tail} }} else {{ local = {right}\n {tail} }}\n alias = copy(f)\n alias(x)\n}}\n"
                );
                assert_eq!(decision(&source, "run"), Ok(false), "{source}");
            }
        }
    }
}

/// A lexical block is not itself a reason to leave an ordinary tensor kernel.
#[test]
fn lexical_block_tensor_choices_preserve_kernel_admission() {
    let source = "def run(flag: bool, x: tensor[2,f64]) -> tensor[2,f64] = {\n y = if flag then { local = abs(x)\n local } else { local = neg(x)\n local }\n y\n}\n";
    assert_eq!(decision(source, "run"), Ok(true));
}

/// A checked noncallable declaration reference is a Load. Its lazy host
/// initializer is not inlined into the consuming numerical kernel.
#[test]
fn captured_checked_tensor_value_is_a_kernel_load_boundary() {
    let source = "weights: tensor[2, f32] = { _ = print(\"initialize\")\n to_tensor([1.0f32,2.0f32]) }\ndef run(x: tensor[2, f32]) -> tensor[2, f32] = add(x, weights)\n";
    assert_eq!(decision(source, "run"), Ok(true));
}

/// A real called function executes its body in the call; host effects are
/// not hidden merely because its result is a noncallable tensor.
#[test]
fn captured_load_boundary_does_not_erase_inline_function_body_effects() {
    let source = "def helper(x: tensor[2, f32]) -> tensor[2, f32] ! { IO } = { _ = print(\"enter\")\n add(x,x) }\ndef run(x: tensor[2, f32]) -> tensor[2, f32] ! { IO } = helper(x)\n";
    assert_eq!(decision(source, "run"), Ok(false));
}

/// A declaration whose value is a runtime callable must retain its host
/// selection/formation boundary; a non-Fn syntax root cannot bypass it.
#[test]
fn captured_load_boundary_does_not_admit_runtime_callable_initializers() {
    let source = "flag = true\npredicate: &tensor[2, f32] -> tensor[2, bool] = { _ = print(\"form\")\n if flag then is_nan else is_finite }\ndef run(x: tensor[2, f32]) -> tensor[2, bool] = predicate(x)\n";
    assert_eq!(decision(source, "run"), Ok(false));
}
