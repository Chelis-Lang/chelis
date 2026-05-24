//! Issue #187 regression: `shrink` and `stride` are registered as 1-arg
//! `tensor_unop` but the IR lowering / evaluator consume window parameters
//! from `args[1..]`. As a result no callable form exists from Surf source:
//!   - bare 1-arg `shrink(x)` / `stride(x)` type-checks but is not
//!     host-runnable (`unsupported builtin in host runtime`).
//!   - parameterized `shrink(x, bounds)` / `stride(x, s0, s1, ...)`
//!     type-rejects with `arity mismatch: expected 1 args`.
//!
//! This file exercises the type-signature side via `check_ir_program`
//! (the `chelis check` entry point). Host-runtime parity is covered in
//! `chelis-compiler-api/tests/issue_187_shrink_stride_host_runtime.rs`.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.4, which documents
//!   shrink: (&tensor[D,p], bounds)  -> tensor[D',p]
//!   stride: (&tensor[D,p], strides) -> tensor[D',p]

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

// ---------------------------------------------------------------------------
// Positive: parameterized forms type-check.
// ---------------------------------------------------------------------------

/// EXPECT: `shrink(&x, [[0, 1], [1, 3]])` on a `tensor[2, 4, f32]` type-checks
/// to `tensor[1, 2, f32]` (the windowed sub-tensor shape).
#[test]
fn issue187_surf_shrink_parameterized_typechecks() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0, 1], [1, 3]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized shrink, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: `stride(&x, 1, 2)` on a `tensor[2, 4, f32]` type-checks to
/// `tensor[2, 2, f32]` (stride 1 on axis 0, stride 2 on axis 1 -> dims
/// 2 and ceil(4/2) = 2).
#[test]
fn issue187_surf_stride_parameterized_typechecks() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 1, 2)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized stride, got {} error(s)",
            rep.errors.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Negative parity: malformed parameterized forms are rejected with clear
// errors. Per the §0 plan, each negative test names ONE pinned failure mode.
// ---------------------------------------------------------------------------

/// EXPECT: bare `shrink(&x)` (no bounds arg) is rejected with an arity
/// error. Before the fix this typed cleanly via the 1-arg `tensor_unop`
/// scheme, but the resulting RISC node had no bounds and the host runtime
/// had no arm for it, so it was a silent-corruption surface.
#[test]
fn issue187_surf_shrink_bare_no_bounds_is_arity_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = shrink(&x)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bare shrink(&x)");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("bounds") || e.message.contains("arity"))),
        "expected a shrink/arity/bounds error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `shrink(&x, [[0, 5], [1, 3]])` on `tensor[2, 4, f32]` is rejected
/// because the axis-0 bound (0, 5) ends past the input dimension (2).
#[test]
fn issue187_surf_shrink_out_of_range_bound_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[5, 2, f32] = shrink(&x, [[0, 5], [1, 3]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on out-of-range shrink bound");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("axis") || e.message.contains("bound"))),
        "expected a shrink axis/bound error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `shrink(&x, [[0, 1]])` on a rank-2 tensor is rejected because
/// the bounds list has fewer pairs than the input rank.
#[test]
fn issue187_surf_shrink_wrong_rank_bounds_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 4, f32] = shrink(&x, [[0, 1]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on rank-mismatched shrink bounds");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("rank") || e.message.contains("bound"))),
        "expected a shrink rank/bound error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, 0, 2)` is rejected because a zero stride is not
/// allowed (it would either be a division-by-zero in the strided index
/// arithmetic or a no-shrink/no-advance footgun).
#[test]
fn issue187_surf_stride_zero_step_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = stride(&x, 0, 2)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on zero stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("zero") || e.message.contains("positive"))),
        "expected a stride zero/positive error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, 1)` on a rank-2 tensor is rejected because the
/// strides list has fewer entries than the input rank.
#[test]
fn issue187_surf_stride_wrong_rank_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = stride(&x, 1)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on rank-mismatched stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("rank") || e.message.contains("axis"))),
        "expected a stride rank/axis error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `stride(&x, "two")` is rejected because stride entries must be
/// int32, not strings.
#[test]
fn issue187_surf_stride_non_int_is_error() {
    let src = r#"
def g(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = stride(&x, "two")
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on non-int stride");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("stride")
                && (e.message.contains("int")
                    || e.message.contains("integer")
                    || e.message.contains("type"))),
        "expected a stride type error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Sibling sweep: `pad` had the same `tensor_unop`-vs-parameterized antipattern
// (registered as 1-arg, lowering reads padding from args[1] and fill from
// args[2]). Fixed in the same PR per the issue #187 sibling-sweep policy.
// ---------------------------------------------------------------------------

/// EXPECT: `pad(&x, [[0, 1], [0, 1]], 0.0)` on a `tensor[2, 4, f32]`
/// type-checks to `tensor[3, 5, f32]` (input dim plus lo plus hi per axis).
#[test]
fn issue187_sibling_pad_parameterized_typechecks() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[0, 1], [0, 1]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for parameterized pad, got {} error(s)",
            rep.errors.len()
        );
    }
}

/// EXPECT: bare `pad(&x)` is rejected with an arity error.
#[test]
fn issue187_sibling_pad_bare_is_arity_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[2, 4, f32] = pad(&x)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bare pad(&x)");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("padding") || e.message.contains("arity"))),
        "expected a pad/arity error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// EXPECT: `pad(&x, [[0, 1]], 0.0)` on a rank-2 tensor is rejected
/// (wrong-rank padding list).
#[test]
fn issue187_sibling_pad_wrong_rank_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 4, f32] = pad(&x, [[0, 1]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on wrong-rank pad padding");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("rank") || e.message.contains("padding"))),
        "expected a pad rank/padding error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Red team round 1 on PR #214: malformed pair literals silently fell through
// to `Dim::Wildcard` at infer time instead of being rejected. Compensating
// host-runtime / IR-verifier safeguards still caught these one layer
// downstream, but the desired behavior is a clean type error at
// `chelis check`.
// ---------------------------------------------------------------------------

/// R1-F1: `shrink(&x, [[0, 1, 2]])` -- inner pair is a triple, not a pair.
/// Must be rejected at infer time with a message naming the offending axis.
#[test]
fn red_team_214_r1_f1_shrink_triple_inner_pair_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0, 1, 2]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on triple-element inner pair");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 3"))),
        "expected a shrink pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `shrink(&x, [[0]])` -- inner pair is a singleton missing the end.
#[test]
fn red_team_214_r1_f1_shrink_singleton_inner_pair_is_error() {
    let src = r#"
def f(x: tensor[2, 4, f32]) -> tensor[1, 2, f32] = shrink(&x, [[0]])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on singleton inner pair");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("shrink")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 1"))),
        "expected a shrink pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `pad(&x, [[1]], 0.0)` -- inner singleton inside pad's pair list.
#[test]
fn red_team_214_r1_f1_pad_singleton_inner_pair_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on singleton inner pair in pad");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 1"))),
        "expected a pad pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F1: `pad(&x, [[1, 0, 99]], 0.0)` -- inner triple inside pad's pair
/// list. Mirror of the shrink triple case.
#[test]
fn red_team_214_r1_f1_pad_triple_inner_pair_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1, 0, 99]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on triple inner pair in pad");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("pair")
                    || e.message.contains("2-element")
                    || e.message.contains("got 3"))),
        "expected a pad pair/length error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// R1-F2: pad fill type was not enforced at infer time. A list fill, a
// bool fill, or a string fill all slipped through to host-runtime; spec
// §2.4 says fill is a scalar of the input precision.
// ---------------------------------------------------------------------------

/// R1-F2: `pad(&x, ..., [0.0])` -- list fill rather than scalar. Must be
/// rejected at infer.
#[test]
fn red_team_214_r1_f2_pad_list_fill_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1, 0], [0, 1]], [0.0])
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on list pad fill");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("fill") || e.message.contains("scalar"))),
        "expected a pad fill/scalar error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F2: `pad(&x, ..., true)` on an f32 tensor -- bool fill, wrong
/// precision. Must be rejected at infer.
#[test]
fn red_team_214_r1_f2_pad_bool_fill_is_error() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1, 0], [0, 1]], true)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("expected check to fail on bool pad fill");
    assert!(
        rep.errors
            .iter()
            .any(|e| e.message.to_lowercase().contains("pad")
                && (e.message.contains("fill")
                    || e.message.contains("scalar")
                    || e.message.contains("precision"))),
        "expected a pad fill/scalar/precision error, got {:?}",
        rep.errors
            .iter()
            .map(|e| e.message.clone())
            .collect::<Vec<_>>()
    );
}

/// R1-F2 positive control: `pad(&x, ..., 0.0)` on f32 tensor still passes.
/// Locks the constraint to "fill matches input precision," not "fill is
/// forbidden."
#[test]
fn red_team_214_r1_f2_pad_matching_scalar_fill_still_typechecks() {
    let src = r#"
def p(x: tensor[2, 4, f32]) -> tensor[3, 5, f32] = pad(&x, [[1, 0], [0, 1]], 0.0)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    if let Err(rep) = res {
        for err in &rep.errors {
            eprintln!("unexpected error: {:?}: {}", err.kind, err.message);
        }
        panic!(
            "expected clean check for matching f32 pad fill, got {} error(s)",
            rep.errors.len()
        );
    }
}
