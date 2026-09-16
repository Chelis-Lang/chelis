//! Recursion-unroll fixtures (Inlining-F1 successor, chelis#620).
//!
//! `lower_plain_callable_app` (`crates/chelis-ir/src/lower.rs`) tracks a
//! per-callee active-inline depth (`inlining_depths`) plus a total active
//! count (`inlining_active`). Recursion lowers by BOUNDED UNROLLING: a
//! re-entrant call to a callee that is mid-inline resolves and inlines
//! normally, and a well-founded recursion terminates through the static
//! `if`/`match` pruning of its base case. Two caps
//! (`MAX_STATIC_RECURSION_DEPTH` per name, `MAX_TOTAL_INLINE_DEPTH`
//! overall) are the loud backstop for chains the pruning cannot bound.
//!
//! History: the original Inlining-F1 guard (`inlining_names`, a
//! refuse-on-reentry set consulted by `resolve_callable_expr_inner`) fixed
//! the fn-typed-parameter alias bug (`outer(f, x) = f(f(x))` wrongly
//! tripping on the non-recursive inner `f(x)`) but made a truly recursive
//! call fall through to `lower_app`'s silently wrong return-last-arg
//! fallback. chelis#620 replaces refuse-on-reentry with depth-bounded
//! unrolling so statically-terminating recursive builders (the School
//! im2col/pool patch collectors) lower for real, and non-terminating
//! recursion errors loudly instead of silently returning its argument.
//!
//! The fixtures pin three contracts:
//!
//! 1. `nested_fn_param_call_lowers_via_substituted_callable` — the
//!    Inlining-F1 target, unchanged: `outer(doubler, seed)` must evaluate
//!    to `doubler(doubler(seed))`; the alias re-entry must not be treated
//!    as recursion.
//!
//! 2. `true_self_recursion_errors_loudly_at_unroll_cap` — CONTROL.
//!    `def loop_self(x) = loop_self(x)` has no base case; lowering must
//!    terminate in bounded time with the named per-callee cap diagnostic,
//!    never hang and never silently collapse to the argument.
//!
//! 3. `static_base_case_recursion_unrolls_within_cap` — a recursion whose
//!    base case resolves through the chelis#620 static `if` pruning
//!    unrolls to completion (~500 levels, near the 512 cap, which also
//!    probes the Rust-stack headroom assumption in debug builds).

use chelis_unord::UnordMap;

use chelis_deep::Expr;
use chelis_ir::dag::{DimInfo, NodeId, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::lower::{lower_subexpr_program, try_lower_subexpr_program};
use chelis_types::types::Prim;

fn f32_vec(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn parse_one(src: &str) -> Expr {
    chelis_deep::parser::parse_str(src)
        .expect("parse failed")
        .into_iter()
        .next()
        .expect("at least one expr")
}

// ─────────────────────────────────────────────────────────────────────────────
// Target: nested fn-typed parameter application.
//
// `def doubler(x) = add(x, x)`                       (concrete callable)
// `def outer(f: tensor -> tensor, x) = f(f(x))`      (nested non-pipe app)
// `def call(seed) = outer(doubler, seed)`            (call site we lower)
//
// Expected: `outer(doubler, seed) = doubler(doubler(seed)) = 4 * seed`.
// Today: the inner `f(x)` resolution trips `inlining_names.contains("f")`
// (the outer `f` call inserted it), so the inner app falls back to "return
// last lowered arg" — the inner `f` is dropped and we get `doubler(seed) =
// 2 * seed` instead of `4 * seed`. (Or zero/partial values depending on
// fallback specifics; the assertion below pins the correct result.)
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn nested_fn_param_call_lowers_via_substituted_callable() {
    // `doubler(x) = add(x, x)` — concrete, non-recursive.
    let doubler_src = r#"
        (fn {}
          (params {}
            (x {type: (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))}))
          (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} add)
            (copy {} (var {} x))
            (copy {} (var {} x))))
    "#;
    // `outer(f, x) = f(f(x))` — nested fn-typed parameter application.
    let outer_src = r#"
        (fn {}
          (params {}
            (f {type: (t-fn {}
                         (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))
                         (t-tensor {} (d-lit {} 3) (t-prim {} f32)))})
            (x {type: (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))}))
          (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} f)
            (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
              (var {} f)
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))))
    "#;
    // Call site: `outer(doubler, seed)`.
    let call_src = r#"
        (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
          (var {} outer)
          (var {} doubler)
          (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} seed))
    "#;

    let mut program_defs = UnordMap::new();
    program_defs.insert("doubler".to_string(), parse_one(doubler_src));
    program_defs.insert("outer".to_string(), parse_one(outer_src));

    let scoped = UnordMap::from([("seed".to_string(), f32_vec(3))]);
    let dag = lower_subexpr_program(&parse_one(call_src), scoped, UnordMap::new(), program_defs);

    let inputs = UnordMap::from([(
        "seed".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    )]);
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(
        !roots.is_empty(),
        "lowered call must produce at least one root"
    );
    let values = eval_tensor_roots_with_strict(&dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds");
    let out = &values[roots.last().unwrap()];
    assert_eq!(out.shape, vec![3]);
    // doubler(doubler([1, 2, 3])) = doubler([2, 4, 6]) = [4, 8, 12].
    let expected = [4.0, 8.0, 12.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-6,
            "nested f(f(seed)) must equal 4*seed elementwise: index {i} \
             expected {want}, got {:?}",
            out.to_f64_lossy_vec(),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Control: real self-recursion errors loudly at the unroll cap.
//
// `def loop_self(x) = loop_self(x)` — body references its own name and has
// no base case, so no amount of static pruning can bound the unroll. The
// contract (chelis#620): lowering terminates in bounded time with the
// named per-callee cap diagnostic. It must NOT hang, and it must NOT
// silently collapse the call to its argument (the pre-#620 fallback).
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn true_self_recursion_errors_loudly_at_unroll_cap() {
    // `loop_self(x) = loop_self(x)` — self-referential.
    let loop_self_src = r#"
        (fn {}
          (params {}
            (x {type: (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))}))
          (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} loop_self)
            (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)))
    "#;
    // Call site: `loop_self(seed)`.
    let call_src = r#"
        (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
          (var {} loop_self)
          (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} seed))
    "#;

    let mut program_defs = UnordMap::new();
    program_defs.insert("loop_self".to_string(), parse_one(loop_self_src));

    let scoped = UnordMap::from([("seed".to_string(), f32_vec(3))]);
    let diagnostic =
        try_lower_subexpr_program(&parse_one(call_src), scoped, UnordMap::new(), program_defs)
            .expect_err("unbounded self-recursion must be rejected, not silently dropped");
    let message = diagnostic.to_string();
    assert!(
        message.contains("static unroll limit"),
        "cap diagnostic must name the unroll limit: {message}"
    );
    assert!(
        message.contains("loop_self"),
        "cap diagnostic must name the recursive callee: {message}"
    );
    assert!(
        message.contains("chelis#620"),
        "cap diagnostic must cite the issue: {message}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Positive: a recursion with a statically-resolvable base case unrolls.
//
// `def count_up(x, k) = if gte(k, 500) then x else count_up(x, k + 1)`
// called as `count_up(seed, 0)`: every level's condition folds (k is a
// literal-rooted Add chain), the base case prunes at k == 500, and the
// whole chain lowers to the identity on `seed`. 500 levels sits just
// under the 512 per-name cap, so this also probes the Rust-stack headroom
// assumption behind the caps in a debug build.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn static_base_case_recursion_unrolls_within_cap() {
    let count_up_src = r#"
        (fn {}
          (params {}
            (x {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))})
            (k {type: (t-prim {} i64)}))
          (if {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (app {} (var {} gte)
              (var {type: (t-prim {} i64)} k)
              (cast {} (lit {} 500) i64))
            (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
            (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
              (var {} count_up)
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
              (app {} (var {} add)
                (var {type: (t-prim {} i64)} k)
                (cast {} (lit {} 1) i64)))))
    "#;
    let call_src = r#"
        (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
          (var {} count_up)
          (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} seed)
          (cast {} (lit {} 0) i64))
    "#;

    let mut program_defs = UnordMap::new();
    program_defs.insert("count_up".to_string(), parse_one(count_up_src));

    let scoped = UnordMap::from([("seed".to_string(), f32_vec(3))]);
    let dag = lower_subexpr_program(&parse_one(call_src), scoped, UnordMap::new(), program_defs);

    let inputs = UnordMap::from([(
        "seed".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    )]);
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(!roots.is_empty(), "unrolled lowering must produce a root");
    let values = eval_tensor_roots_with_strict(&dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds after full static unroll");
    let out = &values[roots.last().unwrap()];
    assert_eq!(out.shape, vec![3]);
    assert_eq!(
        out.to_f64_lossy_vec(),
        vec![1.0, 2.0, 3.0],
        "count_up is the identity on its tensor argument after 500 pruned levels"
    );
}
