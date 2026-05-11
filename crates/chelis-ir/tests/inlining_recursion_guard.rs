//! `inlining_names` recursion-guard fixtures (Inlining-F1).
//!
//! `try_lower_callable_app` (`crates/chelis-ir/src/lower.rs`) inserts the
//! resolved callable's name into `self.inlining_names` before lowering the
//! callable body, then removes it after. The intent is to prevent infinite
//! recursion when a *truly* self-recursive named definition (e.g.
//! `def f(x) = f(x)`) would otherwise loop forever during inlining.
//!
//! The guard fires inside `resolve_callable_expr_inner` at the `var` arm:
//!
//! ```ignore
//! if !visited.insert(name.clone()) || self.inlining_names.contains(&name) {
//!     return None;
//! }
//! ```
//!
//! The bug (Inlining-F1, surfaced by PR #37's call-site parity test
//! comment): the guard tracks the *name* of the callee, not whether the
//! recursion is real. When a fn-typed parameter `f` is substituted into
//! `local_callables["f"] = doubler` by `lower_plain_callable_app` at the
//! call site, and the inlined body contains a nested application like
//! `f(f(x))`, the outer `f` call inserts `"f"` into `inlining_names`
//! before lowering its arguments. When the inner `f(x)` is then lowered
//! as one of those arguments, the resolver sees `"f"` in `inlining_names`
//! and returns `None` — even though `f` resolves to `doubler` (a concrete
//! callable whose body does not reference `f` at all, so no actual
//! recursion is possible).
//!
//! The fixtures here pin both halves:
//!
//! 1. `nested_fn_param_call_lowers_via_substituted_callable` — TARGET.
//!    `def outer(f, x) = f(f(x))` called as `outer(doubler, seed)` must
//!    evaluate to `doubler(doubler(seed))`. Today this silently lowers
//!    the inner `f(x)` to its argument value (the `lower_app` fallback
//!    "Not a recognized built-in — lower func and args, return last")
//!    instead of applying `doubler`, so the result equals
//!    `doubler(seed)` instead of `doubler(doubler(seed))`. The test
//!    asserts the correct doubled-twice result and is gated
//!    `#[ignore]` until the guard is narrowed.
//!
//! 2. `true_self_recursion_still_rejected_by_inlining_guard` — CONTROL.
//!    `def loop_self(x) = loop_self(x)` is genuinely self-recursive: the
//!    body references its own name, the substituted callable IS itself,
//!    and naively inlining would loop forever. The guard must still
//!    reject this case after the fix; otherwise we've disabled the
//!    infinite-recursion protection. Today this lowers without panic
//!    because the guard fires and `lower_app`'s fallback drops the call
//!    silently. After the fix, the guard must still fire here.

use std::collections::HashMap;

use chelis_deep::Expr;
use chelis_ir::dag::{DimInfo, NodeId, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::lower::lower_subexpr_program;
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

    let mut program_defs = HashMap::new();
    program_defs.insert("doubler".to_string(), parse_one(doubler_src));
    program_defs.insert("outer".to_string(), parse_one(outer_src));

    let scoped = HashMap::from([("seed".to_string(), f32_vec(3))]);
    let dag = lower_subexpr_program(&parse_one(call_src), scoped, HashMap::new(), program_defs);

    let inputs = HashMap::from([(
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
            (out.data[i] - want).abs() < 1e-6,
            "nested f(f(seed)) must equal 4*seed elementwise: index {i} \
             expected {want}, got {:?}",
            out.data,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Control: real self-recursion still rejected by the inlining guard.
//
// `def loop_self(x) = loop_self(x)` — body references its own name. The
// substituted callable for `loop_self` is itself; naive inlining would
// recurse forever. The guard must still fire here after the Inlining-F1
// fix, otherwise we've disabled the infinite-recursion protection.
//
// The check: lowering succeeds (no panic, no infinite loop), and the
// resulting DAG node for the outer call evaluates to the *input*, not to
// an infinitely-expanded chain. The guard makes the body lower as
// "lower func and args, return last" — which for `loop_self(x)` returns
// the lowered `x`, an identity. So a single application of `loop_self`
// returns its argument unchanged. Two applications still return the
// argument unchanged. The key behavioral invariant: lowering finishes
// in bounded time and does not panic, which proves the guard fired.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn true_self_recursion_still_rejected_by_inlining_guard() {
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

    let mut program_defs = HashMap::new();
    program_defs.insert("loop_self".to_string(), parse_one(loop_self_src));

    let scoped = HashMap::from([("seed".to_string(), f32_vec(3))]);
    // The contract: lowering finishes without panic or infinite loop. The
    // guard fires; lower_app's fallback returns the lowered `x` argument.
    let dag = lower_subexpr_program(&parse_one(call_src), scoped, HashMap::new(), program_defs);

    let inputs = HashMap::from([(
        "seed".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    )]);
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(
        !roots.is_empty(),
        "guard-protected lowering still produces a root"
    );
    let values = eval_tensor_roots_with_strict(&dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds; guard prevents infinite recursion");
    let out = &values[roots.last().unwrap()];
    assert_eq!(out.shape, vec![3]);
    // Guard fired: the recursive call lowers to "return last arg", i.e. the
    // seed itself. This is the expected fallback behaviour for a callable
    // the DAG genuinely cannot represent (no `RiscOp::Call`). The point of
    // this control is that lowering TERMINATES, not that it produces
    // useful values.
    assert_eq!(
        out.data,
        vec![1.0, 2.0, 3.0],
        "true self-recursion guard must finish lowering with a finite \
         result (the fallback identity); changing this output means the \
         guard has been disabled; fix is too aggressive."
    );
}
