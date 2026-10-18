//! Tensor-DAG lowering inlines a bounded number of function bodies per
//! lowering entry (chelis#3864).
//!
//! Recursion lowers by unrolling (chelis#620), and the depth caps bound how
//! deep an unroll goes. A recursion that lowering cannot decide statically and
//! that calls itself in both arms of a branch lowers both arms, so its unroll
//! is a binary tree: every path stays under the depth caps while the tree
//! doubles at each level. `chelis cost` on nautilus's root-finding examples
//! unrolled such a tree until it ran out of memory. The inlining budget makes
//! the entry a loud lowering rejection instead.
//!
//! The oracle counts the bodies lowering inlines under a test budget, so the
//! tree is cut early, on nautilus's bisection (`Nautilus.Roots`, inlined here
//! as one file).

use chelis_ir::lower::with_inline_budget;
use chelis_types::CheckedProgram;

const TEST_BUDGET: usize = 2_000;

/// nautilus's `Nautilus.Roots` bisection and its `examplerootfind` caller,
/// with the iteration bound left open.
const BISECTION: &str = r#"def r_abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then neg(x) else x
def r_nan_f32() -> f32 = cast(0.0, f32) |> div(cast(0.0, f32))
def r_is_nan_f32(x: f32) -> bool = neq(x, x)
def bisection_rec(f: f32 -> f32, lo: f32, hi: f32, flo: f32, tol: f32, iters: i64) -> f32 = {
  zero_i = cast(0, i64)
  one_i = cast(1, i64)
  if lte(iters, zero_i) then r_nan_f32() else {
    width = sub(hi, lo)
    if lt(width, tol) then (cast(0.5, f32) |> mul(add(lo, hi))) else {
      mid = cast(0.5, f32) |> mul(add(lo, hi))
      if r_is_nan_f32(mid) then r_nan_f32() else if or(eq(mid, lo), eq(mid, hi)) then mid else {
        fmid = f(mid)
        if r_is_nan_f32(fmid) then r_nan_f32() else {
          afmid = r_abs_f32(fmid)
          if lt(afmid, tol) then mid else {
            same_sign = flo |> mul(fmid) |> gt(cast(0.0, f32))
            if same_sign then bisection_rec(f, mid, hi, fmid, tol, sub(iters, one_i)) else bisection_rec(f, lo, mid, flo, tol, sub(iters, one_i))
          }
        }
      }
    }
  }
}
def bisection(f: f32 -> f32, lo: f32, hi: f32, tol: f32, max_iters: i64) -> f32 = {
  nan_input = r_is_nan_f32(lo) |> or(r_is_nan_f32(hi)) |> or(r_is_nan_f32(tol))
  if nan_input then r_nan_f32() else {
    flo = f(lo)
    if r_is_nan_f32(flo) then r_nan_f32() else {
      aflo = r_abs_f32(flo)
      if lt(aflo, tol) then lo else {
        fhi = f(hi)
        if r_is_nan_f32(fhi) then r_nan_f32() else {
          afhi = r_abs_f32(fhi)
          if lt(afhi, tol) then hi else {
            prod = mul(flo, fhi)
            if gt(prod, cast(0.0, f32)) then r_nan_f32() else bisection_rec(f, lo, hi, flo, tol, max_iters)
          }
        }
      }
    }
  }
}
def erf_projectile(theta: f32) -> f32 = {
  two_theta = mul(cast(2.0, f32), theta)
  s = sin(two_theta)
  g = cast(9.81, f32)
  v_sq = mul(cast(20.0, f32), cast(20.0, f32))
  d = cast(30.0, f32)
  sub(mul(div(v_sq, g), s), d)
}
def example() -> f32 = bisection(erf_projectile, cast(0.1, f32), cast(0.7, f32), cast(1e-8, f32), cast(ITERS, i64))"#;

fn checked(source: &str) -> CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let checked = chelis_types::check_typed_program(&deep).expect("the probe checks");
    let checked = chelis_effects::check_program(&checked).expect("the probe has no effect errors");
    chelis_types::check_linearity(&checked).expect("the probe is linear")
}

fn lower_example(iters: usize, budget: usize) -> (bool, chelis_ir::lower::InlineBudgetProbe) {
    let program = checked(&BISECTION.replace("ITERS", &iters.to_string()));
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let names = vec!["example".to_string()];
            let (dags, probe) = with_inline_budget(budget, || {
                chelis_ir::host::lower_named_tensor_entry_dags(&program, &names)
                    .map(|dag| dag.is_some())
                    .collect::<Vec<_>>()
            });
            (dags[0], probe)
        })
        .expect("spawn the lowering thread")
        .join()
        .expect("the lowering thread finishes")
}

#[test]
fn an_unroll_that_branches_at_runtime_is_rejected_at_the_budget() {
    let (lowered, probe) = lower_example(2, TEST_BUDGET);
    assert!(!lowered, "the entry is rejected, not lowered");
    assert_eq!(
        probe.inlined,
        TEST_BUDGET + 1,
        "lowering stopped at the first body past the budget"
    );
    let message = probe.rejection.expect("the budget rejected the entry");
    assert!(
        message.contains(&format!(
            "exceeded the budget of {TEST_BUDGET} inlined function bodies"
        )) && message.contains("while unrolling the recursion of `bisection_rec`")
            && message.contains("chelis#620"),
        "the rejection names the budget, the callee and the unroll rule: {message}"
    );
}

#[test]
fn the_budget_is_what_stops_it() {
    // A larger test budget lets the same entry inline more before it is
    // rejected: the unroll does not end on its own within either.
    let (_, small) = lower_example(2, TEST_BUDGET);
    let (_, large) = lower_example(2, 2 * TEST_BUDGET);
    assert_eq!(small.inlined, TEST_BUDGET + 1);
    assert_eq!(large.inlined, 2 * TEST_BUDGET + 1);
}

#[test]
fn an_entry_under_the_budget_lowers_and_counts_each_body_once() {
    // Each call lowers its callee's body once: `quad` inlines itself and its
    // two `square` calls, so the entry counts three bodies per `quad` call.
    let program = checked(
        "def square(x: f32) -> f32 = (x * x)\n\
         def quad(x: f32) -> f32 = square(square(x))\n\
         def example() -> f32 = (quad(cast(2.0, f32)) + quad(cast(3.0, f32)))\n",
    );
    let names = vec!["example".to_string()];
    let (dags, probe) = with_inline_budget(TEST_BUDGET, || {
        chelis_ir::host::lower_named_tensor_entry_dags(&program, &names)
            .map(|dag| dag.is_some())
            .collect::<Vec<_>>()
    });
    assert_eq!(dags, [true], "the entry lowers");
    assert_eq!(probe.inlined, 6, "each inlined body counts once");
    assert_eq!(probe.rejection, None);
}
