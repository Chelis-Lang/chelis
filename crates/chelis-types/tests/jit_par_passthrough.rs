//! Pin the spec-impl mismatch for `jit` and `par` at the checker level.
//!
//! Per `spec/03-deep-syntax.md`:
//!
//! * §2.7 ``| `jit` | `(jit {} expr)` | Compilation trigger |`` — semantically
//!   a no-op at evaluation; the JIT effect lives in metadata.
//! * §2.3 ``| `par` | `(par {} expr1 expr2 ...)` | Parallel evaluation (v1:
//!   sequential) |`` — v1 promises sequential composition.
//!
//! Today the IR validator rejects both at `crates/chelis-types/src/infer.rs`
//! (search for ``matches!(tag, "par" | "jit")``). Positive fixtures are
//! `#[ignore]`-gated until the rejection is lifted; the matching negative
//! fixtures pin the rejection so we notice if it is moved or renamed without
//! the positive path being flipped on.

use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

fn parse(src: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(src).expect("deep parse")
}

// ── jit ────────────────────────────────────────────────────────────────────

/// Spec §2.7: `jit` is a compilation trigger; at eval it is a no-op.
/// Once the checker stops rejecting it, this fixture flips on.
#[test]
#[ignore = "jit/par spec-impl mismatch, see crates/chelis-types/src/infer.rs `matches!(tag, \"par\" | \"jit\")` rejection"]
fn jit_wrapping_a_value_type_checks_and_carries_inner_type() {
    let exprs = parse(
        "(def {} y (lit {type: (t-prim {} f32)} 1.5))
         (def {} f_jit (jit {} (var {} y)))",
    );

    let res = check_ir_program(&exprs);
    assert!(
        res.is_ok(),
        "jit({{value}}) should type-check (spec §2.7 — compilation trigger, no-op at eval). errors={:?}",
        res.err().map(|r| r.errors)
    );
}

/// Today: the rejection fires. If it stops firing without flipping the positive
/// fixture, we want to notice.
#[test]
fn jit_is_rejected_with_current_ir_lowering_message() {
    let exprs = parse("(def {} y (jit {} (lit {type: (t-prim {} f32)} 1.0)))");
    let res = check_ir_program(&exprs);
    let err = res.expect_err(
        "jit currently rejected by validate_ir_expr; remove this test when the rejection is lifted",
    );
    assert!(
        err.errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::Other)
                && e.message.contains("`jit` is not supported by IR lowering")),
        "expected `jit` rejection from validate_ir_expr, got: {:?}",
        err.errors
    );
}

// ── par ────────────────────────────────────────────────────────────────────

/// Spec §2.3: `par` v1 is sequential. Once the checker stops rejecting it,
/// this fixture flips on.
#[test]
#[ignore = "jit/par spec-impl mismatch, see crates/chelis-types/src/infer.rs `matches!(tag, \"par\" | \"jit\")` rejection"]
fn par_sequential_body_type_checks_and_yields_last_type() {
    let exprs = parse(
        "(def {} a (lit {type: (t-prim {} f32)} 1.0))
         (def {} b (lit {type: (t-prim {} f32)} 2.0))
         (def {} f_par (par {} (var {} a) (var {} b)))",
    );

    let res = check_ir_program(&exprs);
    assert!(
        res.is_ok(),
        "par({{a; b}}) should type-check (spec §2.3 — v1 sequential). errors={:?}",
        res.err().map(|r| r.errors)
    );
}

/// Today: the rejection fires. Pin it so we notice if it moves silently.
#[test]
fn par_is_rejected_with_current_ir_lowering_message() {
    let exprs = parse(
        "(def {} f_par (par {} (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0)))",
    );
    let res = check_ir_program(&exprs);
    let err = res.expect_err(
        "par currently rejected by validate_ir_expr; remove this test when the rejection is lifted",
    );
    assert!(
        err.errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::Other)
                && e.message.contains("`par` is not supported by IR lowering")),
        "expected `par` rejection from validate_ir_expr, got: {:?}",
        err.errors
    );
}
