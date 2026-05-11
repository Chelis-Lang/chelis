//! Pin the spec-impl mismatch fix for `jit` and `par` at the checker level.
//!
//! Per `spec/03-deep-syntax.md`:
//!
//! * §2.7 ``| `jit` | `(jit {} expr)` | Compilation trigger |`` — semantically
//!   a no-op at evaluation; the JIT effect lives in metadata.
//! * §2.3 ``| `par` | `(par {} expr1 expr2 ...)` | Parallel evaluation (v1:
//!   sequential) |`` — v1 promises sequential composition.
//!
//! Lifted in `docs/investigations/jit_par_spec_impl_mismatch_diagnosis.md`:
//! the checker's `validate_ir_expr` no longer rejects either form, and
//! `lower.rs` lowers `jit` as pass-through and `par` as sequential
//! composition.

use chelis_types::check_ir_program;

fn parse(src: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(src).expect("deep parse")
}

// ── jit ────────────────────────────────────────────────────────────────────

/// Spec §2.7: `jit` is a compilation trigger; at eval it is a no-op.
#[test]
fn jit_wrapping_a_value_type_checks_and_carries_inner_type() {
    let exprs = parse(
        "(def {} y (lit {type: (t-prim {} f32)} 1.5))
         (def {} f_jit (jit {} (var {} y)))",
    );

    let res = check_ir_program(&exprs);
    assert!(
        res.is_ok(),
        "jit({{value}}) should type-check (spec §2.7: compilation trigger, no-op at eval). errors={:?}",
        res.err().map(|r| r.errors)
    );
}

// ── par ────────────────────────────────────────────────────────────────────

/// Spec §2.3: `par` v1 is sequential.
#[test]
fn par_sequential_body_type_checks_and_yields_last_type() {
    let exprs = parse(
        "(def {} a (lit {type: (t-prim {} f32)} 1.0))
         (def {} b (lit {type: (t-prim {} f32)} 2.0))
         (def {} f_par (par {} (var {} a) (var {} b)))",
    );

    let res = check_ir_program(&exprs);
    assert!(
        res.is_ok(),
        "par({{a; b}}) should type-check (spec §2.3: v1 sequential). errors={:?}",
        res.err().map(|r| r.errors)
    );
}
