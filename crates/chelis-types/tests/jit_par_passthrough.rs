//! Pin `jit` admission and the temporary `par` checker fence.
//!
//! Per `spec/03-deep-syntax.md`:
//!
//! * §2.7 ``| `jit` | `(jit {} expr)` | Compilation trigger |`` — semantically
//!   a no-op at evaluation; the JIT effect lives in metadata.
//! * §2.3 reserves `par` for scheduler-independent parallel evaluation, but
//!   chelis#2388 shows that the execution lanes do not yet preserve its
//!   effects. Until that implementation is complete, the checker rejects the
//!   construct instead of blessing divergent behavior.
//!
//! The legacy lowerers remain available behind the checker boundary so the
//! eventual implementation can be repaired without changing the syntax.

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

/// Spec §2.3 + chelis#2388: `par` is fenced until every lane implements it.
#[test]
fn par_is_rejected_with_the_typed_issue_fence() {
    let exprs = parse(
        "(def {} a (lit {type: (t-prim {} f32)} 1.0))
         (def {} b (lit {type: (t-prim {} f32)} 2.0))
         (def {} f_par (par {} (var {} a) (var {} b)))",
    );

    let errors = check_ir_program(&exprs)
        .expect_err("par must stay behind the chelis#2388 checker fence")
        .errors;
    let [error] = errors.as_slice() else {
        panic!("expected one par fence diagnostic, got {errors:?}");
    };
    assert_eq!(error.kind.diagnostic_name(), "unsupported_feature");
    assert!(error.message.contains("unimplemented chelis#2388"));
    assert!(error.message.contains("not fully implemented"));
}
