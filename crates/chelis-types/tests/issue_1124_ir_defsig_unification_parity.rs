//! chelis#1124: the IR ingress must run the same `def`-body-vs-`defsig`
//! unification the typed ingress runs.
//!
//! `chelis check` on a `.dp` file drives `check_ir_program`. Before the fix,
//! `infer_ir_program_with_state` prebound each def name to its own BODY type
//! stamp (via `collect_ir_types_with_origins`) and let that overwrite the
//! authoritative binding a `(defsig ...)` had already installed. The later
//! body-vs-declared unification in `infer_top_level` then compared the body
//! against itself, so a `defsig`/body type mismatch was SILENTLY ACCEPTED by
//! the IR ingress while the typed ingress rejected it. This was a verified
//! fail-open in the shipped checker. PP9 gives both entries one function-only
//! metadata prebind that never overwrites an explicit signature.
//!
//! The contract these tests lock: for every input the IR ingress
//! (`check_ir_program`) and the typed ingress (`check_typed_program`) produce
//! the identical ordered diagnostic set. A `defsig`/body mismatch is rejected
//! by BOTH with the same message; a matching `defsig`/body and a `defsig`-less
//! def are accepted by BOTH (the over-rejection guard: the fix must not reject
//! any program the body stamp and the declared signature agree on).

use chelis_deep::parse_and_stamp;
use chelis_types::{check_ir_program, check_typed_program};

/// The ordered `(kind, message)` diagnostic set for the IR ingress.
fn ir_diagnostics(source: &str) -> Vec<String> {
    let exprs = parse_and_stamp(source).expect("fixture must parse and stamp");
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|e| format!("{:?}: {}", e.kind, e.message))
            .collect(),
    }
}

/// The ordered `(kind, message)` diagnostic set for the typed ingress.
fn typed_diagnostics(source: &str) -> Vec<String> {
    let exprs = parse_and_stamp(source).expect("fixture must parse and stamp");
    match check_typed_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|e| format!("{:?}: {}", e.kind, e.message))
            .collect(),
    }
}

/// The blank line between the two top-level forms is required: the un-spaced
/// two-liner is rejected by the style gate before the checker runs. The
/// checker API bypasses that gate, but the fixtures mirror the shipped `.dp`
/// surface, so they keep the blank line.
const MISMATCH: &str = "(defsig {} k (t-prim {} f32))\n\n\
                        (def {} k (lit {type: (t-prim {} int32)} 1))\n";

const MATCHING: &str = "(defsig {} k (t-prim {} f32))\n\n\
                        (def {} k (lit {type: (t-prim {} f32)} 1.0))\n";

const NO_DEFSIG: &str = "(def {} k (lit {type: (t-prim {} int32)} 1))\n";

/// A `defsig`-less def (`use_base`) that forward-references another
/// `defsig`-less def (`base`) declared later. chelis#1134 / [04-INF-4]
/// requires both ingresses to reject it rather than letting serialized body
/// metadata manufacture value scope. The authoritative parity coverage lives
/// in `issue_1134_forward_reference_parity`.
const DEFSIG_LESS_CROSS_REF: &str = "(def {} use_base (var {} base))\n\n\
                                     (def {} base (lit {type: (t-prim {} int32)} 7))\n";

/// Positive rejection: a `defsig`/body type mismatch is rejected by the IR
/// ingress with EXACTLY the diagnostic the typed ingress reports. This is the
/// fail-open chelis#1124 closes.
#[test]
fn ir_ingress_rejects_defsig_body_mismatch_like_typed_ingress() {
    let ir = ir_diagnostics(MISMATCH);
    let typed = typed_diagnostics(MISMATCH);

    assert_eq!(
        ir, typed,
        "IR and typed ingress must report the identical diagnostic set for a \
         defsig/body mismatch (chelis#1124 fail-open)"
    );
    assert_eq!(
        ir,
        vec![
            "TypeMismatch: def 'k' body doesn't match declared signature: \
             body has type `int32`, declared type is `f32`"
                .to_string()
        ],
        "the mismatch must be the exact declared-signature diagnostic"
    );
}

/// Negative parity / over-rejection guard: when the `defsig` type and the body
/// stamp AGREE, both ingresses accept with zero diagnostics. The fix skips the
/// body-stamp prebind only for `defsig`-owning names, so it must not turn any
/// agreeing program into an error.
#[test]
fn ir_ingress_accepts_matching_defsig_body_like_typed_ingress() {
    let ir = ir_diagnostics(MATCHING);
    let typed = typed_diagnostics(MATCHING);

    assert_eq!(
        ir, typed,
        "IR and typed ingress must agree on a matching def"
    );
    assert!(
        ir.is_empty(),
        "a matching defsig/body must be accepted by both ingresses, got: {ir:?}"
    );
}

/// Over-rejection guard: an ordinary `defsig`-less def remains accepted at
/// both ingresses. Function body stamps have the separate forward-call role
/// below; eager values do not receive module-wide metadata scope.
#[test]
fn ir_ingress_still_binds_defsig_less_def_from_body_stamp() {
    let ir = ir_diagnostics(NO_DEFSIG);
    let typed = typed_diagnostics(NO_DEFSIG);

    assert_eq!(
        ir, typed,
        "IR and typed ingress must agree on a defsig-less def"
    );
    assert!(
        ir.is_empty(),
        "a defsig-less def must remain accepted by both ingresses, got: {ir:?}"
    );
}

/// A `defsig`-less def (`caller`) that forward-references another
/// `defsig`-less FUNCTION (`helper`) declared later, where `helper`'s body
/// carries its own type stamp.
///
/// This is chelis#1124's over-rejection sentinel. Resolving the reference is
/// exactly the job of the body-stamp prebind the chelis#1124 fix leaves in
/// place for `defsig`-less names: `helper` is bound from its body stamp before
/// `caller`'s body is inferred, and if the fix's `defsig`-name skip ever
/// stripped a `defsig`-less binding, `helper` would be unbound.
///
/// It reads a FUNCTION deliberately. [04-INF-4] governs non-function `def`s
/// only, so the equivalent value shape below is now a rejection at both
/// ingresses and can no longer observe the prebind. PP9 / [04-TOT-5] makes
/// this function-metadata capability common to both checker entries.
const DEFSIG_LESS_FORWARD_FN: &str = "(def {} caller (fn {} (params {}) (app {} (var {} helper))))\n\n\
     (def {} helper (fn {type: (t-fn {} (t-prim {} int32))} (params {}) \
     (lit {type: (t-prim {} int32)} 1)))\n";

/// Over-rejection sentinel for the prebind's core job (chelis#1124 review
/// condition 3): a `defsig`-less def that forward-references another
/// `defsig`-less function must stay accepted by the IR ingress. The fix's
/// `defsig`-name skip must NOT strip the body-stamp binding that reference
/// depends on.
#[test]
fn ir_ingress_resolves_defsig_less_forward_function_reference() {
    let ir = ir_diagnostics(DEFSIG_LESS_FORWARD_FN);
    let typed = typed_diagnostics(DEFSIG_LESS_FORWARD_FN);

    assert_eq!(
        ir, typed,
        "PP9 requires ingress parity for forward functions"
    );
    assert!(
        ir.is_empty(),
        "a defsig-less forward function reference must remain accepted by the \
         checker (the shared prebind binds the referenced def from its body stamp), \
         got: {ir:?}"
    );
}

/// Cross-issue regression sentinel: chelis#1124's body-stamp prebind must not
/// bypass chelis#1134's sequential top-level value scope.
#[test]
fn both_ingresses_reject_defsig_less_forward_cross_reference() {
    let ir = ir_diagnostics(DEFSIG_LESS_CROSS_REF);
    let typed = typed_diagnostics(DEFSIG_LESS_CROSS_REF);

    assert_eq!(ir, typed, "chelis#1134 requires ingress parity");
    assert!(
        ir.iter()
            .any(|diagnostic| diagnostic.starts_with("UnboundVariable:")),
        "a defsig-less forward cross-reference must reject at both ingresses: {ir:?}"
    );
}
