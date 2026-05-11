//! WS-B2: Contextual tensor-literal inference acceptance suite.
//!
//! Spec references:
//! - `spec/02-surf-syntax.md` §P10b
//! - `spec/04-type-system.md` §5.6
//!
//! When a tensor literal `[e1, e2, ...]` appears in a position with a
//! known element type, the unsuffixed numeric literals in the body
//! adopt that element type instead of the §5.3 / §P10 literal default
//! (int32 for integer literals, f32 for float literals).
//!
//! The closed set of "known-element-type" positions is exactly four:
//!   1. RHS of a let-binding whose declared type is a tensor type
//!   2. Argument position of a call whose callee has a declared signature
//!      with a tensor parameter at that position
//!   3. Body expression of a function with a declared return type that is
//!      a tensor type, when the body is itself a tensor literal
//!   4. First argument of an explicit `cast(literal, p)`
//!
//! Outside the closed set, numeric literals fall back to the WS-0 / D1
//! default: integer literals → int32, float literals → f32.

use chelis_deep::printer::print_canonical_flat;
use chelis_types::infer::infer_program;

/// Parse Surf source, desugar, and run inference. Returns the desugared
/// Deep AST as a printable string vector so tests can pin exact literal
/// type metadata, plus the inference result so tests can pin error
/// counts.
fn pipeline(src: &str) -> (Vec<String>, chelis_types::infer::InferResult) {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let printed: Vec<String> = exprs
        .iter()
        .map(|e| {
            print_canonical_flat(std::slice::from_ref(e))
                .trim()
                .to_string()
        })
        .collect();
    let result = infer_program(&exprs);
    (printed, result)
}

/// True if any printed Deep node contains a `lit` whose type metadata
/// is `(t-prim {} <prec>)`. The desugarer may attach a `span:` entry
/// alongside `type:` in the metadata map, so the match accepts both
/// orderings: `{type: ...}` and `{span: "...", type: ...}`.
fn contains_lit_with_prim(printed: &[String], prec: &str) -> bool {
    let needle = format!("type: (t-prim {{}} {prec})");
    printed.iter().any(|p| {
        // Look for `(lit {...type: (t-prim {} <prec>)...}` — the
        // literal must be inside a `lit` form, not just any context
        // that mentions the prim name (e.g. a `t-tensor` element).
        let mut search_from = 0usize;
        while let Some(idx) = p[search_from..].find(&needle) {
            let absolute = search_from + idx;
            // Walk back to the nearest `(` and check the tag.
            let prefix = &p[..absolute];
            // Find the most recent `(<tag> {` opener before `absolute`.
            // The literal form is `(lit {`. We accept any tag here as
            // long as the tag immediately preceding `{type:` is `lit`.
            if let Some(open_idx) = prefix.rfind("(lit ") {
                // Make sure no closing paren intervenes between
                // `(lit ` and our needle position (so we're inside the
                // same lit form's metadata map).
                let between = &prefix[open_idx..];
                let opens = between.matches('(').count();
                let closes = between.matches(')').count();
                if opens > closes {
                    return true;
                }
            }
            search_from = absolute + 1;
        }
        false
    })
}

fn errors_summary(result: &chelis_types::infer::InferResult) -> String {
    if result.errors.is_empty() {
        "(no errors)".to_string()
    } else {
        result
            .errors
            .iter()
            .map(|e| format!("[{:?}] {}", e.kind, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── Position 1: RHS of a let-binding with declared tensor type ──

#[test]
fn position_1_let_binding_tensor_f64_narrows_float_literals() {
    let (printed, result) = pipeline("xs: tensor[3, f64] = [1.0, 2.0, 3.0]");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    // Each literal must carry (t-prim {} f64), not (t-prim {} f32).
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals, got:\n{}",
        printed.join("\n")
    );
    assert!(
        !contains_lit_with_prim(&printed, "f32"),
        "expected NO f32-typed literals (the §P10 default must be \
         overridden by the contextual rule), got:\n{}",
        printed.join("\n")
    );
}

#[test]
fn position_1_let_binding_tensor_int64_narrows_int_literals() {
    let (printed, result) = pipeline("xs: tensor[3, int64] = [1, 2, 3]");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "int64"),
        "expected int64-typed literals, got:\n{}",
        printed.join("\n")
    );
    assert!(
        !contains_lit_with_prim(&printed, "int32"),
        "expected NO int32-typed literals (the contextual rule must \
         override the §P10 default), got:\n{}",
        printed.join("\n")
    );
}

#[test]
fn position_1_no_annotation_float_default_is_f32() {
    let (printed, result) = pipeline("xs = [1.0, 2.0, 3.0]");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    // Outside the closed contextual set, the §P10 default applies:
    // float literals → f32, no to_tensor wrap (the result is List f32).
    assert!(
        contains_lit_with_prim(&printed, "f32"),
        "expected f32-typed literals (the §P10 default), got:\n{}",
        printed.join("\n")
    );
}

#[test]
fn position_1_no_annotation_int_default_is_int32() {
    // Negative parity for the int-default rule: a bare `[1, 2, 3]` in
    // an unannotated position must produce int32 literals, not int64.
    // Per spec §5.3 / §P10 this is the user-facing contract; silently
    // widening to int64 would violate the WS-0 pin.
    let (printed, result) = pipeline("xs = [1, 2, 3]");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "int32"),
        "expected int32-typed literals (the §P10 default), got:\n{}",
        printed.join("\n")
    );
    assert!(
        !contains_lit_with_prim(&printed, "int64"),
        "expected NO int64-typed literals — silent widening to int64 \
         would violate the WS-0 pin per spec §5.3, got:\n{}",
        printed.join("\n")
    );
}

// ── Position 4: cast(literal, p) ──

#[test]
fn position_4_cast_int_list_to_int8() {
    let (printed, result) = pipeline("xs = cast([1, 2, 3], int8)");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "int8"),
        "expected int8-typed literals, got:\n{}",
        printed.join("\n")
    );
}

#[test]
fn position_4_cast_float_list_to_f64() {
    let (printed, result) = pipeline("xs = cast([1.0, 2.0, 3.0], f64)");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals, got:\n{}",
        printed.join("\n")
    );
}

// ── Position 3: function with declared return type ──

#[test]
fn position_3_fn_return_tensor_f64_narrows_body_literals() {
    let (printed, result) = pipeline("def g() -> tensor[3, f64] = [1.0, 2.0, 3.0]");
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals in the body, got:\n{}",
        printed.join("\n")
    );
    assert!(
        !contains_lit_with_prim(&printed, "f32"),
        "expected NO f32-typed literals — declared return type f64 \
         must override the §P10 default, got:\n{}",
        printed.join("\n")
    );
}

// ── Position 2: call with declared-signature tensor parameter ──

#[test]
fn position_2_call_with_tensor_f64_param_narrows_arg_literals() {
    let src = "def f(xs: tensor[3, f64]) -> tensor[3, f64] = xs\nys = f([1.0, 2.0, 3.0])";
    let (printed, result) = pipeline(src);
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    // The `ys = f([1.0, 2.0, 3.0])` form must narrow argument
    // literals to f64 because f's declared signature is
    // `tensor[3, f64] -> tensor[3, f64]`.
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals for the call-arg position, \
         got:\n{}",
        printed.join("\n")
    );
}

#[test]
fn position_2_call_with_sig_decl_tensor_f64_param_narrows_arg_literals() {
    // Position 2 must also fire when the callee has a separate `sig`
    // declaration (no `def` ascription). The desugarer collects
    // tensor-param prims from `Decl::Sig` Arrow types, so this exercises
    // the Decl::Sig branch of `collect_top_level_fn_tensor_param_prec`.
    let src = "sig f: tensor[3, f64] -> tensor[3, f64]\n\
               def f(xs) = xs\n\
               ys = f([1.0, 2.0, 3.0])";
    let (printed, result) = pipeline(src);
    assert!(
        result.errors.is_empty(),
        "expected no errors, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals via sig declaration, got:\n{}",
        printed.join("\n")
    );
}

// ── Negative coverage ──

#[test]
fn negative_bare_int_list_does_not_silently_default_to_int64() {
    // Per spec §5.3 / §P10, a bare `[1, 2, 3]` in an unannotated
    // position must produce int32 literals, NOT int64. Silent widening
    // to int64 would violate the WS-0 pin.
    let (printed, _result) = pipeline("xs = [1, 2, 3]");
    let combined = printed.join("\n");
    assert!(
        combined.contains("(t-prim {} int32)"),
        "expected int32 literal default, got:\n{}",
        combined
    );
    assert!(
        !combined.contains("(t-prim {} int64)"),
        "bare unannotated [1, 2, 3] must not silently default to \
         int64 — got:\n{}",
        combined
    );
}

#[test]
fn negative_bare_float_list_does_not_silently_default_to_f64() {
    let (printed, _result) = pipeline("xs = [1.0, 2.0, 3.0]");
    let combined = printed.join("\n");
    assert!(
        combined.contains("(t-prim {} f32)"),
        "expected f32 literal default, got:\n{}",
        combined
    );
    assert!(
        !combined.contains("(t-prim {} f64)"),
        "bare unannotated [1.0, 2.0, 3.0] must not silently default \
         to f64 — got:\n{}",
        combined
    );
}

// ── Negative: mixed-suffix in tensor body errors clearly ──
//
// WS-B1 has not yet landed Surf literal suffix grammar (`1.0f32`,
// `2.0f64`, etc.), so the mixed-suffix scenario can only be expressed
// via the Deep canonical form. The Deep equivalent of
// `[1.0, 2.0f64, 3.0]` in an f32 tensor context is a Cons/Nil chain
// where one literal carries an explicit `(lit {type: f64} ...)`
// metadata while the surrounding tensor declares f32.
//
// The current type checker rejects this via element-type unification
// at the `Cons` constructor: `Cons` requires every list element to
// have the same type, so a (lit f32) and (lit f64) in the same chain
// cannot unify. The error surfaces as `TypeMismatch`. WS-B2 does NOT
// add a per-index diagnostic (that requires walking the chain inside
// `to_tensor` inference, which is left for the WS-B1+B2 integration
// pass once the suffix grammar is in place); it relies on the
// existing Cons-unification rejection so mixed-suffix programs do not
// silently type-check.

#[test]
fn negative_deep_mixed_dtype_in_to_tensor_body_errors() {
    // Mixed-dtype Cons chain with explicit type metadata. The first
    // and third literals are (t-prim {} f32); the middle literal is
    // (t-prim {} f64). With the surrounding defsig declaring f32, the
    // type checker must NOT silently accept the chain.
    let src = "(defsig {} xs (t-tensor {} (d-lit {} 3) (t-prim {} f32)))
               (def {} xs (app {} (var {} to_tensor)
                 (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.0)
                   (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 2.0)
                     (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 3.0)
                       (var {} Nil))))))";
    let exprs = chelis_deep::parser::parse_str(src).expect("deep parse");
    let result = infer_program(&exprs);
    assert!(
        !result.errors.is_empty(),
        "expected at least one error for mixed-dtype tensor body in \
         f32 context, got: (no errors)"
    );
    // The error must mention the f32/f64 dtype divergence so the user
    // can locate the offending entry. The classification surfaces as
    // either TypeMismatch (Cons element-type unification path) or
    // PrecisionMismatch (precision-aware unification path); both
    // satisfy the "mixed dtype must not silently type-check" contract.
    let any_dtype_diagnostic = result.errors.iter().any(|e| {
        matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
                | chelis_types::errors::CheckErrorKind::PrecisionMismatch
        ) && e.message.contains("f32")
            && e.message.contains("f64")
    });
    assert!(
        any_dtype_diagnostic,
        "expected TypeMismatch or PrecisionMismatch mentioning both \
         f32 and f64; got:\n{}",
        errors_summary(&result)
    );
}

// ── Outside-the-closed-set fallback (negative parity) ──

#[test]
fn outside_closed_set_let_in_block_no_annotation_uses_default() {
    // A let-binding inside a block with no type annotation is NOT in
    // the closed set; it must use the §P10 default. (Block lets here
    // are local let-bindings inside a `{ x = ...; expr }` form, not
    // module-level LetDef.)
    let src = "def main() = {\n  xs = [1.0, 2.0, 3.0]\n  xs\n}";
    let (printed, _result) = pipeline(src);
    let combined = printed.join("\n");
    assert!(
        combined.contains("(t-prim {} f32)"),
        "expected f32 default in non-contextual let, got:\n{}",
        combined
    );
}

#[test]
fn position_1_block_level_let_with_tensor_type_narrows_literals() {
    // Position 1 also covers block-level `let` bindings, not just
    // module-level LetDef. Spec §P10b enumerates "the right-hand side
    // of a `let`-binding whose declared type is a tensor type" — both
    // top-level and inside a function body satisfy this.
    let src = "def main() = {\n  xs: tensor[3, f64] = [1.0, 2.0, 3.0]\n  xs\n}";
    let (printed, result) = pipeline(src);
    assert!(
        result.errors.is_empty(),
        "expected no errors at block let with tensor type, got:\n{}",
        errors_summary(&result)
    );
    assert!(
        contains_lit_with_prim(&printed, "f64"),
        "expected f64-typed literals at block-let position, got:\n{}",
        printed.join("\n")
    );
}
