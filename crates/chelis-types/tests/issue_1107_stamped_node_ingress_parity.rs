//! chelis#1107 — the two checker ingresses must agree on stamped Deep.
//!
//! Spec authority: spec/design/checker_totality.md §C1.1/§C1.2 and
//! spec/04-type-system.md §10 [04-TOT-1].
//!
//! `check_ir_program` normalizes `Expr::Node` into `Expr::List` before
//! inference (`normalize_nodes_to_lists`), while `check_typed_program` walks
//! the stamped tree directly. `infer_expr`'s Node bridge rebuilds only the node
//! it dispatches on — `Node::to_list` is shallow — so on the typed ingress
//! every CHILD of a bridged node is still an `Expr::Node`. Any reader that
//! destructured `Expr::List` and fell through on anything else therefore
//! skipped its input on one ingress and read it on the other, and the two
//! ingresses returned different verdicts for the same program.
//!
//! The reported symptom (#1107) was the `record` / `record-update` kv-value
//! slot: no field of a record literal was type-checked at all on the stamped
//! ingress, so a bool into an f32 field checked clean there while
//! `check_ir_program` rejected it — and a WELL-FORMED record literal failed the
//! typed ingress with an internal `annotation owner-stamp invariant violated`,
//! because the value's type stamp was never written.
//!
//! The class is wider than that one slot, and it runs in BOTH directions: the
//! same shape silently disabled the duplicate-`def`/`defsig` checks, the §8.6
//! builtin-shadowing gate, the module-reopen and reserved-linker-name forgery
//! guards, and the WS-A8 polymorphic-precision cross-row pass (fail-open), and
//! it made `is_builtin_var` return `false` for every stamped `(var {} name)`,
//! which OVER-rejected a perfectly good literal reduction axis as "not a
//! compile-time constant".
//!
//! Every test here asserts the invariant directly — the two ingresses produce
//! the same diagnostic set for the same program — so a future regression at any
//! of these sites fails here regardless of which direction it drifts.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source)
        .unwrap_or_else(|e| panic!("fixture must stamp: {source}\nstamp error: {e}"))
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

fn typed_errors(exprs: &[Expr]) -> Vec<String> {
    match check_typed_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn ir_errors(exprs: &[Expr]) -> Vec<String> {
    match check_ir_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

/// The chelis#1107 invariant: same program, same verdict, whichever ingress
/// checks it. Returns the agreed diagnostic set so callers can assert on it.
fn agreed_diagnostics(source: &str, label: &str) -> Vec<String> {
    let exprs = stamped(source);
    let typed = typed_errors(&exprs);
    let ir = ir_errors(&exprs);
    assert_eq!(
        typed,
        ir,
        "{label}: the stamped ingress (`check_typed_program`) and the \
         normalizing ingress (`check_ir_program`) must return the same \
         diagnostics for the same program (chelis#1107).\n\
         typed-only: {:?}\nir-only: {:?}",
        typed.iter().filter(|m| !ir.contains(m)).collect::<Vec<_>>(),
        ir.iter().filter(|m| !typed.contains(m)).collect::<Vec<_>>(),
    );
    typed
}

fn assert_agree_and_reject(source: &str, needle: &str, label: &str) {
    let diagnostics = agreed_diagnostics(source, label);
    assert!(
        diagnostics.iter().any(|m| m.contains(needle)),
        "{label}: both ingresses must reject with a diagnostic containing \
         {needle:?}; got {diagnostics:?}"
    );
}

fn assert_agree_and_accept(source: &str, label: &str) {
    let diagnostics = agreed_diagnostics(source, label);
    assert!(
        diagnostics.is_empty(),
        "{label}: this program is well-formed and must check clean on BOTH \
         ingresses (over-rejection control); got {diagnostics:?}"
    );
}

const RECORD_TYPE: &str = "(deftype {} P () (variant {} P (field {} x (t-prim {} f32))))";

fn record_program(kv: &str) -> String {
    format!("{RECORD_TYPE}\n(def {{}} mk (record {{}} P {kv}))")
}

fn record_update_program(kv: &str) -> String {
    format!(
        "{RECORD_TYPE}\n(def {{}} upd (fn {{}} (params {{}} (p {{type: (t-adt {{}} P)}})) \
         (record-update {{}} (var {{}} p) {kv})))"
    )
}

// ── The reported site: `record` / `record-update` kv values ────────────────

#[test]
fn record_field_dtype_mismatch_is_caught_on_both_ingresses() {
    // The #1107 headline: the stamped `kv` is an `Expr::Node`, so the
    // `List`-only destructure sent it to `continue` and NO field was checked.
    assert_agree_and_reject(
        &record_program("(kv {} x (lit {type: (t-prim {} bool)} true))"),
        "precision mismatch",
        "bool into an f32 record field",
    );
}

#[test]
fn record_unknown_field_is_caught_on_both_ingresses() {
    assert_agree_and_reject(
        &record_program("(kv {} nope (lit {type: (t-prim {} f32)} 1.0))"),
        "unknown record field 'nope'",
        "unknown record field",
    );
}

#[test]
fn well_formed_record_literal_checks_clean_on_both_ingresses() {
    // The other half of #1107: because the kv value never got inferred, its
    // authoritative type stamp was never written, and the typed ingress failed
    // a WELL-FORMED literal with an internal owner-stamp invariant violation.
    let source = record_program("(kv {} x (lit {type: (t-prim {} f32)} 1.0))");
    assert_agree_and_accept(&source, "well-formed record literal");
    assert!(
        !typed_errors(&stamped(&source))
            .iter()
            .any(|m| m.contains("owner-stamp invariant violated")),
        "a well-formed record literal must not trip the annotation \
         owner-stamp invariant (chelis#1107)"
    );
}

#[test]
fn record_update_field_dtype_mismatch_is_caught_on_both_ingresses() {
    assert_agree_and_reject(
        &record_update_program("(kv {} x (lit {type: (t-prim {} bool)} true))"),
        "precision mismatch",
        "bool into an f32 record-update field",
    );
}

#[test]
fn well_formed_record_update_checks_clean_on_both_ingresses() {
    assert_agree_and_accept(
        &record_update_program("(kv {} x (lit {type: (t-prim {} f32)} 1.0))"),
        "well-formed record update",
    );
}

// ── Declaration-level guards the same shape had silently disabled ──────────

#[test]
fn duplicate_def_is_reported_on_both_ingresses() {
    assert_agree_and_reject(
        "(def {} f (lit {type: (t-prim {} int32)} 1))\n\
         (def {} f (lit {type: (t-prim {} int32)} 2))",
        "duplicate definition: `f`",
        "duplicate def",
    );
}

#[test]
fn duplicate_defsig_is_reported_on_both_ingresses() {
    assert_agree_and_reject(
        "(defsig {} f (t-prim {} int32))\n\
         (defsig {} f (t-prim {} int32))\n\
         (def {} f (lit {type: (t-prim {} int32)} 1))",
        "duplicate signature: `f`",
        "duplicate defsig",
    );
}

#[test]
fn builtin_shadowing_is_reported_on_both_ingresses() {
    // spec/04-type-system.md §8.6. The gate read declaration names through a
    // `List`-only destructure, so it never fired on the stamped ingress.
    //
    // This one asserts the shared rejection rather than full diagnostic-set
    // equality, because the fixture also trips a SEPARATE, pre-existing
    // divergence that #1107 neither caused nor fixes: `check_ir_program` does
    // not run the def-body-vs-declared-signature unification at all, so the
    // typed ingress additionally reports a `TypeMismatch` against the builtin's
    // signature. That hole reproduces with no builtin in sight -- a `defsig`
    // declaring `f32` over a `def` whose body is `int32` is `score: 1,
    // errors: []` through `chelis check` -- and is tracked separately.
    let source = "(def {} add (lit {type: (t-prim {} int32)} 1))";
    let exprs = stamped(source);
    let needle = "shadows the builtin function `add`";
    for (label, diagnostics) in [("typed", typed_errors(&exprs)), ("ir", ir_errors(&exprs))] {
        assert!(
            diagnostics.iter().any(|m| m.contains(needle)),
            "the §8.6 builtin-shadowing gate must fire on the {label} ingress \
             (chelis#1107); got {diagnostics:?}"
        );
    }
}

#[test]
fn module_reopen_is_reported_on_both_ingresses() {
    assert_agree_and_reject(
        "(module {} m (def {} f (lit {type: (t-prim {} int32)} 1)))\n\
         (module {} m (def {} g (lit {type: (t-prim {} int32)} 2)))",
        "is opened by more than one module wrapper",
        "module reopen",
    );
}

#[test]
fn forged_linker_name_is_reported_on_both_ingresses() {
    // The reef module-identity forgery guard: a top-level declaration may not
    // spell a reserved `pkg__`/`Pkg__` internal name.
    assert_agree_and_reject(
        "(def {} pkg__foo__bar (lit {type: (t-prim {} int32)} 1))",
        "reserved internal-name format",
        "forged linker name",
    );
}

#[test]
fn ordinary_declarations_check_clean_on_both_ingresses() {
    // Over-rejection control for the four guards above: none of them may fire
    // on a program that merely has several distinct declarations.
    assert_agree_and_accept(
        "(defsig {} f (t-prim {} int32))\n\
         (def {} f (lit {type: (t-prim {} int32)} 1))\n\
         (def {} g (lit {type: (t-prim {} int32)} 2))\n\
         (module {} m (def {} h (lit {type: (t-prim {} int32)} 3)))",
        "distinct declarations",
    );
}

// ── Opaque-type metadata: same rejection AND same producer list ────────────

const OPAQUE_MODULE: &str = "(module {} stats.prob\n\
   (export {} probability)\n\
   (deftype {opaque: true} Probability () \
     (variant {} Probability (field {} value (t-prim {} f32))))\n\
   (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))\n\
   (def {} probability (fn {} (params {} (x {type: (t-prim {} f32)})) \
     (record {} Probability (kv {} value (var {} x))))))";

#[test]
fn opaque_violation_names_the_same_producers_on_both_ingresses() {
    // The rejection itself fired on both, but `build_opacity_meta` read its
    // `export`/`def`/`defsig` items through a `List`-only destructure, so on
    // the stamped ingress both maps came back EMPTY and the diagnostic told
    // the user "exported producers: none" for a module that exports one.
    assert_agree_and_reject(
        &format!(
            "{OPAQUE_MODULE}\n(module {{}} agent.strategy\n  \
             (def {{}} bad (fn {{}} (params {{}} (x {{type: (t-prim {{}} f32)}})) \
             (record {{}} Probability (kv {{}} value (var {{}} x))))))"
        ),
        "exported producers of `stats.prob`: probability: (f32) -> Probability",
        "opaque construction outside its defining module",
    );
}

#[test]
fn in_module_opaque_construction_checks_clean_on_both_ingresses() {
    assert_agree_and_accept(OPAQUE_MODULE, "in-module opaque construction");
}

// ── WS-A8 polymorphic-precision cross-row pass ─────────────────────────────

fn poly_mean_program(call_precision: &str) -> String {
    format!(
        "(defsig {{}} my_mean (t-fn {{}} (t-tensor {{}} (d-lit {{}} 4) (t-var {{}} p)) \
           (t-tensor {{}} (t-var {{}} p))))\n\
         (def {{}} my_mean (fn {{}} (params {{}} x) \
           (app {{}} (var {{}} mean) (var {{}} x) (lit {{type: (t-prim {{}} int32)}} 0))))\n\
         (defsig {{}} call (t-fn {{}} (t-tensor {{}} (d-lit {{}} 4) (t-prim {{}} {call_precision})) \
           (t-tensor {{}} (t-prim {{}} {call_precision}))))\n\
         (def {{}} call (fn {{}} (params {{}} y) (app {{}} (var {{}} my_mean) (var {{}} y))))"
    )
}

#[test]
fn polymorphic_integer_mean_is_rejected_on_both_ingresses() {
    // Four separate `List`-only reads (`collect_def_bodies`,
    // `collect_defsig_exprs`, `build_def_param_scope`, and the callee read in
    // `check_app_for_poly_op_constraint`) left this whole pass inert on the
    // stamped ingress, so the chelis#724 capability rejection never fired.
    assert_agree_and_reject(
        &poly_mean_program("int64"),
        "mean on operand precision `int64` is not admitted",
        "polymorphic mean instantiated at int64",
    );
}

#[test]
fn polymorphic_float_mean_checks_clean_on_both_ingresses() {
    // The over-rejection control, and a regression test in the OTHER
    // direction: `is_builtin_var` returned `false` for every stamped
    // `(var {} name)`, so `extract_int_literal` could not see through the
    // literal axis and the typed ingress rejected this valid program with
    // "mean axis must be a compile-time constant".
    assert_agree_and_accept(
        &poly_mean_program("f32"),
        "polymorphic mean instantiated at f32",
    );
}

#[test]
fn literal_reduction_axis_is_accepted_on_both_ingresses() {
    // The narrowest form of that over-rejection: a plain literal axis on a
    // concrete-rank operand.
    assert_agree_and_accept(
        "(defsig {} f (t-fn {} (t-tensor {} (d-lit {} 4) (t-prim {} f32)) \
           (t-tensor {} (t-prim {} f32))))\n\
         (def {} f (fn {} (params {} x) \
           (app {} (var {} mean) (var {} x) (lit {type: (t-prim {} int32)} 0))))",
        "literal reduction axis",
    );
}
