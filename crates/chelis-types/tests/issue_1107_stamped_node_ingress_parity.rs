//! chelis#1107 — the two checker ingresses must agree on stamped Deep.
//!
//! Spec authority: spec/design/checker_totality.md §C1.1/§C1.2 and
//! spec/04-type-system.md §10 [04-TOT-1].
//!
//! `check_ir_program` used to normalize `Expr::Node` into the legacy list
//! carrier before inference, while `check_typed_program` walked the stamped
//! tree directly. `infer_expr`'s Node bridge rebuilt only the node it
//! dispatched on, so on the typed ingress every CHILD of a bridged node was
//! still an `Expr::Node`. Any reader that destructured the legacy list and fell
//! through on anything else therefore skipped its input on one ingress and read
//! it on the other, and the two ingresses returned different verdicts for the
//! same program. The legacy list carrier is now deleted (chelis#1125), so both
//! ingresses hand the checker the same stamped tree; these rows keep the two
//! entries' verdicts pinned to each other and to the expected outcome.
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
        "(def {} f (lit {type: (t-prim {} i32)} 1))\n\
         (def {} f (lit {type: (t-prim {} i32)} 2))",
        "duplicate definition: `f`",
        "duplicate def",
    );
}

#[test]
fn duplicate_defsig_is_reported_on_both_ingresses() {
    assert_agree_and_reject(
        "(defsig {} f (t-prim {} i32))\n\
         (defsig {} f (t-prim {} i32))\n\
         (def {} f (lit {type: (t-prim {} i32)} 1))",
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
    // declaring `f32` over a `def` whose body is `i32` is `score: 1,
    // errors: []` through `chelis check` -- and is tracked separately.
    let source = "(def {} add (lit {type: (t-prim {} i32)} 1))";
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
        "(module {} m (def {} f (lit {type: (t-prim {} i32)} 1)))\n\
         (module {} m (def {} g (lit {type: (t-prim {} i32)} 2)))",
        "is opened by more than one module wrapper",
        "module reopen",
    );
}

#[test]
fn forged_linker_name_is_reported_on_both_ingresses() {
    // The reef module-identity forgery guard: a top-level declaration may not
    // spell a reserved `pkg__`/`Pkg__` internal name.
    assert_agree_and_reject(
        "(def {} pkg__foo__bar (lit {type: (t-prim {} i32)} 1))",
        "reserved internal-name format",
        "forged linker name",
    );
}

#[test]
fn ordinary_declarations_check_clean_on_both_ingresses() {
    // Over-rejection control for the four guards above: none of them may fire
    // on a program that merely has several distinct declarations.
    assert_agree_and_accept(
        "(defsig {} f (t-prim {} i32))\n\
         (def {} f (lit {type: (t-prim {} i32)} 1))\n\
         (def {} g (lit {type: (t-prim {} i32)} 2))\n\
         (module {} m (def {} h (lit {type: (t-prim {} i32)} 3)))",
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
        "(defsig {{dtype_bounds: {{p: float}}}} my_mean (p) (t-fn {{}} (t-tensor {{}} (d-lit {{}} 4) (t-var {{}} p)) \
           (t-tensor {{}} (t-var {{}} p))))\n\
         (def {{}} my_mean (fn {{}} (params {{}} x) \
           (app {{}} (var {{}} mean) (var {{}} x) (lit {{type: (t-prim {{}} i32)}} 0))))\n\
         (defsig {{}} call (t-fn {{}} (t-tensor {{}} (d-lit {{}} 4) (t-prim {{}} {call_precision})) \
           (t-tensor {{}} (t-prim {{}} {call_precision}))))\n\
         (def {{}} call (fn {{}} (params {{}} y) (app {{}} (var {{}} my_mean) (var {{}} y))))"
    )
}

#[test]
fn polymorphic_integer_mean_is_rejected_on_both_ingresses() {
    // Originally the List-only body validator missed stamped ingress. The
    // checked Float contract must now reject this instantiation on both
    // ingresses without looking up the callee body.
    assert_agree_and_reject(
        &poly_mean_program("i64"),
        "dtype family `Float` (the active float dtypes) cannot be instantiated at `i64`",
        "polymorphic mean instantiated at i64",
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
           (app {} (var {} mean) (var {} x) (lit {type: (t-prim {} i32)} 0))))",
        "literal reduction axis",
    );
}

// ── reshape shape-list readers (the red team's amendment finding) ──────────
//
// `reshape_output_dims` reads its shape-list argument through
// `collect_shape_list_elements` and `list_literal_len`. Both were bare
// `let deep::Expr::List(..) else` readers, so on the stamped ingress they
// returned `None` and reshape collapsed to a rank-1 wildcard, while the
// normalizing ingress read the literal shape. Bidirectional and user-reachable
// through `chelis prove`, which drives `check_typed_program`.
//
// This member was LATENT and PRE-EXISTING (it reproduces on 84e84b36, before
// the first #1107 commit). It was missed by the original sweep because that
// sweep was corpus-driven: a probe only proves what the corpus reached. The
// amendment replaced it with a static textual audit of the whole reader class.

/// `[2, 3]` in its desugared `Cons(2, Cons(3, Nil))` form.
const SHAPE_LIST_2_3: &str = "(app {} (var {} Cons) (lit {type: (t-prim {} i64)} 2) \
   (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 3) (var {} Nil)))";

#[test]
fn reshape_to_concrete_dims_is_accepted_on_both_ingresses() {
    // Over-rejection form: the typed ingress inferred `tensor[*, f32]` for the
    // reshape and rejected this valid program against its declared
    // `tensor[2, 3, f32]`, while the IR ingress accepted it.
    assert_agree_and_accept(
        &format!(
            "(defsig {{}} f (t-fn {{}} (t-tensor {{}} (d-lit {{}} 6) (t-prim {{}} f32)) \
               (t-tensor {{}} (d-lit {{}} 2) (d-lit {{}} 3) (t-prim {{}} f32))))\n\
             (def {{}} f (fn {{}} (params {{}} x) \
               (app {{}} (var {{}} reshape) (var {{}} x) {SHAPE_LIST_2_3})))"
        ),
        "reshape to concrete dims matching its declared result",
    );
}

#[test]
fn reshape_shape_mismatch_is_rejected_on_both_ingresses() {
    // Fail-open form: the same reshape declared `-> tensor[n, f32]` is a real
    // rank mismatch. The typed ingress accepted it, because a rank-1 wildcard
    // unified with the declared rank-1 result; the IR ingress rejected it.
    assert_agree_and_reject(
        &format!(
            "(defsig {{}} g (n) (t-fn {{}} (t-tensor {{}} (d-lit {{}} 6) (t-prim {{}} f32)) \
               (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32))))\n\
             (def {{}} g (fn {{}} (params {{}} x) \
               (app {{}} (var {{}} reshape) (var {{}} x) {SHAPE_LIST_2_3})))"
        ),
        "body doesn't match declared signature",
        "reshape whose real rank-2 result contradicts a declared rank-1 signature",
    );
}

/// `[2, 3]` in the host-lane `(list ...)` spelling, which is outside the
/// 62-tag vocabulary.
const SHAPE_LIST_TAGGED_2_3: &str =
    "(list {} (lit {type: (t-prim {} i64)} 2) (lit {type: (t-prim {} i64)} 3))";

#[test]
fn reshape_with_host_lane_list_spelling_is_rejected_on_both_ingresses() {
    // The `(list ...)` shape-list spelling is NOT an ingress divergence, and
    // this test exists to keep it that way.
    //
    // `list` is not in the closed vocabulary, so in expression position both
    // ingresses reject the program outright on `infer_expr`'s `UnknownForm`
    // arm -- the verdict is identical, and neither lane derives a shape from
    // it. PP9 removed the duplicate structural pass, so exact diagnostic-set
    // equality is part of the ingress contract here too.
    let source = format!(
        "(defsig {{}} f (t-fn {{}} (t-tensor {{}} (d-lit {{}} 6) (t-prim {{}} f32)) \
           (t-tensor {{}} (d-lit {{}} 2) (d-lit {{}} 3) (t-prim {{}} f32))))\n\
         (def {{}} f (fn {{}} (params {{}} x) \
           (app {{}} (var {{}} reshape) (var {{}} x) {SHAPE_LIST_TAGGED_2_3})))"
    );
    let exprs = stamped(&source);
    let needle = "unknown Deep tag `list` has no checker disposition";
    let typed = typed_errors(&exprs);
    let ir = ir_errors(&exprs);
    assert_eq!(typed, ir, "PP9 requires exact ingress diagnostic parity");
    assert!(
        typed.iter().any(|message| message.contains(needle)),
        "both ingresses must reject the host-lane `(list ...)` spelling \
         in expression position (chelis#1107); got {typed:?}"
    );
}

// ── round 3: `match`-arm readers with a non-erroring default ───────────────
//
// The round-1/2 audits enumerated only the `let ... Expr::List .. else`
// spelling. The destructure shape was never the correctness boundary -- a
// stamped `Expr::Node` child hits a `match`-arm's `_ =>` default just as it hit
// a `let-else`'s `else` -- so these are the same class, found later only
// because the enumeration pattern was too narrow. All three are pre-existing.

#[test]
fn expand_with_sourceless_runtime_size_is_rejected_on_both_ingresses() {
    // `classify_expand_size` had an `Expr::List`-only match with a
    // `_ => SizeClass::Unknown` default, and `Unknown` is the ACCEPTING class
    // in `check_expand_signature` while `Sourceless` is the rejecting one. So a
    // stamped size argument fell to the fail-OPEN default and the typed ingress
    // accepted a §4.7.2 sourceless size that the IR ingress rejected -- the
    // silent-miscompile class chelis#469 exists to prevent.
    assert_agree_and_reject(
        "(defsig {} f (t-fn {} (t-tensor {} (d-name {} seq) (t-prim {} f32)) (t-prim {} i32) \
           (t-tensor {} (d-name {} seq) (t-prim {} f32))))\n\
         (def {} f (fn {} (params {} x k) \
           (app {} (var {} expand) (var {} x) (lit {type: (t-prim {} i32)} 0) (var {} k))))",
        "no tensor in scope carries it",
        "expand sized by a runtime scalar with no tensor source",
    );
}

#[test]
fn expand_sized_by_a_shape_read_is_accepted_on_both_ingresses() {
    // Over-rejection control for the fix above: a size read off an in-scope
    // tensor is the canonical Form-3 source and must still check clean.
    //
    // The operand carries the unit extent `expand` claims at its axis
    // (spec/05-risc-primitives.md section 2.4.1). The Form-3 source under test
    // is the `shape` read in the size slot, which is unaffected by the
    // operand's extent.
    assert_agree_and_accept(
        "(defsig {} f (t-fn {} (t-tensor {} (d-lit {} 1) (t-prim {} f32)) \
           (t-tensor {} (d-lit {} 1) (t-prim {} f32))))\n\
         (def {} f (fn {} (params {} x) (app {} (var {} expand) (var {} x) \
           (lit {type: (t-prim {} i32)} 0) \
           (app {} (var {} shape) (var {} x) (lit {type: (t-prim {} i32)} 0)))))",
        "expand sized by a shape read of an in-scope tensor",
    );
}

fn to_tensor_pair_program(declared_len: &str) -> String {
    format!(
        "(defsig {{}} h (t-tensor {{}} (d-lit {{}} {declared_len}) (t-prim {{}} f32)))\n\
         (def {{}} h (app {{}} (var {{}} to_tensor) \
           (app {{}} (var {{}} Cons) (app {{}} (var {{}} neg) (lit {{type: (t-prim {{}} f32)}} 1.0)) \
           (app {{}} (var {{}} Cons) (app {{}} (var {{}} neg) (lit {{type: (t-prim {{}} f32)}} 2.0)) \
           (var {{}} Nil)))))"
    )
}

#[test]
fn to_tensor_element_count_mismatch_is_rejected_on_both_ingresses() {
    // `extract_numeric_leaf_for_shape` had the same shape of defect, so a
    // stamped `lit`/`cast`/`neg` leaf defaulted to `None`, the static cons-chain
    // walk gave up, and `to_tensor` produced a rank-1 WILDCARD that masked the
    // real element count. A two-element literal declared `tensor[3, f32]` was
    // accepted by the typed ingress and rejected by the IR one.
    assert_agree_and_reject(
        &to_tensor_pair_program("3"),
        "body doesn't match declared signature",
        "to_tensor with two elements declared as tensor[3]",
    );
}

#[test]
fn to_tensor_matching_element_count_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        &to_tensor_pair_program("2"),
        "to_tensor with two elements declared as tensor[2]",
    );
}

#[test]
fn uniform_like_literal_bounds_are_accepted_on_both_ingresses() {
    // The third member, and the one that bites hardest: `is_static_numeric_bound`
    // had an `Expr::List`-only match with a fail-CLOSED `_ => false` default, so
    // on the stamped ingress EVERY bound read as non-resolvable and ordinary
    // literal bounds were REJECTED by `check_typed_program` while
    // `check_ir_program` accepted them. An over-rejection of a valid program.
    // Covers all three accepted bound spellings: plain literal, `cast`-wrapped,
    // and negated.
    for (label, low) in [
        ("plain literal", "(lit {type: (t-prim {} f32)} 0.0)"),
        (
            "cast-wrapped",
            "(cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32))",
        ),
        (
            "negated",
            "(app {} (var {} neg) (lit {type: (t-prim {} f32)} 1.0))",
        ),
    ] {
        assert_agree_and_accept(
            &format!(
                "(defsig {{}} f (t-fn {{}} (t-prim {{}} key) (t-tensor {{}} (d-lit {{}} 4) (t-prim {{}} f32)) \
                   (t-tensor {{}} (d-lit {{}} 4) (t-prim {{}} f32))))\n\
                 (def {{}} f (fn {{}} (params {{}} k x) (app {{}} (var {{}} uniform_like) (var {{}} k) (var {{}} x) \
                   {low} (lit {{type: (t-prim {{}} f32)}} 1.0))))"
            ),
            &format!("uniform_like with a {label} low bound"),
        );
    }
}

// ── chelis#1125 PP7 E5a: the four in-checker carrier divergences ───────────
//
// Spec authority for this block: spec/04-type-system.md §10 [04-TOT-5] --
// a program's checker verdict does not depend on which entry receives it,
// nor on which admitted representation carries it, and a representation the
// checker admits but a check cannot read is a silent exemption under
// [04-TOT-1] rather than an absent subtree. Design:
// spec/design/checker_totality.md §"PP7. Stamped-ingress reader parity".
//
// PP7 measured these six programs through both shipped surfaces of the same
// file (`chelis check`, the serialized-IR ingress, and `chelis prove`, the
// stamped typed ingress). Four diverged; two agreed and are kept here as
// disposition locks, because they are what makes the other four "a missed
// check" rather than "an absent checker": the stamped ingress does bind
// names and does range-check an in-range literal.
//
// Every row below runs the SAME six programs through `agreed_diagnostics`,
// which drives `check_typed_program` and `check_ir_program` over one stamped
// parse and asserts the diagnostic sets are equal.

/// PP7 row 1, REGRESSION TEST (red before the `infer/expr.rs` repair, green
/// after): `infer_lit`'s `type:` metadata reader destructured the metadata
/// VALUE as `Expr::List`. `Node::to_list` clones the metadata map verbatim,
/// so on the stamped ingress the `(t-prim {} i8)` under `type:` is still an
/// `Expr::Node` and the per-prim range check never selected a row. The §5.6
/// out-of-range literal was accepted by `chelis prove` (exit 0) and rejected
/// by `chelis check` (exit 2) -- the fail-open direction.
#[test]
fn out_of_range_int8_literal_is_rejected_on_both_ingresses() {
    assert_agree_and_reject(
        "(def {} x (lit {type: (t-prim {} i8)} 200))",
        "literal 200 out of range for context-inferred i8",
        "i8 literal 200",
    );
}

/// PP7 row 2, DISPOSITION LOCK (green before and after the repair). Its job
/// is to keep the row above from being satisfied by rejecting every `i8`
/// literal: an in-range value must still check clean on both ingresses, so
/// the repair reads the metadata rather than assuming the worst about it.
#[test]
fn in_range_int8_literal_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        "(def {} x (lit {type: (t-prim {} i8)} 100))",
        "i8 literal 100",
    );
}

/// PP7 row 3, DISPOSITION LOCK (green before and after the repair). Its job
/// is to hold the premise the four divergence rows rest on: the stamped
/// ingress really does bind and resolve names, so a check it does not run is
/// a missed check rather than an absent checker.
#[test]
fn unbound_variable_is_reported_on_both_ingresses() {
    assert_agree_and_reject(
        "(def {} x (var {} nope))",
        "unbound variable: nope",
        "unbound variable control",
    );
}

/// The PP7 `deftype` row's program: an `invariant:` with no `opaque: true`,
/// inside a module wrapper. The module descent is part of the defect -- the
/// D-WF pass reaches a `deftype` only through `flatten_with_modules`.
const INVARIANT_WITHOUT_OPAQUE: &str = "(module {} m.wf \
   (deftype {invariant: (fn {} (params {} p) \
       (app {} (var {} gte) (access {} (var {} p) value) \
         (lit {type: (t-prim {} f32)} 0.0))), \
     invariant_amenability: \"linear\"} \
     T () (variant {} T (field {} value (t-prim {} f32)))))";

/// PP7 row 4, REGRESSION TEST (red before the `invariants.rs` repair, green
/// after): the whole D-WF opaque-invariant pass was inert on the stamped
/// ingress. Its private `tag`, `children`, and `meta_map` helpers are
/// `Expr::List`-only, so `flatten_with_modules` did not recognize the
/// `module` wrapper, never descended into it, and no `deftype` was ever
/// validated. `chelis prove` accepted an invariant on a forgeable type
/// (exit 0) that `chelis check` rejected (exit 2) -- fail-open.
///
/// This is the pass PP7 finding "axis A, not axis B" is about: the pass IS
/// invoked from the typed entry; its readers could not decode the carrier.
#[test]
fn invariant_without_opaque_is_rejected_on_both_ingresses() {
    let error = parse_and_stamp_file(INVARIANT_WITHOUT_OPAQUE)
        .expect_err("invalid metadata cannot create a stamped Node");
    assert!(error.to_string().contains("opaque"));
    let error = chelis_deep::parser::parse_str(INVARIANT_WITHOUT_OPAQUE)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invariant") && error.contains("opaque"));
}

/// PP7 row 4's over-rejection control, DISPOSITION LOCK (green before and
/// after): the same declaration WITH `opaque: true` is well-formed, so the
/// repair must not turn a newly-readable carrier into a new rejection.
#[test]
fn invariant_with_opaque_checks_clean_on_both_ingresses() {
    assert_agree_and_accept(
        &INVARIANT_WITHOUT_OPAQUE
            .replace("(deftype {invariant:", "(deftype {opaque: true, invariant:"),
        "deftype with invariant and opaque",
    );
}

/// PP7 row 5, REGRESSION TEST (red before the `infer/validate.rs` repair,
/// green after): `walk_for_tensor_precision` has an `Expr::Node` arm, so it
/// reads as migrated -- but that arm rebuilds through `Node::to_list`, which
/// copies children verbatim, and the arm's own
/// `let deep::Expr::List(prec_list, _) = last` then fails on the trailing
/// `t-prim`, which is still a `Node`. The §1.1.1 deferred-dtype rejection
/// was therefore skipped entirely on the stamped ingress: `chelis prove`
/// accepted `tensor[2, f8e4m3]` (exit 0) and `chelis check` rejected it
/// (exit 2) -- fail-open, through a walker with a `Node` arm.
#[test]
fn deferred_tensor_precision_is_rejected_on_both_ingresses() {
    assert_agree_and_reject(
        "(def {} f (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 2) \
           (t-prim {} f8e4m3))})) (var {} x)))",
        "cannot use `f8e4m3` as a tensor element dtype: f8e4m3 is deferred per \
         spec/04-type-system.md §1.1.1",
        "t-tensor with f8e4m3 element precision",
    );
}

/// PP7 row 5's over-rejection control, DISPOSITION LOCK (green before and
/// after): the same inline-annotated parameter at an ACTIVE precision must
/// keep checking clean on both ingresses. Without it, "reject every
/// `t-tensor` whose precision the reader cannot decode" would satisfy the
/// row above.
#[test]
fn active_tensor_precision_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        "(def {} f (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 2) \
           (t-prim {} f32))})) (var {} x)))",
        "t-tensor with f32 element precision",
    );
}

/// PP7 row 6, REGRESSION TEST (red before the `infer/expr_record.rs` repair,
/// green after) and the one FAIL-CLOSED direction in the set:
/// `tuple_get_index` read the `lit` index wrapper as `Expr::List`, so a
/// stamped index returned `None` and the sole caller pushed
/// `invalid tuple index: ... found a non-literal expression`. A well-formed
/// projection that `chelis check` accepted at score 1.0 was REJECTED by
/// `chelis prove` (exit 3).
#[test]
fn well_formed_tuple_get_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        "(def {} x (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0) \
           (lit {type: (t-prim {} f32)} 2.0)) (lit {type: (t-prim {} i32)} 0)))",
        "well-formed tuple-get with a literal index",
    );
}

/// PP7 row 6's NEGATIVE TWIN, REGRESSION TEST (red before the
/// `describe_tuple_index` repair, green after). A verdict includes its
/// diagnostic, so a malformed index has to be rejected with the SAME text on
/// both ingresses, not merely rejected on both: the stamped ingress said
/// "found a non-literal expression" for an index that is plainly the integer
/// literal -1. Without this row, "reject every index carrier I cannot decode"
/// would still satisfy the accepted-projection row above.
///
/// The expected substring was updated by chelis#874 Slice 2, which moved the
/// `tuple-get` index read onto the shared role-slot seam. **This row's subject
/// is unchanged**: both properties it protects still hold, and both are still
/// what is asserted. The two ingresses must still agree (`assert_agree_and_reject`
/// is untouched, and it is the agreement, not the wording, that fails first if
/// they diverge), and the payload atom behind the `lit` wrapper must still be
/// named -- which is exactly why Slice 2 kept the seam's caller-detail
/// affordance, with `describe_tuple_index` as its only consumer. Only the
/// surrounding wording moved, from "invalid tuple index: expected a
/// non-negative integer literal, found integer literal -1" to the seam's form.
#[test]
fn negative_tuple_get_index_is_rejected_alike_on_both_ingresses() {
    assert_agree_and_reject(
        "(def {} x (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0) \
           (lit {type: (t-prim {} f32)} 2.0)) (lit {type: (t-prim {} i32)} -1)))",
        "malformed `tuple-get`: expected a non-negative integer index as child 1, \
         found a `lit` form (integer literal -1)",
        "tuple-get with a negative literal index",
    );
}

/// The PP7 finding-1 program: a def with an inline-annotated parameter and
/// NO `defsig`, calling a polymorphic-precision def. `build_def_param_scope`
/// takes its inline-annotation fallback only in exactly this shape.
fn inline_param_poly_program(call_precision: &str) -> String {
    format!(
        "(defsig {{dtype_bounds: {{p: float}}}} my_mean (p) (t-fn {{}} (t-tensor {{}} (d-lit {{}} 4) (t-var {{}} p)) \
           (t-tensor {{}} (t-var {{}} p))))\n\
         (def {{}} my_mean (fn {{}} (params {{}} x) \
           (app {{}} (var {{}} mean) (var {{}} x) (lit {{type: (t-prim {{}} i32)}} 0))))\n\
         (def {{}} call (fn {{}} (params {{}} \
           (y {{type: (t-tensor {{}} (d-lit {{}} 4) (t-prim {{}} {call_precision}))}})) \
           (app {{}} (var {{}} my_mean) (var {{}} y))))"
    )
}

/// PP7 finding 1, REGRESSION TEST (red before the `param_name_and_inline_type`
/// repair, green after). An inline-annotated param stamps to `Expr::BareList`,
/// which `stamped_parts` does not read, so `build_def_param_scope` returned an
/// empty scope, the WS-A8 cross-row pass could not resolve the body's
/// `(var y)`, and the chelis#724 integer-`mean` rejection never fired on the
/// stamped ingress. Fail-open, at a site chelis#1126 had already "migrated to
/// `stamped_parts`" -- which is why PP7 rejects "route it through
/// `stamped_parts`" as a sufficient instruction.
#[test]
fn inline_param_polymorphic_integer_mean_is_rejected_on_both_ingresses() {
    assert_agree_and_reject(
        &inline_param_poly_program("i64"),
        "dtype family `Float` (the active float dtypes) cannot be instantiated at `i64`",
        "inline-annotated i64 param through a polymorphic mean",
    );
}

/// The over-rejection control for the row above, DISPOSITION LOCK (green
/// before and after): the same shape at a float precision must keep checking
/// clean on both ingresses.
#[test]
fn inline_param_polymorphic_float_mean_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        &inline_param_poly_program("f32"),
        "inline-annotated f32 param through a polymorphic mean",
    );
}

/// `key_from_seed` applied to a seed literal of a chosen width. [05-OP-69]
/// takes an `i64` seed, and an integer literal's width is its `type:` `t-prim`
/// in Deep, which both ingresses must read the same way.
fn seeded_key_program(seed_prim: &str) -> String {
    format!(
        "(def {{}} f (app {{}} (var {{}} key_from_seed) \
           (lit {{type: (t-prim {{}} {seed_prim})}} 42)))"
    )
}

/// PP7's EIGHTH in-checker divergence was the retired `with seed` handler's
/// `seed_literal_form`, which read its seed `lit` only as an `Expr::List` and
/// so never fired the i64-suffix rule on the stamped ingress. The handler is
/// gone; its key-form analogue is the seed literal of `key_from_seed`, whose
/// width both ingresses must read from the same `type:` metadata.
/// DISPOSITION LOCK (the defective reader was deleted with the handler).
#[test]
fn unsuffixed_seed_literal_is_rejected_on_both_ingresses() {
    assert_agree_and_reject(
        &seeded_key_program("i32"),
        "precision mismatch: expected i64, got i32",
        "key_from_seed at an i32 literal",
    );
}

/// The over-rejection control for the row above, DISPOSITION LOCK (green
/// before and after): an i64 seed literal must still check clean on both
/// ingresses. Without it, "reject every seed carrier I cannot decode" would
/// satisfy the row above.
#[test]
fn int64_suffixed_seed_literal_is_accepted_on_both_ingresses() {
    assert_agree_and_accept(
        &seeded_key_program("i64"),
        "key_from_seed at an i64 literal",
    );
}
