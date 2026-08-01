//! chelis#731 Phase 0 - the totality invariant as a red harness.
//!
//! [04-TOT-2] (spec/04-type-system.md §10) / §C4.1 of
//! `spec/design/checker_totality.md`: if a check completes with an empty
//! error vector, the typed result SHALL contain no error-typed expression.
//! chelis#709 and chelis#710 violated this before Phase 1; this harness makes
//! the violation executable. Phase 2 (chelis#731) PROMOTED [04-TOT-2] to an
//! on-by-default `finalize_checked_program` validation for fresh checker
//! results, so a missing authoritative owner stamp under an otherwise empty
//! error vector becomes a pushed internal error. `Type::Error` itself is
//! unconstructible without an authoritative diagnostic (the §C3
//! `ErrorWitness` token). This harness remains as the independent test-side
//! mirror: it drives both public checker funnels and asserts the same property.
//!
//! ## How a silent `Type::Error` is detected from the outside
//!
//! The checker's public typed result is `CheckedProgram::annotated_exprs`.
//! The inference pass records canonical owner types in the root's
//! `InferenceProduct`, applies the final substitution at `finish_root`, and
//! annotation consumes the retained types for stamp-required owners. It never
//! semantically re-infers a node. `type_to_deep_expr` encodes `Type::Error` as
//! `(t-var {} _)`, so an `Ok(CheckedProgram)` with either a missing required
//! stamp or that encoding violates the public result invariant. Signature
//! metadata carries `Type` values directly and is scanned as an independent
//! backstop.
//!
//! This harness observes the public checked tree plus signature metadata. It
//! does not replace the structural Phase 2 guarantees: fresh errors can be
//! created only through the authoritative diagnostic sink, Deep type
//! resolution is fallible and witnessed, and the in-checker finalizer validates
//! the complete annotated result before success.
//!
//! Two funnels are driven, both returning `Result<CheckedProgram, _>` through
//! the same diagnostic-session and finalization boundary:
//! `chelis_types::check_ir_program` (used by `chelis build`) and
//! `chelis_types::check_typed_program` (used by `chelis check`).
//!
//! ## Red set and boundary law
//!
//! The four known holes - a `with seed` body, a `with device` body
//! (chelis#709), and two of the chelis#710 forms (`(def {} orphan)`,
//! `(cast {} expr)` with no target) - were `#[ignore]`d red at Phase 0 and
//! are flipped to green here by chelis#731 Phase 1 (the handle-effect case
//! and the `MalformedForm` guard sweep). Per B2.1/B2.2 of the design doc
//! the flip was ONLY by deleting the `#[ignore]` attribute; the assertions
//! never weakened. The full corpus runs in the default suite. A new hole
//! EXTENDS the census and gets filed - it does not edit this set silently.
//!
//! chelis#710 filed FOUR forms, not two. Phase 1 closed forms 1-3 (the two
//! named above plus a `(t-prim {} bogus_dtype)` cast target, covered by
//! `issue_756_cast_type_totality.rs`). Form 4 - a bare `Symbol` / `Keyword`
//! atom in expression position - stayed live until chelis#873 threaded a
//! diagnostic sink through `infer_atom`; its score-surface coverage lives in
//! `issue_731_fitness_honesty_corpus.rs`
//! (`bare_atom_expression_position_scores_below_one`, with the
//! over-application guard in `structural_symbol_positions_still_score_one`).
//! Naming the count here so the census does not understate the filed set.

use chelis_deep::ast as deep;
use chelis_types::types::Type;
use chelis_types::{CheckedProgram, InferResult};

const WELL_TYPED: &str = "add(cast(1.0, f32), cast(2.0, f32))";
/// Same ill-typed expression the chelis#709 canary uses.
const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, int64))";

/// Mirror of `should_attach_type_metadata`'s exclusion list
/// (crates/chelis-types/src/infer.rs). A List node whose tag is NOT in
/// this set gets a `type:` stamp during annotation unless its type
/// inferred to `Type::Error`. If that list changes, this mirror must
/// change in the same PR (the control corpus goes red otherwise, which
/// is the tripwire working as intended). KEEP CHAR-IDENTICAL to that
/// function's list in BOTH directions - do not add tags it lacks (e.g.
/// `d-rank`, which `dim_to_deep_expr` emits but the checker's list
/// omits); divergence either way breaks the missing-stamp <=> Error
/// equivalence this file relies on.
const NON_TYPE_STAMPED_TAGS: &[&str] = &[
    "module",
    "import",
    "import-all",
    "export",
    "let",
    "fn",
    "var",
    "tuple",
    "tuple-get",
    "defsig",
    "deftype",
    "typealias",
    "variant",
    "field",
    "defdim",
    "params",
    "bind",
    "kv",
    "arm",
    "effects",
    "resource",
    "pat-var",
    "pat-lit",
    "pat-ctor",
    "pat-tuple",
    "pat-record",
    "pat-wild",
    "pat-as",
    "t-prim",
    "t-fn",
    "t-tensor",
    "t-adt",
    "t-var",
    "t-ref",
    "t-unit",
    "t-tuple",
    "d-name",
    "d-var",
    "d-lit",
];

fn tag_of(list: &deep::List) -> Option<&str> {
    match list.elements.first() {
        Some(deep::Expr::Atom(deep::Atom::Name(s), _)) => Some(s.as_str()),
        _ => None,
    }
}

/// The `(t-var {} _)` shape `type_to_deep_expr` produces for `Type::Error`.
fn is_error_type_stamp(expr: &deep::Expr) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    tag_of(list) == Some("t-var")
        && matches!(
            list.elements.get(2),
            Some(deep::Expr::Atom(deep::Atom::Name(s), _)) if s == "_"
        )
}

/// Walk the annotated tree collecting silent-Type::Error traces.
///
/// `check_stamp` is false for the children of a `(params ...)` node:
/// param nodes are `(name {type: ...})` lists tagged with the arbitrary
/// parameter NAME, so the vocabulary-based stamp rule does not apply to
/// them. Metadata values under the `type` key are skipped entirely -
/// they are type ENCODINGS (which legitimately contain unstamped
/// `t-*`/`d-*` shapes such as `d-rank`), not program nodes the
/// annotator visits.
fn collect_tree_traces(expr: &deep::Expr, path: &str, check_stamp: bool, out: &mut Vec<String>) {
    match expr {
        deep::Expr::Atom(_, _) => {}
        deep::Expr::Map(map, _) => {
            for (key, value) in &map.entries {
                if key == "type" {
                    continue;
                }
                collect_tree_traces(value, &format!("{path}.{key}"), check_stamp, out);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            collect_tree_traces(&meta.expr, path, check_stamp, out);
            for (key, value) in &meta.entries {
                if key == "type" {
                    continue;
                }
                collect_tree_traces(value, &format!("{path}.{key}"), check_stamp, out);
            }
        }
        deep::Expr::List(list, _) => {
            let tag = tag_of(list);
            if check_stamp
                && let (Some(tag), Some(deep::Expr::Map(meta, _))) = (tag, list.elements.get(1))
                && !NON_TYPE_STAMPED_TAGS.contains(&tag)
            {
                match meta.entries.iter().find(|(key, _)| key == "type") {
                    None => out.push(format!(
                        "{path}/{tag}: stamp-eligible node with no `type:` stamp \
                         (a silent Type::Error verdict)"
                    )),
                    Some((_, value)) if is_error_type_stamp(value) => out.push(format!(
                        "{path}/{tag}: `type:` stamp is `(t-var {{}} _)` \
                         (the Type::Error encoding)"
                    )),
                    Some(_) => {}
                }
            }
            let label = tag.unwrap_or("<untagged>");
            let child_check = tag != Some("params");
            for (index, element) in list.elements.iter().enumerate() {
                collect_tree_traces(
                    element,
                    &format!("{path}/{label}[{index}]"),
                    child_check,
                    out,
                );
            }
        }
    }
}

fn type_contains_error(ty: &Type) -> bool {
    match ty {
        Type::Error(_) => true,
        Type::Fn(args, ret) => args.iter().any(type_contains_error) || type_contains_error(ret),
        Type::Ref(inner) => type_contains_error(inner),
        Type::Adt(_, args) => args.iter().any(type_contains_error),
        Type::Tuple(elems) => elems.iter().any(type_contains_error),
        Type::Prim(_) | Type::Tensor(_, _) | Type::Var(_) | Type::Unit => false,
    }
}

fn collect_signature_traces(checked: &CheckedProgram, out: &mut Vec<String>) {
    for (name, sig) in &checked.signature_inference().functions {
        if type_contains_error(&sig.checked_signature) {
            out.push(format!(
                "signature_inference[{name}].checked_signature carries Type::Error"
            ));
        }
        if type_contains_error(&sig.display_signature) {
            out.push(format!(
                "signature_inference[{name}].display_signature carries Type::Error"
            ));
        }
        for param in &sig.params {
            if type_contains_error(&param.checked_type) || type_contains_error(&param.display_type)
            {
                out.push(format!(
                    "signature_inference[{name}].params[{}] carries Type::Error",
                    param.index
                ));
            }
        }
    }
}

/// One funnel's check outcome, reduced to what [04-TOT-2] speaks about.
enum Verdict {
    /// Errors were reported; the invariant is vacuously satisfied.
    Reported(usize),
    /// Empty error vector and no Type::Error trace in the typed result.
    CleanTotal,
    /// Empty error vector BUT the typed result carries Type::Error
    /// traces - the [04-TOT-2] violation.
    CleanWithTraces(Vec<String>),
}

fn verdict_of(result: Result<CheckedProgram, InferResult>) -> Verdict {
    match result {
        Err(infer_result) => {
            assert!(
                !infer_result.errors.is_empty(),
                "a check that fails must carry at least one error \
                 (Err with an empty vector would itself be a totality bug)"
            );
            Verdict::Reported(infer_result.errors.len())
        }
        Ok(checked) => {
            let mut traces = Vec::new();
            for (index, expr) in checked.annotated_exprs().iter().enumerate() {
                collect_tree_traces(expr, &format!("top[{index}]"), true, &mut traces);
            }
            collect_signature_traces(&checked, &mut traces);
            if traces.is_empty() {
                Verdict::CleanTotal
            } else {
                Verdict::CleanWithTraces(traces)
            }
        }
    }
}

fn funnel_verdicts(exprs: &[deep::Expr]) -> Vec<(&'static str, Verdict)> {
    vec![
        (
            "check_ir_program",
            verdict_of(chelis_types::check_ir_program(exprs)),
        ),
        (
            "check_typed_program",
            verdict_of(chelis_types::check_typed_program(exprs)),
        ),
    ]
}

/// The invariant itself, exactly [04-TOT-2]: every funnel that reports
/// success (empty errors) must have a Type::Error-free typed result.
/// Nothing else is asserted, so after the Phase 1 fix these calls go
/// green whether the program is then rejected loudly or checked fully.
fn assert_totality(name: &str, exprs: &[deep::Expr]) {
    for (funnel, verdict) in funnel_verdicts(exprs) {
        if let Verdict::CleanWithTraces(traces) = verdict {
            panic!(
                "[04-TOT-2] violated for `{name}` via {funnel}: empty error \
                 vector but the typed result carries silent Type::Error \
                 traces:\n  {}",
                traces.join("\n  ")
            );
        }
    }
}

/// Control-corpus form: a clean program must check clean (both funnels
/// Ok) AND be total, so a control can never pass vacuously.
fn assert_checks_clean_and_total(name: &str, exprs: &[deep::Expr]) {
    for (funnel, verdict) in funnel_verdicts(exprs) {
        match verdict {
            Verdict::CleanTotal => {}
            Verdict::Reported(count) => {
                panic!("control `{name}` must check clean via {funnel}, got {count} error(s)")
            }
            Verdict::CleanWithTraces(traces) => panic!(
                "[04-TOT-2] violated for control `{name}` via {funnel}:\n  {}",
                traces.join("\n  ")
            ),
        }
    }
}

/// Reported-error control: both funnels must report (so the invariant is
/// satisfied by loudness, and cascade suppression stays exercised).
fn assert_reports_errors(name: &str, exprs: &[deep::Expr]) {
    for (funnel, verdict) in funnel_verdicts(exprs) {
        match verdict {
            Verdict::Reported(_) => {}
            Verdict::CleanTotal => panic!(
                "ill-typed control `{name}` must report errors via {funnel}, \
                 checked clean instead"
            ),
            Verdict::CleanWithTraces(traces) => panic!(
                "ill-typed control `{name}` checked clean with silent \
                 Type::Error traces via {funnel} (a new chelis#709-class \
                 hole):\n  {}",
                traces.join("\n  ")
            ),
        }
    }
}

/// Surf source -> checked Deep exprs, the same library route the CLI's
/// raw single-file `.ch` check path takes (parse_str ->
/// desugar_program -> macro expansion).
fn surf_to_deep(source: &str) -> Vec<deep::Expr> {
    let decls = chelis_surf::parser::parse_str(source).expect("control/hole Surf must parse");
    assert!(!decls.is_empty(), "program must have declarations");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    chelis_macros::expand_program(&deep_exprs, &chelis_macros::ExpansionOptions::default())
        .expect("macro expansion must succeed")
        .into_exprs()
}

/// `.dp` source -> Deep exprs, the same strict parser the CLI's `.dp`
/// ingestion uses.
fn dp_to_deep(source: &str) -> Vec<deep::Expr> {
    chelis_deep::parser::parse_str_strict(source).expect("probe .dp must parse")
}

// ===========================================================================
// Control corpus: green today, runs in the default suite.
// ===========================================================================

/// The canary's eleven wrapper shapes with a WELL-TYPED body: every one
/// must check clean with a fully-stamped typed tree. If a wrapper here
/// loses its stamps, either a new silent hole opened or the
/// `should_attach_type_metadata` mirror above drifted - both need a look.
///
/// Syntax note (Phase 0 finding, recorded in the chelis#731 census
/// comment): six of the canary's wrapper strings (`let ... in`,
/// `fn (v: f32) -> f32 = ...` lambdas, the pipe stage using that lambda
/// form, `[f32]` return types, brace-only `match`, and the vmap lambda)
/// are PARSE-rejected Surf, so those canary rows score below 1 via the
/// parser, not the checker. This corpus uses the spec/02 forms (block
/// bindings, `fn (v: f32) -> body`, `List[f32]`, `match ... with`,
/// example-style `vmap(f, axis=0)`), all verified parse-clean; the
/// census records that the ill-typed variants of these corrected forms
/// ARE caught by the checker (precision mismatch, score < 1).
#[test]
fn control_wrapper_battery_checks_clean_and_total() {
    let cases: &[(&str, String)] = &[
        ("bare", format!("def f() -> f32 = {WELL_TYPED}\n")),
        (
            "block_binding",
            format!("def f() -> f32 = {{\n  x = {WELL_TYPED}\n  x\n}}\n"),
        ),
        (
            "if_then",
            format!("def f(c: bool) -> f32 = if c then {WELL_TYPED} else 1.0\n"),
        ),
        (
            "lambda_body",
            "def f() -> f32 = (fn (v: f32) -> add(v, cast(2.0, f32)))(1.0)\n".to_string(),
        ),
        (
            "pipe_stage",
            "def f() -> f32 = 1.0 |> fn (v: f32) -> add(v, cast(2.0, f32))\n".to_string(),
        ),
        (
            "tuple_elem",
            format!("def f() -> (f32, f32) = ({WELL_TYPED}, 1.0)\n"),
        ),
        (
            "list_elem",
            format!("def f() -> List[f32] = [{WELL_TYPED}, 1.0]\n"),
        ),
        (
            "match_arm",
            format!(
                "def f(c: bool) -> f32 = match c with {{\n  | true => {WELL_TYPED}\n  | false => 1.0\n}}\n"
            ),
        ),
        (
            "grad_callee",
            format!("def g(x: f32) -> f32 = {WELL_TYPED}\ndef f(x: f32) -> f32 = grad(g)(x)\n"),
        ),
        (
            "vmap_named",
            "def process(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n\
             def f(xs: tensor[2, 4, f32]) -> tensor[2, 4, f32] = xs |> vmap(process, axis=0)\n"
                .to_string(),
        ),
        (
            "jit_callee",
            format!("def g(x: f32) -> f32 = {WELL_TYPED}\ndef f(x: f32) -> f32 = jit(g)(x)\n"),
        ),
    ];
    for (name, program) in cases {
        assert_checks_clean_and_total(name, &surf_to_deep(program));
    }
}

/// Well-formed hand-written `.dp` - the loud siblings of the two
/// chelis#710 hole forms: the same shapes WITH their required children.
#[test]
fn control_wellformed_dp_checks_clean_and_total() {
    let cases: &[(&str, &str)] = &[
        (
            "cast_with_target",
            "(module {} m.main (def {} out (cast {} (lit {type: (t-prim {} f32)} 42.0) (t-prim {} f32))))",
        ),
        (
            "def_with_body",
            "(module {} m.main (def {} answer (lit {type: (t-prim {} int32)} 7)))",
        ),
    ];
    for (name, program) in cases {
        assert_checks_clean_and_total(name, &dp_to_deep(program));
    }
}

/// Ill-typed programs WITHOUT a hole form must report loudly through
/// both funnels - the invariant is then vacuously satisfied, and the
/// cascade case (an error nested under another expression) stays a
/// control per B2.3.
#[test]
fn control_reported_errors_keep_the_invariant_vacuous() {
    let cases: &[(&str, String)] = &[
        (
            "bare_masked_error",
            format!("def f() -> f32 = {MASKED_ERROR}\n"),
        ),
        (
            "cascade_over_reported_error",
            format!("def f() -> f32 = add({MASKED_ERROR}, cast(3.0, f32))\n"),
        ),
    ];
    for (name, program) in cases {
        assert_reports_errors(name, &surf_to_deep(program));
    }
}

// ===========================================================================
// The four known holes: red today, flip by un-ignoring after Phase 1.
// ===========================================================================

/// chelis#709 hole 1 (closed by chelis#731 Phase 1): the `with seed` body used
/// to type as a silent Type::Error (no `handle-effect` case in infer.rs). The
/// handle-effect case now checks the body, so its error is reported and the
/// invariant holds (verdict Reported, no silent Error).
#[test]
fn totality_holds_for_with_seed_body() {
    let program = format!("def f() -> f32 = with seed(42) {{ {MASKED_ERROR} }}\n");
    assert_totality("with_seed_body", &surf_to_deep(&program));
}

/// chelis#731 red team F3: the cell above uses an UNSUFFIXED seed, so its
/// suffix diagnostic alone satisfies [04-TOT-2] (verdict Reported) even if the
/// BODY check regressed - the cell is vacuous w.r.t. the handle-effect body
/// fix. This sibling uses a SUFFIXED seed (`42i64`), so the seed pushes no
/// diagnostic and the ONLY thing that can make the funnel report is the body's
/// masked error. If the body check ever silently exempts again, this cell trips
/// (the handle-effect node carries a silent Type::Error under an empty error
/// vector). The original cell stays untouched per B2.1.
#[test]
fn totality_holds_for_with_seed_suffixed_body_locks_body_check() {
    let program = format!("def f() -> f32 = with seed(42i64) {{ {MASKED_ERROR} }}\n");
    assert_totality("with_seed_suffixed_body", &surf_to_deep(&program));
}

/// chelis#709 hole 2 (closed by chelis#731 Phase 1): same mechanism through
/// `with device`; the device body is now checked.
#[test]
fn totality_holds_for_with_device_body() {
    let program = format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n");
    assert_totality("with_device_body", &surf_to_deep(&program));
}

/// chelis#710 hole 1 (closed by chelis#731 Phase 1): `(def {} orphan)` - the
/// infer_def arity guard now pushes `MalformedForm` instead of a silent
/// Type::Error, so the malformed def is reported.
#[test]
fn totality_holds_for_dp_def_missing_body() {
    assert_totality(
        "dp_def_missing_body",
        &dp_to_deep("(module {} m.main (def {} orphan))"),
    );
}

/// chelis#710 hole 2 (closed by chelis#731 Phase 1): `(cast {} expr)` with no
/// target type - the infer_cast arity guard now pushes `MalformedForm`.
#[test]
fn totality_holds_for_dp_cast_missing_target() {
    assert_totality(
        "dp_cast_missing_target",
        &dp_to_deep("(module {} m.main (def {} out (cast {} (lit {type: (t-prim {} f32)} 42.0))))"),
    );
}
