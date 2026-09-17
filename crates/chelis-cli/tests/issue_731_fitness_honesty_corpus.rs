//! chelis#731 §C4.4 — the fitness-honesty corpus.
//!
//! A continuous, corpus-level enforcement of [04-TOT-2]
//! (spec/04-type-system.md §10 / `spec/design/checker_totality.md`): a suite of
//! KNOWN-ill-typed / malformed programs, each asserted to score strictly below
//! 1.0 through `chelis check`. Any member that ever scores 1.0 fails the build.
//! This stands even after Phase 2's `ErrorWitness` migration makes a silent
//! `Type::Error` unconstructible: it is the behavioral proof that the checker's
//! score cannot read "perfect" on a program that does not type-check.
//!
//! Per Jeff's review (chelis#731) the invariant that matters is score < 1.0 for
//! every known-bad program; the exact margin (e.g. < 0.9) is calibration tied to
//! open question 4's severity weights and is deliberately NOT hard-coded here.
//!
//! Membership (Phase 1): the wrapper battery's ill-typed variants, the four
//! Phase 0 holes (chelis#709 `with seed`/`with device` bodies, chelis#710
//! `(def)`/`(cast)` malformed forms), the seed-suffix and unknown-effect-kind
//! diagnostics, and the chelis#710 census-extension malformed `.dp` family.
//!
//! Membership (Phase 2, chelis#731): the Surf-reachable chelis#755 (field
//! access on a multi-variant / non-record target) and chelis#756 (`cast` to an
//! unknown type name) silent holes JOIN here, now that the `ErrorWitness`
//! migration has made their sites `report(...)` diagnostics -- the property
//! they assert (score < 1.0) is one the tree now has.
//!
//! Membership (chelis#873): bare `Symbol` / `Keyword` atoms in expression
//! position, in the placements where nothing downstream forces their type to
//! meet a concrete one. This is chelis#710 form 4, the residue Phase 1 did not
//! close. `infer_atom` typed them `Type::Unit` from a signature with no
//! diagnostic sink, so `chelis check` scored these programs 1.0 while
//! `chelis build` refused to lower them. The constrained placements (a body
//! under a concrete `defsig`, an argument against a concrete parameter type)
//! were already caught by unification and are deliberately NOT members; they
//! appear in the negative-parity test below so the fix cannot be over-applied
//! into rejecting atoms that legitimately occupy structural positions.
//!
//! Membership (chelis#833): declaration-only ill-typed programs
//! (`total_nodes == 0`, e.g. a duplicate `deftype` / `defsig` / `typealias`).
//! The Phase 1 corpus asserted every member below 1.0 but every member had a
//! runtime body, so the `total_nodes == 0` regime -- where the coverage
//! components are all vacuously 1.0 -- went uncovered while `chelis check`
//! scored those programs a dishonest 1.0. Covered by the §C4.4 honesty cap in
//! `chelis_types::fitness`.
//!
//! Membership (post-Phase-3): chelis#780's deferred shape obligation,
//! chelis#850's unbacked `defsig`, chelis#1131's contradictory literal atom,
//! and chelis#1147's four `scatter_elements` admission holes. Each class also
//! has a well-typed score-1 control below.
//!
//! Membership (chelis#874 / chelis#887 / chelis#1525, PP8): a role-slot child
//! the checker read through a partial extraction that silently defaulted on
//! failure. §C4.1's coverage invariant quantifies over the nodes inference
//! VISITS, so a `vmap` axis it discarded, a `pat-ctor` or `pat-record` head it
//! skipped, a `grad` operand it typed fresh, a `kv` key it stepped past, and a
//! `pat-lit` value it declined to read all satisfied [04-TOT-2] vacuously and
//! scored a perfect 1.0. `spec/04-type-system.md` [04-TOT-4] settles the rule
//! the fix implements: an omitted optional child and a present unreadable one
//! are distinct inputs, and only the omission may take the form's declared
//! default. The paired positive controls below are the readable form of every
//! one of those slots, including bare `vmap(f)` with no axis child at all,
//! which must keep scoring 1.0.
//!
//! Membership (chelis#1494): a `match` arm whose literal pattern cannot denote
//! the scrutinee's primitive. `pattern_bindings` did nothing at `pat-lit`, so
//! `spec/04-type-system.md` [04-PAT-1]'s constraint went unchecked and an
//! `f32` pattern against an `i32` scrutinee scored 1.0 with an empty error
//! list while the arm could never match. The paired positive control is the
//! same program with a matching literal family, which must stay at 1.0 so the
//! rejection cannot creep into a well-formed match.
//!
//! Membership (chelis#2109): a locally bound untyped lambda passed to `vmap`
//! bypassed the launch-core inline-lambda fence and let the checker certify a
//! false result type at score 1.0. The focused CLI matrix owns Surf/Deep
//! parity, the correct-result rejection, nested type holes, alias propagation,
//! and typed local controls; these corpus rows keep the class-wide score
//! invariant explicit.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, i64))";

/// `chelis check` score for a program written with the given extension.
fn check_score(program: &str, ext: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["score"].as_f64().expect("numeric score")
}

/// Every member of the corpus must score strictly below 1.0. A member that
/// scores 1.0 is a fitness-honesty violation (the score lies about a program
/// that does not type-check) and fails the build.
fn assert_below_one(members: &[(&str, String, &str)]) {
    let mut offenders = Vec::new();
    for (name, program, ext) in members {
        let score = check_score(program, ext);
        if score >= 1.0 {
            offenders.push(format!("{name}: score {score} (must be < 1.0)"));
        }
    }
    assert!(
        offenders.is_empty(),
        "[04-TOT-2] fitness-honesty violation: known-bad programs scored 1.0:\n  {}",
        offenders.join("\n  ")
    );
}

/// The wrapper battery (chelis#709 canary) with an ill-typed body, plus the two
/// effect-handler holes and the seed-suffix / return-type diagnostics. Every
/// one is Surf-reachable and must score below 1.0.
#[test]
fn surf_known_bad_programs_score_below_one() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "issue_668_rank_divergent_elementwise",
            "module Repro.RankDivergent\n\
             sig f[n, u]: tensor[n, f32] -> tensor[u, f32]\n\
             def f(x) = {\n\
               s = stride(x, 2i64)\n\
               e = insert(x, 0i32, 2i64)\n\
               add(s, e)\n\
             }\n\
             out = f(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))\n"
                .to_string(),
            ".ch",
        ),
        (
            "issue_2109_untyped_local_vmap_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n"
                .to_string(),
            ".ch",
        ),
        (
            "issue_2109_wildcard_local_vmap_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v: _) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n"
                .to_string(),
            ".ch",
        ),
        (
            "issue_2109_nested_tuple_hole_vmap_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(fn (pair: (tensor[4, 3, f32], _)) -> \
                 sum(pair.1, 0i32))((copy(t), t))\n"
                .to_string(),
            ".ch",
        ),
        (
            "issue_2109_tuple_destructured_vmap_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               (mapped, keep) = (fn (v) -> sum(v, 0i32), 0i32)\n\
               vmap(mapped)(t)\n\
             }\n"
                .to_string(),
            ".ch",
        ),
        (
            "issue_2109_block_forwarded_vmap_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               alias = {\n\
                 forwarded = mapped\n\
                 forwarded\n\
               }\n\
               vmap(alias)(t)\n\
             }\n"
                .to_string(),
            ".ch",
        ),
        ("bare", format!("def f() -> f32 = {MASKED_ERROR}\n"), ".ch"),
        (
            "let_body",
            format!("def f() -> f32 = let x = {MASKED_ERROR} in x\n"),
            ".ch",
        ),
        (
            "if_then",
            format!("def f(c: bool) -> f32 = if c then {MASKED_ERROR} else 1.0\n"),
            ".ch",
        ),
        (
            "lambda_body",
            format!("def f() -> f32 = (fn (v: f32) -> f32 = {MASKED_ERROR})(1.0)\n"),
            ".ch",
        ),
        (
            "pipe_stage",
            format!("def f() -> f32 = 1.0 |> fn (v: f32) -> f32 = {MASKED_ERROR}\n"),
            ".ch",
        ),
        (
            "tuple_elem",
            format!("def f() -> (f32, f32) = ({MASKED_ERROR}, 1.0)\n"),
            ".ch",
        ),
        (
            "list_elem",
            format!("def f() -> [f32] = [{MASKED_ERROR}, 1.0]\n"),
            ".ch",
        ),
        (
            "match_arm",
            format!("def f(c: bool) -> f32 = match c {{ true => {MASKED_ERROR}, false => 1.0 }}\n"),
            ".ch",
        ),
        (
            "grad_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = grad(g)(x)\n"),
            ".ch",
        ),
        (
            "jit_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = jit(g)(x)\n"),
            ".ch",
        ),
        // chelis#709: the two effect-handler bodies (now checked).
        (
            "with_seed_body",
            format!("def f() -> f32 = with seed(42i64) {{ {MASKED_ERROR} }}\n"),
            ".ch",
        ),
        (
            "with_device_body",
            format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n"),
            ".ch",
        ),
        // chelis#709 escalation: an i64 body from an `-> f32` fn.
        (
            "masked_return_type",
            "def f() -> f32 = with seed(42i64) { cast(5, i64) }\n".to_string(),
            ".ch",
        ),
        // chelis#731 §C1.5 / chelis#771: an unsuffixed seed literal.
        (
            "unsuffixed_seed",
            "def f() -> f32 = with seed(42) { add(cast(1.0, f32), cast(2.0, f32)) }\n".to_string(),
            ".ch",
        ),
        // chelis#755 (Phase 2 join): field access on a multi-variant ADT.
        (
            "field_access_multi_variant",
            "type Shape = | Circle(f32) | Square(f32)\n\
             def f(s: Shape) -> f32 = s.radius\nout = print(f(Circle(1.0)))\n"
                .to_string(),
            ".ch",
        ),
        // chelis#755 (Phase 2 join): field access on a scalar.
        (
            "field_access_scalar",
            "def f(x: f32) -> f32 = x.field\nout = print(f(2.0))\n".to_string(),
            ".ch",
        ),
        // chelis#756 (Phase 2 join): cast to an unknown type name.
        (
            "cast_unknown_type",
            "def f() -> f32 = cast(1.0, madeup)\nout = print(f())\n".to_string(),
            ".ch",
        ),
        (
            "deferred_matmul_shape_mismatch",
            "def driver(good: tensor[4, 4, f32]) -> tensor[9, 9, f32] = {\n  \
             f = fn (a) -> matmul(a, good)\n  f(good)\n}\n"
                .to_string(),
            ".ch",
        ),
        (
            "scatter_elements_string_axis",
            "def f(data: tensor[2, 3, f32], indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] = scatter_elements(data, indices, updates, \"bad\")\n".to_string(),
            ".ch",
        ),
        (
            "scatter_elements_oob_axis",
            "def f(data: tensor[2, 3, f32], indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] = scatter_elements(data, indices, updates, 99)\n".to_string(),
            ".ch",
        ),
        (
            "scatter_elements_float_indices",
            "def f(data: tensor[2, 3, f32], indices: tensor[2, 2, f32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] = scatter_elements(data, indices, updates, 1)\n".to_string(),
            ".ch",
        ),
        (
            "scatter_elements_string_data",
            "def f(indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) = scatter_elements(\"bad\", indices, updates, 0)\n".to_string(),
            ".ch",
        ),
        (
            "literal_pattern_float_vs_int_scrutinee",
            "module ScrutineeSigned\n\ndef g(n: i32) -> i32 = add(1, n)\n\n             r: f32 = match g(2) with {\n  | 1.5 => 1.5\n  | _ => 2.5\n}\n"
                .to_string(),
            ".ch",
        ),
        (
            "literal_pattern_out_of_range_vs_int8_scrutinee",
            "def g(n: i8) -> i8 = add(0i8, n)\n\n             r: i32 = match g(1i8) with {\n  | 300 => 10\n  | _ => 20\n}\n"
                .to_string(),
            ".ch",
        ),
    ];
    assert_below_one(&cases);
}

/// The chelis#710 malformed `.dp` family (the two Phase 0 holes plus the
/// census-extension forms), each module-wrapped, plus the bogus-effect-kind
/// handle-effect. Every one reaches the checker malformed and must score below
/// 1.0 via a pushed `MalformedForm`.
#[test]
fn malformed_dp_forms_score_below_one() {
    let wrap = |form: &str| format!("(module {{}} m.main (def {{}} out {form}))\n");
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "dp_def_no_body",
            "(module {} m.main (def {} orphan))\n".to_string(),
            ".dp",
        ),
        (
            "dp_cast_no_target",
            wrap("(cast {} (lit {type: (t-prim {} f32)} 42.0))"),
            ".dp",
        ),
        ("dp_jit_no_child", wrap("(jit {})"), ".dp"),
        ("dp_realize_no_child", wrap("(realize {})"), ".dp"),
        ("dp_copy_no_child", wrap("(copy {})"), ".dp"),
        ("dp_borrow_no_child", wrap("(borrow {})"), ".dp"),
        ("dp_var_no_name", wrap("(var {})"), ".dp"),
        ("dp_lit_no_value", wrap("(lit {})"), ".dp"),
        ("dp_match_no_kids", wrap("(match {})"), ".dp"),
        ("dp_pipe_no_kids", wrap("(pipe {})"), ".dp"),
        (
            "dp_tuple_get_one_kid",
            wrap("(tuple-get {} (lit {type: (t-prim {} i32)} 0))"),
            ".dp",
        ),
        (
            "dp_record_non_symbol_head",
            wrap("(record {} (lit {type: (t-prim {} i32)} 1))"),
            ".dp",
        ),
        (
            "dp_access_one_kid",
            wrap("(access {} (lit {type: (t-prim {} i32)} 1))"),
            ".dp",
        ),
        (
            "dp_record_update_no_kids",
            wrap("(record-update {})"),
            ".dp",
        ),
        ("dp_grad_no_kids", wrap("(grad {})"), ".dp"),
        ("dp_vmap_no_kids", wrap("(vmap {})"), ".dp"),
        // chelis#709 / §I1: an unknown effect kind in a handle-effect.
        (
            "dp_unknown_effect_kind",
            wrap(
                "(handle-effect {effect: teleport} (lit {type: (t-prim {} i64)} 42) \
                 (lit {type: (t-prim {} f32)} 2.5))",
            ),
            ".dp",
        ),
        // A wrong-dtype signed seed remains rejected; negative i64 is valid.
        (
            "dp_negative_int32_seed",
            wrap(
                "(handle-effect {effect: random} (lit {type: (t-prim {} i32)} -1) \
                 (lit {type: (t-prim {} f32)} 2.5))",
            ),
            ".dp",
        ),
        (
            "dp_integer_atom_with_float_primitive",
            wrap("(lit {type: (t-prim {} f32)} 18014399583223809)"),
            ".dp",
        ),
        (
            "dp_nested_typed_literal",
            wrap("(lit {} (lit {type: (t-prim {} f32)} 7.5))"),
            ".dp",
        ),
        // chelis#731 red team: a handle-effect with a THIRD child (spec/03 gives
        // it exactly two); the extra child is an unvisited subtree ([04-TOT-3]).
        (
            "dp_handle_effect_extra_child",
            wrap(
                "(handle-effect {effect: random} (lit {type: (t-prim {} i64)} 42) \
                 (lit {type: (t-prim {} f32)} 2.5) (lit {type: (t-prim {} f32)} 9.0))",
            ),
            ".dp",
        ),
    ];
    assert_below_one(&cases);
}

/// chelis#710 form 4 / chelis#873: a bare `Symbol` or `Keyword` atom in
/// expression position, in the placements where nothing forces the atom's
/// type to meet a concrete one.
///
/// `infer_atom` typed these `Type::Unit` and had no `DiagnosticSink`
/// parameter, so it was structurally incapable of reporting. Where the `Unit`
/// met a concrete declared type, unification caught it; where it did not (a
/// `def` with no `defsig`, a signature returning `t-unit`, a top-level `def`
/// bound straight to an atom, an unused `let` binding) `chelis check`
/// returned `score: 1, errors: [], untyped_nodes: 0` on a program
/// `chelis build` then refused to lower, citing [05-UNS-1] / chelis#730:
/// unhandled or malformed forms cannot become Unit or another value. That is
/// a false 1.0 on the output spec/04-type-system.md §10 designates as the
/// training signal.
#[test]
fn bare_atom_expression_position_scores_below_one() {
    let wrap_body = |body: &str| {
        format!(
            "(module {{}} m.main (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} i32)}})) {body})))\n"
        )
    };
    let wrap_unit_sig = |body: &str| {
        format!(
            "(module {{}} m.main \
             (defsig {{}} f (t-fn {{}} (t-prim {{}} i32) (t-unit {{}}))) \
             (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} i32)}})) {body})))\n"
        )
    };
    let cases: Vec<(&str, String, &str)> = vec![
        ("dp_bare_keyword_body_no_defsig", wrap_body(":oops"), ".dp"),
        ("dp_bare_symbol_body_no_defsig", wrap_body("oops"), ".dp"),
        (
            "dp_bare_keyword_body_unit_defsig",
            wrap_unit_sig(":oops"),
            ".dp",
        ),
        (
            "dp_bare_symbol_body_unit_defsig",
            wrap_unit_sig("oops"),
            ".dp",
        ),
        (
            "dp_bare_keyword_toplevel_def",
            "(module {} m.main (def {} out :oops))\n".to_string(),
            ".dp",
        ),
        (
            "dp_bare_symbol_toplevel_def",
            "(module {} m.main (def {} out oops))\n".to_string(),
            ".dp",
        ),
        (
            "dp_bare_keyword_unused_let_binding",
            wrap_body("(let {} (bind {} unused :oops) (var {} x))"),
            ".dp",
        ),
    ];
    assert_below_one(&cases);
}

/// chelis#885: the seven bare-atom corpus members now hold the TOP rung.
///
/// `bare_atom_expression_position_scores_below_one` above is the check-rung
/// guard (score < 1.0). The #885 domain split moved the defect class up the
/// `docs/agent_quality_architecture.md` ladder: a bare atom in expression
/// position is an INGRESS rejection (`spec/03-deep-syntax.md` [03-ROLE-2]
/// for a bare identifier; the §8.1 metadata-key rule for a bare `:keyword`),
/// identified per [03-PROG-2] discipline. Every member must exit 2 with the
/// identification its class fixes: `keyword` for the four keyword members,
/// `bare name` for the three symbol members. The score-rung test above stays
/// as regression evidence; this one is the rung the class now holds.
#[test]
fn bare_atom_expression_position_rejects_at_parse() {
    let wrap_body = |body: &str| {
        format!(
            "(module {{}} m.main (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} i32)}})) {body})))\n"
        )
    };
    let wrap_unit_sig = |body: &str| {
        format!(
            "(module {{}} m.main \
             (defsig {{}} f (t-fn {{}} (t-prim {{}} i32) (t-unit {{}}))) \
             (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} i32)}})) {body})))\n"
        )
    };
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "dp_bare_keyword_body_no_defsig",
            wrap_body(":oops"),
            "keyword",
        ),
        (
            "dp_bare_symbol_body_no_defsig",
            wrap_body("oops"),
            "bare name",
        ),
        (
            "dp_bare_keyword_body_unit_defsig",
            wrap_unit_sig(":oops"),
            "keyword",
        ),
        (
            "dp_bare_symbol_body_unit_defsig",
            wrap_unit_sig("oops"),
            "bare name",
        ),
        (
            "dp_bare_keyword_toplevel_def",
            "(module {} m.main (def {} out :oops))\n".to_string(),
            "keyword",
        ),
        (
            "dp_bare_symbol_toplevel_def",
            "(module {} m.main (def {} out oops))\n".to_string(),
            "bare name",
        ),
        (
            "dp_bare_keyword_unused_let_binding",
            wrap_body("(let {} (bind {} unused :oops) (var {} x))"),
            "keyword",
        ),
    ];
    for (name, program, identification) in &cases {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("p.dp");
        write_file(&path, program);
        let out = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .expect("chelis check should run");
        assert_eq!(
            out.status.code(),
            Some(2),
            "[{name}] a bare atom in expression position is an ingress rejection (exit 2)"
        );
        let parsed: serde_json::Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("[{name}] check must emit JSON: {e}"));
        let errors = parsed["errors"].to_string();
        assert!(
            !errors.is_empty() && errors != "[]",
            "[{name}] the rejection must carry a diagnostic"
        );
        assert!(
            errors.contains(identification),
            "[{name}] the diagnostic must identify the rejected form as \
             {identification}: {errors}"
        );
    }
}

/// Negative parity for [`bare_atom_expression_position_scores_below_one`]:
/// the chelis#873 rejection must not over-apply.
///
/// `Symbol` atoms are how every Deep form carries its names. A `var`'s name,
/// a `record`'s constructor head, a `kv` key, an `access` field name, a
/// `deftype` / `defsig` / `def` / `export` name are all `Atom::Name`, and
/// their owning forms consume them with `symbol_name` rather than routing
/// them through expression inference. Only an atom that actually reaches
/// `infer_atom` is in expression position. If a future refactor routes a
/// structural symbol through the expression path, this test fails rather than
/// the language quietly losing the ability to name anything.
#[test]
fn structural_symbol_positions_still_score_one() {
    let program = "(module {}\n  \
         stats.prob\n  \
         (export {} probability prob_value)\n  \
         (deftype {opaque: true}\n    \
         Probability\n    \
         ()\n    \
         (variant {} Probability (field {} value (t-prim {} f32))))\n  \
         (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))\n  \
         (def {}\n    \
         probability\n    \
         (fn {}\n      \
         (params {} (x {type: (t-prim {} f32)}))\n      \
         (record {} Probability (kv {} value (var {} x)))))\n  \
         (defsig {} prob_value (t-fn {} (t-adt {} Probability) (t-prim {} f32)))\n  \
         (def {}\n    \
         prob_value\n    \
         (fn {}\n      \
         (params {} (p {type: (t-adt {} Probability)}))\n      \
         (access {} (var {} p) value))))\n";
    let score = check_score(program, ".dp");
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "chelis#873 over-application: a program whose only `Symbol` atoms sit in \
         structural positions (var name, record head, kv key, access field, \
         export / deftype / defsig / def names) must still score exactly 1.0; \
         got {score}"
    );
}

/// chelis#833 regression: declaration-only ill-typed programs
/// (`total_nodes == 0`). These are the fitness-honesty regime the Phase 1
/// corpus missed: with no runtime nodes, every coverage component is
/// vacuously 1.0, so a declaration-level diagnostic (a duplicate
/// `deftype` / `defsig` / `typealias`) let the weighted score read a
/// perfect 1.0 while the error was reported and the exit code was 2. The
/// §C4.4 honesty cap in `chelis_types::fitness` closes it; every member
/// here must score strictly below 1.0.
#[test]
fn declaration_only_known_bad_programs_score_below_one() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "dup_deftype",
            "(deftype {} Foo () (variant {} Foo)) (deftype {} Foo () (variant {} Foo))".to_string(),
            ".dp",
        ),
        (
            "dup_defsig",
            "(defsig {} foo (t-fn {} (t-prim {} i32) (t-prim {} i32))) \
             (defsig {} foo (t-fn {} (t-prim {} i32) (t-prim {} i32)))"
                .to_string(),
            ".dp",
        ),
        (
            "dup_typealias",
            "type Foo = f32\ntype Foo = i32\n".to_string(),
            ".ch",
        ),
        (
            "orphan_defsig",
            "(defsig {} missing (t-fn {} (t-prim {} string)))".to_string(),
            ".dp",
        ),
    ];
    assert_below_one(&cases);
}

#[test]
fn post_phase_checker_controls_still_score_one() {
    let cases = [
        (
            "paired_defsig",
            "(defsig {} present (t-fn {} (t-prim {} string))) \
             (def {} present (fn {} (params {}) (lit {type: (t-prim {} string)} \"ok\")))",
            ".dp",
        ),
        (
            "consistent_deferred_matmul",
            "def driver(good: tensor[4, 4, f32]) -> tensor[4, 4, f32] = {\n  \
             f = fn (a) -> matmul(a, good)\n  f(good)\n}\n",
            ".ch",
        ),
        (
            "valid_scatter_elements",
            "def f(data: tensor[2, 3, f32], indices: tensor[2, 2, i32], updates: tensor[2, 2, f32]) -> tensor[2, 3, f32] = scatter_elements(data, indices, updates, 1)\n",
            ".ch",
        ),
        (
            "matching_literal_pattern_family",
            "module ScrutineeSigned\n\ndef g(n: i32) -> i32 = add(1, n)\n\n             r: f32 = match g(2) with {\n  | 1 => 1.5\n  | _ => 2.5\n}\n",
            ".ch",
        ),
    ];
    for (name, program, extension) in cases {
        let score = check_score(program, extension);
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "{name} must remain a score-1 control, got {score}"
        );
    }
}

/// Membership (chelis#858): a top-level UNTAGGED Deep list. Before the
/// fix, `infer_top_level` silently skipped any top-level list whose
/// element 0 is not a decoded tag, and `check_ir_program`'s clean path
/// fabricated `typed_nodes = count_nodes(exprs)`, so
/// `((var {} f) (var {} x))` scored a vacuous 1.0 with an empty error
/// list while `f`/`x` were unbound and nothing was checked. The skip is
/// now a loud `UnknownForm` and the clean path reports the inference
/// product's own counters. Both polarities: the negative rows must score
/// below 1.0, and the positive control must still score exactly 1.0 so
/// the loud arm cannot creep into well-formed programs.
#[test]
fn top_level_untagged_lists_score_below_one() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "untagged_top_level_app_shape",
            "((var {} f) (var {} x))".to_string(),
            ".dp",
        ),
        (
            "untagged_top_level_beside_valid_def",
            "(def {} out (lit {type: (t-prim {} f32)} 1.0)) ((var {} f) (var {} x))".to_string(),
            ".dp",
        ),
    ];
    assert_below_one(&cases);
}

/// chelis#858 positive control: the sibling well-formed program still
/// scores a perfect 1.0 after the loud-skip fix and the typed-node
/// accounting change.
#[test]
fn well_formed_control_still_scores_one_after_858() {
    let score = check_score("(def {} out (lit {type: (t-prim {} f32)} 1.0))", ".dp");
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "the well-formed control must still score 1.0, got {score}"
    );
}

/// chelis#874 / chelis#887 / chelis#1525 (PP8): the source-coverage slots.
///
/// Each program below submits a child the form's semantics reads and the
/// checker could not read. Before the fix, R1 through R4, the record-pattern
/// `kv` key, and the three `pat-lit` rows all scored exactly 1.0 with an empty
/// error vector; the record-construction, record-update, `pat-var` and `pat-as`
/// rows were rejected, but by §C4.1's owner-stamp tripwire blaming a collateral
/// node rather than by the slot that was wrong. The unit-level assertions on
/// the diagnostics live in
/// `crates/chelis-types/tests/issue_874_source_coverage_totality.rs`; these
/// rows are the corpus-level property that the SCORE cannot read "perfect".
#[test]
fn pp8_source_coverage_slots_score_below_one() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "pp8_vmap_axis_var",
            pp8_vmap("(var {} nonexistent_name_zzz)"),
            ".dp",
        ),
        (
            "pp8_vmap_axis_float_literal",
            pp8_vmap("(lit {type: (t-prim {} f32)} 1.5)"),
            ".dp",
        ),
        (
            "pp8_vmap_axis_type_node",
            pp8_vmap("(t-prim {} f32)"),
            ".dp",
        ),
        ("pp8_vmap_axis_application", pp8_vmap(PP8_APP), ".dp"),
        (
            "pp8_pat_ctor_head_application",
            pp8_pat_ctor(PP8_APP),
            ".dp",
        ),
        (
            "pp8_pat_record_head_var",
            pp8_pat_record("(var {} nonexistent_name_zzz)"),
            ".dp",
        ),
        (
            "pp8_grad_non_function_operand",
            pp8_wrap(
                "(def {} d2 (fn {} (params {} (x {type: (t-prim {} f32)})) (grad {} (var {} x))))",
            ),
            ".dp",
        ),
        (
            "pp8_record_kv_key_var",
            pp8_record_kv("(var {} nonexistent_name_zzz)"),
            ".dp",
        ),
        (
            "pp8_record_update_kv_key_var",
            pp8_record_update_kv("(var {} nonexistent_name_zzz)"),
            ".dp",
        ),
        (
            "pp8_pat_record_kv_key_var",
            pp8_pat_record_kv("(var {} nonexistent_name_zzz)"),
            ".dp",
        ),
        (
            "pp8_pat_var_name_node",
            pp8_wrap(
                "(defsig {} pv (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
                 (def {} pv (fn {} (params {} (q {type: (t-prim {} f32)}))\n\
                 (match {} (var {} q) (arm {} (pat-var {} (var {} zz)) () \
                 (lit {type: (t-prim {} f32)} 1.0)))))",
            ),
            ".dp",
        ),
        (
            "pp8_pat_as_name_node",
            pp8_wrap(
                "(defsig {} pa (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
                 (def {} pa (fn {} (params {} (q {type: (t-prim {} f32)}))\n\
                 (match {} (var {} q) (arm {} (pat-as {} (var {} zz) (pat-wild {})) () \
                 (lit {type: (t-prim {} f32)} 1.0)))))",
            ),
            ".dp",
        ),
        // chelis#1525: spec/03-deep-syntax.md section 6.3 fixes a `pat-lit`
        // value as a literal, not an expression node, so a `lit` wrapper is
        // malformed Deep. The float row is the one that shows the vacuity
        // biting: the same atom unwrapped is an [04-PAT-1] rejection.
        ("pp8_pat_lit_lit_node_int", pp8_pat_lit("(lit {} 1)"), ".dp"),
        (
            "pp8_pat_lit_lit_node_float",
            pp8_pat_lit("(lit {type: (t-prim {} f32)} 1.5)"),
            ".dp",
        ),
        ("pp8_pat_lit_var_node", pp8_pat_lit("(var {} zz)"), ".dp"),
    ];
    assert_below_one(&cases);
}

/// chelis#874 positive controls: the readable form of every slot above stays at
/// a perfect 1.0. Their job is to prove the seam rejects UNREADABILITY rather
/// than non-tagged-ness -- without them, the cheapest wrong fix (rejecting
/// every structural bare list) passes every negative row.
#[test]
fn pp8_readable_slots_still_score_one() {
    let cases: Vec<(&str, String)> = vec![
        // spec/02 section 0.1 spells the zero axis as bare `vmap(f)`, so the
        // OMITTED child must still take the form's declared default.
        ("pp8_vmap_axis_absent", pp8_vmap("")),
        (
            "pp8_vmap_axis_zero",
            pp8_vmap("(lit {type: (t-prim {} i32)} 0)"),
        ),
        (
            "pp8_vmap_axis_cast_wrapped",
            pp8_vmap("(cast {} (lit {type: (t-prim {} i64)} 0) (t-prim {} i32))"),
        ),
        ("pp8_pat_ctor_real_head", pp8_pat_ctor("Non")),
        ("pp8_pat_record_real_head", pp8_pat_record("Circle")),
        (
            "pp8_grad_of_a_function",
            pp8_wrap("(def {} d4 (grad {} (var {} double)))"),
        ),
        ("pp8_record_kv_declared_field", pp8_record_kv("r")),
        (
            "pp8_record_update_declared_field",
            pp8_record_update_kv("r"),
        ),
        ("pp8_pat_record_kv_declared_field", pp8_pat_record_kv("r")),
        ("pp8_pat_lit_bare_atom", pp8_pat_lit("1")),
    ];
    for (name, program) in cases {
        let score = check_score(&program, ".dp");
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "{name} must remain a score-1 control, got {score}"
        );
    }
}

/// An unreadable child the PP8 probe table drives through several slots.
const PP8_APP: &str = "(app {} (var {} missing_fn_qqq) (var {} missing_arg_www))";

/// Declarations the PP8 fixtures share: a scalar function, a rank-1 tensor
/// function to `vmap`, and three ADTs.
const PP8_PRELUDE: &str = concat!(
    "(deftype {} Shape () (variant {} Circle (field {} r (t-prim {} f32)))\n",
    "  (variant {} Square (field {} a (t-prim {} f32))))\n",
    "(deftype {} Box () (variant {} Box (field {} r (t-prim {} f32))))\n",
    "(deftype {} Opt () (variant {} Non) (variant {} Som (field {} value (t-prim {} f32))))\n",
    "(defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
    "(def {} double (fn {} (params {} (x {type: (t-prim {} f32)}))\n",
    "  (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} f32)} 2.0))))\n",
    "(defsig {} process (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))\n",
    "  (t-tensor {} (d-name {} features) (t-prim {} f32))))\n",
    "(def {} process (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) \
     (t-prim {} f32))}))\n",
    "  (app {} (var {} relu) (var {} x))))\n",
);

fn pp8_wrap(body: &str) -> String {
    format!("(module {{}} m.main\n{PP8_PRELUDE}{body})\n")
}

fn pp8_vmap(axis: &str) -> String {
    let child = if axis.is_empty() {
        String::new()
    } else {
        format!(" {axis}")
    };
    pp8_wrap(&format!(
        "(defsig {{}} batched (t-fn {{}} (t-tensor {{}} (d-name {{}} batch) \
         (d-name {{}} features) (t-prim {{}} f32))\n\
         (t-tensor {{}} (d-name {{}} batch) (d-name {{}} features) (t-prim {{}} f32))))\n\
         (def {{}} batched (fn {{}} (params {{}} (xs {{type: (t-tensor {{}} (d-name {{}} batch) \
         (d-name {{}} features) (t-prim {{}} f32))}}))\n\
         (app {{}} (vmap {{}} (var {{}} process){child}) (var {{}} xs))))"
    ))
}

fn pp8_pat_ctor(head: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} gg (t-fn {{}} (t-adt {{}} Opt) (t-prim {{}} f32)))\n\
         (def {{}} gg (fn {{}} (params {{}} (s {{type: (t-adt {{}} Opt)}}))\n\
         (match {{}} (var {{}} s)\n\
         (arm {{}} (pat-ctor {{}} {head}) () (lit {{type: (t-prim {{}} f32)}} 1.0))\n\
         (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    ))
}

fn pp8_pat_record(head: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} hh (t-fn {{}} (t-adt {{}} Shape) (t-prim {{}} f32)))\n\
         (def {{}} hh (fn {{}} (params {{}} (s {{type: (t-adt {{}} Shape)}}))\n\
         (match {{}} (var {{}} s)\n\
         (arm {{}} (pat-record {{}} {head}) () (lit {{type: (t-prim {{}} f32)}} 1.0))\n\
         (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    ))
}

fn pp8_pat_record_kv(key: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} pr (t-fn {{}} (t-adt {{}} Shape) (t-prim {{}} f32)))\n\
         (def {{}} pr (fn {{}} (params {{}} (s {{type: (t-adt {{}} Shape)}}))\n\
         (match {{}} (var {{}} s)\n\
         (arm {{}} (pat-record {{}} Circle (kv {{}} {key} (pat-wild {{}}))) () \
         (lit {{type: (t-prim {{}} f32)}} 1.0))\n\
         (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    ))
}

fn pp8_record_kv(key: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} mk (t-fn {{}} (t-adt {{}} Shape)))\n\
         (def {{}} mk (fn {{}} (params {{}})\n\
         (record {{}} Circle (kv {{}} {key} (lit {{type: (t-prim {{}} f32)}} 1.0)))))"
    ))
}

fn pp8_record_update_kv(key: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} ru (t-fn {{}} (t-adt {{}} Box) (t-adt {{}} Box)))\n\
         (def {{}} ru (fn {{}} (params {{}} (s {{type: (t-adt {{}} Box)}}))\n\
         (record-update {{}} (var {{}} s) (kv {{}} {key} \
         (lit {{type: (t-prim {{}} f32)}} 3.0)))))"
    ))
}

fn pp8_pat_lit(value: &str) -> String {
    pp8_wrap(&format!(
        "(defsig {{}} pl (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} f32)))\n\
         (def {{}} pl (fn {{}} (params {{}} (n {{type: (t-prim {{}} i32)}}))\n\
         (match {{}} (var {{}} n)\n\
         (arm {{}} (pat-lit {{}} {value}) () (lit {{type: (t-prim {{}} f32)}} 1.0))\n\
         (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    ))
}
