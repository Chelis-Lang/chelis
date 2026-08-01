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

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, int64))";

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
        // chelis#709 escalation: an int64 body from an `-> f32` fn.
        (
            "masked_return_type",
            "def f() -> f32 = with seed(42i64) { cast(5, int64) }\n".to_string(),
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
            wrap("(tuple-get {} (lit {type: (t-prim {} int32)} 0))"),
            ".dp",
        ),
        (
            "dp_record_non_symbol_head",
            wrap("(record {} (lit {type: (t-prim {} int32)} 1))"),
            ".dp",
        ),
        (
            "dp_access_one_kid",
            wrap("(access {} (lit {type: (t-prim {} int32)} 1))"),
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
                "(handle-effect {effect: teleport} (lit {type: (t-prim {} int64)} 42) \
                 (lit {type: (t-prim {} f32)} 2.5))",
            ),
            ".dp",
        ),
        // chelis#731 red team F2: a negative int64-literal seed (the RNG lanes
        // fold it to seed 0, breaking [05-RNG-1]).
        (
            "dp_negative_int64_seed",
            wrap(
                "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} -1) \
                 (lit {type: (t-prim {} f32)} 2.5))",
            ),
            ".dp",
        ),
        // chelis#731 red team: a handle-effect with a THIRD child (spec/03 gives
        // it exactly two); the extra child is an unvisited subtree ([04-TOT-3]).
        (
            "dp_handle_effect_extra_child",
            wrap(
                "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) \
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
            "(module {{}} m.main (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} int32)}})) {body})))\n"
        )
    };
    let wrap_unit_sig = |body: &str| {
        format!(
            "(module {{}} m.main \
             (defsig {{}} f (t-fn {{}} (t-prim {{}} int32) (t-unit {{}}))) \
             (def {{}} f (fn {{}} (params {{}} (x {{type: (t-prim {{}} int32)}})) {body})))\n"
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
            "(defsig {} foo (t-fn {} (t-prim {} int32) (t-prim {} int32))) \
             (defsig {} foo (t-fn {} (t-prim {} int32) (t-prim {} int32)))"
                .to_string(),
            ".dp",
        ),
        (
            "dup_typealias",
            "type Foo = f32\ntype Foo = int32\n".to_string(),
            ".ch",
        ),
    ];
    assert_below_one(&cases);
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
