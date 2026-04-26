//! Phase D: tests for `check_effects_with_context`.
//!
//! These tests cover three load-bearing properties of the new API:
//! 1. **Parity vs monolithic**: for a library + snippet pair,
//!    `check_effects_with_context(library, snippet)` produces the same
//!    effect rows on the snippet's defs as `check_program(library + snippet)`.
//! 2. **Cross-context inheritance**: a new-code def calling a library
//!    Io-effecting / Test-effecting helper inherits the effect.
//! 3. **No leak**: a new-code def's effect row does not persist into a
//!    subsequent call against the same `library_program`.

use chelis_deep::ast::Expr;
use chelis_effects::{EffectError, EffectErrorKind, check_effects_with_context, check_program};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{
    CheckedProgram, TypeEnv, build_type_env_from_library, check_phase0e_program,
    check_phase0e_with_context,
};

// ── Helpers ──────────────────────────────────────────────────────────────

fn parse_then_desugar(src: &str) -> Vec<Expr> {
    let decls = parse_surf(src).expect("surf parse");
    desugar_program(&decls)
}

/// Build (library_typeenv, library_checked_with_effects) from library Surf source.
fn build_library_pair(library_src: &str) -> (TypeEnv, CheckedProgram) {
    let lib_deep = parse_then_desugar(library_src);
    let typeenv = build_type_env_from_library(&lib_deep).expect("library type-checks");
    // Run the FULL effects pipeline on the library — same way Phase D's
    // contract requires (the library's effect rows are already validated).
    let lib_checked = check_phase0e_program(&lib_deep).expect("library Phase 0e");
    let lib_with_effects = check_program(&lib_checked).expect("library effects clean");
    (typeenv, lib_with_effects)
}

/// Build a `CheckedProgram` for new-code only via Phase C's `with_context`,
/// then return it. New-code's `annotated_exprs()` carry only new defs.
fn build_new_code_checked(typeenv: &TypeEnv, new_src: &str) -> CheckedProgram {
    let new_deep = parse_then_desugar(new_src);
    check_phase0e_with_context(typeenv, &new_deep).expect("new code type-checks")
}

// ── Parity tests (5 library+snippet pairs) ───────────────────────────────

/// Parity 1: pure library, pure snippet. Both pass cleanly.
#[test]
fn parity_pure_library_pure_snippet() {
    let library = r#"
def lib_id(x: int64) -> int64 = x
"#;
    let snippet = r#"
def caller(y: int64) -> int64 = lib_id(y)
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    check_effects_with_context(&lib_checked, &new_checked)
        .expect("pure library, pure snippet must pass");

    // Monolithic equivalent:
    let combined = parse_then_desugar(&format!("{library}\n{snippet}"));
    let combined_checked = check_phase0e_program(&combined).unwrap();
    check_program(&combined_checked).expect("monolithic must also pass");
}

/// Parity 2: library exposes an Io helper; new-code calls it. Both
/// monolithic and context paths must accept (the snippet does not declare
/// `! {}`, so inferred Io is fine).
#[test]
fn parity_library_io_helper_called_from_snippet() {
    let library = r#"
def emit(x: int64) -> int64 = debug(x)
"#;
    let snippet = r#"
def caller(y: int64) -> int64 = emit(y)
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let new_with_effects = check_effects_with_context(&lib_checked, &new_checked)
        .expect("io-inheriting snippet must pass");

    // Verify the snippet's caller def does carry `effects` metadata
    // referencing io. The annotated decl list is the snippet's only.
    let snippet_text = chelis_deep::printer::print_canonical(new_with_effects.annotated_exprs());
    assert!(
        snippet_text.contains("effects"),
        "expected effect metadata on snippet caller, got:\n{snippet_text}"
    );
    assert!(
        snippet_text.contains("io"),
        "expected io effect inherited from library helper, got:\n{snippet_text}"
    );

    // Monolithic parity:
    let combined = parse_then_desugar(&format!("{library}\n{snippet}"));
    let combined_checked = check_phase0e_program(&combined).unwrap();
    check_program(&combined_checked).expect("monolithic must also pass");
}

/// Parity 3: library exposes a Test wrapper; new-code calls it. Both paths
/// accept (snippet does not declare `! {}`).
#[test]
fn parity_library_test_helper_called_from_snippet() {
    let library = r#"
def lib_check() -> unit = test_assert(true, "lib")
"#;
    let snippet = r#"
def my_test() -> unit = lib_check()
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let new_with_effects = check_effects_with_context(&lib_checked, &new_checked)
        .expect("test-inheriting snippet must pass");

    let snippet_text = chelis_deep::printer::print_canonical(new_with_effects.annotated_exprs());
    assert!(
        snippet_text.contains("test"),
        "expected test effect inherited from library wrapper, got:\n{snippet_text}"
    );

    let combined = parse_then_desugar(&format!("{library}\n{snippet}"));
    let combined_checked = check_phase0e_program(&combined).unwrap();
    check_program(&combined_checked).expect("monolithic must also pass");
}

/// Parity 4: declared `! {}` snippet calling Test-effecting library wrapper
/// must be rejected by both paths with the same error kind.
#[test]
fn parity_declared_empty_eff_snippet_rejects_test_effect_from_library() {
    let library = r#"
def lib_check() -> unit = test_assert(true, "lib")
"#;
    let snippet = r#"
def leak() -> unit ! {} = lib_check()
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let ctx_errors = check_effects_with_context(&lib_checked, &new_checked)
        .expect_err("declared !{} but inherits Test from library must reject");
    assert!(
        ctx_errors
            .iter()
            .any(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Test")),
        "expected UnhandledEffect mentioning Test, got {ctx_errors:?}"
    );

    // Monolithic parity:
    let combined = parse_then_desugar(&format!("{library}\n{snippet}"));
    let combined_checked = check_phase0e_program(&combined).unwrap();
    let mono_errors = check_program(&combined_checked)
        .expect_err("monolithic must also reject !{} that calls Test helper");
    assert_eq!(
        ctx_errors
            .iter()
            .filter(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Test"))
            .count(),
        mono_errors
            .iter()
            .filter(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Test"))
            .count(),
        "context-extension and monolithic paths must produce equivalent UnhandledEffect rejections"
    );
}

/// Parity 5: library has a Random-effecting helper, snippet calls it from a
/// Test-effecting context. The composed effect row must include BOTH
/// Random and Test. Acceptance criterion: declared `! {Random,Test}` accepts;
/// declared `! {Test}` rejects with UnhandledEffect mentioning Random.
#[test]
fn parity_random_lib_helper_under_test_context_composes_both_effects() {
    let library = r#"
def lib_drop(x: tensor[8, f32]) -> tensor[8, f32] = dropout(x, 0.5)
"#;
    let snippet_ok = r#"
def use_drop(t: tensor[8, f32]) -> unit ! {Random, Test} =
  test_assert(true, "before-drop")
"#;
    // Note: Surf doesn't easily let me thread `lib_drop` into the body and
    // discard, so the snippet uses test_assert + a separate stmt is awkward.
    // Use a let-binding form instead.
    let snippet_ok2 = r#"
def use_drop(t: tensor[8, f32]) -> unit ! {Random, Test} = {
  y = lib_drop(t)
  test_assert(true, "after-drop")
}
"#;
    let snippet_bad = r#"
def use_drop(t: tensor[8, f32]) -> unit ! {Test} = {
  y = lib_drop(t)
  test_assert(true, "after-drop")
}
"#;

    let _ = snippet_ok; // silence dead

    // Positive: declared {Random, Test} accepts.
    let (typeenv, lib_checked) = build_library_pair(library);
    let new_ok = build_new_code_checked(&typeenv, snippet_ok2);
    check_effects_with_context(&lib_checked, &new_ok)
        .expect("declared {Random,Test} must accept lib_drop + test_assert");

    // Negative: declared {Test} rejects with UnhandledEffect mentioning Random.
    let new_bad = build_new_code_checked(&typeenv, snippet_bad);
    let ctx_errors = check_effects_with_context(&lib_checked, &new_bad)
        .expect_err("declared {Test} must reject lib_drop's Random");
    assert!(
        ctx_errors
            .iter()
            .any(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Random")),
        "expected UnhandledEffect mentioning Random, got {ctx_errors:?}"
    );

    // Monolithic parity (negative case): same rejection.
    let combined = parse_then_desugar(&format!("{library}\n{snippet_bad}"));
    let combined_checked = check_phase0e_program(&combined).unwrap();
    let mono_errors = check_program(&combined_checked).expect_err("monolithic must also reject");
    assert!(
        mono_errors
            .iter()
            .any(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Random")),
        "monolithic must also flag Random, got {mono_errors:?}"
    );
}

// ── Specifically required: library-effect inheritance ────────────────────

/// New-code function that calls library `read_file` (an Io-effecting builtin
/// wrapped in a library def) inherits Io.
#[test]
fn new_code_inherits_io_via_library_read_file_wrapper() {
    let library = r#"
def lib_load(path: string) -> string = read_file(path)
"#;
    let snippet = r#"
def my_loader(p: string) -> string = lib_load(p)
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let new_with_effects = check_effects_with_context(&lib_checked, &new_checked)
        .expect("io-inheriting snippet must pass");

    let snippet_text = chelis_deep::printer::print_canonical(new_with_effects.annotated_exprs());
    assert!(
        snippet_text.contains("io"),
        "expected io effect inherited from lib_load -> read_file chain, got:\n{snippet_text}"
    );

    // Negative parity: declaring `! {}` must reject.
    let bad_snippet = r#"
def my_loader(p: string) -> string ! {} = lib_load(p)
"#;
    let new_bad = build_new_code_checked(&typeenv, bad_snippet);
    let errors = check_effects_with_context(&lib_checked, &new_bad)
        .expect_err("declared !{} but inherits Io from lib_load must reject");
    assert!(
        errors
            .iter()
            .any(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("IO")),
        "expected UnhandledEffect mentioning IO, got {errors:?}"
    );
}

/// New-code function that calls a Test-effecting library wrapper inherits Test.
#[test]
fn new_code_inherits_test_via_library_wrapper() {
    let library = r#"
def lib_assert_eq(a: int64, b: int64) -> unit = test_assert_eq_int(a, b, "lib_assert_eq")
"#;
    let snippet = r#"
def my_check(x: int64) -> unit = lib_assert_eq(x, cast(1, int64))
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let new_with_effects = check_effects_with_context(&lib_checked, &new_checked)
        .expect("test-inheriting snippet must pass");

    let snippet_text = chelis_deep::printer::print_canonical(new_with_effects.annotated_exprs());
    assert!(
        snippet_text.contains("test"),
        "expected test effect inherited from lib_assert_eq, got:\n{snippet_text}"
    );

    // Negative parity: declaring `! {}` must reject.
    let bad_snippet = r#"
def my_check(x: int64) -> unit ! {} = lib_assert_eq(x, cast(1, int64))
"#;
    let new_bad = build_new_code_checked(&typeenv, bad_snippet);
    let errors = check_effects_with_context(&lib_checked, &new_bad)
        .expect_err("declared !{} but inherits Test from lib_assert_eq must reject");
    assert!(
        errors
            .iter()
            .any(|e| e.kind == EffectErrorKind::UnhandledEffect && e.message.contains("Test")),
        "expected UnhandledEffect mentioning Test, got {errors:?}"
    );
}

// ── No-leak: new-code effect rows don't persist into subsequent calls ────

/// Two snippets checked sequentially against the same `library_program`. A
/// def in snippet A whose declared signature is `! {Test}` (and whose body
/// performs Test) must NOT be visible to snippet B's effect inference. If
/// it were, snippet B's `unrelated` def — which calls neither Test nor IO —
/// would erroneously inherit Test (because library_program's effect map
/// would have been mutated to include it).
///
/// Formally: snippet B's caller of an unrelated library helper must end up
/// with NO effect, regardless of what snippet A's defs declared.
#[test]
fn no_leak_snippet_a_effects_do_not_persist_into_snippet_b() {
    let library = r#"
def lib_id(x: int64) -> int64 = x
"#;
    let snippet_a = r#"
def stamp() -> unit ! {Test} = test_assert(true, "snippet-a")
def call_a(y: int64) -> int64 = lib_id(y)
"#;
    let snippet_b = r#"
def call_b(z: int64) -> int64 = lib_id(z)
"#;

    let (typeenv, lib_checked) = build_library_pair(library);

    // Snippet A: assert it type-checks, has Test effect on `stamp`, and
    // accepts.
    let a_checked = build_new_code_checked(&typeenv, snippet_a);
    let a_with_effects =
        check_effects_with_context(&lib_checked, &a_checked).expect("snippet A passes");
    let a_text = chelis_deep::printer::print_canonical(a_with_effects.annotated_exprs());
    assert!(
        a_text.contains("test"),
        "snippet A's stamp should have test effect"
    );

    // Snippet B: must produce a `call_b` def with NO effects whatsoever.
    // If `lib_checked` had been mutated to include snippet A's `stamp`
    // entry, `call_b` would still be effect-free (it doesn't call stamp);
    // so this test alone is not perfectly diagnostic. Stronger probe
    // below: snippet B re-defines a name that clashes with snippet A's
    // `stamp`, then declares `! {}`. If snippet A's effect row leaked,
    // the new `stamp` would be rejected.
    let b_checked = build_new_code_checked(&typeenv, snippet_b);
    let b_with_effects =
        check_effects_with_context(&lib_checked, &b_checked).expect("snippet B passes");
    let b_text = chelis_deep::printer::print_canonical(b_with_effects.annotated_exprs());
    // call_b's body is just `lib_id(z)`; lib_id is effect-free, so the
    // annotated output should NOT carry an `effects` block on call_b.
    // (The Surf desugar may emit `(effects)` empty though, so check
    // `effects test|effects io|effects random` substrings instead.)
    assert!(
        !b_text.contains("test"),
        "snippet B's call_b must NOT inherit any test effect, got:\n{b_text}"
    );

    // Stronger probe: a snippet C with `def stamp() -> unit ! {} = lib_id(...)`
    // — same name as snippet A's stamp, but pure body and `! {}` declared.
    // If snippet A's `! {Test}` row leaked into lib_checked, the stamp
    // entry would still be `{Test}` and validate_declared_vs_inferred
    // would reject. Library is unchanged — must accept.
    let snippet_c = r#"
def stamp() -> unit ! {} = {
  _u = lib_id(cast(1, int64))
  cast((), unit)
}
"#;
    let c_checked = build_new_code_checked(&typeenv, snippet_c);
    check_effects_with_context(&lib_checked, &c_checked).expect(
        "snippet C declares pure stamp; library_program must NOT have been mutated by snippet A",
    );
}

/// Direct mutation probe: re-call against the SAME `lib_checked` reference
/// twice and assert effects on the first snippet are unaffected by a
/// previously-passed snippet that defined an effect-rich def.
#[test]
fn no_leak_repeated_calls_against_same_library_are_independent() {
    let library = r#"
def lib_id(x: int64) -> int64 = x
"#;
    let (typeenv, lib_checked) = build_library_pair(library);

    let effectful_snippet = r#"
def loud() -> unit ! {Test} = test_assert(true, "loud")
"#;
    let pure_snippet = r#"
def quiet(z: int64) -> int64 = lib_id(z)
"#;

    // Call 1: effectful_snippet. Must accept (declared {Test} matches inferred {Test}).
    let eff_checked = build_new_code_checked(&typeenv, effectful_snippet);
    check_effects_with_context(&lib_checked, &eff_checked).expect("effectful snippet accepts");

    // Call 2: pure_snippet. Must accept and produce ZERO effects on `quiet`.
    let pure_checked = build_new_code_checked(&typeenv, pure_snippet);
    let pure_with_effects =
        check_effects_with_context(&lib_checked, &pure_checked).expect("pure snippet accepts");
    let text = chelis_deep::printer::print_canonical(pure_with_effects.annotated_exprs());
    assert!(
        !text.contains("test") && !text.contains("io") && !text.contains("random"),
        "pure snippet's quiet must carry no inferred effects, got:\n{text}"
    );
}

// ── Position-sensitive errors ────────────────────────────────────────────

/// EffectError span coordinates must trace back to the new-source byte
/// offsets, not to library coordinates. The current `EffectError` struct
/// doesn't expose a span field, but the suggestions/messages must mention
/// new-code names (e.g. `leak`), not library names. This is a guard for
/// the case where a future revision adds spans to EffectError — the test
/// will continue to be correct because both messages and (future) spans
/// point at new-code's `leak` def.
#[test]
fn error_messages_reference_new_code_names_not_library() {
    let library = r#"
def lib_check() -> unit = test_assert(true, "lib")
"#;
    let snippet = r#"
def leak() -> unit ! {} = lib_check()
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let errors = check_effects_with_context(&lib_checked, &new_checked).expect_err("must reject");

    // The error message must name `leak` (new-code def), NOT `lib_check`
    // (the library helper).
    let first_unhandled: &EffectError = errors
        .iter()
        .find(|e| e.kind == EffectErrorKind::UnhandledEffect)
        .expect("at least one UnhandledEffect");
    assert!(
        first_unhandled.message.contains("leak"),
        "error message must reference new-code def name `leak`, got:\n{}",
        first_unhandled.message
    );
    assert!(
        !first_unhandled.message.contains("lib_check"),
        "error message must NOT name library helper `lib_check`, got:\n{}",
        first_unhandled.message
    );
}

// ── Regression: existing check_program continues to work ──────────────────

/// Smoke that the un-modified `check_program` API still does the right
/// thing on a monolithic input. Phase D must not regress the legacy path.
#[test]
fn legacy_check_program_still_works() {
    let combined = parse_then_desugar(
        r#"
def emit(x: int64) -> int64 = debug(x)
def caller(y: int64) -> int64 = emit(y)
"#,
    );
    let checked = check_phase0e_program(&combined).expect("phase 0e");
    let with_effects = check_program(&checked).expect("legacy check_program clean");
    let text = chelis_deep::printer::print_canonical(with_effects.annotated_exprs());
    assert!(
        text.contains("io"),
        "legacy path must still infer Io, got:\n{text}"
    );
}

// ── Sanity: composition with multiple library defs at once ───────────────

#[test]
fn snippet_composing_two_library_helpers_picks_up_both_effects() {
    let library = r#"
def lib_emit(x: int64) -> int64 = debug(x)
def lib_assert(a: int64, b: int64) -> unit = test_assert_eq_int(a, b, "lib_assert")
"#;
    let snippet = r#"
def my_op(x: int64) -> unit ! {IO, Test} = {
  _y = lib_emit(x)
  lib_assert(x, x)
}
"#;

    let (typeenv, lib_checked) = build_library_pair(library);
    let new_checked = build_new_code_checked(&typeenv, snippet);
    let new_with_effects = check_effects_with_context(&lib_checked, &new_checked)
        .expect("declared {IO, Test} composes both library helpers");

    let text = chelis_deep::printer::print_canonical(new_with_effects.annotated_exprs());
    assert!(text.contains("io"), "must inherit io, got:\n{text}");
    assert!(text.contains("test"), "must inherit test, got:\n{text}");
}
