//! Issue #2442: a literal pattern against a rigid dtype-binder scrutinee is
//! decided at every instantiation the binder admits.
//!
//! `spec/04-type-system.md` [04-PAT-1] makes a literal pattern a typing
//! constraint on the scrutinee, [04-LIT-2] binds it at the scrutinee's
//! primitive, and [04-INF-6] makes an authored binder denote every admissible
//! instantiation. Before the fix `check_literal_pattern` returned early on any
//! `Type::Var` scrutinee, so `| 300 =>` under `p: Int` (out of range at `i8`),
//! `| 70000.0 =>` under `p: Float` (infinite at `f16`), and `| 0 =>` under
//! `p: Numeric` (dead at every float) all checked at score 1.
//!
//! Every fixture is asserted on both checker ingresses (chelis#1107). Each
//! rejection must name the binder, its family, the member at which the
//! pattern fails, and a repair; the repair's spelling is asserted here, and
//! `crates/chelis-cli/tests/issue_2442_binder_literal_pattern.rs` runs it.
//!
//! What is claimed is the forms below: a scrutinee that an authored binder of
//! the enclosing declaration resolves to when the pattern is checked, directly,
//! through a block alias, and nested in tuple and constructor patterns. A
//! flexible inference variable is not a binder; its separate declaration
//! obligation is covered by issue #2448.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("surf must parse: {source}\n{e:?}"));
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn expanded(source: &str) -> Vec<Expr> {
    expand_program(&desugared(source), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

/// One diagnostic as `[kind] message` plus its suggestions, so the ingress
/// comparison covers the repair text too.
fn rendered(errors: &[CheckError]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = errors
        .iter()
        .map(|e| {
            (
                format!("[{:?}] {}", e.kind, e.message),
                e.suggestions.clone(),
            )
        })
        .collect();
    out.sort();
    out
}

fn agreed_diagnostics(source: &str) -> Vec<(String, Vec<String>)> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

/// The one [04-PAT-1] rejection for `source`, as `(message, repair)`.
fn sole_pattern_rejection(source: &str) -> (String, String) {
    let diagnostics = agreed_diagnostics(source);
    let [(message, suggestions)] = diagnostics.as_slice() else {
        panic!("expected exactly one diagnostic for:\n{source}\ngot {diagnostics:?}");
    };
    assert!(
        message.starts_with("[TypeMismatch] ") && message.contains("[04-PAT-1]"),
        "a literal-pattern rejection is a TypeMismatch citing [04-PAT-1], got {message}"
    );
    let [repair] = suggestions.as_slice() else {
        panic!("the rejection must carry exactly one repair, got {suggestions:?}");
    };
    (message.clone(), repair.clone())
}

fn accepts(source: &str) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics.is_empty(),
        "must type-check:\n{source}\ngot {diagnostics:?}"
    );
}

/// `def f<binders>(x: p) -> i32` with one literal arm and a wildcard.
fn binder_program(binders: &str, pattern: &str) -> String {
    format!(
        "def f{binders}(x: p) -> i32 =\n  match x with {{\n    | {pattern} => 1\n    | _ => 0\n  }}\n"
    )
}

/// `(binder list, pattern, member the rejection names, repair fragment)`.
/// The repair fragments are the spellings `binder_pattern_repair` owes for each
/// branch of its rule; the CLI suite runs them.
const BOUNDED_REJECTIONS: &[(&str, &str, &str, &str)] = &[
    // The issue's three programs.
    (
        "[p: Int]",
        "300",
        "`i8`",
        "`if eq(cast(x, i64), 300i64) then",
    ),
    (
        "[p: Float]",
        "70000.0",
        "`f16`",
        "`if eq(cast(x, f64), 70000.0f64) then",
    ),
    ("[p: Numeric]", "0", "`f32`", "`if eq(x, cast(0, p)) then"),
    // A family that admits the same value in its own kind gets that literal.
    (
        "[p: Float]",
        "0",
        "`f32`",
        "Write the literal as a float, `0.0`,",
    ),
    (
        "[p: Int]",
        "1.0",
        "`i8`",
        "Write the literal as an integer, `1`,",
    ),
    // A value every member holds exactly, under Numeric, whatever its kind.
    (
        "[p: Numeric]",
        "-1.0",
        "`i8`",
        "`if eq(x, cast(-1, p)) then",
    ),
    // A value some member cannot hold exactly widens instead. A large float
    // is spelled in exponent form, as [04-LIT-2]'s diagnostics spell it.
    (
        "[p: Float]",
        "3.4e38",
        "`f16`",
        "`if eq(cast(x, f64), 3.4e38f64) then",
    ),
    (
        "[p: Float]",
        "257",
        "`f32`",
        "`if eq(cast(x, f64), 257.0f64) then",
    ),
    (
        "[p: Int]",
        "300.0",
        "`i8`",
        "`if eq(cast(x, i64), 300i64) then",
    ),
    (
        "[p: Numeric]",
        "300",
        "`i8`",
        "`if eq(cast(x, f64), 300.0f64) then",
    ),
    (
        "[p: Numeric]",
        "0.5",
        "`i8`",
        "`if eq(cast(x, f64), 0.5f64) then",
    ),
    // No member holds the value: the arm matches nothing anywhere.
    ("[p: Int]", "0.5", "`i8`", "delete it"),
    ("[p: Numeric]", "true", "`i8`", "delete it"),
    ("[p: Float]", "\"a\"", "`f32`", "delete it"),
    // At `f64`, an `i64` near this value rounds onto it, so no trap-free
    // comparison is exact at every member of Numeric.
    (
        "[p: Numeric]",
        "9007199254740993",
        "`i8`",
        "Declare `p` with the family",
    ),
];

/// REGRESSION TEST. Every bounded row scored 1 before the fix. Each now names
/// the binder, its family, the first member that refuses the pattern, and a
/// repair spelling.
#[test]
fn a_bounded_binder_decides_the_pattern_at_every_member() {
    for (binders, pattern, member, repair_fragment) in BOUNDED_REJECTIONS {
        let (message, repair) = sole_pattern_rejection(&binder_program(binders, pattern));
        let family = binders.trim_start_matches("[p: ").trim_end_matches(']');
        assert!(
            message.contains("scrutinee of type `p`")
                && message.contains(&format!("`p: {family}`"))
                && message.contains(&format!("at {member}"))
                && message.contains("[04-INF-6]"),
            "`| {pattern} =>` under `{binders}` must name the binder, `{family}`, and \
             {member}, got {message}"
        );
        assert!(
            repair.contains(repair_fragment),
            "`| {pattern} =>` under `{binders}`: expected the repair to contain \
             {repair_fragment}, got {repair}"
        );
        // A match guard is ignored at run time today (chelis#2445), so a
        // guard repair would check clean and answer wrongly.
        let spells_a_guard = repair
            .split('`')
            .skip(1)
            .step_by(2)
            .any(|span| span.starts_with("| ") && span.contains(" if "));
        assert!(
            !spells_a_guard,
            "`| {pattern} =>` under `{binders}`: the repair must not be a guard, got {repair}"
        );
    }
}

/// REGRESSION TEST. An unbounded binder admits non-primitive types, against
/// which no literal pattern is admissible ([04-DTYPE-2], [04-PAT-1]).
#[test]
fn an_unbounded_binder_refuses_every_literal_pattern() {
    let rows: &[(&str, &str)] = &[
        ("1", "Declare `p: Int`"),
        ("1.5", "Declare `p: Float`"),
        (
            "true",
            "No dtype family contains `bool`; write that type in place of `p`",
        ),
        (
            "\"a\"",
            "No dtype family contains `string`; write that type in place of `p`",
        ),
    ];
    for (pattern, repair_fragment) in rows {
        let (message, repair) = sole_pattern_rejection(&binder_program("[p]", pattern));
        assert!(
            message.contains("`p` declares no dtype-family bound")
                && message.contains("non-primitive"),
            "`| {pattern} =>` under `[p]`: got {message}"
        );
        assert!(
            repair.contains(repair_fragment),
            "`| {pattern} =>` under `[p]`: expected {repair_fragment}, got {repair}"
        );
    }
}

/// An unbounded binder whose pattern the declared family would still refuse
/// gets both repairs: the bound, then that family's spelling.
#[test]
fn an_unbounded_repair_chains_the_family_repair_when_the_family_refuses_too() {
    let (_, repair) = sole_pattern_rejection(&binder_program("[p]", "300"));
    assert!(
        repair.contains("Declare `p: Int`") && repair.contains("`if eq(cast(x, i64), 300i64) then"),
        "got {repair}"
    );
}

/// REGRESSION TEST. The binder is recognized through the variable it
/// currently resolves to, so a block alias, a tuple component, a constructor
/// argument, and a standalone signature's binder are each decided.
#[test]
fn the_binder_is_decided_through_aliases_nesting_and_signatures() {
    let programs = [
        "def f[p: Int](x: p) -> i32 = {\n  y = x\n  match y with {\n    | 300 => 1\n    | _ => 0\n  }\n}\n",
        "def f[p: Int](x: p, n: i32) -> i32 =\n  match (x, n) with {\n    | (300, _) => 1\n    | _ => 0\n  }\n",
        "type Tag[a] =\n  | Tag(a)\n\ndef f[p: Int](t: Tag[p]) -> i32 =\n  match t with {\n    | Tag(300) => 1\n    | _ => 0\n  }\n",
        "sig f[p: Int]: p -> i32\ndef f(x) =\n  match x with {\n    | 300 => 1\n    | _ => 0\n  }\n",
    ];
    for program in programs {
        let (message, _) = sole_pattern_rejection(program);
        assert!(
            message.contains("`p: Int`") && message.contains("at `i8`"),
            "got {message} for:\n{program}"
        );
    }
}

/// The repair names the matched value only where an expression names it.
///
/// When the literal is the arm's whole pattern and the scrutinee is a
/// variable, the comparison runs in an `if` ahead of the match over that
/// variable, by its own name. Anywhere else no expression names the value: a
/// nested position, an as-pattern, or a scrutinee that is a call. There the
/// repair binds the position to a fresh variable and compares that in the
/// arm's body, so it never spells a comparison over the wrong value.
#[test]
fn the_repair_names_the_matched_value_only_where_an_expression_names_it() {
    // A scrutinee variable not named `x` is spelled by its own name.
    let (_, repair) = sole_pattern_rejection(
        "def f[p: Int](n: p) -> i32 =\n  match n with {\n    | 300 => 1\n    | _ => 0\n  }\n",
    );
    assert!(
        repair.contains(
            "`if eq(cast(n, i64), 300i64) then <this arm's body> else match n with \
             { <the other arms> }`"
        ),
        "got {repair}"
    );

    let elsewhere: &[(&str, &str)] = &[
        (
            "type Tag[a] =\n  | Tag(a)\n\ndef f[p: Numeric](o: Tag[p]) -> i32 =\n  match o with {\n    | Tag(0) => 1\n    | _ => 0\n  }\n",
            "`if eq(v, cast(0, p)) then",
        ),
        (
            "def f[p: Numeric](o: Option[p]) -> i32 =\n  match o with {\n    | Some(0) => 1\n    | _ => 0\n  }\n",
            "`if eq(v, cast(0, p)) then",
        ),
        (
            "def f[p: Int](x: p, n: i32) -> i32 =\n  match (x, n) with {\n    | (300, _) => 1\n    | _ => 0\n  }\n",
            "`if eq(cast(v, i64), 300i64) then",
        ),
        (
            "type Tag[a] =\n  | Tag(a)\n\ndef f[p: Int](t: Tag[p]) -> i32 =\n  match t with {\n    | Tag(300) => 1\n    | _ => 0\n  }\n",
            "`if eq(cast(v, i64), 300i64) then",
        ),
        (
            "def f[p: Int](x: p) -> i32 =\n  match add(x, x) with {\n    | 300 => 1\n    | _ => 0\n  }\n",
            "`if eq(cast(v, i64), 300i64) then",
        ),
        (
            "def f[p: Int](x: p) -> i32 =\n  match x with {\n    | q @ 300 => 1\n    | _ => 0\n  }\n",
            "`if eq(cast(v, i64), 300i64) then",
        ),
    ];
    for (program, comparison) in elsewhere {
        let (_, repair) = sole_pattern_rejection(program);
        assert!(
            repair.contains("put a fresh variable `v` where the literal is")
                && repair.contains(comparison)
                && repair.contains("else <what the remaining arms give>`")
                && !repair.contains(" with {"),
            "the repair must bind the literal's position rather than name a scrutinee, \
             got {repair} for:\n{program}"
        );
    }
}

/// Legal controls: a pattern every member admits checks clean, at the
/// boundaries of the narrowest member.
#[test]
fn a_pattern_every_member_admits_checks_clean() {
    for (binders, pattern) in [
        ("[p: Int]", "100"),
        ("[p: Int]", "-128"),
        ("[p: Int]", "127"),
        ("[p: Float]", "0.5"),
        ("[p: Float]", "65504.0"),
        ("[p: Float]", "-65504.0"),
    ] {
        accepts(&binder_program(binders, pattern));
    }
}

/// Legal control: the generic comparison the repairs are built from.
#[test]
fn the_generic_comparison_stays_legal() {
    accepts("def is_zero[p: Numeric](x: p) -> bool = if eq(x, cast(0, p)) then true else false\n");
}

/// A concrete scrutinee keeps its existing decision and wording: an in-range
/// pattern is admitted and an out-of-range one is refused at the scrutinee's
/// own width, naming no binder.
#[test]
fn a_concrete_scrutinee_is_unchanged() {
    accepts("def f(x: i8) -> i32 =\n  match x with {\n    | 100 => 1\n    | _ => 0\n  }\n");
    let (message, repair) = sole_pattern_rejection(
        "def f(x: i8) -> i32 =\n  match x with {\n    | 300 => 1\n    | _ => 0\n  }\n",
    );
    assert!(
        message.contains("integer literal pattern `300` is outside the `i8` range [-128, 127]")
            && !message.contains("[04-INF-6]"),
        "got {message}"
    );
    assert_eq!(
        repair,
        "Use a value the scrutinee's `i8` width can hold, or widen the scrutinee"
    );
}

/// A flexible inference variable is not a binder. Its pattern is decided
/// after the local lambda's first application under [04-INF-1].
#[test]
fn a_flexible_scrutinee_is_decided_by_its_first_application() {
    accepts(
        "def pick(x: i32) -> i32 = {\n  k = fn (y) -> match y with {\n    | 300 => 1\n    | _ => 0\n  }\n  k(x)\n}\n",
    );
}
