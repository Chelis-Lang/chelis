// chelis#2443: a dtype binder could be bounded only by `Float`, `Int` or
// `Numeric`, so a generic body could not hold a constant that one narrow
// member cannot represent, and the checker's own diagnostic named a repair the
// language could not express. `[04-INF-6]` makes a bounded binder denote every
// admissible instantiation and `[04-LIT-2]` makes a literal that rounds to
// infinity at its dtype a type error, so `cast(57568490574.0, p)` under
// `p: Float` is rejected because `Float` admits `f16` -- and "narrow the
// declaration's dtype domain" had no spelling.
//
// spec/04-type-system.md §5.9 now admits two bound forms: a family name, whose
// membership follows §1.1's active set, and an explicit dtype set `{f32, f64}`,
// which admits exactly its listed members. spec/02-surf-syntax.md §P4c carries
// the surface grammar and the formatter's printing rule.
//
// The two forms are deliberately NOT interchangeable, and §5.9 says so: a
// family widens when §1.1 activates a dtype, an explicit set never does. The
// tests below therefore pin the set form's admission, its rejection, the
// intersection rules of [04-DTYPE-2], and each declaration error -- with the
// positive control beside every rejection, because a bound that rejects
// everything would pass a rejection-only suite.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write(source: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    fs::write(&path, source).expect("write probe");
    (dir, path)
}

fn check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check output for {} is not JSON ({error}): stdout={} stderr={}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

fn score(report: &Value) -> f64 {
    report["score"].as_f64().expect("score")
}

fn messages(report: &Value) -> String {
    report["errors"]
        .as_array()
        .map(|errors| {
            errors
                .iter()
                .map(|error| error["message"].as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Every diagnostic's message AND its suggestions. A repair hint lives in
/// `suggestions`, so a helper that reads only `message` silently cannot see
/// the text a repair test is about.
fn messages_and_hints(report: &Value) -> String {
    report["errors"]
        .as_array()
        .map(|errors| {
            errors
                .iter()
                .flat_map(|error| {
                    std::iter::once(error["message"].as_str().unwrap_or_default().to_string())
                        .chain(
                            error["suggestions"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .map(|hint| hint.as_str().unwrap_or_default().to_string()),
                        )
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn assert_clean(source: &str, what: &str) {
    let (_dir, path) = write(source);
    let report = check(&path);
    assert_eq!(
        score(&report),
        1.0,
        "{what} must check clean, got score {} with:\n{}",
        score(&report),
        messages(&report),
    );
}

fn assert_rejected(source: &str, needles: &[&str], what: &str) {
    let (_dir, path) = write(source);
    let report = check(&path);
    assert!(
        score(&report) < 1.0,
        "{what} must be rejected, but scored 1.0",
    );
    let text = messages(&report);
    for needle in needles {
        assert!(
            text.contains(needle),
            "{what} diagnostic must name {needle:?}, got:\n{text}",
        );
    }
}

// 1. The motivating case. A literal finite at f32 and f64 but not f16 is
//    admitted under the set that excludes f16, and still rejected under the
//    family that admits it. This pair is the whole point of the issue.

#[test]
fn wide_literal_is_admitted_under_a_set_excluding_f16() {
    assert_clean(
        "module Probe\n\
         def widen[p: {f32, f64}](x: p) -> p = add(x, cast(57568490574.0, p))\n",
        "a literal finite at every member of `{f32, f64}`",
    );
}

#[test]
fn wide_literal_is_still_rejected_under_the_float_family() {
    assert_rejected(
        "module Probe\n\
         def widen[p: Float](x: p) -> p = add(x, cast(57568490574.0, p))\n",
        &["57568490574.0", "f16"],
        "a literal that rounds to infinity at f16 under `Float`",
    );
}

#[test]
fn a_literal_outside_every_set_member_is_rejected() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, f64}](x: p) -> p = add(x, cast(1e40, p))\n",
        &["1e40"],
        "a literal that rounds to infinity at f32 under `{f32, f64}`",
    );
}

#[test]
fn a_literal_only_f64_holds_is_admitted_under_the_f64_singleton() {
    assert_clean(
        "module Probe\n\
         def widen[p: {f64}](x: p) -> p = add(x, cast(1e40, p))\n",
        "a literal finite at f64 under the one-member set `{f64}`",
    );
}

// 2. The integer rule of #1545 follows the same shape.

#[test]
fn integer_literal_admitted_under_a_narrow_int_set() {
    assert_clean(
        "module Probe\n\
         def bump[p: {i32, i64}](x: p) -> p = add(x, cast(1000000, p))\n",
        "an integer literal every member of `{i32, i64}` holds",
    );
}

#[test]
fn integer_literal_rejected_under_the_int_family() {
    assert_rejected(
        "module Probe\n\
         def bump[p: Int](x: p) -> p = add(x, cast(1000000, p))\n",
        &["1000000"],
        "an integer literal i8 cannot hold under `Int`",
    );
}

// 3. Instantiation. A set admits its members and refuses everything else,
//    with `PrecisionMismatch` naming the required bound.

#[test]
fn instantiation_at_a_set_member_is_accepted() {
    assert_clean(
        "module Probe\n\
         def widen[p: {f32, f64}](x: p) -> p = x\n\
         def single(v: f32) -> f32 = widen(v)\n\
         def double(v: f64) -> f64 = widen(v)\n",
        "instantiation at each member of `{f32, f64}`",
    );
}

#[test]
fn instantiation_outside_the_set_is_a_precision_mismatch() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, f64}](x: p) -> p = x\n\
         def use_f16(v: f16) -> f16 = widen(v)\n",
        &["f16"],
        "instantiation at a dtype the set excludes",
    );
}

#[test]
fn instantiation_at_a_non_numeric_is_a_precision_mismatch() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, f64}](x: p) -> p = x\n\
         def use_bool(v: bool) -> bool = widen(v)\n",
        &["bool"],
        "instantiation at `bool`, which belongs to no bound",
    );
}

// 4. Intersection, per [04-DTYPE-2]: family with family, family with set, and
//    set with set. An empty intersection names both bounds.

#[test]
fn a_float_declaration_cannot_call_a_narrower_set_callee() {
    // [04-INF-9]: the body requires `{f32, f64}` while the signature promises
    // every `Float`, including f16, so the declaration is too WIDE for its
    // body. The intersection is `{f32, f64}`; the violation is that the
    // declared bound is not at least as narrow as what the body needs.
    assert_rejected(
        "module Probe\n\
         def narrow[p: {f32, f64}](x: p) -> p = x\n\
         def wide[p: Float](x: p) -> p = narrow(x)\n",
        &["`p`", "{f32, f64}"],
        "a `Float` declaration whose body requires `{f32, f64}`",
    );
}

#[test]
fn a_set_declaration_may_call_a_wider_family_callee() {
    // The sound direction: a `{f32, f64}` declaration satisfies a `Float`
    // body requirement without equalling it, which is why [04-INF-9] is a
    // subset relation rather than equality.
    assert_clean(
        "module Probe\n\
         def anyfloat[p: Float](x: p) -> p = x\n\
         def narrow[p: {f32, f64}](x: p) -> p = anyfloat(x)\n",
        "a `{f32, f64}` declaration calling a `Float` callee",
    );
}

#[test]
fn a_set_intersected_with_a_disjoint_family_is_empty() {
    assert_rejected(
        "module Probe\n\
         def narrow[p: {f32, f64}](x: p) -> p = x\n\
         def ints[p: Int](x: p) -> p = narrow(x)\n",
        &["Int"],
        "`Int` intersected with `{f32, f64}` is empty",
    );
}

#[test]
fn two_disjoint_sets_intersect_to_empty() {
    assert_rejected(
        "module Probe\n\
         def lhs[p: {f32}](x: p) -> p = x\n\
         def rhs[p: {f64}](x: p) -> p = lhs(x)\n",
        &["f32", "f64"],
        "`{f32}` intersected with `{f64}` is empty",
    );
}

#[test]
fn two_overlapping_sets_intersect_to_their_common_members() {
    assert_clean(
        "module Probe\n\
         def lhs[p: {f32, f64, bf16}](x: p) -> p = x\n\
         def rhs[p: {f32, f64}](x: p) -> p = lhs(x)\n",
        "`{f32, f64, bf16}` intersected with `{f32, f64}`",
    );
}

// 5. Declaration errors, each named by §5.9 / [04-DTYPE-2].

#[test]
fn an_empty_set_is_a_declaration_error() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {}](x: p) -> p = x\n",
        &["non-empty dtype set"],
        "an empty dtype set",
    );
}

#[test]
fn a_repeated_member_is_a_declaration_error() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, f32}](x: p) -> p = x\n",
        &["f32"],
        "a dtype set repeating a member",
    );
}

#[test]
fn a_non_primitive_member_is_a_declaration_error() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, Widget}](x: p) -> p = x\n",
        &["Widget"],
        "a dtype set naming a non-primitive",
    );
}

#[test]
fn bool_and_string_are_not_admissible_members() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, bool}](x: p) -> p = x\n",
        &["bool"],
        "a dtype set naming `bool`, which belongs to no bound",
    );
}

#[test]
fn a_bare_dtype_without_braces_is_a_syntax_error() {
    assert_rejected(
        "module Probe\n\
         def widen[p: f64](x: p) -> p = x\n",
        &["cannot be parsed as Surf"],
        "a bare dtype in the bound position, per §P4c",
    );
}

#[test]
fn a_set_bound_in_a_dimension_slot_is_an_error() {
    assert_rejected(
        "module Probe\n\
         def widen[n: {f32, f64}](x: tensor[n, f32]) -> tensor[n, f32] = x\n",
        &["`n`", "cannot be used as a dimension slot"],
        "a bounded binder used as a dimension",
    );
}

#[test]
fn a_set_bound_that_does_not_occur_in_the_type_is_an_error() {
    assert_rejected(
        "module Probe\n\
         def widen[p: {f32, f64}](x: f32) -> f32 = x\n",
        &["`p`", "does not occur in defsig"],
        "a bounded binder absent from the declared type",
    );
}

// 6. The bound is part of the scheme, so it survives the routes
//    [04-DTYPE-2] enumerates. Each of these must keep rejecting the
//    excluded dtype rather than losing the bound along the way.

#[test]
fn a_set_bound_survives_an_alias_and_a_recursive_call() {
    assert_rejected(
        "module Probe\n\
         def go[p: {f32, f64}](x: p, n: i64) -> p = if eq(n, cast(0, i64)) then x else go(x, sub(n, cast(1, i64)))\n\
         def use_f16(v: f16) -> f16 = go(v, cast(1, i64))\n",
        &["f16"],
        "a set bound propagated through a recursive call",
    );
}

#[test]
fn a_set_bound_survives_a_wrapper() {
    assert_rejected(
        "module Probe\n\
         def inner[p: {f32, f64}](x: p) -> p = x\n\
         def outer[q: Float](x: q) -> q = inner(x)\n\
         def use_f16(v: f16) -> f16 = outer(v)\n",
        &["f16"],
        "a set bound propagated out through a `Float` wrapper",
    );
}

// 7. The formatter's §P4c rule: a set prints with §1.1 declaration order,
//    one space after the colon and after each comma, and no space inside the
//    braces. Authored order is not preserved for members.

#[test]
fn the_formatter_normalizes_set_member_order_and_spacing() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    fs::write(
        &path,
        "module Probe\ndef widen[p: {f64,f32}](x: p) -> p = x\n",
    )
    .expect("write probe");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
    let formatted = fs::read_to_string(&path).expect("read formatted");
    assert!(
        formatted.contains("[p: {f32, f64}]"),
        "the formatter must print `{{f32, f64}}` in §1.1 order, got:\n{formatted}",
    );
}

#[test]
fn formatting_a_set_bound_is_idempotent() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    let canonical = "module Probe\ndef widen[p: {f32, f64}](x: p) -> p = x\n";
    fs::write(&path, canonical).expect("write probe");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&path).expect("read formatted"),
        canonical,
        "canonical set-bound output must be a fixed point of the formatter",
    );
}

// 8. Controls: the family forms must keep working exactly as before, so the
//    amendment is additive rather than a change to existing declarations.

#[test]
fn the_family_forms_are_unchanged() {
    assert_clean(
        "module Probe\n\
         def scale[p: Float](x: p) -> p = add(x, cast(1.0, p))\n\
         def bump[p: Int](x: p) -> p = add(x, cast(1, p))\n\
         def any[p: Numeric](x: p) -> p = x\n",
        "the three family bounds",
    );
}

#[test]
fn an_unbounded_binder_is_still_not_a_dtype_binder() {
    assert_clean(
        "module Probe\n\
         def identity[a](x: a) -> a = x\n\
         def use_bool(v: bool) -> bool = identity(v)\n",
        "an unbounded binder admitting a non-dtype",
    );
}

// 9. Red-team round 1 regressions. `ActiveSet` was added as a seventh
//    `TypeVarRestriction` variant while several matches stayed keyed on the
//    three families, so a set fell into `_` arms. The root cause was an
//    infallible `family_name()` returning `""` for a set; it is now
//    `Option`, which made the compiler enumerate all fifteen call sites.

/// A set restricts WHICH dtypes are admissible, never which programs
/// type-check. The `to_tensor` element-peel gate listed the three families
/// explicitly, so a set-bounded `to_tensor` stayed suspended past the
/// declaration boundary and was rejected where its equivalent family was
/// accepted. The set here enumerates exactly `Float`'s current members.
#[test]
fn a_set_bound_accepts_every_program_its_equivalent_family_accepts() {
    for bound in ["Float", "Numeric", "{f32, f64}", "{f32, f64, bf16, f16}"] {
        assert_clean(
            &format!(
                "module Probe\n\
                 def f[p: {bound}](xs: List[p]) -> tensor[2, p] = reshape(to_tensor(xs), [2i64])\n"
            ),
            &format!("`to_tensor` + `reshape` under `[p: {bound}]`"),
        );
    }
}

/// Every diagnostic naming a bound must name it. An `ActiveSet(_) => ""` arm
/// on `family_name()` let a set reach at least seven diagnostics as an empty
/// name, while the family control rendered correctly. This mirrors the family
/// invariant at `chelis-types/tests/unresolved_operand_matrix.rs`.
#[test]
fn a_set_bound_is_named_in_the_rejection_not_left_empty() {
    let (_dir, path) = write(
        "module Probe\n\
         def f[n, p: {f32, i32}](x: tensor[n, p]) -> tensor[n, p] = exp(x)\n",
    );
    let report = check(&path);
    let text = messages(&report);
    assert!(score(&report) < 1.0, "a mixed set must not admit `exp`");
    assert!(
        text.contains("dtype set `{f32, i32}`"),
        "the rejection must name the set, not an empty family name, got:\n{text}",
    );
    assert!(
        !text.contains("dtype family ``") && !text.contains("`p: `"),
        "no diagnostic may render an empty bound name, got:\n{text}",
    );
}

/// The literal-pattern repair chose its widest member and its f64-collision
/// guard by matching the three family variants, so an all-integer SET was
/// advised to compare at `f64` — advice that is false, because two distinct
/// i64 values share one f64 — and a MIXED set skipped the guard entirely.
/// Both now key on what the bound admits.
#[test]
fn the_pattern_repair_keys_on_what_the_bound_admits() {
    let colliding = "9007199254740994";
    // An all-integer bound, family or set, must advise i64.
    for bound in ["Int", "{i8, i16, i32, i64}", "{i8, i64}"] {
        let (_dir, path) = write(&format!(
            "module Probe\ndef f[p: {bound}](x: p) -> i32 =\n  match x with {{\n    | {colliding} => 1i32\n    | _ => 0i32\n  }}\n"
        ));
        let text = messages_and_hints(&check(&path));
        assert!(
            text.contains("Compare at `i64`"),
            "`[p: {bound}]` must advise i64, got:\n{text}",
        );
        assert!(
            !text.contains("Compare at `f64`"),
            "`[p: {bound}]` must never advise f64: two i64 values share one f64, so the \
             suggested comparison would match a different value. Got:\n{text}",
        );
    }
    // A bound admitting both an integer and a float cannot offer an exact
    // non-trapping comparison in f64's collision range, however it is spelled.
    for bound in ["Numeric", "{f64, i8}", "{f64, i16}"] {
        let (_dir, path) = write(&format!(
            "module Probe\ndef f[p: {bound}](x: p) -> i32 =\n  match x with {{\n    | {colliding} => 1i32\n    | _ => 0i32\n  }}\n"
        ));
        let text = messages_and_hints(&check(&path));
        assert!(
            text.contains("No comparison that cannot trap is exact"),
            "`[p: {bound}]` must refuse rather than advise a lossy comparison, got:\n{text}",
        );
    }
}
