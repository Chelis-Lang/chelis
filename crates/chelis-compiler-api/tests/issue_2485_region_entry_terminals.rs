//! chelis#2485 and chelis#2458: compiled C must emit every terminal the
//! verified ownership schedule places at the entry of a match arm or a loop
//! body, even when that arm or body branches.
//!
//! The schedule releases an owner right after its last use. A payload nothing
//! reads, and a scrutinee whose last use is the payload extraction, are
//! therefore released in the arm's entry block, and a loop item nothing reads
//! in the body's entry block. The C emitter laid out a branching arm or body
//! from its completion block alone: a match arm never emitted those releases,
//! and a loop emitted them after the loop, where the item is out of C scope.
//!
//! PR #2454's match planner made the first case reachable from an ordinary
//! `None`-first `Option` match: the `None` test extracts a payload that only
//! the rest of the match follows, and the rest of the match branches.
//!
//! Emitting those releases also exposed a naming defect from the same planner.
//! A match draws its scrutinee and payload names after its arms have lowered,
//! and a match nested in an arm drew the same names from its own supply. In C
//! the nested binder then shadows the enclosing payload, so a release of the
//! enclosing payload inside the nested arm named the nested value: a type
//! error in C, or, when the types agree, a release of the wrong object.
//!
//! Oracle: the compiled program runs against the `ownership-ledger` runtime,
//! every allocation must be finalized with no live owner left, and stdout must
//! equal what `chelis eval` prints for the same program.

mod ownership_support;

fn assert_balanced(name: &str, source: &str, expected: &str) {
    let generated = ownership_support::emit(source, name);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, expected, "{name}");
}

/// REGRESSION TEST. The `option-string` fixture of the compiled value
/// ownership oracle, with a parameter so the only root is `out`. On
/// `82f2a62d5` the ledger ended with 1 live `String` owner: the payload the
/// `None` test extracted was never released.
#[test]
fn a_none_first_option_match_releases_the_payload_its_none_test_extracts() {
    assert_balanced(
        "option_none_first",
        "def option_length(text: string) -> i64 = {\n  \
         value: Option[string] = Some(string_concat(text, \"def\"))\n  \
         match value with {\n    | None => 0i64\n    | Some(inner) => string_len(inner)\n  }\n}\n\
         out = option_length(\"abc\")\n",
        "out = 6\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the ledger ended with 2 live `String`
/// owners: `Some(_)` extracts a payload nothing reads, and the arm branches.
#[test]
fn a_wildcard_some_arm_with_a_branching_body_releases_its_payload() {
    assert_balanced(
        "option_wild_branching",
        "def classify(flag: bool) -> i64 = {\n  \
         value: Option[string] = Some(string_concat(\"abc\", \"def\"))\n  \
         match value with {\n    | Some(_) => if flag then 1i64 else 2i64\n    | None => 0i64\n  }\n}\n\
         a = classify(true)\n\
         b = classify(false)\n",
        "a = 1\nb = 2\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the ledger ended with 2 live owners.
#[test]
fn a_nested_option_match_releases_every_payload() {
    assert_balanced(
        "option_nested",
        "def option_length(text: string) -> i64 = {\n  \
         value: Option[Option[string]] = Some(Some(string_concat(text, \"def\")))\n  \
         match value with {\n    | None => 0i64\n    | Some(None) => 1i64\n    \
         | Some(Some(inner)) => string_len(inner)\n  }\n}\n\
         out = option_length(\"abc\")\n",
        "out = 6\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the ledger ended with 2 live `String`
/// owners: the guarded arm follows a `None` arm, so it sits in the branching
/// rest of the match.
#[test]
fn a_guarded_some_arm_after_a_none_arm_releases_the_payload() {
    assert_balanced(
        "option_guard_none_first",
        "def classify(text: string) -> i64 = {\n  \
         value: Option[string] = Some(string_concat(text, \"def\"))\n  \
         match value with {\n    | None => 0i64\n    \
         | Some(s) if gt(string_len(s), 4i64) => string_len(s)\n    | Some(_) => 1i64\n  }\n}\n\
         a = classify(\"abc\")\n\
         b = classify(\"\")\n",
        "a = 6\nb = 1\n",
    );
}

/// REGRESSION TEST, chelis#2458's reproduction. On `82f2a62d5` the ledger
/// ended with 20 live owners, two per call: the scrutinee's last use is the
/// field extraction, and the arm branches.
#[test]
fn an_adt_arm_with_a_branching_body_releases_its_scrutinee() {
    assert_balanced(
        "adt_branching_arm",
        "type Box =\n  | Box(i64)\n  | NoBox\n\
         def f(b: Box) -> i32 =\n  match b with {\n    \
         | Box(n) => if gt(n, 2i64) then 1 else 2\n    | _ => 0\n  }\n\
         def repeat(k: i64, acc: i32) -> i32 =\n  \
         if eq(k, 0i64) then acc else repeat(sub(k, 1i64), add(acc, f(Box(k))))\n\
         a = repeat(10i64, 0)\n",
        "a = 12\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the ledger ended with 4 live owners: the
/// string field nothing reads, and the scrutinee.
#[test]
fn an_unused_adt_string_field_is_released_when_the_arm_branches() {
    assert_balanced(
        "adt_unused_string_field",
        "type Named =\n  | Named(string, i64)\n  | Anon\n\
         def classify(n: Named) -> i64 =\n  match n with {\n    \
         | Named(s, k) => if gt(k, 2i64) then 1i64 else 2i64\n    | Anon => 0i64\n  }\n\
         a = classify(Named(string_concat(\"abc\", \"def\"), 3i64))\n\
         b = classify(Anon)\n",
        "a = 1\nb = 0\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the ledger ended with 4 live owners: the
/// selected pass extracts the `Option` field, and its nested `Some` branches.
#[test]
fn a_guarded_nested_constructor_arm_releases_its_scrutinee() {
    assert_balanced(
        "adt_guarded_nested",
        "type Holder =\n  | Holder(Option[string])\n  | Empty\n\
         def classify(h: Holder) -> i64 =\n  match h with {\n    \
         | Holder(Some(s)) if gt(string_len(s), 2i64) => string_len(s)\n    | _ => 0i64\n  }\n\
         a = classify(Holder(Some(string_concat(\"abc\", \"def\"))))\n\
         b = classify(Holder(Some(\"x\")))\n\
         c = classify(Holder(None))\n\
         d = classify(Empty)\n",
        "a = 6\nb = 0\nc = 0\nd = 0\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the generated C did not compile: the
/// release of the unread item was emitted after the loop, where the item is
/// out of scope.
#[test]
fn a_branching_map_body_releases_an_unread_item_inside_the_loop() {
    assert_balanced(
        "map_unread_item",
        "def classify(flag: bool) -> List[i64] =\n  \
         map(fn (s: string) -> if flag then 1i64 else 2i64, [string_concat(\"abc\", \"def\"), \"x\"])\n\
         a = classify(true)\n",
        "a = [1, 1]\n",
    );
}

/// REGRESSION TEST. On `82f2a62d5` the generated C did not compile, as for
/// `map`.
#[test]
fn a_branching_fold_body_releases_an_unread_item_inside_the_loop() {
    assert_balanced(
        "fold_unread_item",
        "def tally(flag: bool) -> i64 =\n  \
         fold(fn (acc: i64, s: string) -> if flag then add(acc, 1i64) else acc, 0i64, \
         [string_concat(\"abc\", \"def\"), \"x\"])\n\
         a = tally(true)\n\
         b = tally(false)\n",
        "a = 2\nb = 0\n",
    );
}

/// REGRESSION TEST, the oracle's `option-nested-string` shape. On `7c7b58c7f`
/// the generated C did not compile: the release of the outer payload, inside
/// the inner match's arm, named the inner match's `string` payload.
#[test]
fn a_match_nested_in_an_arm_releases_the_enclosing_payload() {
    assert_balanced(
        "nested_on_binding",
        "def option_length(text: string) -> i64 = {\n  \
         value: Option[Option[string]] = Some(Some(string_concat(text, \"def\")))\n  \
         match value with {\n    | None => 0i64\n    \
         | Some(inner) => match inner with {\n      | None => 0i64\n      \
         | Some(s) => string_len(s)\n    }\n  }\n}\n\
         out = option_length(\"abc\")\n",
        "out = 6\n",
    );
}

/// REGRESSION TEST. When the enclosing and nested payloads have the same type
/// the shadowed release compiles and names the wrong object: on `7c7b58c7f`
/// the ledger ended with 3 live owners, and with only the region-entry fix
/// applied the run aborted with a heap kind mismatch.
#[test]
fn a_nested_option_match_releases_its_own_payload_not_the_enclosing_one() {
    assert_balanced(
        "nested_same_type_option",
        "def pick(a: Option[string], b: Option[string]) -> i64 =\n  \
         match a with {\n    | None => 0i64\n    \
         | Some(x) => match b with {\n      | None => string_len(x)\n      \
         | Some(y) => add(string_len(x), string_len(y))\n    }\n  }\n\
         a = pick(Some(string_concat(\"abc\", \"def\")), Some(string_concat(\"gh\", \"i\")))\n\
         b = pick(Some(string_concat(\"abc\", \"def\")), None)\n\
         c = pick(None, Some(\"z\"))\n",
        "a = 9\nb = 6\nc = 0\n",
    );
}

/// REGRESSION TEST. On `7c7b58c7f` the run aborted with a heap kind mismatch:
/// a release of the outer field named the inner match's field.
#[test]
fn a_nested_adt_match_releases_its_own_field_not_the_enclosing_one() {
    assert_balanced(
        "nested_same_type_adt",
        "type Named =\n  | Named(string)\n  | Anon\n\
         def pick(a: Named, b: Named) -> i64 =\n  match a with {\n    | Anon => 0i64\n    \
         | Named(x) => match b with {\n      | Anon => string_len(x)\n      \
         | Named(y) => add(string_len(x), string_len(y))\n    }\n  }\n\
         a = pick(Named(string_concat(\"abc\", \"def\")), Named(string_concat(\"gh\", \"i\")))\n\
         b = pick(Named(string_concat(\"abc\", \"def\")), Anon)\n",
        "a = 9\nb = 6\n",
    );
}
