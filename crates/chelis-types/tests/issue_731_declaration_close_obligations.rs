//! chelis#731: every operand obligation a declaration's inference leaves open
//! is decided or reported when the declaration closes, and the authored-binder
//! contract is checked after the last step that can narrow a binder.
//!
//! Each program below checked at score 1 on `main` `32e4122e4`. They reached
//! the same defect by different routes: a call suspended on an operand whose
//! type never bound was dropped at the declaration boundary when that type was
//! a flexible variable (chelis#2518) or a variable nested in an operand
//! (chelis#2523's `to_tensor` and `dict_of`); a route admitted a variable
//! operand and never revisited it (chelis#2523's dictionary operations and
//! axis slots); a tuple projection or record-field read on a variable target
//! kept no obligation at the boundary and did not keep its lambda monomorphic
//! (chelis#2523); and the authored-binder contract ran before the boundary
//! replays that narrowed a binder (chelis#2537).
//!
//! Every program is asserted on both checker ingresses (chelis#1107), and each
//! rejection has an accepted control. Gather also requires a literal axis
//! under [05-AXIS-2], so its positive control uses that static spelling.

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

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

fn agreed_diagnostics(source: &str) -> Vec<String> {
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

fn accepts(source: &str) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics.is_empty(),
        "must type-check:\n{source}\ngot {diagnostics:#?}"
    );
}

/// A rejection carrying a diagnostic that contains every fragment.
fn rejects_with(source: &str, fragments: &[&str]) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {diagnostics:#?}"
    );
}

/// REGRESSION TEST (chelis#2537). A `cast_trunc` inside a returned closure
/// narrows `p: Numeric` to `Float` only when the declaration boundary replays
/// the closure's `index`, whose operand only the declared type fixes. The
/// authored-binder contract ran before that replay and never saw the
/// narrowing; `eval` then reached chelis#2158's panic.
#[test]
fn a_binder_narrowed_by_a_boundary_replay_is_reported() {
    for element in [
        "index(xs, 0i64)",
        "(fn (pair) -> pair.0)((index(xs, 0i64), 0i32))",
    ] {
        let program = format!(
            "module P.Main\nexport (main)\n\
             def f[p: Numeric]() -> (List[tensor[2, p]]) -> tensor[2, i32] = \
             fn (xs) -> cast_trunc({element}, i32)\n\
             def main() -> tensor[2, i32] = (f())([to_tensor([7i32, -8i32])])\n"
        );
        rejects_with(
            &program,
            &[
                "[PrecisionMismatch] declared type parameter `p` of `f` requires dtype family `Float`",
                "the declared `Numeric` family",
            ],
        );
        accepts(
            &program
                .replace("p: Numeric", "p: Float")
                .replace("to_tensor([7i32, -8i32])", "to_tensor([7.5f32, -8.5f32])"),
        );
    }
}

/// REGRESSION TEST (chelis#2518). An operand whose type nothing in the
/// declaration determines was dropped at the boundary together with the check
/// waiting on it. [04-INF-1] and [04-INF-9] give it no implicit contract, so the
/// operation must hold at an arbitrary type, and these do not.
#[test]
fn an_undetermined_operand_is_decided_at_an_arbitrary_type() {
    rejects_with(
        "def size(k) -> i64 = numel(k)",
        &[
            "[TypeMismatch] `numel` admits only some operand types",
            "never determined within `size`",
            "[04-INF-9]",
        ],
    );
    accepts("def size(k: tensor[3, f32]) -> i64 = numel(k)");

    // #2518's second witness: the count is a never-applied lambda parameter
    // and the list an unbounded binder.
    rejects_with(
        "def f[a](k: a) -> i64 = {\n  g = fn (x) -> take(k, x)\n  0i64\n}\n",
        &[
            "`take` admits only some operand types",
            "never determined within `f`",
        ],
    );
    accepts("def f[a](k: List[a]) -> List[a] = (fn (x) -> take(k, x))(1i64)");

    // An operation that holds at every type stays accepted in a lambda nothing
    // applies: only an operation with a requirement is an obligation.
    accepts("def f() -> i32 = {\n  g = fn (t) -> t\n  h = fn (t) -> debug(t)\n  0i32\n}\n");
}

/// REGRESSION TEST (chelis#2523). The dictionary operations admitted a
/// variable first operand and returned; nothing revisited it, so a binder there
/// was never checked. Each now suspends and is decided at the binder's
/// instantiations.
#[test]
fn dictionary_operations_on_a_binder_are_decided_at_its_instantiations() {
    for (call, result, diagnostic) in [
        (
            "dict_get(k, k)",
            "Option[i32]",
            "dict_get expects Dict[K, V] and K, got f32",
        ),
        (
            "dict_contains(k, k)",
            "bool",
            "dict_contains expects Dict[K, V] and K, got f32",
        ),
        (
            "dict_remove(k, k)",
            "Dict[string, i32]",
            "dict_remove expects Dict[K, V] and K, got f32",
        ),
        (
            "dict_insert(k, k, k)",
            "Dict[string, i32]",
            "dict_insert expects Dict[K, V], K, and V, got f32",
        ),
    ] {
        rejects_with(
            &format!("def bad[p: Float](k: p) -> {result} = {call}"),
            &[diagnostic, "authored type binder `p`", "`p := f32`"],
        );
    }
    accepts("def ok(d: Dict[string, i32], k: string) -> Option[i32] = dict_get(d, k)");
    accepts("def ok(d: Dict[string, i32], k: string) -> bool = dict_contains(d, k)");
    accepts("def ok(d: Dict[string, i32], k: string) -> Dict[string, i32] = dict_remove(d, k)");
    accepts(
        "def ok(d: Dict[string, i32], k: string) -> Dict[string, i32] = dict_insert(d, k, 1i32)",
    );

    // A late-bound dictionary operand is decided by the route's own rule.
    rejects_with(
        "def f(k: i32) -> Option[i32] = (fn (t) -> dict_get(t, 1i64))(k)",
        &["dict_get expects Dict[K, V] and K, got i32 and i64"],
    );
    accepts("def f(d: Dict[i64, i32]) -> Option[i32] = (fn (t) -> dict_get(t, 1i64))(d)");
}

/// REGRESSION TEST (chelis#2523). `dict_of` and `to_tensor` suspended on a
/// variable inside their list operand, and the boundary looked only for an
/// operand that was itself a variable, so the element type an unbounded binder
/// denotes was never checked.
#[test]
fn a_variable_inside_a_list_operand_is_decided_at_the_boundary() {
    rejects_with(
        "def g[a](k: a) -> Dict[a, i32] = dict_of([(k, 1i32)])",
        &[
            "`dict_of` admits only some operand types",
            "`a` := an arbitrary type",
        ],
    );
    accepts("def ok(k: string) -> Dict[string, i32] = dict_of([(k, 1i32)])");
    rejects_with(
        "def f[a](xs: List[a]) -> tensor[*, a] = to_tensor(xs)",
        &[
            "`to_tensor` admits only some operand types",
            "`a` := an arbitrary type",
        ],
    );
    // A declared result fixes the element type, and pins the binder to it.
    rejects_with(
        "def f[a](xs: List[a]) -> tensor[*, f32] = to_tensor(xs)",
        &["declared type parameter `a` of `f` was narrowed to `f32`"],
    );
    accepts("def ok[p: Numeric](xs: List[p]) -> tensor[*, p] = to_tensor(xs)");

    // An empty list's element is determined by the declared result, as the
    // standard library's JSON reader and the `coral` shell rely on.
    accepts("def ok() -> Dict[string, i32] = dict_of([])");
    accepts("def ok() -> tensor[*, i64] = to_tensor([])");
    accepts("def ok[p: Float]() -> tensor[*, p] = to_tensor([])");
    rejects_with(
        "def bad[a]() -> tensor[*, a] = to_tensor([])",
        &[
            "`to_tensor` admits only some operand types",
            "`a` := an arbitrary type",
        ],
    );
}

/// REGRESSION TEST (#2584 round 4). `dict_of` admitted only `i64` and `string`
/// keys, narrower than [05-OP-56], so deciding it at every instantiation of an
/// `Int`-bounded binder rejected `k := i8`, which `main` checked and both lanes
/// run. The key domain is the atom's: string, bool, and every active
/// signed-integer dtype. No unsigned type reaches `dict_of`: the `uint*` names
/// are reserved and rejected where they are written ([04-DTYPE-1]).
#[test]
fn dictionary_keys_are_the_atom_domain_at_every_binder_instantiation() {
    for (key, literal) in [
        ("string", "\"k\""),
        ("bool", "true"),
        ("i8", "1i8"),
        ("i16", "1i16"),
        ("i32", "1i32"),
        ("i64", "1i64"),
    ] {
        accepts(&format!(
            "def ok() -> Dict[{key}, i32] = dict_of([({literal}, 1i32)])"
        ));
    }

    // The two round-4 witnesses, over a binder bounded by `Int`, and each
    // called at every member.
    for (declaration, call) in [
        (
            "def mk[k: Int, v](pairs: List[(k, v)]) -> Dict[k, v] = dict_of(pairs)",
            "mk([(1KEY, 1i32)])",
        ),
        (
            "def keyed[k: Int](x: k) -> Dict[k, i32] = dict_insert(dict_of([]), x, 1i32)",
            "keyed(1KEY)",
        ),
    ] {
        accepts(declaration);
        for key in ["i8", "i16", "i32", "i64"] {
            accepts(&format!(
                "{declaration}\ndef main() -> Dict[{key}, i32] = {}\n",
                call.replace("KEY", key)
            ));
        }
    }

    const KEY_RULE: &str =
        "dict_of keys must be string, bool, or a signed integer scalar ([05-OP-56])";
    rejects_with(
        "def bad() -> Dict[f32, i32] = dict_of([(1.0f32, 1i32)])",
        &[KEY_RULE, "got f32"],
    );
    rejects_with(
        "def bad() -> Dict[(i32, i32), i32] = dict_of([((1i32, 2i32), 1i32)])",
        &[KEY_RULE, "got (i32, i32)"],
    );
    rejects_with(
        "def mk[k: Float, v](pairs: List[(k, v)]) -> Dict[k, v] = dict_of(pairs)",
        &[KEY_RULE, "declared `k: Float`", "`k := f32`"],
    );
}

/// REGRESSION TEST (chelis#2523). A tuple projection or field read on a binder
/// kept no obligation at the boundary.
#[test]
fn an_access_on_a_binder_is_decided_at_its_instantiations() {
    rejects_with(
        "def bad[p: Float](k: p) -> i32 = k.0",
        &["expected tuple type, got f32", "`p := f32`"],
    );
    accepts("def ok[p: Float](k: (p, i32)) -> p = k.0");
    rejects_with(
        "def bad[p: Float](k: p) -> i32 = k.x",
        &["field access `.x` expects a record value", "`p := f32`"],
    );
    accepts("type Point =\n  | Point { x: i32, y: i32 }\ndef ok(k: Point) -> i32 = k.x\n");
    rejects_with(
        "def fst(t) = t.0",
        &[
            "the access `.0` admits only some operand types",
            "never determined within `fst`",
        ],
    );
}

/// REGRESSION TEST (chelis#2523). A projection on a lambda parameter is an
/// obligation like a suspended call ([04-INF-1]), so it keeps its lambda
/// monomorphic until the first application. A `let`-generalized lambda gave
/// every use a fresh, unconstrained result, and a false declared result
/// checked and then failed at run time.
#[test]
fn a_projection_keeps_its_lambda_monomorphic() {
    let source = "def f() -> f32 = {\n  g = fn (t) -> t.0\n  g((1i32, 2i32))\n}\n";
    let diagnostics = agreed_diagnostics(source);
    assert!(!diagnostics.is_empty(), "the mismatched result must reject");
    let errors = check_typed_program(&desugared(source)).unwrap_err().errors;
    assert!(
        errors.iter().any(|e| {
            e.expected.as_deref() == Some("() -> f32") && e.got.as_deref() == Some("() -> i32")
        }),
        "the declaration must reject the inferred i32 result: {errors:?}"
    );
    accepts("def f() -> i32 = {\n  g = fn (t) -> t.0\n  g((1i32, 2i32))\n}\n");
}

/// REGRESSION TEST (chelis#2523). [05-DIM-3] makes an axis-domain argument
/// `i32`. An axis whose type was still a variable was admitted and never
/// revisited, so a binder or a later `i64` binding went unchecked. It is now
/// constrained to `i32`, which pins a binder for its rigidity check.
#[test]
fn an_unresolved_axis_is_constrained_to_i32() {
    for call in ["cumsum(k, ax)", "sort(k, ax).0"] {
        rejects_with(
            &format!("def bad[q: Int](k: tensor[3, f32], ax: q) -> tensor[3, f32] = {call}"),
            &["declared type parameter `q` of `bad` was narrowed to `i32`"],
        );
        accepts(&format!(
            "def ok(k: tensor[3, f32], ax: i32) -> tensor[3, f32] = {call}"
        ));
    }
    rejects_with(
        "def bad[q: Int](x: tensor[3, f32], ax: q) -> i64 = shape(x, ax)",
        &["declared type parameter `q` of `bad` was narrowed to `i32`"],
    );
    for (call, valid) in [
        ("cumsum(x, v)", "0i32"),
        ("gather(x, to_tensor([0i64, 1i64]), v)", "0i32"),
    ] {
        let program =
            format!("def f(x: tensor[3, f32]) -> tensor[*, f32] = (fn (v) -> {call})(0i64)");
        for checked in [
            check_typed_program(&desugared(&program)),
            check_ir_program(&expanded(&program)),
        ] {
            let report = checked.expect_err("an i64 axis must not satisfy an i32 slot");
            assert!(
                report.errors.iter().any(|error| {
                    error.kind.diagnostic_name() == "PrecisionMismatch"
                        && error.expected.as_deref() == Some("i32")
                        && error.got.as_deref() == Some("i64")
                        && error.span_offset == program.find("fn (v)")
                }),
                "the later binding must conflict with the axis dtype: {:?}",
                report.errors
            );
        }
        let correctly_typed = program.replace("0i64)", &format!("{valid})"));
        if call.starts_with("gather(") {
            // [05-AXIS-2]: the lambda parameter remains a variable even
            // when its application supplies a literal. Its i32 constraint
            // is necessary but does not establish gather's static geometry.
            rejects_with(
                &correctly_typed,
                &["gather axis must be an i32 integer constant", "[05-AXIS-2]"],
            );
            accepts(
                "def f(x: tensor[3, f32]) -> tensor[2, f32] = gather(x, to_tensor([0i64, 1i64]), 0i32)",
            );
            rejects_with(
                "def f(x: tensor[3, f32]) -> tensor[2, f32] = gather(x, to_tensor([0i64, 1i64]), 0i64)",
                &["gather expects i32 axis, got i64"],
            );
        } else {
            accepts(&correctly_typed);
        }
    }
}

/// REGRESSION TEST (chelis#2584 round 1, P1-2). A recursive group's
/// in-group reference is typed at the member's provisional type, which its
/// body determines ([04-INF-5], [04-INF-2]). A declared header with an
/// inference hole (here the omitted result) quantified the hole, so each
/// recursive call saw a fresh variable nothing tied to the body, and a
/// member's obligations were decided when it closed, before a sibling's body
/// had filled the type they waited on. Both are decided when the group
/// completes, on the shared hole.
#[test]
fn a_recursive_group_decides_its_obligations_on_the_bodies_it_infers() {
    for program in [
        // Self-recursion through a tuple, a collection operation, and a field.
        "def step(n: i32) = if eq(n, 0) then (0i32, 1i32) else {\n  (x, y) = step(n - 1)\n  (y, x + y)\n}\n",
        "def f(n: i32) = if eq(n, 0) then [1i32] else take(f(n - 1), 1i64)\n",
        "type Pt =\n  | Pt { x: i32, y: i32 }\ndef walk(n: i32) = if eq(n, 0) then Pt { x: 0i32, y: 0i32 } else {\n  p = walk(n - 1)\n  Pt { x: p.x + 1i32, y: p.y }\n}\n",
        // Mutual recursion through the omitted result of a sibling inferred
        // after the reader, in both declaration orders.
        "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\ndef b(n: i32) = if a(n) then 0 else 1\n",
        "def b(n: i32) = if a(n) then 0 else 1\ndef a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\n",
        "def a(n: i32) -> i32 = if eq(n, 0) then 0i32 else b(n - 1).0\ndef b(n: i32) = (a(n), 1i32)\n",
    ] {
        accepts(program);
    }
    // The body-determined type decides the access: `step` returns `i32`.
    rejects_with(
        "def step(n: i32) = if eq(n, 0) then 5i32 else step(n - 1).0\n",
        &["expected tuple type, got i32"],
    );
}

/// Diagnostics of a program that is rejected, empty when it checks.
fn verdict(source: &str) -> Vec<String> {
    agreed_diagnostics(source)
}

/// The repair a rejection names when a recursive call reaches a member that
/// omits a type at an instantiation other than its own.
const POLYMORPHIC_RECURSION_REPAIR: &str = "Write the omitted types to allow a call at another instantiation of its own binders ([04-INF-2]).";

/// Suggestions of every diagnostic, on the typed ingress.
fn suggestions(source: &str) -> Vec<String> {
    match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .flat_map(|e| e.suggestions.clone())
            .collect(),
    }
}

/// REGRESSION TEST (chelis#2584 round 2). [04-INF-5]: inside a group
/// inferred as one unit, an in-group reference is typed at the member's
/// provisional monomorphic type, as [04-INF-2] provides, and an authored
/// binder admits no substitute. So every in-group reference to a member that
/// omits a type shares that member's authored binders, and a recursive call
/// that swaps two of them identifies them, which [04-INF-6] rejects (its
/// omitted types are chelis#2590's, in `issue_2590_recursive_group_holes`).
/// Round 1 shared only the omitted types and instantiated the binders afresh
/// at each call, so these checked with score 1 and failed in `eval`: the
/// omitted type never followed the call's instantiation.
#[test]
fn a_group_member_that_omits_a_type_is_referenced_at_one_instantiation() {
    for program in [
        "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, n) else f(y, x, n - 1)\n",
        "def f[n, m](x: tensor[n, f32], y: tensor[m, f32], k: i32) = if eq(k, 0) then x else f(y, x, k - 1)\n",
        "def f[a](x: a, n: i32) = if eq(n, 0) then (x, n) else g(x, x, n - 1)\n\
         def g[b, c](y: b, z: c, n: i32) = if eq(n, 0) then (z, n) else f(y, n - 1)\n",
        "def g[b, c](y: b, z: c, n: i32) = if eq(n, 0) then (z, n) else f(y, n - 1)\n\
         def f[a](x: a, n: i32) = if eq(n, 0) then (x, n) else g(x, x, n - 1)\n",
    ] {
        rejects_with(program, &["were unified by the function body"]);
        assert!(
            suggestions(program)
                .iter()
                .any(|suggestion| suggestion.contains(POLYMORPHIC_RECURSION_REPAIR)),
            "the rejection must name the repair that allows polymorphic recursion:\n{program}"
        );
    }
    // A member whose every type is written is referenced at its declared
    // scheme, so its recursive calls may instantiate it afresh.
    accepts(
        "def swap[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else {\n  \
         (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
    );
    // Omitted types still generalize once the group is complete.
    accepts(
        "def up[a](x: a, y, n: i32) = if eq(n, 0) then y else down(x, y, n - 1)\n\
         def down[b](x: b, y, n: i32) = if eq(n, 0) then y else up(x, y, n - 1)\n\
         def main() -> (string, f32) = (up(1i32, \"s\", 2), down(\"t\", 1.5f32, 1))\n",
    );
}

/// How a fixture that omits a type may differ from its twin with the
/// body-determined type written out as authored binders ([04-INF-2]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TwinDifference {
    /// Both forms get the same verdict.
    None,
    /// An in-group call instantiates an omitted type at a fully concrete
    /// type, which an inference-introduced variable admits and an authored
    /// binder does not: only the omitted form checks.
    ConcreteArgument,
    /// An in-group call instantiates the member's own type parameters at one
    /// another (swapped or merged), which a fully written member's
    /// polymorphic recursion admits and a member that omits a type does not:
    /// only the written form checks.
    OwnParametersPermuted,
}

/// ORACLE (chelis#2584 round 2, made bidirectional for chelis#2590): a
/// recursive-group fixture that omits a type and its twin with the
/// body-determined type written out get the same verdict, except in the two
/// directions [04-INF-2] creates. The exact set of differing pairs is
/// asserted, so neither direction can grow or vanish unnoticed.
#[test]
fn an_omitted_type_and_its_written_twin_differ_only_where_the_instantiation_rule_does() {
    use TwinDifference::*;
    let pairs: [(&str, &str, &str, TwinDifference); 14] = [
        (
            "self-recursion reading its own result",
            "def step(n: i32) = if eq(n, 0) then (0i32, 1i32) else {\n  (x, y) = step(n - 1)\n  (y, x + y)\n}\n",
            "def step(n: i32) -> (i32, i32) = if eq(n, 0) then (0i32, 1i32) else {\n  (x, y) = step(n - 1)\n  (y, x + y)\n}\n",
            None,
        ),
        (
            "an access on a non-tuple result",
            "def step(n: i32) = if eq(n, 0) then 5i32 else step(n - 1).0\n",
            "def step(n: i32) -> i32 = if eq(n, 0) then 5i32 else step(n - 1).0\n",
            None,
        ),
        (
            "mutual recursion through a sibling's omitted result",
            "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\ndef b(n: i32) = if a(n) then 0 else 1\n",
            "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\ndef b(n: i32) -> i32 = if a(n) then 0 else 1\n",
            None,
        ),
        (
            "authored binders across three members, at one instantiation",
            "def f[a](x: a, n: i32) = if eq(n, 0) then (x, n) else g(x, n - 1)\n\
             def g[b](y: b, n: i32) = if eq(n, 0) then (y, n) else h(y, n - 1)\n\
             def h[c](z: c, n: i32) = if eq(n, 0) then (z, n) else f(z, n - 1)\n",
            "def f[a](x: a, n: i32) -> (a, i32) = if eq(n, 0) then (x, n) else g(x, n - 1)\n\
             def g[b](y: b, n: i32) -> (b, i32) = if eq(n, 0) then (y, n) else h(y, n - 1)\n\
             def h[c](z: c, n: i32) -> (c, i32) = if eq(n, 0) then (z, n) else f(z, n - 1)\n",
            None,
        ),
        (
            "omitted types flowing around a ring",
            "def a(x, n: i32) = if eq(n, 0) then x else b(x, n - 1)\n\
             def b(y, n: i32) = if eq(n, 0) then y else a(y, n - 1)\n",
            "def a[t](x: t, n: i32) -> t = if eq(n, 0) then x else b(x, n - 1)\n\
             def b[u](y: u, n: i32) -> u = if eq(n, 0) then y else a(y, n - 1)\n",
            None,
        ),
        (
            "a swapped call whose result the body joins with its own",
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, n) else f(y, x, n - 1)\n",
            "def f[a, b](x: a, y: b, n: i32) -> (a, i32) = if eq(n, 0) then (x, n) else f(y, x, n - 1)\n",
            None,
        ),
        (
            "a variable embedded in a larger type",
            "def f(x, n: i32) = if eq(n, 0) then 0i32 else f([x], n - 1)\n",
            "def f[a](x: a, n: i32) -> i32 = if eq(n, 0) then 0i32 else f([x], n - 1)\n",
            None,
        ),
        (
            "round 3's mk3: the member's own call at a concrete type",
            "def pick(x, n: i32) = if eq(n, 0) then x else {\n  k = pick(3i32, n - 1)\n  x\n}\n",
            "def pick[a](x: a, n: i32) -> a = if eq(n, 0) then x else {\n  k = pick(3i32, n - 1)\n  x\n}\n",
            ConcreteArgument,
        ),
        (
            "round 3's mk5: a sibling's call at a concrete type",
            "def show(x, n: i32) = if eq(n, 0) then x else {\n  k = helper(n - 1)\n  x\n}\n\
             def helper(n: i32) -> i32 = if eq(n, 0) then 0i32 else show(1i32, n - 1)\n",
            "def show[a](x: a, n: i32) -> a = if eq(n, 0) then x else {\n  k = helper(n - 1)\n  x\n}\n\
             def helper(n: i32) -> i32 = if eq(n, 0) then 0i32 else show(1i32, n - 1)\n",
            ConcreteArgument,
        ),
        (
            "round 3's mk11: a concrete list",
            "def total(xs, n: i32) = if eq(n, 0) then xs else {\n  k = total([1i32], n - 1)\n  xs\n}\n",
            "def total[a](xs: a, n: i32) -> a = if eq(n, 0) then xs else {\n  k = total([1i32], n - 1)\n  xs\n}\n",
            ConcreteArgument,
        ),
        (
            "round 3's mk10: a concrete omitted slot beside an authored binder",
            "def f[a](x: a, y, n: i32) = if eq(n, 0) then (x, y) else {\n  k = f(x, 3i32, n - 1)\n  (x, y)\n}\n",
            "def f[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else {\n  k = f(x, 3i32, n - 1)\n  (x, y)\n}\n",
            ConcreteArgument,
        ),
        (
            "round 2's r10: swapped authored binders",
            "def swap[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
            "def swap[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
            OwnParametersPermuted,
        ),
        (
            "round 2's r9: authored binders merged through a sibling",
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, y) else (g(x, n - 1), g(y, n - 1))\n\
             def g[c](z: c, n: i32) = if eq(n, 0) then z else f(z, z, n - 1).0\n",
            "def f[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else (g(x, n - 1), g(y, n - 1))\n\
             def g[c](z: c, n: i32) -> c = if eq(n, 0) then z else f(z, z, n - 1).0\n",
            OwnParametersPermuted,
        ),
        (
            "swapped omitted types",
            "def swap(x, y, n: i32) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
            "def swap[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
            OwnParametersPermuted,
        ),
    ];
    let mut differing = Vec::new();
    for (label, omitted, written, expected) in pairs {
        let omitted_checks = verdict(omitted).is_empty();
        let written_checks = verdict(written).is_empty();
        let found = match (omitted_checks, written_checks) {
            (true, true) | (false, false) => None,
            (true, false) => ConcreteArgument,
            (false, true) => OwnParametersPermuted,
        };
        assert_eq!(
            found,
            expected,
            "{label}:\n{omitted}\n{written}\nomitted: {:#?}\nwritten: {:#?}",
            verdict(omitted),
            verdict(written)
        );
        match found {
            None => {}
            ConcreteArgument => {
                // The written form's rejection is the authored binder's.
                assert!(
                    verdict(written)
                        .iter()
                        .any(|message| message.contains("polymorphic recursion")
                            || message.contains("[04-INF-6]")),
                    "{label}: {:#?}",
                    verdict(written)
                );
                differing.push(label);
            }
            OwnParametersPermuted => {
                // The omitted form's rejection is the instantiation rule's.
                assert!(
                    verdict(omitted)
                        .iter()
                        .any(|message| message.contains("polymorphic recursion"))
                        || suggestions(omitted)
                            .iter()
                            .any(|suggestion| suggestion.contains(POLYMORPHIC_RECURSION_REPAIR)),
                    "{label}: {:#?}",
                    verdict(omitted)
                );
                differing.push(label);
            }
        }
    }
    assert_eq!(
        differing,
        [
            "round 3's mk3: the member's own call at a concrete type",
            "round 3's mk5: a sibling's call at a concrete type",
            "round 3's mk11: a concrete list",
            "round 3's mk10: a concrete omitted slot beside an authored binder",
            "round 2's r10: swapped authored binders",
            "round 2's r9: authored binders merged through a sibling",
            "swapped omitted types",
        ]
    );
}
