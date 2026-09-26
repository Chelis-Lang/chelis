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
//! rejection has an accepted twin that differs only in the operand's type.

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
    rejects_with(
        "def f() -> f32 = {\n  g = fn (t) -> t.0\n  g((1i32, 2i32))\n}\n",
        &["body has type `() -> i32`, declared type is `() -> f32`"],
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
        rejects_with(&program, &["expected i32, got i64"]);
        accepts(&program.replace("0i64)", &format!("{valid})")));
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
