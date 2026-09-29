//! chelis#2590: a recursive group whose members omit types is typed by
//! [04-INF-2] and [04-INF-5] together.
//!
//! At an in-group reference, each authored binder of the callee is the
//! caller's own, and each omitted type is instantiated afresh. When the group
//! completes, each instance must be the member's own type (it unified with it,
//! or stayed unconstrained and is chosen as it) or a fully concrete type;
//! anything else is [04-INF-3]'s polymorphic recursion, and so is an in-group
//! call that swaps or merges two of a member's omitted types. The reference
//! is typed at the type the member's body determines, which it never narrows,
//! and the member's omitted types generalize once the group completes. The
//! verdict is decided on the group's solved types, so it is the same in every
//! declaration order.
//!
//! Each test is labelled with the heads it fails on: `main` (`a116a9e10`, the
//! pull request's base, which instantiated every omitted type afresh and never
//! tied the instance to the body), the round-3 head `c2ec7adca` (which typed
//! every in-group reference at one monomorphic instantiation), or the round-4
//! candidate `375598343` (which decided each reference as it linked it, so
//! the verdict depended on the declaration order). A fixture that fails on
//! none is a lock, and says so.

use std::collections::BTreeSet;

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

/// The diagnostics of a program, the same on both checker ingresses
/// (chelis#1107), with `?N` ids renumbered so they compare across runs.
fn diagnostics(source: &str) -> Vec<String> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let typed = typed.iter().map(|m| normalized(m)).collect::<Vec<_>>();
    let ir = ir.iter().map(|m| normalized(m)).collect::<Vec<_>>();
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

fn accepts(source: &str) {
    let found = diagnostics(source);
    assert!(
        found.is_empty(),
        "must type-check:\n{source}\ngot {found:#?}"
    );
}

/// A rejection carrying a diagnostic that contains every fragment.
fn rejects_with(source: &str, fragments: &[&str]) {
    let found = diagnostics(source);
    assert!(
        found
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {found:#?}"
    );
}

fn rejects_vmap_scalar_claim(source: &str) {
    let found = diagnostics(source);
    assert!(
        found.iter().any(|message| {
            message.contains("[TypeMismatch]")
                && message.contains("f32")
                && message.contains("tensor")
        }),
        "the mapped tensor must contradict the scalar claim:\n{source}\ngot {found:#?}"
    );
    assert!(
        found.iter().all(|message| !message.contains("unsupported")),
        "the retired vmap fence must not decide:\n{source}\ngot {found:#?}"
    );
}

/// Renumber `?N` type variables in order of first occurrence.
fn normalized(text: &str) -> String {
    let mut seen: Vec<String> = Vec::new();
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '?' && chars.peek().is_some_and(char::is_ascii_digit) {
            let mut id = String::new();
            while let Some(d) = chars.peek().copied().filter(char::is_ascii_digit) {
                id.push(d);
                chars.next();
            }
            let index = seen.iter().position(|s| *s == id).unwrap_or_else(|| {
                seen.push(id);
                seen.len() - 1
            });
            out.push_str(&format!("?{index}"));
        } else {
            out.push(c);
        }
    }
    out
}

/// The signature `chelis check --show-inferred` publishes for `name`.
fn published(source: &str, name: &str) -> String {
    let checked = check_typed_program(&desugared(source))
        .unwrap_or_else(|e| panic!("must type-check:\n{source}\n{:#?}", rendered(&e.errors)));
    let function = checked
        .signature_inference()
        .functions
        .values()
        .find(|function| function.name == name)
        .unwrap_or_else(|| panic!("`{name}` has an inferred signature"));
    normalized(&function.checked_signature.to_string())
}

/// Every declaration order of the program's `def`s, with `main` kept last.
fn declaration_orders(source: &str) -> Vec<String> {
    let blocks: Vec<&str> = source
        .split("\n\n")
        .map(str::trim)
        .filter(|block| !block.is_empty())
        .collect();
    let (main, members): (Vec<&str>, Vec<&str>) = blocks
        .into_iter()
        .partition(|block| block.starts_with("def main"));
    let mut orders = Vec::new();
    permute(&members, &mut Vec::new(), &mut orders);
    orders
        .into_iter()
        .map(|order| {
            let mut all: Vec<&str> = order;
            all.extend(main.iter().copied());
            all.join("\n\n") + "\n"
        })
        .collect()
}

fn permute<'a>(rest: &[&'a str], prefix: &mut Vec<&'a str>, out: &mut Vec<Vec<&'a str>>) {
    if rest.is_empty() {
        out.push(prefix.clone());
        return;
    }
    for index in 0..rest.len() {
        let mut remaining = rest.to_vec();
        let chosen = remaining.remove(index);
        prefix.push(chosen);
        permute(&remaining, prefix, out);
        prefix.pop();
    }
}

// Round 3's `mk3`, `mk5`, `mk11`, `mk1` and `mk10`: an in-group reference that
// instantiates an omitted type at a fully concrete type.
const PICK: &str = "def pick(x, n: i32) = if eq(n, 0) then x else {\n  k = pick(3i32, n - 1)\n  x\n}\n\n\
def main() -> (string, i32) = (pick(\"s\", 2), pick(4i32, 1))\n";
const SHOW: &str = "def show(x, n: i32) = if eq(n, 0) then x else {\n  k = helper(n - 1)\n  x\n}\n\n\
def helper(n: i32) -> i32 = if eq(n, 0) then 0i32 else show(1i32, n - 1)\n\n\
def main() -> (string, i32) = (show(\"a\", 2), helper(2))\n";
const TOTAL: &str = "def total(xs, n: i32) = if eq(n, 0) then xs else {\n  k = total([1i32], n - 1)\n  xs\n}\n\n\
def main() -> (List[string], List[i32]) = (total([\"a\"], 2), total([2i32], 1))\n";
const MK: &str = "def mk(n: i32) = if eq(n, 0) then [] else {\n  xs = mk(n - 1)\n  ys = append(xs, 1i32)\n  []\n}\n\n\
def main() -> (List[i32], List[string]) = (mk(1), mk(2))\n";
const HALF_WRITTEN: &str = "def f[a](x: a, y, n: i32) = if eq(n, 0) then (x, y) else {\n  k = f(x, 3i32, n - 1)\n  (x, y)\n}\n\n\
def main() -> ((i32, string), (f32, i32)) = (f(1i32, \"s\", 2), f(1.5f32, 4i32, 1))\n";

/// REGRESSION TEST (fails on `c2ec7adca`, and on `main`, which rejected `mk`
/// and the half-written `f`). [04-INF-2] admits a fully concrete
/// type argument for a signature variable that inference introduced, so a
/// sibling's or the member's own call at a concrete type neither fixes the
/// omitted type nor narrows the published signature. `c2ec7adca` bound it at
/// the call's type, so `show("a", 2)` was rejected, or the signature was
/// published as `(i32, i32) -> i32`.
#[test]
fn a_concrete_in_group_argument_leaves_the_omitted_type_generic() {
    for program in [PICK, SHOW, TOTAL, MK, HALF_WRITTEN] {
        accepts(program);
    }
    // `--show-inferred` publishes main's generic signatures.
    assert_eq!(published(PICK, "pick"), "(?0, i32) -> ?0");
    assert_eq!(published(SHOW, "show"), "(?0, i32) -> ?0");
    assert_eq!(published(TOTAL, "total"), "(?0, i32) -> ?0");
    assert_eq!(published(MK, "mk"), "(i32) -> List ?0");
    // Round 3's `mk9`: with no other use, the signature is still generic.
    let only_helper = SHOW.replace(
        "def main() -> (string, i32) = (show(\"a\", 2), helper(2))",
        "def main() -> i32 = helper(2)",
    );
    assert_eq!(published(&only_helper, "show"), "(?0, i32) -> ?0");
}

/// REGRESSION TEST (fails on `main`). The concrete instance is the call's own:
/// its result is the body-determined type at that instance, never the
/// member's generic result. Here the body returns the call's result, so the
/// body itself determines `x: i32`, and `pick("s", 2)` is a type error at
/// `main`'s reference. `main` left the call's result unrelated to its
/// argument and returned `3i32` as a `string`.
#[test]
fn a_concrete_instance_types_the_result_of_its_own_call() {
    rejects_with(
        "def pick(x, n: i32) = if eq(n, 0) then x else pick(3i32, n - 1)\n\n\
         def main() -> string = pick(\"s\", 2)\n",
        &["precision mismatch: expected i32, got string"],
    );
    accepts(
        "def pick(x, n: i32) = if eq(n, 0) then x else pick(3i32, n - 1)\n\n\
         def main() -> i32 = pick(4i32, 2)\n",
    );
    rejects_with(
        "def pick(x, n: i32) = if eq(n, 0) then x else {\n  k = pick(3i32, n - 1)\n  m = numel(k)\n  x\n}\n",
        &["numel expects tensor input, got i32"],
    );
}

/// REGRESSION TEST (fails on `main` and on `c2ec7adca`). An instance that is
/// neither the member's own type nor fully concrete is [04-INF-3]'s polymorphic
/// recursion, and the diagnostic names the function, the body-determined
/// type, and the reference's instantiation. Choosing a variable as the
/// member's own type cannot identify two of one member's own types: that
/// would narrow a published signature to fit a reference.
#[test]
fn an_instance_neither_own_nor_concrete_is_polymorphic_recursion() {
    // A variable embedded in a larger type.
    rejects_with(
        "def f(x, n: i32) = if eq(n, 0) then 0i32 else f([x], n - 1)\n",
        &[
            "polymorphic recursion: `f` makes an in-group reference to `f` at `(List ?0, i32) -> i32`",
            "`f`'s body determines `(?0, i32) -> i32`",
            "[04-INF-3]",
        ],
    );
    // The member's own omitted types, swapped.
    rejects_with(
        "def swap(x, y, n: i32) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n",
        &["polymorphic recursion: `swap` makes an in-group reference to `swap`"],
    );
    // One member's two omitted types, each chosen as a sibling's one.
    rejects_with(
        "def g(x, y, n: i32) = if eq(n, 0) then (x, y) else (f(x, n), f(y, n))\n\n\
         def f(z, n: i32) = if eq(n, 0) then z else g(z, 1i32, n - 1).0\n",
        &["polymorphic recursion: `g` makes an in-group reference to `f`"],
    );
    // Its accepted twin: only one of them reaches the sibling, which shares
    // it, and the other is instantiated at a concrete type.
    accepts(
        "def g(x, y, n: i32) = if eq(n, 0) then (x, y) else (f(x, n), y)\n\n\
         def f(z, n: i32) = if eq(n, 0) then z else g(z, 1i32, n - 1).0\n",
    );
}

/// REGRESSION TEST (fails on `c2ec7adca` and on `main`). [04-INF-5]: a
/// reference whose use
/// disagrees with the body-determined type is a type error at the reference,
/// and the slot is the one the body determines. `c2ec7adca` let the use fix the
/// hole first and reported the body against a "declared type" carrying `?N`.
#[test]
fn a_disagreeing_use_is_reported_at_the_reference() {
    for program in [
        // The use precedes the body in declaration order, then follows it.
        "def f(n: i32) = if eq(n, 0) then 1i32 else {\n  k = g(n - 1)\n  1i32\n}\n\n\
         def g(n: i32) -> bool = if eq(n, 0) then true else f(n - 1)\n",
        "def g(n: i32) -> bool = if eq(n, 0) then true else f(n - 1)\n\n\
         def f(n: i32) = if eq(n, 0) then 1i32 else {\n  k = g(n - 1)\n  1i32\n}\n",
    ] {
        let found = diagnostics(program);
        assert!(
            found.iter().any(|m| m.contains(
                "`g` uses `f` at `(i32) -> bool`, which disagrees with the type `f`'s body \
                 determines, `(i32) -> i32`"
            ) && m.contains("[04-INF-5]")),
            "{program}\n{found:#?}"
        );
        assert!(
            found.iter().all(|m| !m.contains("declared type")),
            "an omitted slot is not declared:\n{program}\n{found:#?}"
        );
    }
}

/// LOCK (passes on `main` and on `c2ec7adca`): round 1's and round 2's
/// group fixtures check in every declaration order. Round 1 found `72ff073af`
/// rejecting `step`, `t4` and `t10`, and `step`, `t9` and `t10` fail when a
/// `let` generalizes over a reference's instance.
#[test]
fn the_round_one_group_fixtures_check() {
    for program in [
        // `step` (round 1's `t2`), `t9`, `t10`, and `t4`.
        "def step(n: i32) = if eq(n, 0) then (0i32, 1i32) else {\n  (x, y) = step(n - 1)\n  (y, x + y)\n}\n\n\
         def main() -> i32 = step(5).0\n",
        "def fib2(n: i32) = if eq(n, 0) then (0i32, 1i32) else {\n  prev = fib2(n - 1)\n  (prev.1, add(prev.0, prev.1))\n}\n\n\
         def main() -> i32 = fib2(10).0\n",
        "type Pt =\n  | Pt { x: i32, y: i32 }\n\ndef walk(n: i32) = if eq(n, 0) then Pt { x: 0i32, y: 0i32 } else {\n  p = walk(n - 1)\n  Pt { x: p.x + 1i32, y: p.y }\n}\n\n\
         def main() -> i32 = walk(3).x\n",
        "def f(n: i32) = if eq(n, 0) then [1i32] else take(f(n - 1), 1i64)\n\n\
         def main() -> List[i32] = f(2)\n",
        // `t1`, `t3`, `t7`.
        "def a(n: i32) -> i32 = if eq(n, 0) then 0i32 else b(n - 1).0\n\n\
         def b(n: i32) = (a(n), 1i32)\n\n\
         def main() -> i32 = a(3)\n",
        "def step(n: i32) -> (i32, i32) = if eq(n, 0) then (0i32, 1i32) else {\n  (x, y) = step(n - 1)\n  (y, x + y)\n}\n\n\
         def main() -> i32 = step(5).0\n",
        "def f(n) = if eq(n, 0i32) then (0i32, 1i32) else {\n  (x, y) = f(n - 1i32)\n  (y, x + y)\n}\n",
        // `rec1`, `rec1e`, `rec1g`, and `o1`, `o3`.
        "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\n\n\
         def b(n: i32) = if a(n) then 0 else 1\n\n\
         def main() -> bool = a(3)\n",
        "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), 1)\n\n\
         def b(n: i32) = if a(n) then 0 else 1\n",
        "def a(n: i32) = if eq(n, 0) then 0i32 else b(n - 1)\n\n\
         def b(n: i32) -> i32 = if eq(a(n), a(n)) then 1i32 else 0i32\n",
        "def a(n: i32) -> bool = if eq(n, 0) then true else eq(b(n - 1), b(n - 1))\n\n\
         def b(n: i32) = if a(n) then 0i32 else 1i32\n",
        // The written twins `r10d` and `r11`: polymorphic recursion over the
        // binders of a member whose every type is written.
        "def swap[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n\n\
         def main() -> (i32, string) = swap(3i32, \"s\", 2)\n",
        "def f[a, b](x: a, y: b, n: i32) -> (a, b) = if eq(n, 0) then (x, y) else (g(x, n - 1), g(y, n - 1))\n\n\
         def g[c](z: c, n: i32) -> c = if eq(n, 0) then z else f(z, z, n - 1).0\n\n\
         def main() -> (i32, string) = f(3i32, \"s\", 2)\n",
    ] {
        for order in declaration_orders(program) {
            accepts(&order);
        }
    }
}

/// REGRESSION TEST (each fails on `main`, which checked it at score 1 and
/// then failed in `eval`, or accepted it by dropping the obligation):
/// round 1's `t8`, round 2's `u5`, `u7`, `s1`, `s2`, `u6`, `u9`, `u3`, `u4`,
/// `perm2u_*`, `r9` and `r10`, in every declaration order.
#[test]
fn the_ill_typed_group_fixtures_are_rejected_in_every_order() {
    for (program, fragment) in [
        (
            "def step(n: i32) = if eq(n, 0) then 5i32 else step(n - 1).0\n\n\
             def main() -> i32 = step(2)\n",
            "expected tuple type, got i32",
        ),
        (
            "def f[a](x: a, n: i32) = if eq(n, 0) then x else g(x, x, n - 1)\n\n\
             def g[b, c](y: b, z: c, n: i32) = if eq(n, 0) then z else f(y, n - 1)\n\n\
             def main() -> string = g(1i32, \"s\", 1)\n",
            "distinct declared type parameters `b` and `c` of `g` were unified",
        ),
        (
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then x else f(y, x, n - 1)\n\n\
             def main() -> i32 = f(1i32, \"s\", 1)\n",
            "distinct declared type parameters `a` and `b` of `f` were unified",
        ),
        (
            "def a(x, n: i32) = if eq(n, 0) then numel(x) else b(x, n - 1)\n\n\
             def b(y, n: i32) = a(y, n)\n\n\
             def main() -> i64 = a(to_tensor([1.0f32]), 2)\n",
            "`numel` admits only some operand types",
        ),
        (
            "def a(n: i32) = if eq(n, 0) then 0i64 else numel(b(n - 1))\n\n\
             def b(n: i32) = if eq(n, 0) then [1i32] else c(n)\n\n\
             def c(n: i32) = if eq(a(n - 1), 0i64) then [2i32] else b(n - 1)\n\n\
             def main() -> i64 = a(2)\n",
            "numel expects tensor input, got List i32",
        ),
        (
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, n) else f(y, x, n - 1)\n\n\
             def main() -> i32 = f(1i32, \"s\", 1).0\n",
            "distinct declared type parameters `a` and `b` of `f` were unified",
        ),
        (
            "def f[n, m](x: tensor[n, f32], y: tensor[m, f32], k: i32) = if eq(k, 0) then x else f(y, x, k - 1)\n\n\
             def main() -> tensor[3, f32] = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([4.0f32, 5.0f32]), 1)\n",
            "distinct declared dim parameters `n` and `m` were unified",
        ),
        (
            "def f[a](x: a, n: i32) = if eq(n, 0) then (x, n) else g(x, x, n - 1)\n\n\
             def g[b, c](y: b, z: c, n: i32) = if eq(n, 0) then (z, n) else f(y, n - 1)\n\n\
             def main() -> string = g(1i32, \"s\", 1).0\n",
            "distinct declared type parameters `b` and `c` of `g` were unified",
        ),
        (
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, n) else g(y, x, n - 1)\n\n\
             def g[c, d](z: c, w: d, n: i32) = if eq(n, 0) then (z, n) else f(z, w, n - 1)\n\n\
             def main() -> i32 = f(1i32, \"s\", 1).0\n",
            "were unified by the function body",
        ),
        (
            "def f[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, y) else (g(x, n - 1), g(y, n - 1))\n\n\
             def g[c](z: c, n: i32) = if eq(n, 0) then z else f(z, z, n - 1).0\n\n\
             def main() -> (i32, string) = f(3i32, \"s\", 2)\n",
            "distinct declared type parameters `a` and `b` of `f` were unified",
        ),
        (
            "def swap[a, b](x: a, y: b, n: i32) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n\n\
             def main() -> (i32, string) = swap(3i32, \"s\", 2)\n",
            "distinct declared type parameters `a` and `b` of `swap` were unified",
        ),
    ] {
        for order in declaration_orders(program) {
            rejects_with(&order, &[fragment]);
        }
    }
}

/// Round 2's `h1`, `h2` and `v5`: omitted types generalize once the group
/// completes (`h1` fails on `main`, which rejected it as polymorphic
/// recursion; `h2` and `v5` are locks).
#[test]
fn omitted_types_generalize_after_the_group_completes() {
    let keep = "def keep[a](x: a, y, n: i32) = if eq(n, 0) then (y, x) else keep(x, y, n - 1)\n\n\
                def main() -> ((string, i32), (f32, string)) = (keep(1i32, \"s\", 2), keep(\"t\", 1.5f32, 1))\n";
    accepts(keep);
    assert_eq!(published(keep, "keep"), "(?0, ?1, i32) -> (?1, ?0)");
    let up = "def up[a](x: a, y, n: i32) = if eq(n, 0) then y else down(x, y, n - 1)\n\n\
              def down[b](x: b, y, n: i32) = if eq(n, 0) then y else up(x, y, n - 1)\n\n\
              def main() -> (string, f32) = (up(1i32, \"s\", 2), down(\"t\", 1.5f32, 1))\n";
    for order in declaration_orders(up) {
        accepts(&order);
        assert_eq!(published(&order, "up"), "(?0, ?1, i32) -> ?1");
        assert_eq!(published(&order, "down"), "(?0, ?1, i32) -> ?1");
    }
    let ring = "def a(x, n: i32) = if eq(n, 0) then x else b(x, n - 1)\n\n\
                def b(y, n: i32) = if eq(n, 0) then y else c(y, n - 1)\n\n\
                def c(z, n: i32) = if eq(n, 0) then z else a(z, n - 1)\n\n\
                def main() -> (i32, string, tensor[2, f32]) = (a(1i32, 2), c(\"s\", 3), b(to_tensor([1.0f32, 2.0f32]), 4))\n";
    for order in declaration_orders(ring) {
        accepts(&order);
        for member in ["a", "b", "c"] {
            assert_eq!(published(&order, member), "(?0, i32) -> ?0", "{order}");
        }
    }
}

/// The `a`/`b` pair whose calls swap `a`'s two omitted types: round 3's `tw`
/// (with a use) and `two` (without one).
const TWO: &str = "def a(x, y, n: i32) = if eq(n, 0) then (x, y) else b(y, x, n - 1)\n\n\
def b(u, v, n: i32) = if eq(n, 0) then (u, v) else a(u, v, n - 1)\n\n\
def main() -> i32 = 0i32\n";
/// Round 3's `nm3`: the swap around a three-member ring.
const THREE: &str = "def a(x, y, n: i32) = if eq(n, 0) then (x, y) else b(y, x, n - 1)\n\n\
def b(u, v, n: i32) = if eq(n, 0) then (u, v) else c(u, v, n - 1)\n\n\
def c(p, q, n: i32) = if eq(n, 0) then (p, q) else a(p, q, n - 1)\n\n\
def main() -> i32 = 0i32\n";

/// The repair a merge or swap of omitted types names.
fn names_the_binders_repair(program: &str, member: &str) -> bool {
    let repair = format!("write `{member}`'s signature with explicit type binders");
    match check_typed_program(&desugared(program)) {
        Ok(_) => false,
        Err(result) => result
            .errors
            .iter()
            .flat_map(|error| error.suggestions.iter())
            .any(|suggestion| suggestion.contains(&repair)),
    }
}

/// REGRESSION TEST (fails on `375598343`, which accepted `tw` and `bad3` with
/// `a`'s two omitted types merged in some declaration orders and rejected
/// them in others; and on `main` and `c2ec7adca` for `swh` and `gxy2`, which
/// they accepted). An in-group call that swaps or merges two of a member's
/// omitted types is rejected in every declaration order ([04-INF-2],
/// [04-INF-3]), and the diagnostic names the repair: write the binders. A
/// published signature is never narrowed by an in-group use, so `a` is not
/// published with `x` and `y` at one type.
#[test]
fn swapped_or_merged_omitted_types_are_rejected_in_every_order() {
    let with_use = |program: &str, main: &str| program.replace("def main() -> i32 = 0i32", main);
    for (program, member) in [
        // `two` and `tw`.
        (TWO.to_string(), "a"),
        (
            with_use(TWO, "def main() -> (i32, i32) = a(1i32, 2i32, 3)"),
            "a",
        ),
        // `nm3` and `bad3`.
        (THREE.to_string(), "a"),
        (
            with_use(THREE, "def main() -> (i32, string) = a(1i32, \"s\", 3)"),
            "a",
        ),
        // `swh`: one member swaps its own omitted types.
        (
            "def swap(x, y, n: i32) = if eq(n, 0) then (x, y) else {\n  (u, v) = swap(y, x, n - 1)\n  (v, u)\n}\n\n\
             def main() -> (i32, string) = swap(3i32, \"s\", 2)\n"
                .to_string(),
            "swap",
        ),
        // `gxy2`: a sibling's call gives `g`'s two omitted types one type.
        (
            "def g(x, y, n: i32) = if eq(n, 0) then (x, y) else (f(x, n), f(y, n))\n\n\
             def f(z, n: i32) = if eq(n, 0) then z else g(z, z, n - 1).0\n\n\
             def main() -> (i32, i32) = g(1i32, 2i32, 2)\n"
                .to_string(),
            "g",
        ),
        // Round 3's `s9`: the call passes `x` for `y`.
        (
            "def f(x, y, n: i32) = if eq(n, 0) then y else f(x, x, n - 1)\n\n\
             def main() -> (i32, string) = (f(1i32, 2i32, 2), f(\"a\", \"b\", 1))\n"
                .to_string(),
            "f",
        ),
        // A sibling whose body gives its own two types one type.
        (
            "def f(x, y, n: i32) = if eq(n, 0) then x else g(x, y, n - 1)\n\n\
             def g(u, v, n: i32) = if eq(n, 0) then u else if eq(n, 1) then v else f(u, v, n - 1)\n\n\
             def main() -> i32 = f(1i32, 2i32, 3)\n"
                .to_string(),
            "f",
        ),
        // Two omitted element types of a result, swapped by a sibling.
        (
            "def f(n: i32) = if eq(n, 0) then ([], []) else g(n)\n\n\
             def g(n: i32) = {\n  (p, q) = f(n - 1)\n  (q, p)\n}\n\n\
             def main() -> i32 = 0i32\n"
                .to_string(),
            "f",
        ),
        // An omitted type given an authored binder's type.
        (
            "def f[a](x: a, y, n: i32) = if eq(n, 0) then (x, y) else {\n  k = f(x, x, n - 1)\n  (x, y)\n}\n\n\
             def main() -> i32 = 0i32\n"
                .to_string(),
            "f",
        ),
    ] {
        for order in declaration_orders(&program) {
            rejects_with(
                &order,
                &["polymorphic recursion", "[04-INF-2]", "[04-INF-3]"],
            );
            assert!(
                names_the_binders_repair(&order, member),
                "the rejection must tell the author to write `{member}`'s binders:\n{order}"
            );
        }
    }
}

/// REGRESSION TEST (`fwd`, `dup` and `step` fail on `375598343`; `tw`-like
/// orders there accepted too much). A member's type that only an in-group
/// call determines is determined by it, not merged: a member that forwards a
/// sibling's result, or reads it twice, is accepted in every order, and round
/// 3's `ok3`, `sig3`, `s3`, `s4`, `s7` and `s8` publish main's generic
/// signatures. `s4`'s `h` is published at the type its call to `f` gives it,
/// `(i32, i32) -> i32`, where `main` published an unrelated result variable.
#[test]
fn determined_group_types_check_with_generic_signatures_in_every_order() {
    let generic = "(?0, i32) -> ?0";
    for (program, signatures) in [
        (
            "def f(x, n: i32) = g(x, n)\n\n\
             def g(y, n: i32) = if eq(n, 0) then y else f(y, n - 1)\n\n\
             def main() -> (i32, string) = (f(1i32, 2), g(\"s\", 1))\n",
            vec![("f", generic), ("g", generic)],
        ),
        (
            "def f(x, n: i32) = if eq(n, 0) then g(x, n) else g(x, n - 1)\n\n\
             def g(y, n: i32) = if eq(n, 0) then y else f(y, n - 1)\n\n\
             def main() -> (i32, string) = (f(1i32, 2), g(\"s\", 1))\n",
            vec![("f", generic), ("g", generic)],
        ),
        (
            "def f(n: i32) = {\n  (p, q) = g(n)\n  (p, q)\n}\n\n\
             def g(n: i32) = if eq(n, 0) then (fn (e) -> (e, e))([]) else f(n - 1)\n\n\
             def main() -> i32 = 0i32\n",
            vec![
                ("f", "(i32) -> (List ?0, List ?0)"),
                ("g", "(i32) -> (List ?0, List ?0)"),
            ],
        ),
        (
            "def f(n: i32) = (g(n), g(n))\n\n\
             def g(n: i32) = if eq(n, 0) then 1i32 else f(n - 1).0\n\n\
             def main() -> (i32, i32) = f(2)\n",
            vec![("f", "(i32) -> (i32, i32)"), ("g", "(i32) -> i32")],
        ),
        // `ok3`.
        (
            "def a(x, n: i32) = if eq(n, 0) then x else {\n  k = c(3i32, n - 1)\n  b(x, n - 1)\n}\n\n\
             def b(y, n: i32) = if eq(n, 0) then y else c(y, n - 1)\n\n\
             def c(z, n: i32) = if eq(n, 0) then z else a(z, n - 1)\n\n\
             def main() -> (string, f32) = (a(\"s\", 3), c(1.5f32, 2))\n",
            vec![("a", generic), ("b", generic), ("c", generic)],
        ),
        // `sig3`.
        (
            "sig a: _ -> i32 -> _\ndef a(x, n) = if eq(n, 0) then x else {\n  k = b(2.5f32, n - 1)\n  x\n}\n\n\
             def b(y, n: i32) = if eq(n, 0) then y else {\n  k = a(y, n - 1)\n  y\n}\n\n\
             def main() -> (string, i32) = (a(\"s\", 2), b(4i32, 1))\n",
            vec![("a", generic), ("b", generic)],
        ),
        // `s3`.
        (
            "def f(x, n: i32) = if eq(n, 0) then x else {\n  k = f(to_tensor([1.0f32]), n - 1)\n  x\n}\n\n\
             def main() -> tensor[3, f32] = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), 2)\n",
            vec![("f", generic)],
        ),
        // `s4`: the group closes through a top-level value.
        (
            "def f(x, n: i32) = if eq(n, 0) then x else {\n  k = h(3i32, n - 1)\n  x\n}\n\n\
             h = fn (y, m) -> f(y, m)\n\n\
             def main() -> (string, i32) = (f(\"s\", 2), h(4i32, 1))\n",
            vec![("f", generic), ("h", "(i32, i32) -> i32")],
        ),
        // `s7`.
        (
            "def g(y, n: i32) = if eq(n, 0) then y else f(y, n - 1)\n\n\
             def f(x, n: i32) = if eq(n, 0) then x else {\n  k = g(3i32, n - 1)\n  g(x, n - 1)\n}\n\n\
             def main() -> (string, f32) = (f(\"s\", 3), g(1.5f32, 2))\n",
            vec![("f", generic), ("g", generic)],
        ),
        // `s8`.
        (
            "def f(x, n: i32) = if eq(n, 0) then x else g(x, n - 1)\n\n\
             def g(y, n: i32) = if eq(n, 0) then y else {\n  k = f(y, n - 1)\n  j = f(2.5f32, n - 1)\n  y\n}\n\n\
             def main() -> (string, i32) = (f(\"s\", 3), g(1i32, 2))\n",
            vec![("f", generic), ("g", generic)],
        ),
    ] {
        for order in reorderings(program) {
            accepts(&order);
            for (name, signature) in &signatures {
                assert_eq!(published(&order, name), *signature, "`{name}` in:\n{order}");
            }
        }
    }
}

/// REGRESSION TEST (fails on `c2ec7adca` and `375598343`, which reported
/// these as an instance neither own nor concrete, and on `main`, which
/// reported them as a recursive call at another instantiation). A group
/// whose types grow with every link never reaches a solution, and the bound
/// on linking rounds reports it as [04-INF-3] polymorphic recursion. This is
/// the test that reaches that bound.
#[test]
fn a_group_whose_types_grow_without_bound_reaches_the_round_bound() {
    for program in [
        "def f(n: i32) = if eq(n, 0) then [] else [f(n - 1)]\n\n\
         def main() -> i64 = len(f(2))\n",
        "def f(n: i32) = if eq(n, 0) then [] else [g(n - 1)]\n\n\
         def g(n: i32) = if eq(n, 0) then [] else [f(n - 1)]\n\n\
         def main() -> i64 = len(f(2))\n",
    ] {
        for order in declaration_orders(program) {
            rejects_with(&order, &["grows without bound", "[04-INF-2]", "[04-INF-3]"]);
        }
    }
}

// ---------------------------------------------------------------------------
// chelis#2626: `grad` decided on the group's types.

/// Round 3's `o1`: `g` differentiates a lambda whose result is `f`'s, which
/// only `f`'s body determines.
const GRAD_OF_SIBLING: &str = "def f(x, n) = if eq(n, 0i32) then mul(x, x) else g(x, n - 1i32)\n\n\
     def g(y: f32, n: i32) = if eq(n, 0) then y else {\n  d = grad(fn (z: f32) -> f(z, 0i32))\n  d(y)\n}\n\n\
     def main() -> f32 = f(3.0f32, 2i32)\n";

/// REGRESSION TEST (fails on `main`, `375598343` and `0db7e3c8e`, which
/// rejected round 3's `o1` with `g` declared first: `grad` decided that `f`'s
/// result, a variable until `f`'s body was inferred, was not a floating
/// scalar). chelis#2626: `grad` waits for a type that only the group
/// determines, so `o1` is accepted in every order and publishes its
/// written-type twin's signatures; the twin is accepted in every order on
/// every head (a lock). A result that is not a floating scalar is rejected in
/// every order with `grad`'s own diagnostic naming the type, where those heads
/// named a variable in one order.
#[test]
fn grad_decides_a_sibling_determined_output_on_the_groups_types() {
    let written = GRAD_OF_SIBLING
        .replace("def f(x, n) =", "def f(x: f32, n: i32) -> f32 =")
        .replace("def g(y: f32, n: i32) =", "def g(y: f32, n: i32) -> f32 =");
    for program in [GRAD_OF_SIBLING, written.as_str()] {
        for order in declaration_orders(program) {
            accepts(&order);
            for member in ["f", "g"] {
                assert_eq!(
                    published(&order, member),
                    "(f32, i32) -> f32",
                    "`{member}`:\n{order}"
                );
            }
        }
    }
    let tensor_result = "def f(x, n) = if eq(n, 0i32) then to_tensor([x, x]) else to_tensor([g(x, n - 1i32), x])\n\n\
         def g(y: f32, n: i32) -> f32 = if eq(n, 0) then y else {\n  d = grad(fn (z: f32) -> f(z, 0i32))\n  d(y)\n}\n\n\
         def main() -> f32 = g(3.0f32, 2i32)\n";
    for order in declaration_orders(tensor_result) {
        rejects_with(
            &order,
            &["grad requires a scalar floating output, got tensor"],
        );
    }
}

/// REGRESSION TEST (the acceptances fail on `main`, `375598343` and
/// `0db7e3c8e` with `g` declared first, where `grad` decided that `f`'s first
/// parameter, still a variable, was not differentiable, dropping it from the
/// gradient or rejecting `wrt=x`, and on `main` the integer programs got a
/// different diagnostic in each order). chelis#2626: a parameter of `f`'s
/// provisional type is decided on the type the group gives it, in every
/// order. When the group makes it an integer, the rule rejects `wrt=x`, and
/// without `wrt` leaves the parameter out of the gradient, in every order.
#[test]
fn grad_differentiates_a_sibling_determined_parameter_in_every_order() {
    for program in [
        "def g(y: f32, n: i32) -> f32 = if eq(n, 0i32) then y else grad(fn (z) -> f(z, 0i32))(y)\n\n\
         def f(x, n) = if eq(n, 0i32) then mul(x, mul(x, x)) else g(x, n - 1i32)\n\n\
         def main() -> f32 = f(3.0f32, 2i32)\n",
        "def g(y: f32, n: i32) -> f32 = if eq(n, 0i32) then y else grad(f, wrt=x)(y, 0i32)\n\n\
         def f(x, n) = if eq(n, 0i32) then mul(x, mul(x, x)) else g(x, n - 1i32)\n\n\
         def main() -> f32 = f(3.0f32, 2i32)\n",
    ] {
        for order in declaration_orders(program) {
            accepts(&order);
        }
    }
    for (program, fragment) in [
        (
            "def g(k: i32, n: i32) -> f32 = if eq(n, 0i32) then 1.0f32 else grad(f, wrt=x)(k, 0i32)\n\n\
             def f(x, n) = if eq(n, 0i32) then 2.0f32 else g(x, n - 1i32)\n\n\
             def main() -> f32 = f(3i32, 2i32)\n",
            "grad `wrt` index 0 is not differentiable",
        ),
        (
            "def g(k: i32, n: i32) -> f32 = if eq(n, 0i32) then 1.0f32 else grad(fn (z) -> f(z, 0i32))(k)\n\n\
             def f(x, n) = if eq(n, 0i32) then 2.0f32 else g(x, n - 1i32)\n\n\
             def main() -> f32 = f(3i32, 2i32)\n",
            "type mismatch: f32 vs ()",
        ),
    ] {
        for order in declaration_orders(program) {
            rejects_with(&order, &[fragment]);
        }
    }
}

/// chelis#2647: both float parameters contribute, including the lambda's
/// previously unknown second parameter. The authored scalar result is wrong
/// and must reject in every recursive declaration order.
#[test]
fn grad_decides_a_parameter_no_sibling_determines_where_it_is_inferred() {
    let program = "def g(y: f32, n: i32) -> f32 = if eq(n, 0i32) then y else grad(fn (z, w) -> f(z, 0i32))(y, y)\n\n\
         def f(x, n) = if eq(n, 0i32) then mul(x, mul(x, x)) else g(x, n - 1i32)\n\n\
         def main() -> f32 = f(3.0f32, 2i32)\n";
    let orders = declaration_orders(program);
    let first = outcome(&orders[0]);
    assert!(
        first.typed.is_err(),
        "the scalar signature must reject a tuple gradient"
    );
    for order in &orders[1..] {
        assert_eq!(outcome(order), first, "{order}");
    }
}

/// REGRESSION TEST (the rejections fail on `main`, `375598343` and
/// `0db7e3c8e`, which accepted both: `grad` of an operand whose type was a
/// variable published a fresh variable that nothing ever checked).
/// chelis#2626: the rule waits for the operand to bind and is decided then,
/// so a function whose result is a tensor is rejected with `grad`'s own
/// diagnostic, and an operand that never binds is rejected at the declaration
/// boundary. A function with a floating scalar result is accepted (a lock).
#[test]
fn grad_of_an_unresolved_operand_is_decided_when_it_binds() {
    let applied = |function: &str| {
        format!("def main() -> f32 = {{\n  k = fn (g) -> grad(g)(3.0f32)\n  k({function})\n}}\n")
    };
    accepts(&applied("fn (x: f32) -> mul(x, mul(x, x))"));
    rejects_with(
        &applied("fn (x: f32) -> to_tensor([x, x])"),
        &["grad requires a scalar floating output, got tensor"],
    );
    rejects_with(
        "def h(g) = grad(g)\n\ndef main() -> f32 = 1.0f32\n",
        &[
            "`grad` admits only some operand types",
            "never determined within `h`",
        ],
    );
}

/// Generic output instantiation remains chelis#2460. Parameter inference
/// with a known scalar output is admitted by chelis#2647, even when only
/// part of a recursive member's signature is authored.
#[test]
fn grad_waits_on_no_type_outside_a_provisional_group_type() {
    for program in [
        "def sq[a: Float](x: a) -> a = mul(x, mul(x, x))\n\ndef main() -> f32 = grad(sq)(3.0f32)\n",
        "def k[a: Float](x: a) -> a = grad(fn (z: a) -> mul(z, mul(z, z)))(x)\n\n\
         def main() -> f32 = k(3.0f32)\n",
        "def main() -> f32 = grad(fn (x) -> mul(x, mul(x, x)))(3.0f32)\n",
    ] {
        rejects_with(program, &["grad requires a scalar floating output"]);
    }
    let generic_output = "def g(y: f32, n: i32) -> f32 = if eq(n, 0i32) then y else grad(ev)(y, 0i32)\n\n\
             def ev(x, n: i32) = if eq(n, 0i32) then x else {\n  u = g(1.0f32, 0i32)\n  x\n}\n\n\
             def main() -> f32 = g(3.0f32, 1i32)\n";
    for order in declaration_orders(generic_output) {
        rejects_with(&order, &["grad requires a scalar floating output"]);
    }
    let inferred_parameter = "def g(y: f32, n: i32) -> f32 = if eq(n, 0i32) then y else grad(fn (z) -> f(z, 0i32))(y)\n\n\
             def f(x, n: i32) -> f32 = if eq(n, 0i32) then mul(x, mul(x, x)) else g(x, n - 1i32)\n\n\
             def main() -> f32 = f(3.0f32, 2i32)\n";
    for order in declaration_orders(inferred_parameter) {
        accepts(&order);
    }
}

/// Round 4's `tup`, `ten` and `lst`: `h` determines a variable nested in the
/// type of `f`'s first parameter, a tuple component, a tensor's precision or
/// a list element, and `g` differentiates `f`.
const GRAD_OF_NESTED: [&str; 3] = [
    "def fst2[a](p: (a, i32)) -> a = p.0\n\n\
     def f(x, n) = {\n  k = fst2(x)\n  if eq(n, 0i32) then 1.0f32 else add(g(n - 1i32), h(n - 1i32))\n}\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  d = grad(f)\n  r = d((2.0f32, 1i32), 0i32)\n  r.0\n}\n\n\
     def h(n: i32) -> f32 = f((2.0f32, 1i32), n)\n\n\
     def main() -> f32 = g(1)\n",
    "def first3[p](t: tensor[3, p]) -> tensor[3, p] = t\n\n\
     def f(x, n) = {\n  k = first3(x)\n  if eq(n, 0i32) then 1.0f32 else add(g(n - 1i32), h(n - 1i32))\n}\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  d = grad(f)\n  r = d(to_tensor([1.0f32, 2.0f32, 3.0f32]), 0i32)\n  tensor_to_scalar(sum(r, 0))\n}\n\n\
     def h(n: i32) -> f32 = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), n)\n\n\
     def main() -> f32 = g(1)\n",
    "def firstl[a](xs: List[a]) -> List[a] = xs\n\n\
     def f(x, n) = {\n  k = firstl(x)\n  if eq(n, 0i32) then 1.0f32 else add(g(n - 1i32), h(n - 1i32))\n}\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  d = grad(f)\n  r = d([1.0f32], 0i32)\n  index(r, 0i64)\n}\n\n\
     def h(n: i32) -> f32 = f([2.0f32], n)\n\n\
     def main() -> f32 = g(1)\n",
];

/// REGRESSION TEST (fails on `main` and `448018919`, which rejected every
/// program with `f`, `g` and `h` declared in that order: `grad` read the
/// variable nested in `f`'s parameter before `h` had determined it, decided
/// the parameter was not differentiable, and published `()` for its
/// gradient). chelis#2626: `grad` waits on a variable of the group anywhere in
/// a type it reads, so each program is accepted in every order, both where
/// `grad` is applied to `f` and to a lambda over it. `chelis eval` gives the
/// lambda forms 0.0 in every order and `chelis build --target c` builds the
/// tensor one and prints 0.0; the forms that apply `grad` to `f` itself run
/// out of memory in `chelis eval` in every order, as the typed `grad` of a
/// group member does on `main`.
#[test]
fn grad_waits_on_a_group_variable_nested_in_a_parameter() {
    for program in GRAD_OF_NESTED {
        let over_lambda = program.replace("grad(f)", "grad(fn (z) -> f(z, 0i32))");
        let over_lambda = over_lambda
            .replace("d((2.0f32, 1i32), 0i32)", "d((2.0f32, 1i32))")
            .replace(
                "d(to_tensor([1.0f32, 2.0f32, 3.0f32]), 0i32)",
                "d(to_tensor([1.0f32, 2.0f32, 3.0f32]))",
            )
            .replace("d([1.0f32], 0i32)", "d([1.0f32])");
        for source in [program, over_lambda.as_str()] {
            for order in declaration_orders(source) {
                accepts(&order);
            }
        }
    }
}

/// `g` differentiates a lambda whose parameter `p` holds a variable of the
/// group, `f`'s parameter type, beside `b`, which no sibling determines and
/// which the application after `grad` binds to `f32`.
const GRAD_BESIDE_A_LOCAL_VARIABLE: &str = "def fst2b[a, b](p: (a, b)) -> a = p.0\n\n\
     def f(x, n) = if eq(n, 0i32) then 1.0f32 else add(g(n - 1i32), h(n - 1i32))\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  d = grad(fn (p) -> f(fst2b(p), 0i32))\n  r = d((2.0f32, 3.0f32))\n  r.0\n}\n\n\
     def h(n: i32) -> f32 = f(2.0f32, n)\n\n\
     def main() -> f32 = g(1)\n";

/// `w`'s type is a variable no sibling determines, and the application
/// binds it to the component that `f`'s parameter type, a variable of the
/// group, resolves to.
const GRAD_BESIDE_A_LINKED_VARIABLE: &str = "def f(x, n) = if eq(n, 0i32) then 1.0f32 else add(g(n - 1i32), h(n - 1i32))\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  d = grad(fn (z, w) -> f(z, 0i32))\n  k = fn (u) -> d((u, 1i32), u)\n  r = k(2.0f32)\n  r.0\n}\n\n\
     def h(n: i32) -> f32 = f((2.0f32, 1i32), n)\n\n\
     def main() -> f32 = g(1)\n";

/// chelis#2647: local and group variables both use their settled types.
/// The local float component receives a float cotangent, and the separate
/// float parameter contributes a separate result slot.
#[test]
fn grad_decides_a_local_variable_where_it_is_inferred_beside_a_group_variable() {
    for program in [
        GRAD_BESIDE_A_LOCAL_VARIABLE.to_string(),
        GRAD_BESIDE_A_LOCAL_VARIABLE.replace("\n  r.0\n}", "\n  add(r.0, r.1)\n}"),
        GRAD_BESIDE_A_LINKED_VARIABLE.replace("\n  r.0\n}", "\n  add((r.0).0, r.1)\n}"),
    ] {
        for order in declaration_orders(&program) {
            accepts(&order);
        }
    }
    for order in declaration_orders(GRAD_BESIDE_A_LINKED_VARIABLE) {
        rejects_with(&order, &["TypeMismatch"]);
    }
}

// ---------------------------------------------------------------------------
// chelis#2651: `vmap` over a member whose signature is not written.

/// Round 4's `vA`, `vB`, `vC` and `vD`: `g` maps a lambda that calls `f`, or
/// `f` itself, and `f`'s parameter and result types are its provisional ones,
/// which `f`'s body determines.
const VMAP_OVER_A_SIGNATURE_LESS_MEMBER: [&str; 4] = [
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else {\n  k = g(n - 1i32)\n  1.0f32\n}\n\n\
     def g(n: i32) = if eq(n, 0) then to_tensor([0.0f32, 0.0f32]) else vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n\n\
     def main() -> tensor[2, f32] = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  add(v, 1.0f32)\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x) = {\n  k = g(0i32)\n  tensor_to_scalar(sum(x, 0))\n}\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(f)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
];

/// [04-INF-5] and spec/06 section 3.4: after the group finishes, both orders
/// use the same body-determined row function type. The first two programs
/// have a scalar row result and hence a tensor-valued mapped result.
#[test]
fn vmap_batches_body_determined_group_types_in_every_order() {
    for program in &VMAP_OVER_A_SIGNATURE_LESS_MEMBER[..2] {
        for order in declaration_orders(program) {
            accepts(&order);
        }
    }
}

/// [04-INF-1] and spec/06 section 3.4: a use of the mapped result cannot
/// settle the row result itself. `h`'s unknown result must be decided from
/// within `app`, or `app` must reject at its declaration boundary.
#[test]
fn vmap_of_unresolved_function_parameter_does_not_certify_a_scalar_result() {
    rejects_with(
        "def app(h) = {\n\
           v = vmap(fn (row: tensor[3, f32]) -> h(row))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n\
           add(v, 1.0f32)\n\
         }\n\n\
         def main() -> f32 = app(fn (row: tensor[3, f32]) -> tensor_to_scalar(sum(row, 0)))\n",
        &["vmap"],
    );
}

/// spec/06 section 3.4: an untyped row parameter receives one slice of the
/// batched actual, so the row reduction returns tensor[3], and mapping it
/// returns tensor[5,3]. Both a direct lambda and a local alias obey that rule.
#[test]
fn vmap_types_an_untyped_row_parameter_against_the_slice() {
    accepts(
        "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = \
           vmap(fn (v) -> sum(v, 0i32))(t)\n",
    );
    accepts(
        "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
           mapped = fn (v) -> sum(v, 0i32)\n\
           vmap(mapped)(t)\n\
         }\n",
    );
    accepts(
        "def probe(t: tensor[4, 5, 3, f32]) -> tensor[4, 5, 3, f32] = \
           vmap(fn (v) -> v, axis=1)(t)\n",
    );
    accepts(
        "def probe(t: tensor[4, 3, 5, f32]) -> tensor[4, 3, 5, f32] = \
           vmap(fn (v) -> v, axis=2)(t)\n",
    );
    rejects_with(
        "def probe(t: tensor[4, 5, 3, f32]) -> tensor[4, 5, 3, f32] = \
           vmap(fn (v) -> v, axis=3)(t)\n",
        &["vmap axis 3 is out of bounds"],
    );
}

/// [04-INF-1]: annotating a declaration's result does not bind an untyped
/// parameter of a lambda inside that declaration.
#[test]
fn vmap_result_annotation_does_not_infer_an_untyped_row_parameter() {
    rejects_with(
        "def make() -> (tensor[2, 3, f32] -> tensor[2, 3, f32]) = \
           vmap(fn (row) -> row)\n",
        &["`vmap` has an unresolved row parameter"],
    );
}

/// chelis#2651: the completed group determines `f`'s row type in either
/// declaration order. `vC` claims a scalar after adding to a mapped tensor
/// and must reject, while the three correctly typed forms accept.
#[test]
fn vmap_over_a_signature_less_member_uses_the_solved_row_type_in_every_order() {
    for (index, program) in VMAP_OVER_A_SIGNATURE_LESS_MEMBER.into_iter().enumerate() {
        let orders = declaration_orders(program);
        let first = outcome(&orders[0]);
        for order in &orders {
            if index == 2 {
                rejects_vmap_scalar_claim(order);
            } else {
                accepts(order);
            }
            assert_eq!(outcome(order), first, "{order}");
        }
    }
}

/// Round 5's witnesses: the mapped function reaches `f` through a binding
/// rather than in the operand's own syntax. A `let`-bound lambda that calls
/// `f` (its result summed, then added to a scalar), a `let` alias of `f`
/// itself, and a lambda that calls a member whose signature omits only its
/// result.
const VMAP_THROUGH_A_BINDING: [&str; 5] = [
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  k = fn (row: tensor[3, f32]) -> f(row, 0i32)\n  v = vmap(k)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  k = fn (row: tensor[3, f32]) -> f(row, 0i32)\n  v = vmap(k)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  add(v, 1.0f32)\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x) = {\n  k = g(0i32)\n  tensor_to_scalar(sum(x, 0))\n}\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  h = f\n  v = vmap(h)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  add(v, 1.0f32)\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x: tensor[3, f32], n: i32) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  add(v, 1.0f32)\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x: tensor[3, f32], n: i32) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
];

/// The mapped callable may pass through a local lambda or alias, and an
/// omitted result is as deferred as an omitted parameter. The scalar claims
/// after `add` are invalid; the row-sum cases accept in every order.
#[test]
fn vmap_over_a_type_its_group_has_yet_to_determine_resolves_through_any_binding() {
    for (index, program) in VMAP_THROUGH_A_BINDING.into_iter().enumerate() {
        let orders = declaration_orders(program);
        let first = outcome(&orders[0]);
        for order in &orders {
            if matches!(index, 1..=3) {
                rejects_vmap_scalar_claim(order);
            } else {
                accepts(order);
            }
            assert_eq!(outcome(order), first, "{order}");
        }
    }
}

/// A member whose signature is written, a function outside the group, a
/// lambda parameter named like the member, a local binding that shadows the
/// member's name, a lambda that calls the member but whose result does not
/// depend on it, and a `vmap` over a member once its group has completed.
const VMAP_OVER_A_FUNCTION_THE_GROUP_DOES_NOT_TYPE: [&str; 6] = [
    "def f(x: tensor[3, f32], n: i32) -> f32 = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else {\n  v = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def rowsum(x: tensor[3, f32]) = tensor_to_scalar(sum(x, 0))\n\n\
     def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then f(to_tensor([1.0f32, 1.0f32, 1.0f32]), 0i32) else {\n  v = vmap(fn (row: tensor[3, f32]) -> rowsum(row))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then f(to_tensor([1.0f32, 1.0f32, 1.0f32]), 0i32) else {\n  v = vmap(fn (f: tensor[3, f32]) -> tensor_to_scalar(sum(f, 0)))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then f(to_tensor([1.0f32, 1.0f32, 1.0f32]), 0i32) else {\n  f = fn (r: tensor[3, f32]) -> tensor_to_scalar(sum(r, 0))\n  v = vmap(fn (row: tensor[3, f32]) -> f(row))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then f(to_tensor([1.0f32, 1.0f32, 1.0f32]), 0i32) else {\n  v = vmap(fn (row: tensor[3, f32]) -> {\n    k = f(to_tensor([1.0f32, 1.0f32, 1.0f32]), 0i32)\n    tensor_to_scalar(sum(row, 0))\n  })(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n  tensor_to_scalar(sum(v, 0))\n}\n\n\
     def main() -> f32 = g(1)\n",
    "def f(x, n) = if eq(n, 0i32) then tensor_to_scalar(sum(x, 0)) else g(n - 1i32)\n\n\
     def g(n: i32) -> f32 = if eq(n, 0) then 0.0f32 else f(to_tensor([1.0f32, 2.0f32, 3.0f32]), n)\n\n\
     def main() -> tensor[2, f32] = vmap(fn (row: tensor[3, f32]) -> f(row, 0i32))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n",
];

/// REGRESSION TEST for the fifth program (fails on `4ea492b6e`, whose fence
/// read the operand's syntax and so rejected a lambda that calls `f` even
/// though `vmap` batches nothing `f` determines); a LOCK for the rest, which
/// `448018919` and `4ea492b6e` accept in every order (`main` rejected the
/// `f`-first order of the first four but the first, before the group rule).
/// The fence reads the types `vmap` batches, so none of these is fenced.
/// `chelis eval` and `chelis build --target c` agree on each.
#[test]
fn vmap_over_a_function_the_group_does_not_type_is_not_fenced() {
    for program in VMAP_OVER_A_FUNCTION_THE_GROUP_DOES_NOT_TYPE {
        for order in declaration_orders(program) {
            accepts(&order);
        }
    }
}

// ---------------------------------------------------------------------------
// The order-independence oracle.

/// What the oracle compares across declaration orders: the verdict, each
/// diagnostic's kind with the atoms its message cites, and every published
/// signature, on both checker ingresses. Spans and messages are left out:
/// they name positions and declarations, which move with the order.
#[derive(Debug, PartialEq)]
struct Outcome {
    typed: Result<Vec<(String, String)>, Vec<Kind>>,
    ir: Result<(), Vec<Kind>>,
}

/// A diagnostic's kind and the atoms its message cites.
type Kind = (String, Vec<String>);

/// The spec atoms (`[04-INF-3]`) a message cites, in order.
fn atoms(message: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (start, _) in message.match_indices('[') {
        let rest = &message[start + 1..];
        let Some(end) = rest.find(']') else {
            continue;
        };
        let candidate = &rest[..end];
        let parts: Vec<&str> = candidate.split('-').collect();
        if let [chapter, family, number] = parts.as_slice()
            && chapter.len() == 2
            && chapter.chars().all(|c| c.is_ascii_digit())
            && !family.is_empty()
            && family.chars().all(|c| c.is_ascii_uppercase())
            && !number.is_empty()
            && number.chars().all(|c| c.is_ascii_digit())
        {
            found.push(candidate.to_string());
        }
    }
    found
}

fn kinds(errors: &[CheckError]) -> Vec<Kind> {
    let mut kinds = errors
        .iter()
        .map(|error| (format!("{:?}", error.kind), atoms(&error.message)))
        .collect::<Vec<_>>();
    kinds.sort();
    kinds
}

fn outcome(source: &str) -> Outcome {
    let desugared = desugared(source);
    let typed = match check_typed_program(&desugared) {
        Ok(checked) => {
            let mut signatures = checked
                .signature_inference()
                .functions
                .values()
                .map(|function| {
                    (
                        function.name.clone(),
                        normalized(&function.checked_signature.to_string()),
                    )
                })
                .collect::<Vec<_>>();
            signatures.sort();
            Ok(signatures)
        }
        Err(result) => Err(kinds(&result.errors)),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Ok(()),
        Err(result) => Err(kinds(&result.errors)),
    };
    Outcome { typed, ir }
}

/// A program's top-level declarations, split at every line that starts in
/// the first column other than a closing bracket. A `sig` travels with the
/// `def` after it.
fn top_level_declarations(program: &str) -> Vec<String> {
    let mut declarations: Vec<String> = Vec::new();
    let mut signature: Option<String> = None;
    for line in program.lines() {
        let starts = line
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace() && !matches!(c, '}' | ')' | ']'));
        if starts {
            if let Some(last) = declarations.last()
                && last.starts_with("sig ")
                && !last.contains('\n')
                && line.starts_with("def ")
            {
                signature = declarations.pop();
            }
            let mut declaration = signature.take().map(|sig| sig + "\n").unwrap_or_default();
            declaration.push_str(line);
            declarations.push(declaration);
        } else if let Some(last) = declarations.last_mut() {
            last.push('\n');
            last.push_str(line);
        }
    }
    declarations
        .into_iter()
        .map(|declaration| declaration.trim_end().to_string())
        .filter(|declaration| !declaration.is_empty())
        .collect()
}

/// The name a top-level declaration binds, and the text after its header.
fn binding(declaration: &str) -> Option<(String, &str)> {
    let header = declaration.lines().find(|line| !line.starts_with("sig "))?;
    let offset = declaration.find(header)?;
    let text = &declaration[offset..];
    let name_start = if let Some(rest) = header.strip_prefix("def ") {
        header.len() - rest.len()
    } else if header.starts_with("type ") || header.starts_with("import ") {
        return None;
    } else {
        0
    };
    let name: String = header[name_start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let body = text.find(" = ").map(|at| &text[at + 3..])?;
    (!name.is_empty()).then_some((name, body))
}

/// Whether `text` mentions the identifier `name`, not as part of a longer
/// identifier or as a field.
fn mentions(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        !before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            && !after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// The top-level declarations that lie on a reference cycle: the members of
/// the program's recursive groups, including a component closed through a
/// value.
fn cycle_members(declarations: &[String]) -> BTreeSet<usize> {
    let bindings = declarations
        .iter()
        .enumerate()
        .filter_map(|(index, declaration)| {
            binding(declaration).map(|(name, body)| (index, name, body.to_string()))
        })
        .collect::<Vec<_>>();
    let edges = bindings
        .iter()
        .map(|(_, _, body)| {
            bindings
                .iter()
                .enumerate()
                .filter(|(_, (_, name, _))| mentions(body, name))
                .map(|(to, _)| to)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (0..bindings.len())
        .filter(|start| {
            let mut seen = BTreeSet::new();
            let mut stack = edges[*start].clone();
            while let Some(next) = stack.pop() {
                if next == *start {
                    return true;
                }
                if seen.insert(next) {
                    stack.extend(&edges[next]);
                }
            }
            false
        })
        .map(|member| bindings[member].0)
        .collect()
}

fn has_recursive_group(program: &str) -> bool {
    !cycle_members(&top_level_declarations(program)).is_empty()
}

/// Every order the oracle checks a program in: the `def`s on a reference
/// cycle are permuted among their own positions, every permutation of up to
/// four and every rotation and the reversal of more, and every other
/// declaration keeps its place. A value keeps its place because it is
/// visible only after its declaration ([04-INF-4]).
fn reorderings(program: &str) -> Vec<String> {
    let declarations = top_level_declarations(program);
    let members = cycle_members(&declarations)
        .into_iter()
        .filter(|index| {
            declarations[*index]
                .lines()
                .find(|line| !line.starts_with("sig "))
                .is_some_and(|line| line.starts_with("def "))
        })
        .collect::<Vec<_>>();
    let movable = members
        .iter()
        .map(|index| declarations[*index].as_str())
        .collect::<Vec<_>>();
    let mut orders = Vec::new();
    if movable.len() <= 4 {
        permute(&movable, &mut Vec::new(), &mut orders);
    } else {
        for rotation in 0..movable.len() {
            let mut order = movable.clone();
            order.rotate_left(rotation);
            orders.push(order);
        }
        orders.push(movable.iter().rev().copied().collect());
    }
    orders
        .into_iter()
        .map(|order| {
            let mut placed = declarations.iter().map(String::as_str).collect::<Vec<_>>();
            for (slot, declaration) in members.iter().zip(order) {
                placed[*slot] = declaration;
            }
            placed.join("\n\n") + "\n"
        })
        .collect()
}

/// The contents of every string literal in a Rust source file, unescaped.
/// Comments, character literals and lifetimes are skipped.
fn string_literals(source: &str) -> Vec<String> {
    let chars: Vec<char> = source.chars().collect();
    let mut literals = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        match chars[at] {
            '/' if chars.get(at + 1) == Some(&'/') => {
                while at < chars.len() && chars[at] != '\n' {
                    at += 1;
                }
            }
            '\'' => {
                // A character literal is `'x'` or an escape `'\..'`; anything
                // else is a lifetime.
                if chars.get(at + 1) == Some(&'\\') {
                    at += 2;
                    while at < chars.len() && chars[at] != '\'' {
                        at += 1;
                    }
                    at += 1;
                } else if chars.get(at + 2) == Some(&'\'') {
                    at += 3;
                } else {
                    at += 1;
                }
            }
            '"' => {
                let mut literal = String::new();
                at += 1;
                while at < chars.len() && chars[at] != '"' {
                    if chars[at] == '\\' {
                        at += 1;
                        match chars.get(at) {
                            Some('n') => literal.push('\n'),
                            Some('t') => literal.push('\t'),
                            Some('0') => literal.push('\0'),
                            Some('\n') => {
                                while chars.get(at + 1).is_some_and(|c| c.is_whitespace()) {
                                    at += 1;
                                }
                            }
                            Some('u') => {
                                let close = chars[at..]
                                    .iter()
                                    .position(|c| *c == '}')
                                    .map_or(at, |offset| at + offset);
                                let digits: String = chars[at + 2..close].iter().collect();
                                if let Some(c) = u32::from_str_radix(&digits, 16)
                                    .ok()
                                    .and_then(char::from_u32)
                                {
                                    literal.push(c);
                                }
                                at = close;
                            }
                            Some(other) => literal.push(*other),
                            None => {}
                        }
                    } else {
                        literal.push(chars[at]);
                    }
                    at += 1;
                }
                literals.push(literal);
                at += 1;
            }
            _ => at += 1,
        }
    }
    literals
}

/// Every recursive-group fixture the oracle can find: the programs written in
/// this file and in `issue_731_declaration_close_obligations.rs`, and the
/// review rounds' group probes in `fixtures/recursive_group_orders.txt`, each
/// named for where it came from.
fn group_fixtures() -> Vec<(String, String)> {
    let mut fixtures = Vec::new();
    for (file, source) in [
        (
            "issue_2590_recursive_group_holes.rs",
            include_str!("issue_2590_recursive_group_holes.rs"),
        ),
        (
            "issue_731_declaration_close_obligations.rs",
            include_str!("issue_731_declaration_close_obligations.rs"),
        ),
    ] {
        for (index, literal) in string_literals(source).into_iter().enumerate() {
            let program = literal.trim_start();
            if (program.starts_with("def ")
                || program.starts_with("type ")
                || program.starts_with("sig "))
                && parse_surf(program).is_ok()
                && has_recursive_group(program)
            {
                fixtures.push((format!("{file} literal {index}"), program.to_string()));
            }
        }
    }
    let probes = include_str!("fixtures/recursive_group_orders.txt");
    for section in probes.split("==== ").filter(|section| !section.is_empty()) {
        let (name, program) = section
            .split_once('\n')
            .expect("a fixture is a `==== name` line and a program");
        assert!(
            parse_surf(program).is_ok() && has_recursive_group(program),
            "fixture `{name}` must parse and hold a recursive group:\n{program}"
        );
        fixtures.push((name.to_string(), program.to_string()));
    }
    fixtures
}

/// ORACLE (fails on `375598343` for round 3's `tw`, `bad3` and `nm3`, which it
/// accepted in some declaration orders and rejected in others, and on
/// `0db7e3c8e` for round 3's `o1` and the chelis#2626 programs above, whose
/// `grad` it decided before a sibling had determined its types). Every
/// recursive-group fixture gets one outcome in every declaration order: the
/// same verdict, the same diagnostic kinds citing the same atoms, and the same
/// published signatures, on both checker ingresses. No fixture is exempt.
#[test]
fn every_group_fixture_has_one_outcome_in_every_declaration_order() {
    let fixtures = group_fixtures();
    assert!(
        fixtures.len() > 200,
        "the oracle must find the fixtures it runs over; it found {}",
        fixtures.len()
    );
    let mut divergent = Vec::new();
    let mut reordered = 0;
    for (name, program) in &fixtures {
        let orders = reorderings(program);
        assert!(
            orders.iter().all(|order| parse_surf(order).is_ok()),
            "`{name}`: every reordering of a fixture must parse:\n{program}"
        );
        reordered += usize::from(orders.len() > 1);
        let first = outcome(&orders[0]);
        let divergence = orders[1..]
            .iter()
            .map(|order| (order, outcome(order)))
            .find(|(_, other)| *other != first);
        if let Some((order, other)) = divergence {
            divergent.push(format!(
                "`{name}`:\n{}\n  {first:?}\nversus\n{order}\n  {other:?}",
                orders[0]
            ));
        }
    }
    // A self-recursive fixture has one order; the oracle's force is in the
    // groups of two or more members.
    assert!(
        reordered > 100,
        "the oracle must reorder the groups it finds; it reordered {reordered}"
    );
    assert!(
        divergent.is_empty(),
        "{} of {} group fixtures get a different outcome in another declaration order:\n\n{}",
        divergent.len(),
        fixtures.len(),
        divergent.join("\n\n")
    );
}
