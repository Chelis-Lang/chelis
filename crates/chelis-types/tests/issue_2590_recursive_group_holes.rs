//! chelis#2590: a recursive group whose members omit types is typed by
//! [04-INF-2] and [04-INF-5] together.
//!
//! At an in-group reference, each authored binder of the callee is the
//! caller's own, and each omitted type is instantiated afresh. When the group
//! completes, each instance must be the member's own type (it unified with it,
//! or stayed unconstrained and is chosen as it) or a fully concrete type;
//! anything else is [04-INF-3]'s polymorphic recursion. The reference is typed
//! at the type the member's body determines, which it never narrows, and the
//! member's omitted types generalize once the group completes.
//!
//! Each test is labelled with the heads it fails on: `main` (`a116a9e10`, the
//! pull request's base, which instantiated every omitted type afresh and never
//! tied the instance to the body) or the round-3 head `c2ec7adca` (which typed every in-group
//! reference at one monomorphic instantiation). A fixture that fails on
//! neither is a lock, and says so.

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
