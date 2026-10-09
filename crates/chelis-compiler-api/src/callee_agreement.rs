//! [04-INF-9]: a lexical binding shadows every def of the same spelling, so
//! evaluation and the compiled C lane call the binding, or the C lane refuses
//! the program loudly; it never calls the def (chelis#3484). The C lane
//! previously called the def and printed its result without any diagnostic.
//!
//! These are default-feature unit tests, so they run on every pull request,
//! not only with the ownership-ledger targets.

use crate::compiler::{compile, eval_selected};
use crate::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::fs;

const SHADOWING: &str = "def mix(k: key, n: i64) -> key = k\n\
def seed(x: i64) -> key = key_from_seed(0i64)\n\
def main() = {\n  \
  mix: key -> i64 -> key = fold_in\n  \
  seed: i64 -> key = key_from_seed\n  \
  (mix(seed(7i64), -1i64), map(seed, [1i64, -1i64]))\n\
}\n";

/// `main.N = <display>` for each root of `main`, as evaluation prints it.
fn evaluated_lines(source: &str) -> Vec<String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .unwrap_or_else(|error| panic!("evaluate: {error:?}"));
    result
        .roots
        .iter()
        .enumerate()
        .map(|(index, root)| {
            format!(
                "main.{index} = {}",
                root.display.as_deref().expect("evaluated root display")
            )
        })
        .collect()
}

/// The debug rendering of the compile error for `source`.
fn compile_error(source: &str) -> String {
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("agreement".into()),
    })
    .expect_err("the C lane must refuse this program");
    format!("{error:?}")
}

/// The standard output of the compiled C program for `source`.
fn compiled_stdout(source: &str) -> String {
    let artifact = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("agreement".into()),
    })
    .unwrap_or_else(|error| panic!("compile: {error:?}"));
    let dir = tempfile::tempdir().expect("temporary build directory");
    for file in &artifact.files {
        fs::write(dir.path().join(&file.path), &file.contents).expect("write generated file");
    }
    let staged = chelis_runtime_bundle::stage(dir.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    let source_path = dir.path().join("agreement.c");
    let binary = dir.path().join("agreement");
    let toolchain = chelis_backend_c::toolchain::test_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements::default(),
    );
    let built = chelis_backend_c::toolchain::tool_command(&toolchain.compiler)
        .arg("-I")
        .arg(dir.path())
        .args(chelis_backend_c::toolchain::link_args(
            &artifact.compile_flags,
            &[source_path.as_os_str(), staged.archive.as_os_str()],
            &artifact.link_flags,
            binary.as_os_str(),
        ))
        .output()
        .expect("run the C compiler");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = std::process::Command::new(&binary)
        .output()
        .expect("run the compiled program");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 program output")
}

/// Require `source`'s compiled C to print every root evaluation prints, and
/// evaluation to print `expected`.
fn assert_eval_and_c_agree(source: &str, expected: &[&str]) {
    let evaluated = evaluated_lines(source);
    assert_eq!(evaluated, expected, "evaluation");
    let compiled = compiled_stdout(source);
    for line in &evaluated {
        assert!(
            compiled.lines().any(|printed| printed == line),
            "compiled C must print `{line}`:\n{compiled}"
        );
    }
}

#[test]
fn key_alias_shadowing_a_def_agrees_in_eval_and_compiled_c() {
    // Evaluation calls the aliases: fold_in(7, -1) and key_from_seed of each
    // item.
    assert_eval_and_c_agree(
        SHADOWING,
        &[
            "main.0 = key(45c80b557fb94ddb)",
            "main.1 = [key(0000000000000001), key(ffffffffffffffff)]",
        ],
    );
}

/// Unannotated and pattern-bound key-builtin aliases shadow a def in a
/// direct call, which applies the operation the alias carries.
#[test]
fn unannotated_key_aliases_shadowing_a_def_agree_on_direct_calls() {
    let expected = [
        "main.0 = key(0000000000000001)",
        "main.1 = key(b2fe04bac90dd534)",
    ];
    for source in [
        "def seed(x: i64) -> key = key_from_seed(0i64)\n\
         def mix(k: key, n: i64) -> key = k\n\
         def go() -> (key, key) = {\n  \
           seed = key_from_seed\n  \
           mix = fold_in\n  \
           (seed(1i64), mix(seed(1i64), 2i64))\n\
         }\n\
         def main() = go()\n",
        "def seed(x: i64) -> key = key_from_seed(0i64)\n\
         def mix(k: key, n: i64) -> key = k\n\
         def go() -> (key, key) = {\n  \
           (seed, mix, n) = (key_from_seed, fold_in, 1i64)\n  \
           (seed(n), mix(seed(n), 2i64))\n\
         }\n\
         def main() = go()\n",
    ] {
        assert_eval_and_c_agree(source, &expected);
    }
}

/// An unannotated or pattern-bound key-builtin alias has no function type
/// for a C loop callback. When it shadows a def, the C lane refuses the
/// program exactly as it does when no def has that spelling, rather than
/// calling the def.
#[test]
fn unannotated_key_alias_shadowing_a_def_is_refused_as_a_c_loop_callback() {
    for (source, operation) in [
        (
            "def seed(x: i64) -> key = key_from_seed(0i64)\n\
             def go() -> List[key] = {\n  \
               seed = key_from_seed\n  \
               map(seed, [2i64])\n\
             }\n\
             def main() = (go(), 1i64)\n",
            "map",
        ),
        (
            "def mix(k: key, n: i64) -> key = k\n\
             def go() -> key = {\n  \
               mix = fold_in\n  \
               fold(mix, key_from_seed(1i64), [2i64, 3i64])\n\
             }\n\
             def main() = (go(), 1i64)\n",
            "fold",
        ),
        (
            "def seed(x: i64) -> key = key_from_seed(0i64)\n\
             def go() -> List[key] = {\n  \
               (seed, n) = (key_from_seed, 2i64)\n  \
               map(seed, [n])\n\
             }\n\
             def main() = (go(), 1i64)\n",
            "map",
        ),
        (
            "def mix(k: key, n: i64) -> key = k\n\
             def go() -> key = {\n  \
               (mix, n) = (fold_in, 1i64)\n  \
               fold(mix, key_from_seed(n), [2i64, 3i64])\n\
             }\n\
             def main() = (go(), 1i64)\n",
            "fold",
        ),
    ] {
        let error = compile_error(source);
        assert!(
            error.contains(&format!("`{operation}` requires a lowerable callback")),
            "{error}"
        );
    }
}

/// A local alias of one def that shadows another def is not a callable the
/// C lane admits, so it is refused like an unshadowed alias of a def.
#[test]
fn local_def_alias_shadowing_a_def_is_refused_in_c() {
    let error = compile_error(
        "def inc(x: i64) -> i64 = x + 1i64\n\
         def dbl(x: i64) -> i64 = x * 2i64\n\
         def main() = {\n  \
           inc: i64 -> i64 = dbl\n  \
           (inc(3i64), 1i64)\n\
         }\n",
    );
    assert!(
        error.contains("ownership lowering in `main` names unknown callee `inc`"),
        "{error}"
    );
}

/// Shadowing bindings in higher-order positions: a def parameter used as a
/// map callback, a closure capture, an alias passed to higher-order defs, a
/// shadowed recursive def, and a def call that precedes the shadowing alias.
#[test]
fn shadowing_bindings_in_higher_order_positions_agree_in_eval_and_compiled_c() {
    assert_eval_and_c_agree(
        "def inc(x: i64) -> i64 = x + 1i64\n\
         def twice(x: i64) -> i64 = x * 2i64\n\
         def apply_all(inc: i64 -> i64, xs: List[i64]) -> List[i64] = map(inc, xs)\n\
         def seed(n: i64) -> key = if n <= 0i64 then key_from_seed(0i64) else seed(n - 1i64)\n\
         def apply(f: i64 -> key, x: i64) -> key = f(x)\n\
         def apply_keys(f: i64 -> key, xs: List[i64]) -> List[key] = map(f, xs)\n\
         def main() = {\n  \
           before = seed(1i64)\n  \
           seed: i64 -> key = key_from_seed\n  \
           (apply_all(twice, [1i64, 2i64, 3i64]), seed(4i64), map(fn (x: i64) -> seed(x), [4i64, 5i64]), apply(seed, 3i64), apply_keys(seed, [1i64, 2i64]), before)\n\
         }\n",
        &[
            "main.0 = [2, 4, 6]",
            "main.1 = key(0000000000000004)",
            "main.2 = [key(0000000000000004), key(0000000000000005)]",
            "main.3 = key(0000000000000003)",
            "main.4 = [key(0000000000000001), key(0000000000000002)]",
            "main.5 = key(0000000000000000)",
        ],
    );
}

/// A typed key-builtin alias shadows a def of every lowering shape, in a
/// direct call and as a `map` callback. Each shape has its own definition
/// route in application lowering; none may see the lexical callee.
#[test]
fn typed_key_alias_shadows_every_def_shape_in_eval_and_compiled_c() {
    for (shape, def) in [
        (
            "monomorphic",
            "def seed(x: i64) -> key = key_from_seed(100i64)\n",
        ),
        (
            "generic",
            "def seed[t](x: t) -> key = key_from_seed(100i64)\n",
        ),
        (
            "precision-polymorphic",
            "def seed[p: Float](x: p) -> key = key_from_seed(100i64)\n",
        ),
        (
            "rank-polymorphic",
            "def seed[r](x: tensor[..r, i64]) -> key = key_from_seed(100i64)\n",
        ),
        (
            "nullary generic constructor wrapper",
            "type Box[a] =\n  | Empty\n  | Full { value: a }\ndef seed[a]() -> Box[a] = Empty\n",
        ),
        (
            "callable-parameter",
            "def seed(f: i64 -> key, xs: List[i64]) -> List[key] = map(f, xs)\n",
        ),
        (
            "summary-dispatched",
            "def seed(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, 2, f32] = matmul(a, b)\n",
        ),
        (
            "recursive",
            "def seed(n: i64) -> key = if n <= 0i64 then key_from_seed(100i64) else seed(n - 1i64)\n",
        ),
        (
            "recursive generic",
            "def seed[t](x: t, n: i64) -> key = if n <= 0i64 then key_from_seed(100i64) else seed(x, n - 1i64)\n",
        ),
    ] {
        let source = format!(
            "{def}def go() -> (key, List[key]) = {{\n  \
               seed: i64 -> key = key_from_seed\n  \
               (seed(7i64), map(seed, [-1i64]))\n\
             }}\n\
             def main() = go()\n"
        );
        let evaluated = evaluated_lines(&source);
        assert_eq!(
            evaluated,
            [
                "main.0 = key(0000000000000007)",
                "main.1 = [key(ffffffffffffffff)]",
            ],
            "evaluation, {shape} def"
        );
        let compiled = compiled_stdout(&source);
        for line in &evaluated {
            assert!(
                compiled.lines().any(|printed| printed == line),
                "{shape} def: compiled C must print `{line}`:\n{compiled}"
            );
        }
    }
}

/// The reverse direction: a def whose name a compiler-generated binder also
/// takes stays the callee. Positional callback parameters (`arg0`, `arg1`),
/// the parameters of a point-free binding, and a staged alias renamed to its
/// def's spelling inside a scope that binds that spelling must not capture
/// the def.
#[test]
fn defs_named_like_compiler_binders_stay_the_callee_in_eval_and_compiled_c() {
    for (shape, source, expected) in [
        (
            "map callback named arg0",
            "def arg0(x: i64) -> i64 = x + 1i64\n\
             def main() = (map(arg0, [1i64, 2i64]), 1i64)\n",
            "main.0 = [2, 3]",
        ),
        (
            "fold callback named arg1",
            "def arg1(acc: i64, x: i64) -> i64 = acc + x\n\
             def main() = (fold(arg1, 0i64, [1i64, 2i64, 3i64]), 1i64)\n",
            "main.0 = 6",
        ),
        (
            "fold callback named arg0",
            "def arg0(acc: i64, x: i64) -> i64 = acc * 10i64 + x\n\
             def main() = (fold(arg0, 0i64, [1i64, 2i64, 3i64]), 1i64)\n",
            "main.0 = 123",
        ),
        (
            "point-free binding of arg0",
            "def arg0(x: i64) -> i64 = x + 1i64\n\
             h: i64 -> i64 = arg0\n\
             def main() = (h(4i64), 1i64)\n",
            "main.0 = 5",
        ),
        (
            "point-free binding of arg1",
            "def arg1(a: i64, b: i64) -> i64 = (a * 10i64) + b\n\
             h: i64 -> i64 -> i64 = arg1\n\
             def main() = (h(4i64, 2i64), 1i64)\n",
            "main.0 = 42",
        ),
        (
            "point-free binding as a map callback",
            "def arg0(x: i64) -> i64 = x + 1i64\n\
             h: i64 -> i64 = arg0\n\
             def main() = (map(h, [4i64, 5i64]), 1i64)\n",
            "main.0 = [5, 6]",
        ),
        (
            "staged alias renamed over a local",
            "def halve(e: i64) -> i64 = floor_div(e, 2i64)\n\
             def go[n](x: tensor[n, f32]) -> tensor[2, 2, f32] = {\n  \
               g = halve\n  \
               halve = numel(x)\n  \
               reshape(x, [g(halve), 2i64])\n\
             }\n\
             def main() = (go(to_tensor([1.0, 2.0, 3.0, 4.0], f32)), 1i64)\n",
            "main.0 = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
        ),
    ] {
        let evaluated = evaluated_lines(source);
        assert_eq!(evaluated, [expected, "main.1 = 1"], "evaluation, {shape}");
        let compiled = compiled_stdout(source);
        for line in &evaluated {
            assert!(
                compiled.lines().any(|printed| printed == line),
                "{shape}: compiled C must print `{line}`:\n{compiled}"
            );
        }
    }
}
