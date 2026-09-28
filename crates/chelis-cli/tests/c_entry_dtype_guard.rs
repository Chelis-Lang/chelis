//! chelis#2490, [04-NUM-11]: a generated C public entry compares each supplied
//! tensor's dtype tag with the declared parameter dtype before it reads any
//! element. A mismatch traps `Domain` in `load` at the declared dtype under
//! [04-NUM-9], after a context line naming the input and both dtypes. It is
//! never reinterpreted: before this guard every mismatched row below ran to
//! completion, reading the supplied storage at the declared dtype.
//!
//! One program carries every host entry kind that takes a tensor: a tensor
//! result, a tuple result, a record result, and a tensor nested in a tuple,
//! record, data-type or list parameter. The direct four-argument DAG entry is
//! covered beside the other direct-entry guards in
//! `chelis-backend-c/tests/exec_compile.rs`.
//!
//! chelis#2506: a nested tensor gets the null, dtype, rank and extent checks a
//! tensor parameter gets, before the body reads it, and each names the
//! parameter path (`p.1`, `r.square`, `c.Ints.0`, `xs[1]`).
//!
//! A kernel the entry calls carries its own guard, prefixed with the kernel's
//! name, so an entry that reaches a kernel would still trap without the host
//! entry's guard. The kernel-free rows (a pass-through, a `debug`-only body, a
//! tuple of the parameter) reach no kernel, and every direct row must print
//! the host rendering, a line with no function-name prefix.

mod common;

use assert_cmd::Command;
use common::{authored_c_symbol, gcc_available, link_generated};
use std::fs;
use std::path::Path;

const SOURCE: &str = "def ints(x: tensor[3, i64]) -> tensor[3, i64] = x + x\n\
def floats(x: tensor[3, f64]) -> tensor[3, f64] = x + x\n\
def singles(x: tensor[3, f32]) -> tensor[3, f32] = x + x\n\
def int_pair(x: tensor[3, i64]) -> (tensor[3, i64], tensor[3, i64]) = (x + x, x * x)\n\
type IntRecord =\n  | IntRecord { twice: tensor[3, i64], square: tensor[3, i64] }\n\
def int_record(x: tensor[3, i64]) -> IntRecord = IntRecord { twice: x + x, square: x * x }\n\
def nested(p: (tensor[3, i64], tensor[3, i64])) -> tensor[3, i64] = p.0 + p.1\n\
def pass_through(x: tensor[3, i64]) -> tensor[3, i64] = x\n\
def shown(x: tensor[3, i64]) -> tensor[3, i64] ! {IO} = debug(x)\n\
def pass_pair(x: tensor[3, i64]) -> (tensor[3, i64], tensor[3, i64]) = (x, x)\n\
def mixed(p: (tensor[3, i64], tensor[3, i64])) -> tensor[3, i64] ! {IO} = debug(p.0) + p.1\n\
def record_sum(r: IntRecord) -> tensor[3, i64] = r.twice + r.square\n\
type Choice =\n  | Ints(tensor[3, i64])\n  | Floats(tensor[2, f64])\n\
def choice_total(c: Choice) -> i64 = match c with {\n  | Ints(t) => tensor_to_scalar(sum(t, 0))\n  | Floats(u) => cast(0, i64)\n}\n\
def named_pair[n](p: (tensor[n, i64], tensor[n, i64])) -> tensor[n, i64] = p.0 + p.1\n\
def list_count(xs: List[tensor[3, i64]]) -> i64 = len(xs)\n\
type Tree =\n  | Leaf(tensor[3, i64])\n  | Node(Tree, Tree)\n\
def tree_leaves(t: Tree) -> i64 = match t with {\n  | Leaf(x) => 1i64\n  | Node(l, r) => tree_leaves(l) + tree_leaves(r)\n}\n\
def list_walk(xs: List[tensor[3, i64]], i: i64, acc: i64) -> i64 = if (i >= len(xs)) then acc else list_walk(xs, i + 1i64, acc + 1i64)\n";

/// How the harness hands one supplied tensor to one entry and releases the
/// result. The C local `x` is the supplied tensor.
#[derive(Clone, Copy)]
enum Entry {
    Tensor(&'static str),
    Tuple(&'static str),
    Record(&'static str),
    NestedInTuple(&'static str),
}

impl Entry {
    fn call(self) -> String {
        match self {
            Entry::Tensor(name) => format!(
                "chelis_tensor *result = {}(x);\nchelis_tensor_release(result);",
                authored_c_symbol(name)
            ),
            Entry::Tuple(name) => format!(
                "chelis_tuple *result = {}(x);\nchelis_tuple_release(result);",
                authored_c_symbol(name)
            ),
            Entry::Record(name) => format!(
                "chelis_adt *result = {}(x);\nchelis_adt_release(result);",
                authored_c_symbol(name)
            ),
            Entry::NestedInTuple(name) => format!(
                "chelis_tensor_retain(x); chelis_tensor_retain(x);\n\
                 chelis_value items[2] = {{ chelis_value_take_tensor(x), chelis_value_take_tensor(x) }};\n\
                 chelis_tuple *p = chelis_tuple_from_values(items, 2);\n\
                 chelis_value_release(items[0]); chelis_value_release(items[1]);\n\
                 chelis_tensor *result = {}(p);\n\
                 chelis_tensor_release(result); chelis_tuple_release(p);",
                authored_c_symbol(name)
            ),
        }
    }
}

fn build(dir: &Path) -> std::path::PathBuf {
    build_source(dir, SOURCE).unwrap_or_else(|stderr| panic!("{stderr}"))
}

/// Build `source` to C in `dir`, returning the output directory or the
/// compiler's stderr.
fn build_source(dir: &Path, source: &str) -> Result<std::path::PathBuf, String> {
    let path = dir.join("entry.ch");
    fs::write(&path, source).expect("source");
    let out = dir.join("c");
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&out)
        .output()
        .expect("build C");
    if built.status.success() {
        Ok(out)
    } else {
        Err(String::from_utf8_lossy(&built.stderr).into_owned())
    }
}

/// Link the C `harness` against the generated program in `out`, run it, and
/// return (exit success, stdout plus stderr).
fn link_and_run(out: &Path, name: &str, harness: &str) -> (bool, String) {
    let file = format!("{name}.c");
    fs::write(out.join(&file), harness).expect("harness");
    assert!(link_generated(out, &file, name).success(), "{name} link");
    let run = std::process::Command::new(out.join(name))
        .output()
        .expect("run");
    (
        run.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ),
    )
}

/// Link a harness that supplies a zero-filled rank-1 tensor of dtype
/// `CHELIS_DTYPE_{supplied}` to `entry`, and return (exit success, stdout plus
/// stderr).
fn run(out: &Path, name: &str, entry: Entry, supplied: &str) -> (bool, String) {
    let harness = format!(
        "#define main generated_main\n#include \"entry.c\"\n#undef main\n\
         int main(void) {{\n\
         chelis_tensor *x = chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_{supplied});\n\
         {}\n\
         chelis_tensor_release(x);\n\
         puts(\"completed\"); return 0;\n}}\n",
        entry.call()
    );
    link_and_run(out, name, &harness)
}

#[test]
fn mismatched_dtypes_trap_at_every_host_entry_kind() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    // Every row runs before any assertion, so a failure names each entry kind
    // the guard misses rather than only the first.
    let mut failures = Vec::new();
    for (name, entry, supplied, declared) in [
        ("f64_for_i64", Entry::Tensor("ints"), "F64", "i64"),
        ("i32_for_i64", Entry::Tensor("ints"), "I32", "i64"),
        ("i64_for_f64", Entry::Tensor("floats"), "I64", "f64"),
        ("bool_for_f32", Entry::Tensor("singles"), "BOOL", "f32"),
        ("f16_for_f32", Entry::Tensor("singles"), "F16", "f32"),
        ("tuple", Entry::Tuple("int_pair"), "F64", "i64"),
        ("record", Entry::Record("int_record"), "F64", "i64"),
        ("nested", Entry::NestedInTuple("nested"), "F64", "i64"),
        ("pass_through", Entry::Tensor("pass_through"), "F64", "i64"),
        ("debug_only", Entry::Tensor("shown"), "F64", "i64"),
        ("pass_pair", Entry::Tuple("pass_pair"), "F64", "i64"),
    ] {
        // Each supplied tag's language spelling is its lowercase name.
        let actual = supplied.to_lowercase();
        let (succeeded, output) = run(&out, name, entry, supplied);
        let host_line = format!("input `x` expected dtype {declared}, got {actual}");
        // chelis#2506: the nested row now names its parameter path.
        let host_line = if matches!(entry, Entry::NestedInTuple(_)) {
            format!("input `p.0` expected dtype {declared}, got {actual}")
        } else {
            host_line
        };
        let names_parameter = output.lines().any(|line| line == host_line);
        if succeeded
            || output.contains("completed")
            || !output.contains(&format!("expected dtype {declared}, got {actual}"))
            || !output.contains(&format!("numeric trap: domain in load at {declared}\n"))
            || !names_parameter
        {
            failures.push(format!("{name}: exit success {succeeded}: {output}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn matching_dtypes_complete_at_every_host_entry_kind() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    for (name, entry, supplied) in [
        ("i64_ints", Entry::Tensor("ints"), "I64"),
        ("f64_floats", Entry::Tensor("floats"), "F64"),
        ("f32_singles", Entry::Tensor("singles"), "F32"),
        ("i64_pair", Entry::Tuple("int_pair"), "I64"),
        ("i64_record", Entry::Record("int_record"), "I64"),
        ("i64_nested", Entry::NestedInTuple("nested"), "I64"),
        ("i64_pass_through", Entry::Tensor("pass_through"), "I64"),
        ("i64_debug_only", Entry::Tensor("shown"), "I64"),
        ("i64_pass_pair", Entry::Tuple("pass_pair"), "I64"),
    ] {
        let (succeeded, output) = run(&out, name, entry, supplied);
        assert!(succeeded, "{name}: {output}");
        assert!(output.contains("completed"), "{name}: {output}");
        assert!(!output.contains("numeric trap"), "{name}: {output}");
    }
}

/// How a harness hands a parameter carrying one supplied tensor, beside
/// well-formed `tensor[3, i64]` siblings, to an entry.
#[derive(Clone, Copy)]
enum Carrier {
    /// `mixed(p)` with `p = (good, supplied)`: path `p.1`.
    Tuple,
    /// `record_sum(r)` with `r = IntRecord { twice: good, square: supplied }`.
    Record,
    /// `choice_total(c)` with `c = Ints(supplied)`.
    DataType,
    /// `list_count(xs)` with `xs = [good, supplied]`.
    List,
    /// `named_pair(p)` with `p = (good, supplied)`, whose axes share `n`.
    NamedPair,
    /// `tree_leaves(t)` with `t = Node(Leaf(good), Leaf(supplied))`, a
    /// recursive data type.
    Tree,
}

impl Carrier {
    fn path(self) -> &'static str {
        match self {
            Carrier::Tuple | Carrier::NamedPair => "p.1",
            Carrier::Record => "r.square",
            Carrier::DataType => "c.Ints.0",
            Carrier::List => "xs[1]",
            Carrier::Tree => "t.Node.1.Leaf.0",
        }
    }

    /// C that builds the parameter from `good` and `supplied`, calls the
    /// entry, and releases everything.
    fn call(self) -> String {
        let pair = "chelis_value items[2] = { chelis_value_take_tensor(good), chelis_value_take_tensor(supplied) };\n";
        let release_pair = "chelis_value_release(items[0]); chelis_value_release(items[1]);\n";
        match self {
            Carrier::Tuple | Carrier::NamedPair => {
                let name = if matches!(self, Carrier::Tuple) {
                    "mixed"
                } else {
                    "named_pair"
                };
                format!(
                    "{pair}chelis_tuple *p = chelis_tuple_from_values(items, 2);\n{release_pair}\
                     chelis_tensor *result = {}(p);\n\
                     chelis_tensor_release(result); chelis_tuple_release(p);",
                    authored_c_symbol(name)
                )
            }
            Carrier::Record => format!(
                "{pair}chelis_string ctor = chelis_string_from_cstr(\"IntRecord\");\n\
                 chelis_adt *r = chelis_adt_construct(ctor, items, 2);\n\
                 chelis_string_release(ctor);\n{release_pair}\
                 chelis_tensor *result = {}(r);\n\
                 chelis_tensor_release(result); chelis_adt_release(r);",
                authored_c_symbol("record_sum")
            ),
            Carrier::DataType => format!(
                "chelis_tensor_release(good);\n\
                 chelis_value field = chelis_value_take_tensor(supplied);\n\
                 chelis_string ctor = chelis_string_from_cstr(\"Ints\");\n\
                 chelis_adt *c = chelis_adt_construct(ctor, &field, 1);\n\
                 chelis_string_release(ctor); chelis_value_release(field);\n\
                 printf(\"total %lld\\n\", (long long){}(c));\n\
                 chelis_adt_release(c);",
                authored_c_symbol("choice_total")
            ),
            Carrier::List => format!(
                "{pair}chelis_list *xs = chelis_list_from_values(items, 2);\n{release_pair}\
                 printf(\"count %lld\\n\", (long long){}(xs));\n\
                 chelis_list_release(xs);",
                authored_c_symbol("list_count")
            ),
            Carrier::Tree => format!(
                "chelis_string leaf = chelis_string_from_cstr(\"Leaf\");\n\
                 chelis_string node = chelis_string_from_cstr(\"Node\");\n\
                 chelis_value first = chelis_value_take_tensor(good);\n\
                 chelis_value second = chelis_value_take_tensor(supplied);\n\
                 chelis_value leaves[2] = {{\n\
                 chelis_value_take_adt(chelis_adt_construct(leaf, &first, 1)),\n\
                 chelis_value_take_adt(chelis_adt_construct(leaf, &second, 1)) }};\n\
                 chelis_value_release(first); chelis_value_release(second);\n\
                 chelis_adt *t = chelis_adt_construct(node, leaves, 2);\n\
                 chelis_value_release(leaves[0]); chelis_value_release(leaves[1]);\n\
                 chelis_string_release(leaf); chelis_string_release(node);\n\
                 printf(\"leaves %lld\\n\", (long long){}(t));\n\
                 chelis_adt_release(t);",
                authored_c_symbol("tree_leaves")
            ),
        }
    }
}

/// Link a harness that supplies a zero-filled tensor of dtype
/// `CHELIS_DTYPE_{dtype}` and `shape` inside `carrier`, and return (exit
/// success, stdout plus stderr).
fn run_nested(
    out: &Path,
    name: &str,
    carrier: Carrier,
    dtype: &str,
    shape: &[i64],
) -> (bool, String) {
    let extents = shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let harness = format!(
        "#define main generated_main\n#include \"entry.c\"\n#undef main\n\
         int main(void) {{\n\
         chelis_tensor *good = chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_I64);\n\
         chelis_tensor *supplied = chelis_alloc({}, (int64_t[]){{{extents}}}, CHELIS_DTYPE_{dtype});\n\
         {}\n\
         puts(\"completed\"); return 0;\n}}\n",
        shape.len(),
        carrier.call()
    );
    link_and_run(out, name, &harness)
}

/// chelis#2506 REGRESSION TEST: at the pre-fix tree a tensor nested in a
/// tuple, record, data-type or list parameter was validated only if it later
/// reached a kernel entry, which named `__host_tensor_arg_N`; the rows below
/// that reach no kernel ran to completion on a wrong dtype, rank or extent.
/// Every row now stops at the host entry with the parameter path, including
/// one two levels down a recursive data type.
#[test]
fn a_mismatched_nested_tensor_stops_at_the_entry_with_its_parameter_path() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    let mut failures = Vec::new();
    for carrier in [
        Carrier::Tuple,
        Carrier::Record,
        Carrier::DataType,
        Carrier::List,
        Carrier::Tree,
    ] {
        let path = carrier.path();
        for (kind, dtype, shape, expected) in [
            (
                "dtype",
                "F64",
                &[3i64][..],
                format!(
                    "input `{path}` expected dtype i64, got f64\nnumeric trap: domain in load at i64\n"
                ),
            ),
            (
                "rank",
                "I64",
                &[3, 1][..],
                format!("input `{path}` expected rank 1, got 2\n"),
            ),
            (
                "extent",
                "I64",
                &[4][..],
                format!(
                    "input `{path}` axis 0 expected 3, got 4\nnumeric trap: domain in load at i64\n"
                ),
            ),
        ] {
            let name = format!("nested_{}_{kind}", path.replace(['.', '[', ']'], "_"));
            let (succeeded, output) = run_nested(&out, &name, carrier, dtype, shape);
            if succeeded || output.contains("completed") || !output.contains(&expected) {
                failures.push(format!("{path} {kind}: exit success {succeeded}: {output}"));
            }
        }
    }
    let (succeeded, output) =
        run_nested(&out, "nested_named_extent", Carrier::NamedPair, "I64", &[4]);
    if succeeded
        || output.contains("completed")
        || !output.contains(
            "extent `n`: p.0 axis 0 = 3, p.1 axis 0 = 4\nnumeric trap: domain in load at i64\n",
        )
    {
        failures.push(format!(
            "p.1 named extent: exit success {succeeded}: {output}"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_matching_nested_tensor_runs_at_every_carrier() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    for (name, carrier, result) in [
        ("tuple", Carrier::Tuple, None),
        ("record", Carrier::Record, None),
        ("data_type", Carrier::DataType, Some("total 0")),
        ("list", Carrier::List, Some("count 2")),
        ("named_pair", Carrier::NamedPair, None),
        ("tree", Carrier::Tree, Some("leaves 2")),
    ] {
        let (succeeded, output) =
            run_nested(&out, &format!("matching_{name}"), carrier, "I64", &[3]);
        assert!(succeeded, "{name}: {output}");
        assert!(output.contains("completed"), "{name}: {output}");
        assert!(!output.contains("numeric trap"), "{name}: {output}");
        if let Some(result) = result {
            assert!(output.contains(result), "{name}: {output}");
        }
    }
}

/// chelis#2506 REGRESSION TEST: an exported entry walks a nested value once,
/// and the recursive calls its body makes do not walk it again. The harness
/// counts every `chelis_tensor_shape` read in the generated program; walking
/// four elements reads four literal extents. At the pre-fix tree the walk ran
/// in the body every call entered, so `list_walk` over four elements read
/// twenty, one walk per call.
#[test]
fn an_exported_entry_walks_a_nested_value_once_and_its_internal_calls_do_not() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    let harness = format!(
        "#include \"chelis_runtime.h\"\n\
         static long long shape_reads = 0;\n\
         static int64_t counted_shape(const chelis_tensor *tensor, int32_t axis) {{\n\
         ++shape_reads; return chelis_tensor_shape(tensor, axis);\n}}\n\
         #define chelis_tensor_shape(tensor, axis) counted_shape(tensor, axis)\n\
         #define main generated_main\n#include \"entry.c\"\n#undef main\n\
         int main(void) {{\n\
         chelis_value items[4];\n\
         for (int k = 0; k < 4; ++k) items[k] = chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_I64));\n\
         chelis_list *xs = chelis_list_from_values(items, 4);\n\
         for (int k = 0; k < 4; ++k) chelis_value_release(items[k]);\n\
         long long steps = (long long){}(xs, 0, 0);\n\
         printf(\"steps %lld shape reads %lld\\n\", steps, shape_reads);\n\
         chelis_list_release(xs);\n\
         return 0;\n}}\n",
        authored_c_symbol("list_walk")
    );
    let (succeeded, output) = link_and_run(&out, "walk_once", &harness);
    assert!(succeeded, "{output}");
    assert!(
        output.contains("steps 4 shape reads 4\n"),
        "one walk of four elements at the exported entry: {output}"
    );
    let (succeeded, output) = run_nested(&out, "walk_once_trap", Carrier::List, "F64", &[3]);
    assert!(
        !succeeded
            && output.contains(
                "input `xs[1]` expected dtype i64, got f64\nnumeric trap: domain in load at i64\n"
            ),
        "the exported entry still traps: {output}"
    );
}

/// chelis#2506: a path longer than the 512-byte rendering buffer ends in a
/// visible marker rather than stopping mid-segment. At the pre-fix tree it
/// was cut silently.
#[test]
fn a_nested_path_too_long_to_render_ends_in_a_truncation_marker() {
    if !gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = build(dir.path());
    // Eighty `Node(Leaf(good), .)` levels above `Leaf(supplied)` render
    // `t` + 80 x `.Node.1` + `.Leaf.0`, 568 bytes.
    let harness = format!(
        "#define main generated_main\n#include \"entry.c\"\n#undef main\n\
         int main(void) {{\n\
         chelis_string leaf = chelis_string_from_cstr(\"Leaf\");\n\
         chelis_string node = chelis_string_from_cstr(\"Node\");\n\
         chelis_value field = chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_F64));\n\
         chelis_value tree = chelis_value_take_adt(chelis_adt_construct(leaf, &field, 1));\n\
         chelis_value_release(field);\n\
         for (int level = 0; level < 80; ++level) {{\n\
         chelis_value good = chelis_value_take_tensor(chelis_alloc(1, (int64_t[]){{3}}, CHELIS_DTYPE_I64));\n\
         chelis_value children[2] = {{ chelis_value_take_adt(chelis_adt_construct(leaf, &good, 1)), tree }};\n\
         chelis_value_release(good);\n\
         tree = chelis_value_take_adt(chelis_adt_construct(node, children, 2));\n\
         chelis_value_release(children[0]); chelis_value_release(children[1]);\n\
         }}\n\
         printf(\"leaves %lld\\n\", (long long){}(chelis_adt_take_value(tree)));\n\
         puts(\"completed\"); return 0;\n}}\n",
        authored_c_symbol("tree_leaves")
    );
    let (succeeded, output) = link_and_run(&out, "long_path", &harness);
    let prefix = format!("input `t{}", ".Node.1".repeat(70));
    let line = output
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no deep path line: {output}"));
    assert!(
        !succeeded
            && line.ends_with("...(truncated)` expected dtype i64, got f64")
            && line.len() == "input ``".len() + 511 + " expected dtype i64, got f64".len(),
        "{line}"
    );
}

/// chelis#2526: `Result` is not a built-in type. It was admitted as a
/// checker-native nominal type with no constructors, so a parameter or field
/// could carry one that no program could build or read, and the C entry walker
/// needed a special case for its payload. It is now an unknown type at check in
/// every position a parameter can carry it, and the build stops there.
#[test]
fn result_is_an_unknown_type_at_check() {
    let mut accepted = Vec::new();
    for (name, definition, position) in [
        (
            "scalar",
            "def f(x: Result[i64, string]) -> i64 = 1i64\n",
            "defsig",
        ),
        (
            "tensor",
            "def f(x: Result[tensor[3, f32], string]) -> i64 = 1i64\n",
            "defsig",
        ),
        (
            "list",
            "def f(x: List[Result[i64, string]]) -> i64 = len(x)\n",
            "defsig",
        ),
        (
            "option",
            "def f(x: Option[Result[i64, string]]) -> i64 = 1i64\n",
            "defsig",
        ),
        (
            "tuple",
            "def f(x: (i64, Result[i64, string])) -> i64 = x.0\n",
            "defsig",
        ),
        (
            "field",
            "type Wrap =\n  | Wrap(Result[i64, string])\ndef f(x: Wrap) -> i64 = 1i64\n",
            "deftype field",
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = format!("{definition}def g(x: i64) -> i64 = x\n");
        match build_source(dir.path(), &source) {
            Ok(_) => accepted.push(format!("{name}: built")),
            Err(stderr) => assert!(
                stderr.contains(&format!("unknown nominal type `Result` in {position}")),
                "{name}: {stderr}"
            ),
        }
    }
    assert!(accepted.is_empty(), "{}", accepted.join("\n"));

    // A program may still declare its own `Result`, at any arity; its
    // constructors give the entry walker a layout like any other data type.
    // The zero-arity form was rejected before, because the built-in header
    // demanded two arguments.
    for declared in [
        "type Result[a, b] =\n  | Ok(a)\n  | Err(b)\n\
         def f(x: Result[tensor[3, f32], string]) -> i64 = 1i64\n",
        "type Result =\n  | HomeWin\n  | Draw\n  | AwayWin\n\
         def f(x: Result) -> i64 = 1i64\n",
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        build_source(dir.path(), &format!("{declared}def g(x: i64) -> i64 = x\n"))
            .unwrap_or_else(|stderr| panic!("a declared Result builds: {stderr}\n{declared}"));
    }
}
