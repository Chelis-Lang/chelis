//! CLI coverage for binder-target literal adoption (#1544/#1553).

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

const EXACT_F64: &str = "0.30000000000000004";
const SCALAR: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: p) -> p = mul(x, cast(0.1, p))\n\
 def at_f32() -> f32 = scale(3.0f32)\n\
 def at_f64() -> f64 = scale(3.0f64)\n\
 def main() -> f64 = at_f64()\n";
const TENSOR: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: tensor[1, p]) -> tensor[1, p] = mul(x, cast(0.1, p))\n\
 def at_f32() -> tensor[1, f32] = scale(to_tensor([3.0f32]))\n\
 def at_f64() -> tensor[1, f64] = scale(to_tensor([3.0f64]))\n\
 def main() -> tensor[1, f64] = at_f64()\n";
const TENSOR_NATIVE: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: tensor[3, p]) -> tensor[3, p] = mul(x, insert(scalar_to_tensor(cast(0.1, p)), cast(0, int32), cast(3, int64)))\n\
 def at_f64() -> tensor[3, f64] = scale(to_tensor([3.0f64, 3.0f64, 3.0f64]))\n\
 def main() -> tensor[3, f64] = at_f64()\n";

fn package(name: &str, source: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("src");
    let manifest = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Bind\"\n",
        chelis_compiler_api::COMPILER_VERSION
    );
    fs::write(root.join("reef.toml"), manifest).expect("manifest");
    fs::write(root.join("src/main.ch"), source).expect("source");
    (dir, root)
}

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run")
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn success(output: &std::process::Output) {
    assert!(output.status.success(), "{}", text(output));
}

fn eval(root: &Path) -> String {
    let check = run(root, &["check", "src/main.ch"]);
    success(&check);
    assert!(root.join("reef.lock").exists());
    let output = run(root, &["eval", "--file", "src/main.ch"]);
    success(&output);
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn reject(source: &str, citation: &str) {
    let (_dir, root) = package("rejected", source);
    for args in [
        &["check", "src/main.ch"][..],
        &["eval", "--file", "src/main.ch"],
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    ] {
        let output = run(&root, args);
        assert!(!output.status.success() && text(&output).contains(citation));
    }
}

fn assert_native(name: &str, source: &str, expected: &str) {
    let (_dir, root) = package(name, source);
    let interpreted = eval(&root);
    let native = common::build_and_run(source, name);
    assert!(
        interpreted.contains(expected) && native.contains(expected),
        "expected {expected:?}; eval={interpreted:?}; native={native:?}"
    );
}

/// The round-5 P1 witness, in Deep. Its resugared Surf is character-for-character
/// the scalar witness this suite already documents; the only difference is that
/// the `lit` carries its `type` stamp TWICE.
const DUPLICATED_TYPE_STAMP: &str = "(module {surf_path: \"Bind.Main\"}\n\
     bind.main\n\
     (export {} main)\n\
     (defsig {dtype_bounds: {p: numeric}} addk (t-fn {} (t-var {} p) (t-var {} p)))\n\
     (def {} addk\n\
       (fn {} (params {} (x {type: (t-var {} p)}))\n\
         (app {} (var {} add) (var {} x)\n\
           (cast {}\n\
             (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p), type: (t-var {} p)} 16777217.0)\n\
             (t-var {} p)))))\n\
     (defsig {} main (t-fn {} (t-prim {} int64)))\n\
     (def {} main (fn {} (params {}) (app {} (var {} addk) (lit {type: (t-prim {} int64)} 0)))))\n";

/// Round 5 P1, regression test. One accepted program, one answer, both lanes.
///
/// The host-specialization join required EXACTLY ONE `type` metadata entry,
/// which made it stricter than the two readers that decide whether a program is
/// accepted at all: the checker's `visit_binder_literal_uses` and the
/// interpreter's `lit_meta_type_var_name` both take the FIRST entry. So this
/// program type-checked, evaluated with the f32 narrow applied, and compiled
/// WITHOUT it, giving `16777216` from `chelis eval` and `16777217` from the
/// compiled C. All three readers now take the first entry.
///
/// Failing closed in the lowering lane instead would have traded a wrong value
/// for a lane split, with C rejecting what the checker and eval accept. Whether
/// a duplicated `type` key should be malformed Deep at the ingress under
/// [04-TOT-3] is a well-formedness question for its own slice.
///
/// `surf_literal_style` deliberately keeps its exactly-one requirement: the
/// checker consumes the same predicate and rejects a duplicate of that key, so
/// strictness there has a rejecting counterpart and cannot fail open.
#[test]
fn a_duplicated_type_stamp_gives_one_answer_on_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let unit = root.join("dup_type.dp");
    fs::write(&unit, DUPLICATED_TYPE_STAMP).expect("write fixture");

    let interpreted = text(&run(root, &["eval", "--file", "dup_type.dp"]));
    assert!(
        interpreted.contains("main = 16777216"),
        "eval must apply the f32 narrow: {interpreted}"
    );

    let out_dir = root.join("dup-out");
    success(&run(
        root,
        &[
            "build",
            unit.to_str().expect("utf8"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("utf8"),
        ],
    ));
    let emitted = fs::read_to_string(out_dir.join("dup_type.c")).expect("emitted C");
    assert!(
        emitted.contains("f64 -> f32"),
        "the emitted C must carry the f32 narrow plan"
    );

    let status = common::link_generated(&out_dir, "dup_type.c", "dup_type");
    assert!(status.success(), "link failed: {status}");
    let native = std::process::Command::new(out_dir.join("dup_type"))
        .output()
        .expect("compiled binary should run");
    let native = String::from_utf8_lossy(&native.stdout).to_string();
    assert!(
        native.contains("main = 16777216"),
        "compiled C must agree with eval, not return 16777217: {native}"
    );
}

/// The same duplicated stamp on the DAG/TENSOR lowering lane, in the
/// explicit-broadcast form. `lower.rs::binder_float_literal_source_default`
/// reads the stamp through its own copy of the predicate, so the scalar row
/// above measures only the host join.
const DUPLICATED_TYPE_STAMP_TENSOR: &str = "(module {surf_path: \"Bind.Main\"}\n\
     bind.main\n\
     (export {} main)\n\
     (defsig {dtype_bounds: {p: numeric}} addk\n\
       (t-fn {} (t-tensor {} (d-lit {} 1) (t-var {} p)) (t-tensor {} (d-lit {} 1) (t-var {} p))))\n\
     (def {} addk\n\
       (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 1) (t-var {} p))}))\n\
         (app {} (var {} add) (var {} x)\n\
           (app {} (var {} insert)\n\
             (app {} (var {} scalar_to_tensor)\n\
               (cast {}\n\
                 (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p), type: (t-var {} p)} 16777217.0)\n\
                 (t-var {} p)))\n\
             (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} int32)} 0) (t-prim {} int32))\n\
             (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} int64)} 1) (t-prim {} int64))))))\n\
     (defsig {} main (t-fn {} (t-tensor {} (d-lit {} 1) (t-prim {} int64))))\n\
     (def {} main\n\
       (fn {} (params {})\n\
         (app {} (var {} addk)\n\
           (app {} (var {} to_tensor)\n\
             (app {} (var {} Cons) (lit {type: (t-prim {} int64)} 0) (var {} Nil)))))))\n";

/// Round 5 P1 on the tensor lane, regression test.
///
/// The scalar row above proves the host join; this one proves `lower.rs`,
/// which reads the same stamp through a separate predicate and had made the
/// same exactly-one choice. The two lanes reach the f32 source differently, so
/// the assertion here is the VALUE rather than a plan line: the tensor lane
/// finalizes the literal at f32 directly and emits no `f64 -> f32` narrow.
///
/// Proved red on f86f479e6, where `chelis check` accepts this unit, `chelis
/// eval` prints `[16777216]`, and the compiled C prints `[16777217]`.
#[test]
fn a_duplicated_type_stamp_gives_one_answer_on_the_tensor_lane() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let unit = root.join("dup_tensor.dp");
    fs::write(&unit, DUPLICATED_TYPE_STAMP_TENSOR).expect("write fixture");

    let expected = "main = tensor(shape=[1], data=[16777216])";
    let interpreted = text(&run(root, &["eval", "--file", "dup_tensor.dp"]));
    assert!(
        interpreted.contains(expected),
        "eval must apply the f32 narrow: {interpreted}"
    );

    let out_dir = root.join("dup-tensor-out");
    success(&run(
        root,
        &[
            "build",
            unit.to_str().expect("utf8"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("utf8"),
        ],
    ));
    let status = common::link_generated(&out_dir, "dup_tensor.c", "dup_tensor");
    assert!(status.success(), "link failed: {status}");
    let native = std::process::Command::new(out_dir.join("dup_tensor"))
        .output()
        .expect("compiled binary should run");
    let native = String::from_utf8_lossy(&native.stdout).to_string();
    assert!(
        native.contains(expected),
        "compiled C must agree with eval, not return 16777217: {native}"
    );
}

#[test]
fn scalar_and_tensor_binders_compute_at_each_instantiation() {
    for family in ["Float", "Numeric"] {
        let source = SCALAR.replace("Float", family);
        let (_dir, root) = package(family, &source);
        let output = eval(&root);
        for expected in ["at_f32 = 0.3".to_string(), format!("at_f64 = {EXACT_F64}")] {
            assert!(output.contains(&expected), "{output}");
        }
    }
    let (_dir, root) = package("tensor", TENSOR);
    let output = eval(&root);
    assert!(output.contains("at_f32 = tensor(shape=[1], data=[0.3])"));
    assert!(output.contains(&format!("at_f64 = tensor(shape=[1], data=[{EXACT_F64}])")));
}

#[test]
fn native_scalar_and_tensor_match_eval_exactly() {
    assert_native("scalar", SCALAR, &format!("at_f64 = {EXACT_F64}"));
    assert_native(
        "tensor",
        TENSOR_NATIVE,
        &format!("at_f64 = tensor(shape=[3], data=[{EXACT_F64}, {EXACT_F64}, {EXACT_F64}])"),
    );
}

#[test]
fn undeclared_and_unbounded_targets_fail_all_lanes() {
    reject(
        "module Bind.Main\nexport (main)\ndef typo(x: f32) -> f32 = cast(x, flt32)\ndef main() -> f32 = typo(1.0f32)\n",
        "cast target `flt32` is not a recognized primitive type",
    );
    for literal in ["0.1", "-0.1", "0.1f64", "-0.1f64"] {
        reject(
            &format!(
                "module Bind.Main\nexport (main)\ndef scale[p](x: p) -> p = mul(x, cast({literal}, p))\ndef main() -> f64 = scale(3.0f64)\n"
            ),
            "[04-DTYPE-1]",
        );
    }
}

#[test]
fn integer_literals_adopt_every_compatible_family() {
    for family in ["Int", "Float", "Numeric"] {
        let ty = if family == "Int" { "int64" } else { "f64" };
        let source = format!(
            "module Bind.Main\nexport (main)\ndef addk[p: {family}](x: p) -> p = add(x, cast(1, p))\ndef main() -> {ty} = addk(41{suffix})\n",
            suffix = if family == "Int" { "i64" } else { ".0f64" }
        );
        let (_dir, root) = package(family, &source);
        assert!(eval(&root).contains("main = 42"));
    }
}

#[test]
fn adopted_integer_literals_must_fit_every_family_member() {
    for family in ["Int", "Numeric"] {
        for literal in ["128", "-129"] {
            reject(
                &format!(
                    "module Bind.Main\nexport (main)\ndef value[p: {family}](x: p) -> p = cast({literal}, p)\ndef main() -> int64 = value(0i64)\n"
                ),
                "§5.6",
            );
        }
    }
}

#[test]
fn numeric_cross_family_float_literal_preserves_source_default() {
    for (literal, expected, name) in [
        ("16777217.0", "16777216", "positive"),
        ("-16777217.0", "-16777216", "negative"),
    ] {
        let source = format!(
            "module Bind.Main\nexport (main)\n\
             def addk[p: Numeric](x: p) -> p = add(x, cast({literal}, p))\n\
             def main() -> int64 = addk(0i64)\n"
        );
        assert_native(name, &source, &format!("main = {expected}"));
    }

    let tensor = "module Bind.Main\nexport (main)\n\
                  def addk[p: Numeric](x: tensor[1, p]) -> tensor[1, p] = add(x, insert(scalar_to_tensor(cast(16777217.0, p)), cast(0, int32), cast(1, int64)))\n\
                  def main() -> tensor[1, int64] = addk(to_tensor([0i64]))\n";
    assert_native(
        "tensor_cross_family",
        tensor,
        "main = tensor(shape=[1], data=[16777216])",
    );
}

#[test]
fn float_family_literal_finalization_remains_total() {
    let source = "module Bind.Main\nexport (main)\n\
                  def value[p: Float](x: p) -> p = cast(10000000000.0, p)\n\
                  def main() -> f16 = value(0.0f16)\n";
    let (_dir, root) = package("float_range", source);
    success(&run(&root, &["check", "src/main.ch"]));
}

#[test]
fn ascription_is_not_literal_adoption() {
    reject(
        "module Bind.Main\nexport (main)\ndef scale[p: Float](x: p) -> p = mul(x, (0.1 : p))\ndef main() -> f64 = scale(3.0f64)\n",
        "[04-INF-6]",
    );
}
