//! chelis#3164 (A1): a numeric literal's dtype is written at its site, through
//! `chelis check`, `eval`, `build`, `deep` and `surf`.
//!
//! `spec/04-type-system.md` §5.3, §5.6 and §8.6, `spec/02-surf-syntax.md`
//! §P9 and §P10b, `spec/03-deep-syntax.md` §6.4 and `spec/05-risc-primitives.md`
//! [05-OP-57]. Each positive is compared with a program that spells every
//! dtype with a suffix, which the rule says denotes the same value, and with
//! its native C build. Each negative is rejected by `check`, `eval` and
//! `build` before any lane runs, with the diagnostic the rule names.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::tempdir;

fn chelis(directory: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(directory);
    command
}

fn eval_json(directory: &Path, file: &str) -> Value {
    let output = chelis(directory)
        .args(["eval", "--json", "--file", file])
        .output()
        .expect("chelis eval runs");
    assert!(
        output.status.success(),
        "{file} must evaluate: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("eval JSON")
}

fn eval_text(directory: &Path, file: &str) -> String {
    let output = chelis(directory)
        .args(["eval", "--file", file])
        .output()
        .expect("chelis eval runs");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).expect("utf-8 eval output")
}

/// Root name to evaluated value.
fn roots(report: &Value) -> Vec<(String, Value)> {
    report["roots"]
        .as_array()
        .expect("roots")
        .iter()
        .map(|root| {
            (
                root["name"].as_str().expect("root name").to_string(),
                root["value"].clone(),
            )
        })
        .collect()
}

fn root<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["roots"]
        .as_array()
        .expect("roots")
        .iter()
        .find(|root| root["name"] == name)
        .map(|root| &root["value"])
        .unwrap_or_else(|| panic!("no root `{name}` in {report}"))
}

/// `chelis deep`, then `chelis surf`, into `resugared.ch` beside `file`.
fn resugar(directory: &Path, file: &str) -> String {
    let deep = chelis(directory)
        .args(["deep", file])
        .output()
        .expect("chelis deep runs");
    assert!(deep.status.success(), "{deep:?}");
    std::fs::write(directory.join("program.dp"), &deep.stdout).expect("write Deep");
    let surf = chelis(directory)
        .args(["surf", "program.dp"])
        .output()
        .expect("chelis surf runs");
    assert!(surf.status.success(), "{surf:?}");
    let text = String::from_utf8(surf.stdout).expect("utf-8 Surf");
    std::fs::write(directory.join("resugared.ch"), &text).expect("write Surf");
    text
}

/// Runs `check`, `eval` and `build --emit-c` on `file`; each must fail and
/// report every needle.
fn assert_rejected_everywhere(directory: &Path, file: &str, needles: &[&str], source: &str) {
    let output = directory.join("native");
    for args in [
        vec!["check", file],
        vec!["eval", "--file", file],
        vec![
            "build",
            "--emit-c",
            file,
            "--target",
            "c",
            "--output",
            output.to_str().expect("utf-8 path"),
        ],
    ] {
        let result = chelis(directory).args(&args).output().expect("chelis runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            !result.status.success(),
            "{args:?} accepted:\n{source}\n{text}"
        );
        for needle in needles {
            assert!(
                text.contains(needle),
                "{args:?}: expected `{needle}`:\n{source}\n{text}"
            );
        }
    }
}

/// Each A1 spelling, and the same values spelled with suffixes only.
const SITES: &str = "\
a = cast(1.1, f64)
b: f64 = 1.1
sig c: f64
c = 1.1
def d() -> i8 = -128
dd = d()
e: i64 = 3000000000
f = to_tensor([1.1, 2.2], f64)
g = cast(to_tensor([1.1, 2.2], f32), f64)
h: f64 = 5
k = to_tensor([[1, 2], [3, 4]], i64)
m = {
  f64 = 2
  to_tensor([1.5], f64)
}
n: tensor[2, f64] = [1.1, 2.2]
q = cast(-1.1, f64)
r = cast(1.1f32, f64)
t: i8 = -128
u = cast(-3000000000, i64)
v = to_tensor([-1.5, 2], f64)
w: f16 = 0.1
y = cast(1, bool)
def kb[p: Float](x: p) -> tensor[1, p] = to_tensor([cast(1.5, p)], p)
kb_value = kb(1.0f64)
pa = 1.1 |> cast(f64)
pb = [1.1, 2.2] |> to_tensor(f64)
";

const SUFFIXED: &str = "\
a = 1.1f64
b = 1.1f64
c = 1.1f64
dd = sub(neg(127i8), 1i8)
e = 3000000000i64
f = to_tensor([1.1f64, 2.2f64])
g = cast(to_tensor([1.1f32, 2.2f32]), f64)
h = 5f64
k = to_tensor([[1i64, 2i64], [3i64, 4i64]])
m = to_tensor([1.5f64])
n = to_tensor([1.1f64, 2.2f64])
q = neg(1.1f64)
r = 1.100000023841858f64
t = sub(neg(127i8), 1i8)
u = neg(3000000000i64)
v = to_tensor([neg(1.5f64), 2f64])
w = 0.1f16
y = true
kb_value = to_tensor([1.5f64])
pa = 1.1f64
pb = to_tensor([1.1f64, 2.2f64])
";

#[test]
fn every_site_denotes_its_suffixed_value_in_eval_and_native_c() {
    let directory = tempdir().expect("tempdir");
    common::write_file(&directory.path().join("sites.ch"), SITES);
    common::write_file(&directory.path().join("suffixed.ch"), SUFFIXED);
    let sites = eval_json(directory.path(), "sites.ch");
    let suffixed = eval_json(directory.path(), "suffixed.ch");
    assert_eq!(
        roots(&sites),
        roots(&suffixed),
        "A1 spellings change values"
    );

    // The exact bits, not only agreement: no f32 rounding where none is
    // written, and the stated rounding where it is.
    assert_eq!(root(&sites, "b")["value"]["bits"], "3ff199999999999a");
    assert_eq!(
        root(&sites, "f")["value"]["data"]["bits"],
        serde_json::json!(["3ff199999999999a", "400199999999999a"])
    );
    assert_eq!(
        root(&sites, "g")["value"]["data"]["bits"],
        serde_json::json!(["3ff19999a0000000", "40019999a0000000"])
    );
    assert_eq!(root(&sites, "q")["value"]["bits"], "bff199999999999a");

    assert_eq!(
        common::build_and_run(SITES, "sites"),
        eval_text(directory.path(), "sites.ch"),
        "eval and native C disagree"
    );
}

#[test]
fn every_rejected_spelling_is_refused_by_check_eval_and_build() {
    for (index, (source, needles)) in [
        (
            "x = to_tensor([1.1, 2.2])\n",
            &["states no dtype", "to_tensor([1.1, 2.2], f32)"][..],
        ),
        (
            "x = cast(to_tensor([1.1, 2.2]), f64)\n",
            &["states no dtype", "to_tensor([1.1, 2.2], f64)"][..],
        ),
        (
            "xs: tensor[2, f32] = to_tensor([1.1, 2.2])\n",
            &["states no dtype"][..],
        ),
        ("x = [1.1, 2.2] |> to_tensor\n", &["states no dtype"][..]),
        ("x: i32 = 1.5\n", &["cannot bind at i32"][..]),
        ("x = to_tensor([1.0f64], f32)\n", &["dtype argument"][..]),
        ("xs = [1.5, 2.5]\nx = to_tensor(xs, f64)\n", &["never converts"][..]),
        ("v = 2\nx = to_tensor([1.5], v)\n", &["must be a dtype"][..]),
        (
            "def k[p: Float](x: p) -> tensor[1, p] = to_tensor([1.5], p)\n",
            &["chelis#3148"][..],
        ),
        (
            "def f(to_tensor: i32) -> i32 = to_tensor\nr = f(1)\n",
            &["`to_tensor` is reserved"][..],
        ),
        (
            "import Std.Sort (to_tensor)\nr = 1\n",
            &["`to_tensor` is reserved"][..],
        ),
        // chelis#3152
        (
            "def mk(l: List[f32]) -> tensor[2, f32] = to_tensor(l)\n\
             def g2(to_tensor: List[f32] -> tensor[2, f32]) -> tensor[2, f32] = to_tensor([1.1f32, 2.2f32])\n\
             r2 = g2(mk)\n",
            &["`to_tensor` is reserved"][..],
        ),
        // chelis#3167
        (
            "def tt[p: Float](l: List[p]) -> tensor[2, p] = to_tensor(l)\n\
             macro m() = {\n  xs: tensor[2, f64] = [1.1, 2.2]\n  xs\n}\n\
             via_macro = {\n  to_tensor = tt\n  m()\n}\n",
            &["`to_tensor` is reserved"][..],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let file = format!("rejected{index}.ch");
        common::write_file(&directory.path().join(&file), source);
        assert_rejected_everywhere(directory.path(), &file, needles, source);
    }
}

/// §8.6: the reef-package exemption from builtin shadowing does not reach a
/// reserved name, so a package module cannot bind `to_tensor` either.
#[test]
fn a_reef_package_module_cannot_bind_to_tensor() {
    for (index, body) in [
        "def to_tensor(x: i32) -> i32 = x\n\nr = to_tensor(1)\n",
        "def f(to_tensor: i32) -> i32 = to_tensor\n\nr = f(1)\n",
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let package = directory.path().join(format!("demo{index}"));
        common::write_file(
            &package.join("reef.toml"),
            &format!(
                "schema = \"1\"\n\n[package]\nname = \"demo{index}\"\nversion = \"0.1.0\"\n\
                 compiler = \"={}\"\nmodule_prefix = \"Demo\"\n",
                common::COMPILER_VERSION
            ),
        );
        let source = format!("module Demo.Main\n\n{body}");
        common::write_file(&package.join("src/main.ch"), &source);
        assert_rejected_everywhere(
            &package,
            "src/main.ch",
            &["`to_tensor` is reserved"],
            &source,
        );
    }
}

/// spec/03 §6.4 and §8.6 at Deep ingress: the three-child `to_tensor` runs
/// in both lanes, an untyped literal element is rejected, and so is a
/// `to_tensor` binder.
#[test]
fn hand_written_deep_follows_the_same_rules() {
    let directory = tempdir().expect("tempdir");
    let three_child = "(def {} f (app {} (var {} to_tensor)\n\
         (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 1.1)\n\
           (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 2.2) (var {} Nil)))\n\
         (t-prim {} f64)))\n";
    common::write_file(&directory.path().join("three.dp"), three_child);
    let report = eval_json(directory.path(), "three.dp");
    assert_eq!(
        root(&report, "f")["value"]["data"]["bits"],
        serde_json::json!(["3ff199999999999a", "400199999999999a"])
    );
    let native_dir = directory.path().join("native");
    chelis(directory.path())
        .args(["build", "three.dp", "--target", "c", "--output"])
        .arg(&native_dir)
        .assert()
        .success();
    let native = std::process::Command::new(native_dir.join("three"))
        .output()
        .expect("native binary runs");
    assert!(native.status.success(), "{native:?}");
    assert_eq!(
        String::from_utf8(native.stdout).expect("utf-8"),
        eval_text(directory.path(), "three.dp"),
        "eval and native C disagree on the three-child form"
    );

    for (index, (source, needle)) in [
        (
            "(def {} a (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {} 1.5) (var {} Nil))))\n",
            "no `type` metadata",
        ),
        (
            "(def {} g (fn {} (params {} (to_tensor {type: (t-prim {} i32)})) (var {} to_tensor)))\n",
            "`to_tensor` is reserved",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let file = format!("rejected{index}.dp");
        common::write_file(&directory.path().join(&file), source);
        assert_rejected_everywhere(directory.path(), &file, &[needle], source);
    }
}

/// spec/03 §6.3.1: `chelis surf` keeps every literal's dtype, including the
/// macro-produced cast operands of chelis#3165, so values survive a
/// deep-to-surf round trip.
#[test]
fn deep_to_surf_keeps_every_value() {
    let directory = tempdir().expect("tempdir");
    let source = "\
macro mh() = neg(1.1)
macro half() = 1.1
def k[p: Float](x: p) -> p = cast(half(), p)
a = cast(mh(), f64)
b = k(1.0f64)
c = cast(half(), f64)
d = cast(1.1f32, f64)
e: f64 = 1.1
f = to_tensor([1.1, 2.2], f64)
g = cast(-128, i8)
h = cast(1.1, f64)
";
    common::write_file(&directory.path().join("program.ch"), source);
    let original = eval_json(directory.path(), "program.ch");
    assert_eq!(root(&original, "a")["value"]["bits"], "bff19999a0000000");
    assert_eq!(root(&original, "c")["value"]["bits"], "3ff19999a0000000");
    let printed = resugar(directory.path(), "program.ch");
    assert_eq!(
        roots(&eval_json(directory.path(), "resugared.ch")),
        roots(&original),
        "chelis surf changed a value:\n{printed}"
    );
}
