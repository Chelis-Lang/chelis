//! chelis#3419: the sign of a zero gradient follows spec/06 §2.4.
//!
//! Each float leaf's adjoint is the balanced addition tree "beginning with an
//! exact positive-zero base leaf", so under [04-NUM-2] round-to-nearest a
//! lone `-0` contribution finalizes to `+0`: `+0 + -0 = +0`. The forward
//! product keeps its `-0`, so the normalization belongs to the accumulation,
//! not to printing. The evaluator and the C lane agree at every float dtype.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::build_and_run;

fn write_file(path: &std::path::Path, source: &str) {
    common::write_file(
        path,
        &chelis_surf::format::format_source(source).expect("canonical Surf"),
    );
}

fn eval(source: &str, stem: &str) -> String {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{stem}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The printed element strings of `name`, so the sign of a zero survives.
fn printed_elements(stdout: &str, name: &str) -> Vec<String> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` line:\n{stdout}"));
    let start = line.find("data=[").expect("data") + "data=[".len();
    let end = start + line[start..].find(']').expect("closing bracket");
    line[start..end]
        .split(',')
        .map(|element| element.trim().to_string())
        .collect()
}

#[test]
fn a_lone_negative_zero_contribution_accumulates_to_positive_zero() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            r#"
def loss(a: tensor[2, {dtype}], w: tensor[2, {dtype}]) -> tensor[{dtype}] = sum(mul(a, w), 0i32)
def product(a: tensor[2, {dtype}], w: tensor[2, {dtype}]) -> tensor[2, {dtype}] = mul(a, w)
forward = product(to_tensor([1.0{dtype}, 2.0{dtype}]), to_tensor([-0.0{dtype}, 1.0{dtype}]))
out = grad(loss, wrt=a)(to_tensor([1.0{dtype}, 2.0{dtype}]), to_tensor([-0.0{dtype}, 1.0{dtype}]))
"#
        );
        let stem = format!("signed_zero_{dtype}");
        let lanes = [
            ("eval", eval(&source, &stem)),
            ("C", build_and_run(&source, &stem)),
        ];
        for (lane, stdout) in lanes {
            assert_eq!(
                printed_elements(&stdout, "forward"),
                ["-0.0", "2.0"],
                "{dtype} {lane}: the forward product keeps its signed zero"
            );
            assert_eq!(
                printed_elements(&stdout, "out"),
                ["0.0", "1.0"],
                "{dtype} {lane}: the positive-zero base leaf absorbs the -0 contribution"
            );
        }
    }
}
