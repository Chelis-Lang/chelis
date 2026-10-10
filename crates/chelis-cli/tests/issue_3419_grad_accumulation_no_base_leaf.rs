//! chelis#3419: spec/06 §2.4 accumulates a float leaf's contributions with no
//! base leaf. One contribution is the adjoint unchanged, so a lone `-0` stays
//! `-0`; m >= 2 contributions take [05-OP-30]'s adjacent-pair tree at the
//! leaf's dtype; no contribution is exact `+0`. The evaluator and the C lane
//! agree at every float dtype.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::build_and_run;

const DTYPES: [&str; 4] = ["f16", "bf16", "f32", "f64"];

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

/// The printed element strings of `name`, so the sign of a zero survives. A
/// rank-zero result prints as a bare scalar.
fn printed_elements(stdout: &str, name: &str) -> Vec<String> {
    let prefix = format!("{name} = ");
    let value = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` line:\n{stdout}"));
    let Some(line) = value.strip_prefix("tensor(") else {
        return vec![value.trim().to_string()];
    };
    let start = line.find("data=[").expect("data") + "data=[".len();
    let end = start + line[start..].find(']').expect("closing bracket");
    line[start..end]
        .split(',')
        .map(|element| element.trim().to_string())
        .filter(|element| !element.is_empty())
        .collect()
}

/// Run `source` on the evaluator and the C lane and return both outputs.
fn both_lanes(source: &str, stem: &str) -> [(&'static str, String); 2] {
    [
        ("eval", eval(source, stem)),
        ("C", build_and_run(source, stem)),
    ]
}

/// A magnitude `b` with `b + 1 == b` and `-b + 1 == -b` at `dtype`, so the
/// pairing of the tree decides whether the ones survive. 1e20 overflows f16.
fn absorbing(dtype: &str) -> &'static str {
    if dtype == "f16" {
        "4096.0"
    } else {
        "100000000000000000000.0"
    }
}

fn signed_zero_program(dtype: &str) -> String {
    let d = dtype;
    let b = absorbing(dtype);
    let negative_b = format!("-{b}");
    // The six coefficients of `six_uses`, each one use's contribution.
    let six = [b, "1.0", negative_b.as_str(), "1.0", "1.0", "1.0"];
    let lit = |value: &str| format!("scalar_to_tensor({value}{d})");
    let host_six = if matches!(dtype, "f32" | "f64") {
        // `sum` accumulates f32 and f64 at their own width, so it is the same
        // adjacent-pair tree at the same dtype.
        format!(
            "sum(to_tensor([{}]), 0i32)",
            six.iter()
                .map(|value| format!("{value}{d}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        // `sum` accumulates f16 and bf16 in f32; §2.4 finalizes every addition
        // at the leaf's dtype, so the host tree is spelled out pair by pair.
        format!(
            "add(add(add({}, {}), add({}, {})), add({}, {}))",
            lit(six[0]),
            lit(six[1]),
            lit(six[2]),
            lit(six[3]),
            lit(six[4]),
            lit(six[5])
        )
    };
    format!(
        r#"
def loss(a: tensor[2, {d}], w: tensor[2, {d}]) -> tensor[{d}] = sum(mul(a, w), 0i32)
def product(a: tensor[2, {d}], w: tensor[2, {d}]) -> tensor[2, {d}] = mul(a, w)
def loss_twice(a: tensor[2, {d}], w: tensor[2, {d}]) -> tensor[{d}] = add(sum(mul(a, w), 0i32), sum(mul(a, w), 0i32))
def loss_mixed(a: tensor[2, {d}], w: tensor[2, {d}], v: tensor[2, {d}]) -> tensor[{d}] = add(sum(mul(a, w), 0i32), sum(mul(a, v), 0i32))
def loss_relu(a: tensor[2, {d}], v: tensor[2, {d}]) -> tensor[{d}] = sum(relu(mul(a, v)), 0i32)
def loss_unused(a: tensor[2, {d}], b: tensor[2, {d}]) -> tensor[{d}] = sum(mul(a, to_tensor([-0.0{d}, 1.0{d}])), 0i32)
def loss_compare(a: tensor[2, {d}], w: tensor[2, {d}]) -> tensor[{d}] = sum(where(lt(a, w), w, w), 0i32)
def loss_pad(x: tensor[2, {d}], w: tensor[4, {d}]) -> tensor[{d}] = sum(mul(pad(x, [[1i64, 1i64]], 0.0{d}), w), 0i32)
def loss_shared(t: tensor[2, {d}], y: tensor[2, {d}], w: tensor[2, {d}]) -> tensor[{d}] = sum(mul(add(t, y), w), 0i32)
def four_uses(x: tensor[{d}]) -> tensor[{d}] = add(add(mul(x, {c1}), mul(x, {c2})), add(mul(x, {c3}), mul(x, {c4})))
def six_uses(x: tensor[{d}]) -> tensor[{d}] = add(add(add(mul(x, {s0}), mul(x, {s1})), add(mul(x, {s2}), mul(x, {s3}))), add(mul(x, {s4}), mul(x, {s5})))
def loss_capture[n](theta: tensor[n, {d}], scale: tensor[1, {d}]) -> {d} = {{
  s = tensor_to_scalar(cast(sum(scale, 0i32), {d}))
  weights = to_tensor(map(fn (i: i64) -> mul(s, 1.0{d}), range(0i64, shape(theta, 0i32))))
  tensor_to_scalar(sum(mul(theta, weights), 0i32))
}}
forward = product(to_tensor([1.0{d}, 2.0{d}]), to_tensor([-0.0{d}, 1.0{d}]))
single = grad(loss, wrt=a)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([-0.0{d}, 1.0{d}]))
twice = grad(loss_twice, wrt=a)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([-0.0{d}, 1.0{d}]))
mixed = grad(loss_mixed, wrt=a)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([-0.0{d}, 1.0{d}]), to_tensor([0.0{d}, 1.0{d}]))
dead_relu = grad(loss_relu, wrt=a)(to_tensor([1.0{d}, 1.0{d}]), to_tensor([-1.0{d}, 2.0{d}]))
unused = grad(loss_unused, wrt=b)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([1.0{d}, 2.0{d}]))
compared = grad(loss_compare, wrt=a)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([-0.0{d}, 3.0{d}]))
shared = grad(loss_shared)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([3.0{d}, 4.0{d}]), to_tensor([-0.0{d}, 1.0{d}]))
padded = grad(loss_pad, wrt=x)(to_tensor([1.0{d}, 2.0{d}]), to_tensor([5.0{d}, -0.0{d}, 1.0{d}, 7.0{d}]))
four = grad(four_uses)(scalar_to_tensor(1.0{d}))
six = grad(six_uses)(scalar_to_tensor(1.0{d}))
host_six = {host_six}
capture_one = grad(loss_capture, wrt=scale)(to_tensor([-0.0{d}]), to_tensor([2.0{d}]))
capture_two = grad(loss_capture, wrt=scale)(to_tensor([-0.0{d}, -0.0{d}]), to_tensor([2.0{d}]))
capture_mixed = grad(loss_capture, wrt=scale)(to_tensor([-0.0{d}, 0.0{d}]), to_tensor([2.0{d}]))
capture_empty = grad(loss_capture, wrt=scale)(insert(scalar_to_tensor(0.0{d}), 0i32, 0i64), to_tensor([2.0{d}]))
"#,
        c1 = lit(b),
        c2 = lit(&negative_b),
        c3 = lit("1.0"),
        c4 = lit("1.0"),
        s0 = lit(six[0]),
        s1 = lit(six[1]),
        s2 = lit(six[2]),
        s3 = lit(six[3]),
        s4 = lit(six[4]),
        s5 = lit(six[5]),
    )
}

#[test]
fn gradient_accumulation_has_no_base_leaf_on_every_lane_and_float_width() {
    for dtype in DTYPES {
        let source = signed_zero_program(dtype);
        for (lane, stdout) in both_lanes(&source, &format!("no_base_leaf_{dtype}")) {
            let check = |name: &str, expected: &[&str], why: &str| {
                assert_eq!(
                    printed_elements(&stdout, name),
                    expected,
                    "{dtype} {lane} `{name}`: {why}\n{stdout}"
                );
            };
            check("forward", &["-0.0", "2.0"], "the forward product keeps its -0");
            check("single", &["-0.0", "1.0"], "one -0 contribution is the adjoint unchanged");
            check("twice", &["-0.0", "2.0"], "(-0) + (-0) is -0");
            check("dead_relu", &["-0.0", "2.0"], "a dead unit's +0 times a negative input is -0");
            for target in ["shared.0", "shared.1"] {
                check(target, &["-0.0", "1.0"], "targets sharing one adjoint node each keep it");
            }
            check("padded", &["-0.0", "1.0"], "pad's adjoint shrinks the -0 interior cotangent");
            check("shared.2", &["4.0", "6.0"], "the third root is its own adjoint");
            check("four", &["2.0"], "(b + -b) + (1 + 1): no base leaf absorbs a one");
            check("six", &["2.0"], "((b + 1) + (-b + 1)) + (1 + 1) at the leaf's dtype");
            assert_eq!(
                printed_elements(&stdout, "six"),
                printed_elements(&stdout, "host_six"),
                "{dtype} {lane}: the adjoint is the host tree over the same contributions"
            );
            check("capture_one", &["-0.0"], "one List-capture row is the adjoint unchanged");
            check("capture_two", &["-0.0"], "two -0 List-capture rows add to -0");
            // Negative parity: +0 wherever a +0 enters or nothing is queued.
            check("mixed", &["0.0", "2.0"], "(-0) + (+0) is +0");
            check("unused", &["0.0", "0.0"], "an unused parameter receives exact +0");
            check("compared", &["0.0", "0.0"], "a comparison contributes exact +0");
            check("capture_mixed", &["0.0"], "(-0) + (+0) List-capture rows add to +0");
            check("capture_empty", &["0.0"], "an empty List capture receives exact +0");
        }
    }
}
