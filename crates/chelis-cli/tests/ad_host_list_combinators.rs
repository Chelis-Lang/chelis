// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#![allow(clippy::disallowed_methods)]
use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write source");
}

fn eval_source(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        output.status.success(),
        "chelis eval failed for {name}: status={}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

fn build_and_run_c_source(source: &str, name: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("C build should run");
    assert!(
        built.status.success(),
        "C build failed for {name}: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "C link failed for {name}: {status}");
    std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled C executable should run")
}

fn parse_tensor_data(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|line| line.starts_with(&prefix) || line.starts_with("tensor("))
        .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}"));
    let data_marker = "data=[";
    let start = line
        .find(data_marker)
        .unwrap_or_else(|| panic!("no data marker in line: {line}"))
        + data_marker.len();
    let end = line[start..]
        .find(']')
        .unwrap_or_else(|| panic!("no data terminator in line: {line}"));
    line[start..start + end]
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<f64>().expect("numeric tensor data"))
        .collect()
}

fn parse_numeric_result(stdout: &str, name: &str) -> f64 {
    let tensor_prefix = format!("{name} = tensor(");
    if stdout
        .lines()
        .any(|line| line.starts_with(&tensor_prefix) || line.starts_with("tensor("))
    {
        let values = parse_tensor_data(stdout, name);
        assert_eq!(
            values.len(),
            1,
            "{name}: expected scalar or rank-0 tensor output, got {values:?}"
        );
        return values[0];
    }

    let prefix = format!("{name} = ");
    let payload = match stdout.lines().find(|line| line.starts_with(&prefix)) {
        Some(line) => &line[prefix.len()..],
        // [05-OBS-4] (chelis#732 P1): a single scalar/rank-0 root renders
        // bare, so the anonymous single-root output is the value line.
        None => stdout
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}")),
    };
    payload
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|err| panic!("numeric output parse failed for `{payload}`: {err}"))
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={actual:?} expected={expected:?}"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{label}: element {index} actual={actual} expected={expected} diff={diff} tol={tol}",
        );
    }
}

fn cdf_identity_black_scholes_price(spot: f64) -> f64 {
    let strike: f64 = 100.0;
    let rate: f64 = 0.05;
    let time: f64 = 1.25;
    let sigma: f64 = 0.3;
    let root_time = time.sqrt();
    let vol_root_time = sigma * root_time;
    let carry = (rate + 0.5 * sigma * sigma) * time;
    let discount = (-rate * time).exp();
    let d1 = ((spot / strike).ln() + carry) / vol_root_time;
    let d2 = d1 - vol_root_time;
    spot * d1 - strike * discount * d2
}

fn cdf_identity_black_scholes_delta(spot: f64) -> f64 {
    let strike: f64 = 100.0;
    let rate: f64 = 0.05;
    let time: f64 = 1.25;
    let sigma: f64 = 0.3;
    let root_time = time.sqrt();
    let vol_root_time = sigma * root_time;
    let carry = (rate + 0.5 * sigma * sigma) * time;
    let discount = (-rate * time).exp();
    let d1 = ((spot / strike).ln() + carry) / vol_root_time;
    d1 + (1.0 / vol_root_time) - (strike * discount) / (spot * vol_root_time)
}

fn cdf_identity_black_scholes_gamma(spot: f64) -> f64 {
    let strike: f64 = 100.0;
    let rate: f64 = 0.05;
    let time: f64 = 1.25;
    let sigma: f64 = 0.3;
    let root_time = time.sqrt();
    let vol_root_time = sigma * root_time;
    let discount = (-rate * time).exp();
    (1.0 / (spot * vol_root_time)) + (strike * discount) / (spot * spot * vol_root_time)
}

fn finite_difference_delta(spot: f64) -> f64 {
    let step = 1e-3;
    (cdf_identity_black_scholes_price(spot + step) - cdf_identity_black_scholes_price(spot - step))
        / (2.0 * step)
}

fn finite_difference_gamma(spot: f64) -> f64 {
    let step = 1e-3;
    (cdf_identity_black_scholes_delta(spot + step) - cdf_identity_black_scholes_delta(spot - step))
        / (2.0 * step)
}

const BLACK_SCHOLES_BODY: &str = "\
def bs_price(spot: f32) -> f32 = {
  strike = cast(100.0, f32)
  rate = cast(0.05, f32)
  time = cast(1.25, f32)
  sigma = cast(0.3, f32)
  root_time = sqrt(time)
  vol_root_time = mul(sigma, root_time)
  carry = mul(add(rate, mul(cast(0.5, f32), mul(sigma, sigma))), time)
  discount = exp(neg(mul(rate, time)))
  d1 = div(add(log(div(spot, strike)), carry), vol_root_time)
  d2 = sub(d1, vol_root_time)
  sub(mul(spot, d1), mul(mul(strike, discount), d2))
}

def bs_total(spots: tensor[3, f32]) -> f32 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f32) -> bs_price(spot), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one(spots: tensor[1, f32]) -> f32 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f32) -> bs_price(spot), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one_delta(spots: tensor[1, f32]) -> f32 =
  tensor_to_scalar(sum(grad(bs_total_one)(spots), cast(0, i32)))
";

const BLACK_SCHOLES_F64_ARGS_BODY: &str = "\
def bs_price(spot: f64, strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  root_time = sqrt(time)
  vol_root_time = mul(sigma, root_time)
  carry = mul(add(rate, mul(cast(0.5, f64), mul(sigma, sigma))), time)
  discount = exp(neg(mul(rate, time)))
  d1 = div(add(log(div(spot, strike)), carry), vol_root_time)
  d2 = sub(d1, vol_root_time)
  sub(mul(spot, d1), mul(mul(strike, discount), d2))
}

def bs_total(spots: tensor[3, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f64) -> bs_price(spot, strike, rate, sigma, time), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f64) -> bs_price(spot, strike, rate, sigma, time), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one_delta(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 =
  tensor_to_scalar(sum(grad(bs_total_one, wrt=spots)(spots, strike, rate, sigma, time), cast(0, i32)))
";

#[test]
fn eval_grad_through_to_list_to_tensor_identity_returns_identity_cotangent() {
    let stdout = eval_source(
        "\
def loss(x: tensor[3, f32]) -> f32 = {
  y = to_tensor(to_list(copy(x)))
  tensor_to_scalar(sum(y, cast(0, i32)))
}
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
",
        "ad_boundary_identity",
    );
    assert_close(
        &parse_tensor_data(&stdout, "out"),
        &[1.0, 1.0, 1.0],
        1e-6,
        "boundary identity",
    );
}

#[test]
fn eval_grad_through_to_list_map_black_scholes_f64_args_matches_analytic_delta() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_F64_ARGS_BODY}
out = grad(bs_total, wrt=spots)(
  to_tensor([cast(90.0, f64), cast(100.0, f64), cast(110.0, f64)]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
        ),
        "ad_map_black_scholes_f64_args_delta",
    );
    let actual = parse_tensor_data(&stdout, "out");
    let expected = [90.0, 100.0, 110.0]
        .into_iter()
        .map(cdf_identity_black_scholes_delta)
        .collect::<Vec<_>>();
    assert_close(&actual, &expected, 1e-10, "Black-Scholes f64 args delta");
}

#[test]
fn eval_grad_through_to_list_map_black_scholes_matches_analytic_and_fd_delta() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_BODY}
out = grad(bs_total)(to_tensor([cast(90.0, f32), cast(100.0, f32), cast(110.0, f32)]))
"
        ),
        "ad_map_black_scholes_delta",
    );
    let actual = parse_tensor_data(&stdout, "out");
    let spots = [90.0, 100.0, 110.0];
    let analytic = spots
        .iter()
        .copied()
        .map(cdf_identity_black_scholes_delta)
        .collect::<Vec<_>>();
    let finite_difference = spots
        .iter()
        .copied()
        .map(finite_difference_delta)
        .collect::<Vec<_>>();
    assert_close(&actual, &analytic, 2e-4, "Black-Scholes delta analytic");
    assert_close(
        &actual,
        &finite_difference,
        2e-4,
        "Black-Scholes delta finite difference",
    );
}

#[test]
fn eval_nested_grad_named_black_scholes_f64_args_matches_analytic_gamma() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_F64_ARGS_BODY}
out = grad(bs_total_one_delta, wrt=spots)(
  to_tensor([cast(100.0, f64)]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
        ),
        "ad_map_black_scholes_f64_args_gamma",
    );
    let actual = parse_tensor_data(&stdout, "out");
    let expected = [cdf_identity_black_scholes_gamma(100.0)];
    assert_close(&actual, &expected, 1e-10, "Black-Scholes f64 args gamma");
}

#[test]
fn eval_nested_grad_through_to_list_map_black_scholes_matches_analytic_and_fd_gamma() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_BODY}
out = grad(bs_total_one_delta)(to_tensor([cast(100.0, f32)]))
"
        ),
        "ad_map_black_scholes_gamma",
    );
    let actual = parse_tensor_data(&stdout, "out");
    let analytic = [cdf_identity_black_scholes_gamma(100.0)];
    let finite_difference = [finite_difference_gamma(100.0)];
    assert_close(&actual, &analytic, 2e-4, "Black-Scholes gamma analytic");
    assert_close(
        &actual,
        &finite_difference,
        2e-4,
        "Black-Scholes gamma finite difference",
    );
}

#[test]
fn eval_scalar_black_scholes_f64_delta_can_be_read_as_rank0_greek() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_F64_ARGS_BODY}
out = grad(bs_price, wrt=spot)(
  cast(100.0, f64),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
        ),
        "ad_scalar_black_scholes_f64_delta",
    );
    let actual = parse_numeric_result(&stdout, "out");
    let expected = cdf_identity_black_scholes_delta(100.0);
    assert_close(
        &[actual],
        &[expected],
        1e-10,
        "scalar Black-Scholes f64 delta",
    );
}

#[test]
fn eval_vmap_grad_black_scholes_f64_args_matches_unbatched_deltas() {
    let stdout = eval_source(
        &format!(
            "{BLACK_SCHOLES_F64_ARGS_BODY}
def bs_total_one_batched(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 =
  bs_total_one(spots, strike, rate, sigma, time)
out = vmap(grad(bs_total_one_batched, wrt=spots))(
  to_tensor([[cast(90.0, f64)], [cast(100.0, f64)], [cast(110.0, f64)]]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
        ),
        "ad_vmap_black_scholes_f64_args",
    );
    let actual = parse_tensor_data(&stdout, "out");
    let expected = [90.0, 100.0, 110.0]
        .into_iter()
        .map(cdf_identity_black_scholes_delta)
        .collect::<Vec<_>>();
    assert_close(
        &actual,
        &expected,
        1e-10,
        "vmap Black-Scholes f64 args delta",
    );
}

#[test]
fn eval_vmap_of_grad_composed_with_map_matches_unbatched_gradients() {
    let stdout = eval_source(
        "\
def loss(row: tensor[3, f32]) -> f32 = {
  xs = to_list(copy(row))
  ys = map(fn (v: f32) -> mul(v, v), xs)
  tensor_to_scalar(sum(to_tensor(ys), cast(0, i32)))
}
rows = to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]])
batched = vmap(grad(loss))(rows)
single0 = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
single1 = grad(loss)(to_tensor([cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]))
",
        "ad_vmap_map",
    );
    assert_close(
        &parse_tensor_data(&stdout, "batched"),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        1e-5,
        "batched gradient",
    );
    assert_close(
        &parse_tensor_data(&stdout, "single0"),
        &[2.0, 4.0, 6.0],
        1e-5,
        "single0",
    );
    assert_close(
        &parse_tensor_data(&stdout, "single1"),
        &[8.0, 10.0, 12.0],
        1e-5,
        "single1",
    );
}

#[test]
fn eval_grad_through_filter_uses_primal_mask_as_constant() {
    let stdout = eval_source(
        "\
def loss(x: tensor[4, f32]) -> f32 = {
  selected = to_tensor(filter(fn (v: f32) -> gt(v, cast(0.0, f32)), to_list(copy(x))))
  squares = map(fn (v: f32) -> mul(v, v), to_list(selected))
  tensor_to_scalar(sum(to_tensor(squares), cast(0, i32)))
}
out = grad(loss)(to_tensor([cast(-2.0, f32), cast(3.0, f32), cast(0.5, f32), cast(-1.0, f32)]))
",
        "ad_filter_mask",
    );
    assert_close(
        &parse_tensor_data(&stdout, "out"),
        &[0.0, 6.0, 1.0, 0.0],
        1e-5,
        "filter mask",
    );
}

#[test]
fn eval_grad_through_fold_uses_reverse_scan_cotangents() {
    let stdout = eval_source(
        "\
def loss(x: tensor[3, f32]) -> f32 =
  fold(fn (acc: f32, v: f32) -> add(mul(acc, cast(0.5, f32)), mul(v, v)), cast(1.0, f32), to_list(copy(x)))
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
",
        "ad_fold_recurrence",
    );
    assert_close(
        &parse_tensor_data(&stdout, "out"),
        &[0.5, 2.0, 6.0],
        1e-5,
        "fold recurrence",
    );
}

fn eval_failure(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        !output.status.success(),
        "chelis eval unexpectedly succeeded for {name}:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

// chelis#3343: [05-OP-55] fold/scan over a List-typed `grad` parameter reverse
// the executed recurrence. These Lists are staged as Cons/Nil spines of tensor
// leaves, which the scalar `to_list(tensor)` fold path cannot read.

#[test]
fn eval_grad_through_fold_over_tensor_list_matches_literal_index_unrolling() {
    let stdout = eval_source(
        "\
def fold_loss(x: tensor[1, 2, f32], ws: List[tensor[2, 2, f32]]) -> f32 = {
  o = fold(fn (acc: tensor[1, 2, f32], w: tensor[2, 2, f32]) -> matmul(acc, w), x, ws)
  tensor_to_scalar(sum(sum(o, 1i32), 0i32))
}
def index_loss(x: tensor[1, 2, f32], ws: List[tensor[2, 2, f32]]) -> f32 = {
  o = matmul(matmul(x, index(ws, 0i64)), index(ws, 1i64))
  tensor_to_scalar(sum(sum(o, 1i32), 0i32))
}
def ws() -> List[tensor[2, 2, f32]] = [to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), to_tensor([[1.0f32, 0.0f32], [0.0f32, 1.0f32]])]
def x() -> tensor[1, 2, f32] = to_tensor([[1.0f32, 1.0f32]])
gf = grad(fold_loss, wrt=ws)(x(), ws())
gi = grad(index_loss, wrt=ws)(x(), ws())
f0 = index(gf, 0i64)
f1 = index(gf, 1i64)
i0 = index(gi, 0i64)
i1 = index(gi, 1i64)
",
        "ad_fold_tensor_list",
    );
    let f0 = parse_tensor_data(&stdout, "f0");
    let f1 = parse_tensor_data(&stdout, "f1");
    // d/dW0 = x^T (W1 1)^T and d/dW1 = (x W0)^T 1^T for loss = sum(x W0 W1).
    assert_close(&f0, &[1.0, 1.0, 1.0, 1.0], 1e-6, "fold d/dw0");
    assert_close(&f1, &[4.0, 4.0, 6.0, 6.0], 1e-6, "fold d/dw1");
    assert_close(
        &f0,
        &parse_tensor_data(&stdout, "i0"),
        0.0,
        "fold == index d/dw0",
    );
    assert_close(
        &f1,
        &parse_tensor_data(&stdout, "i1"),
        0.0,
        "fold == index d/dw1",
    );
}

#[test]
fn eval_grad_through_fold_over_tensor_list_with_tanh_agrees_with_finite_differences() {
    let stdout = eval_source(
        "\
def fold_loss[bt, h](x: tensor[bt, h, f32], ws: List[tensor[h, h, f32]]) -> f32 = {
  o = fold(fn (acc: tensor[bt, h, f32], w: tensor[h, h, f32]) -> tanh(matmul(acc, w)), x, ws)
  tensor_to_scalar(sum(sum(mul(o, o), 1i32), 0i32))
}
def index_loss[bt, h](x: tensor[bt, h, f32], ws: List[tensor[h, h, f32]]) -> f32 = {
  o = tanh(matmul(tanh(matmul(x, index(ws, 0i64))), index(ws, 1i64)))
  tensor_to_scalar(sum(sum(mul(o, o), 1i32), 0i32))
}
def px() -> tensor[1, 2, f32] = to_tensor([[0.5f32, -0.25f32]])
def pws() -> List[tensor[2, 2, f32]] = [to_tensor([[0.5f32, 0.25f32], [-0.5f32, 1.0f32]]), to_tensor([[1.0f32, -0.25f32], [0.125f32, 0.5f32]])]
gf = grad(fold_loss, wrt=ws)(px(), pws())
gi = grad(index_loss, wrt=ws)(px(), pws())
f0 = index(gf, 0i64)
f1 = index(gf, 1i64)
i0 = index(gi, 0i64)
i1 = index(gi, 1i64)
",
        "ad_fold_tensor_list_tanh",
    );
    let x = [0.5_f64, -0.25];
    let ws = [[[0.5_f64, 0.25], [-0.5, 1.0]], [[1.0, -0.25], [0.125, 0.5]]];
    let loss = |ws: &[[[f64; 2]; 2]; 2]| {
        let mut acc = x;
        for w in ws {
            acc = [
                (acc[0] * w[0][0] + acc[1] * w[1][0]).tanh(),
                (acc[0] * w[0][1] + acc[1] * w[1][1]).tanh(),
            ];
        }
        acc[0] * acc[0] + acc[1] * acc[1]
    };
    let h = 1e-6;
    for (layer, name) in [(0, "f0"), (1, "f1")] {
        let mut fd = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let (mut plus, mut minus) = (ws, ws);
                plus[layer][i][j] += h;
                minus[layer][i][j] -= h;
                fd.push((loss(&plus) - loss(&minus)) / (2.0 * h));
            }
        }
        assert_close(&parse_tensor_data(&stdout, name), &fd, 1e-5, name);
    }
    assert_close(
        &parse_tensor_data(&stdout, "f0"),
        &parse_tensor_data(&stdout, "i0"),
        0.0,
        "fold == index d/dw0",
    );
    assert_close(
        &parse_tensor_data(&stdout, "f1"),
        &parse_tensor_data(&stdout, "i1"),
        0.0,
        "fold == index d/dw1",
    );
}

#[test]
fn eval_grad_through_fold_over_scalar_list_parameter() {
    let stdout = eval_source(
        "\
def loss(ws: List[f32]) -> f32 = fold(fn (acc: f32, w: f32) -> mul(acc, w), 2.0f32, ws)
g = grad(loss, wrt=ws)([3.0f32, 5.0f32])
g0 = index(g, 0i64)
g1 = index(g, 1i64)
",
        "ad_fold_scalar_list",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g0")],
        &[10.0],
        1e-6,
        "d/dw0",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g1")],
        &[6.0],
        1e-6,
        "d/dw1",
    );
}

#[test]
fn eval_grad_through_fold_with_tuple_accumulator_over_mapped_list() {
    let stdout = eval_source(
        "\
def loss(x: tensor[2, f32], ws: List[tensor[2, f32]]) -> f32 = {
  pair = fold(fn (acc: (tensor[2, f32], tensor[2, f32]), w: tensor[2, f32]) -> (mul(acc.0, w), add(acc.1, w)), (x, x), map(fn (v: tensor[2, f32]) -> mul(v, v), ws))
  tensor_to_scalar(sum(add(pair.0, pair.1), 0i32))
}
g = grad(loss, wrt=ws)(to_tensor([1.0f32, 2.0f32]), [to_tensor([3.0f32, 4.0f32]), to_tensor([5.0f32, 6.0f32])])
g0 = index(g, 0i64)
g1 = index(g, 1i64)
",
        "ad_fold_tuple_accumulator",
    );
    // loss = sum(x w0^2 w1^2) + sum(x + w0^2 + w1^2).
    assert_close(
        &parse_tensor_data(&stdout, "g0"),
        &[156.0, 584.0],
        1e-4,
        "d/dw0",
    );
    assert_close(
        &parse_tensor_data(&stdout, "g1"),
        &[100.0, 396.0],
        1e-4,
        "d/dw1",
    );
}

#[test]
fn eval_grad_through_scan_over_tensor_list_differentiates_every_state() {
    let stdout = eval_source(
        "\
def loss(x: tensor[2, f32], ws: List[tensor[2, f32]]) -> f32 = {
  states = scan(fn (acc: tensor[2, f32], w: tensor[2, f32]) -> mul(acc, w), x, ws)
  tensor_to_scalar(sum(index(states, 0i64), 0i32)) + tensor_to_scalar(sum(index(states, 1i64), 0i32))
}
g = grad(loss, wrt=ws)(to_tensor([1.0f32, 2.0f32]), [to_tensor([3.0f32, 4.0f32]), to_tensor([5.0f32, 6.0f32])])
g0 = index(g, 0i64)
g1 = index(g, 1i64)
",
        "ad_scan_tensor_list",
    );
    // States are x w0 and x w0 w1, so d/dw0 = x + x w1 and d/dw1 = x w0.
    assert_close(
        &parse_tensor_data(&stdout, "g0"),
        &[6.0, 14.0],
        1e-6,
        "d/dw0",
    );
    assert_close(
        &parse_tensor_data(&stdout, "g1"),
        &[3.0, 8.0],
        1e-6,
        "d/dw1",
    );
}

#[test]
fn eval_grad_through_fold_over_runtime_length_list_is_rejected_loudly() {
    let stderr = eval_failure(
        "\
def loss(x: tensor[2, f32], wss: List[List[tensor[2, f32]]], k: tensor[i64]) -> f32 = tensor_to_scalar(sum(fold(fn (acc: tensor[2, f32], w: tensor[2, f32]) -> mul(acc, w), x, index(wss, tensor_to_scalar(k))), 0i32))
g = grad(loss, wrt=wss)(to_tensor([1.0f32, 2.0f32]), [[to_tensor([3.0f32, 4.0f32])], [to_tensor([5.0f32, 6.0f32]), to_tensor([7.0f32, 8.0f32])]], scalar_to_tensor(1i64))
",
        "ad_fold_runtime_length_list",
    );
    assert!(
        stderr.contains(&format!("unsupported: `fold` {UNSTAGED_LIST_REJECTION}")),
        "a runtime-length List must be a loud fold rejection, not a placeholder:\n{stderr}"
    );
}

const UNSTAGED_LIST_REJECTION: &str = "over a List that is not staged as known items on the \
     List staging of a `grad` or `vmap` body (lowering); unimplemented chelis#3366";

#[test]
fn vmap_fold_over_runtime_length_list_names_the_vmap_context() {
    let stderr = eval_failure(
        "\
def k(t: tensor[i64]) -> tensor[f32] = scalar_to_tensor(fold(fn (acc: f32, i: i64) -> acc + 1.0f32, 0.0f32, range(0i64, tensor_to_scalar(t))))
v = vmap(k)(to_tensor([1i64, 2i64]))
",
        "vmap_fold_runtime_length_list",
    );
    assert!(
        stderr.contains("could not lower `vmap(...)`")
            && stderr
                .contains("runtime range inside vmap requires a shared cardinality representation"),
        "the rejection must not claim a differentiated body under plain vmap:\n{stderr}"
    );
}

// chelis#3366: [05-OP-55] reverses exactly the runtime recurrence. The bound
// is an argument of the differentiated function, so lowering cannot stage a
// fixed number of callback applications.
#[test]
fn eval_grad_fold_over_runtime_range_uses_executed_trip_count() {
    let stdout = eval_source(
        "\
def loss(x: f32, n: i64) -> f32 = fold(fn (acc: f32, i: i64) -> acc * x, 1.0f32, range(0i64, n))
g0 = grad(loss, wrt=x)(2.0f32, 0i64)
g1 = grad(loss, wrt=x)(2.0f32, 1i64)
g3 = grad(loss, wrt=x)(2.0f32, 3i64)
",
        "ad_fold_runtime_range",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g0")],
        &[0.0],
        1e-6,
        "zero steps",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g1")],
        &[1.0],
        1e-6,
        "one step",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g3")],
        &[12.0],
        1e-6,
        "three steps",
    );
}

#[test]
fn eval_grad_scan_over_runtime_range_routes_every_recorded_state() {
    let stdout = eval_source(
        "\
def loss(x: f32, n: i64) -> f32 = {
  states = scan(fn (acc: f32, i: i64) -> acc * x, 1.0f32, range(0i64, n))
  fold(fn (acc: f32, state: f32) -> acc + state, 0.0f32, states)
}
g = grad(loss, wrt=x)(2.0f32, 3i64)
",
        "ad_scan_runtime_range",
    );
    assert_close(
        &[parse_numeric_result(&stdout, "g")],
        &[17.0],
        1e-6,
        "scan states",
    );
}

#[test]
fn c_grad_fold_over_runtime_range_matches_executed_trip_count() {
    let source = "\
def loss(theta: tensor[1, f32], n: i64) -> f32 = fold(fn (acc: f32, i: i64) -> acc * tensor_to_scalar(sum(theta, 0)), 1.0f32, range(0i64, n))
def derivative(model: tensor[1, f32] -> i64 -> f32, n: i64, theta: tensor[1, f32]) -> tensor[1, f32] = {
  target = fn (local: tensor[1, f32]) -> model(local, n)
  grad(target, wrt=local)(theta)
}
g0 = derivative(loss, 0i64, to_tensor([2.0], f32))
g1 = derivative(loss, 1i64, to_tensor([2.0], f32))
g3 = derivative(loss, 3i64, to_tensor([2.0], f32))
";
    let evaluated = eval_source(source, "runtime_range_c_parity_eval");
    let compiled = build_and_run_c_source(source, "runtime_range_c_parity");
    assert!(
        compiled.status.success(),
        "C gradient failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let c_stdout = String::from_utf8(compiled.stdout).expect("C stdout utf-8");
    for (name, expected) in [("g0", 0.0), ("g1", 1.0), ("g3", 12.0)] {
        assert_close(
            &parse_tensor_data(&evaluated, name),
            &[expected],
            1e-6,
            name,
        );
        assert_close(&parse_tensor_data(&c_stdout, name), &[expected], 1e-6, name);
    }
}

#[test]
fn c_grad_scan_over_runtime_range_reverses_recorded_states() {
    let source = "\
def loss(theta: tensor[1, f32], n: i64) -> f32 = {
  x = tensor_to_scalar(sum(theta, 0))
  states = scan(fn (acc: f32, i: i64) -> acc * x, 1.0f32, range(0i64, n))
  fold(fn (acc: f32, state: f32) -> acc + state, 0.0f32, states)
}
def derivative(model: tensor[1, f32] -> i64 -> f32, n: i64, theta: tensor[1, f32]) -> tensor[1, f32] = {
  target = fn (local: tensor[1, f32]) -> model(local, n)
  grad(target, wrt=local)(theta)
}
g = derivative(loss, 3i64, to_tensor([2.0], f32))
";
    let evaluated = eval_source(source, "runtime_scan_c_parity_eval");
    let compiled = build_and_run_c_source(source, "runtime_scan_c_parity");
    assert!(
        compiled.status.success(),
        "C scan gradient failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let c_stdout = String::from_utf8(compiled.stdout).expect("C stdout utf-8");
    assert_close(
        &parse_tensor_data(&evaluated, "g"),
        &[17.0],
        1e-6,
        "eval scan",
    );
    assert_close(&parse_tensor_data(&c_stdout, "g"), &[17.0], 1e-6, "C scan");
}

#[test]
fn runtime_range_callback_trap_occurs_only_at_visited_step() {
    let source = "\
def loss(theta: tensor[1, f32], n: i64) -> f32 = fold(fn (acc: f32, i: i64) -> acc * tensor_to_scalar(sum(theta, 0)) + cast(trunc_div(1i64, sub(1i64, i)), f32), 1.0f32, range(0i64, n))
def derivative(model: tensor[1, f32] -> i64 -> f32, n: i64, theta: tensor[1, f32]) -> tensor[1, f32] = {
  target = fn (local: tensor[1, f32]) -> model(local, n)
  grad(target, wrt=local)(theta)
}
g = derivative(loss, __TRIP_COUNT__i64, to_tensor([2.0], f32))
";
    for (count, expected) in [("0", Some(0.0)), ("1", Some(1.0)), ("2", None)] {
        let source = source.replace("__TRIP_COUNT__", count);
        let name = format!("runtime_range_trap_{count}");
        let eval = {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join(format!("{name}.ch"));
            write_file(&path, &source);
            Command::cargo_bin("chelis")
                .expect("binary")
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file", path.to_str().unwrap()])
                .output()
                .expect("eval should run")
        };
        let compiled = build_and_run_c_source(&source, &name);
        if let Some(gradient) = expected {
            assert!(
                eval.status.success(),
                "eval failed: {}",
                String::from_utf8_lossy(&eval.stderr)
            );
            assert!(
                compiled.status.success(),
                "C failed: {}",
                String::from_utf8_lossy(&compiled.stderr)
            );
            assert_close(
                &parse_tensor_data(&String::from_utf8_lossy(&eval.stdout), "g"),
                &[gradient],
                1e-6,
                "eval visited steps",
            );
            assert_close(
                &parse_tensor_data(&String::from_utf8_lossy(&compiled.stdout), "g"),
                &[gradient],
                1e-6,
                "C visited steps",
            );
        } else {
            assert!(!eval.status.success(), "eval visited step must trap");
            assert!(!compiled.status.success(), "C visited step must trap");
            for stderr in [&eval.stderr, &compiled.stderr] {
                assert!(
                    String::from_utf8_lossy(stderr)
                        .contains("division by zero in trunc_div at i64"),
                    "wrong trap: {}",
                    String::from_utf8_lossy(stderr)
                );
            }
        }
    }
}

#[test]
fn c_grad_fold_over_runtime_adt_columns_preserves_field_cotangents() {
    let source = "\
type Block =
  | Block { w: f32, bias: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = fold(fn (acc: f32, blk: Block) -> add(mul(acc, blk.w), blk.bias), 2.0f32, copy(p.blocks))
def step(p: Model) -> Model = grad(loss)(p)
two = step(Model { blocks: [Block { w: 3.0f32, bias: 1.0f32 }, Block { w: 4.0f32, bias: 2.0f32 }] })
empty = step(Model { blocks: [] })
";
    let evaluated = eval_source(source, "runtime_adt_columns_eval");
    assert!(
        evaluated.contains("two = Model([Block(8.0, 4.0), Block(7.0, 1.0)])"),
        "{evaluated}"
    );
    assert!(evaluated.contains("empty = Model([])"), "{evaluated}");
    let compiled = build_and_run_c_source(source, "runtime_adt_columns");
    assert!(
        compiled.status.success(),
        "C ADT gradient failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8(compiled.stdout).expect("C stdout utf-8"),
        evaluated
    );
}

#[test]
fn c_grad_runtime_adt_fold_with_two_axis_means_matches_eval() {
    let source = "\
type Block =
  | Block { w: tensor[1, 1, 7, 7, f32] }
type Model =
  | Model { blocks: List[Block] }
def loss(p: Model) -> f32 = {
  init = reshape(insert(scalar_to_tensor(1.0f32), 0i32, 49i64), [1i64, 1i64, 7i64, 7i64])
  y = fold(fn (acc: tensor[1, 1, 7, 7, f32], blk: Block) -> add(acc, blk.w), init, copy(p.blocks))
  pooled = mean(mean(y, 3i32), 2i32)
  tensor_to_scalar(sum(sum(pooled, 0i32), 0i32))
}
def gradient_total(p: Model) -> f32 = {
  g = grad(loss)(p)
  match index(g.blocks, 0i64) with {
    | Block { w } => tensor_to_scalar(sum(sum(sum(sum(w, 0i32), 0i32), 0i32), 0i32))
  }
}
p = Model { blocks: [Block { w: reshape(insert(scalar_to_tensor(2.0f32), 0i32, 49i64), [1i64, 1i64, 7i64, 7i64]) }] }
value = loss(p)
g = gradient_total(p)
";
    let evaluated = eval_source(source, "runtime_adt_two_axis_means_eval");
    let compiled = build_and_run_c_source(source, "runtime_adt_two_axis_means");
    assert!(
        compiled.status.success(),
        "C two-axis mean gradient failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let c_stdout = String::from_utf8(compiled.stdout).expect("C stdout utf-8");
    for (name, expected) in [("value", 3.0), ("g", 1.0)] {
        assert_close(
            &[parse_numeric_result(&evaluated, name)],
            &[expected],
            1e-5,
            name,
        );
        assert_close(
            &[parse_numeric_result(&c_stdout, name)],
            &[expected],
            1e-5,
            name,
        );
    }
}
