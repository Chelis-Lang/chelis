use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

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
    let sigma: f64 = 0.30;
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
    let sigma: f64 = 0.30;
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
    let sigma: f64 = 0.30;
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
  sigma = cast(0.30, f32)
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
  tensor_to_scalar(sum(to_tensor(prices), cast(0, int32)))
}

def bs_total_one(spots: tensor[1, f32]) -> f32 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f32) -> bs_price(spot), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, int32)))
}

def bs_total_one_delta(spots: tensor[1, f32]) -> f32 =
  tensor_to_scalar(sum(grad(bs_total_one)(spots), cast(0, int32)))
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
  tensor_to_scalar(sum(to_tensor(prices), cast(0, int32)))
}

def bs_total_one(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f64) -> bs_price(spot, strike, rate, sigma, time), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, int32)))
}

def bs_total_one_delta(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 =
  tensor_to_scalar(sum(grad(bs_total_one, wrt=spots)(spots, strike, rate, sigma, time), cast(0, int32)))
";

#[test]
fn eval_grad_through_to_list_to_tensor_identity_returns_identity_cotangent() {
    let stdout = eval_source(
        "\
def loss(x: tensor[3, f32]) -> f32 = {
  y = to_tensor(to_list(copy(x)))
  tensor_to_scalar(sum(y, cast(0, int32)))
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
  cast(0.30, f64),
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
  cast(0.30, f64),
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
  cast(0.30, f64),
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
  cast(0.30, f64),
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
  tensor_to_scalar(sum(to_tensor(ys), cast(0, int32)))
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
  tensor_to_scalar(sum(to_tensor(squares), cast(0, int32)))
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
