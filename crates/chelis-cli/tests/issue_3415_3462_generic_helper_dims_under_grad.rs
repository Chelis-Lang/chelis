//! chelis#3462 and chelis#3415: a generic helper's dimension variables under
//! `grad`.
//!
//! spec/04-type-system.md [04-ADT-2]: "Every use of a generic function
//! signature SHALL instantiate a fresh substitution", so one use's extents
//! cannot constrain another's. The checker leaves its own dimension
//! variables in a generic body's types: the signature's binders, and any
//! extent no binder constrains, such as a `conv` result's. Each activation
//! of the body instantiates all of them, the activation `grad` makes of the
//! function it differentiates included. A dimension-generic `gather` helper
//! called from a generic loss therefore differentiates (#3462), and one
//! helper used at two sizes inside one `grad` carries each use's own extents
//! (#3415). Within one use, actuals that disagree still trap. Gradients are
//! checked against their closed forms and a central difference, on the
//! evaluator and, where it lowers the program, the C lane.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::{build_and_run, parse_tensor_data};

fn write_file(path: &std::path::Path, source: &str) {
    common::write_file(
        path,
        &chelis_surf::format::format_source(source).expect("canonical Surf"),
    );
}

fn eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap()
}

fn eval_stdout(source: &str, stem: &str) -> String {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// A printed scalar `name = value`.
fn parse_scalar(stdout: &str, name: &str) -> f64 {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line:\n{stdout}"))
        .trim()
        .parse()
        .expect("numeric scalar")
}

/// `fd`, a central difference of a quadratic loss, equals `dot`, the
/// gradient's projection on the same direction, up to f32 rounding.
fn assert_central_difference(lane: &str, stdout: &str) {
    let (fd, dot) = (parse_scalar(stdout, "fd"), parse_scalar(stdout, "dot"));
    assert!(
        (fd - dot).abs() <= 1e-5 * dot.abs().max(1.0),
        "{lane}: central difference {fd} against gradient projection {dot}"
    );
}

fn assert_refused_on_eval(source: &str, needle: &str, stem: &str) {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "must refuse: {stderr}");
    assert!(stderr.contains(needle), "{stderr}");
    assert!(
        !String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.starts_with("out = ")),
        "no gradient may be printed"
    );
}

/// The table gradient of `sum(gather(t, ids, 0)^2)` for the reproducers'
/// table and ids: twice each row, once per time `ids` selects it.
const TABLE_GRADIENT: [f64; 12] = [
    4.0, 8.0, 12.0, 8.0, 10.0, 12.0, 28.0, 32.0, 36.0, 20.0, 22.0, 24.0,
];

const TABLE_AND_IDS: &str = r#"
def t0() -> tensor[4, 3, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32]), [4i64, 3i64])
def ids0() -> tensor[2, 3, i64] = reshape(to_tensor([0i64, 2i64, 2i64, 1i64, 0i64, 3i64]), [2i64, 3i64])
"#;

/// #3462's reproducer: the helper is generic over a record, the loss is
/// generic over the same record, and the record argument is concrete. Before
/// the fix the backward graph failed verification (`gather ... output dims
/// [.., Named("d61", Some(3))], expected [.., Lit(3)]`). The concrete-caller
/// and inline controls give the same gradient. Evaluator only: the C lane
/// does not lower `grad` over a record argument, on the release before this
/// change too.
#[test]
fn generic_gather_helper_over_a_record_differentiates_from_a_generic_loss() {
    let source = format!(
        r#"
type Embed[v, d] =
  | Embed {{ table: tensor[v, d, f32] }}
def lookup[a, s, v, d](ids: &tensor[a, s, i64], e: &Embed[v, d]) -> tensor[a, s, d, f32] =
  match e with {{
    | Embed {{ table }} => gather(table, ids, 0i32)
  }}
def eloss_generic[v, d](e: &Embed[v, d], ids: &tensor[2, 3, i64]) -> f32 = {{
  x = lookup(ids, e)
  tensor_to_scalar(sum(sum(sum(mul(&x, &x), 2i32), 1i32), 0i32))
}}
def eloss_concrete(e: &Embed[4, 3], ids: &tensor[2, 3, i64]) -> f32 = {{
  x = lookup(ids, e)
  tensor_to_scalar(sum(sum(sum(mul(&x, &x), 2i32), 1i32), 0i32))
}}
def eloss_inline[v, d](e: &Embed[v, d], ids: &tensor[2, 3, i64]) -> f32 =
  match e with {{
    | Embed {{ table }} => {{
    x = gather(table, ids, 0i32)
    tensor_to_scalar(sum(sum(sum(mul(&x, &x), 2i32), 1i32), 0i32))
  }}
  }}
{TABLE_AND_IDS}
def e0() -> Embed[4, 3] = Embed {{ table: t0() }}
def tbl(g: Embed[4, 3]) -> tensor[12, f32] =
  match g with {{
    | Embed {{ table }} => reshape(table, [12i64])
  }}
out = tbl(grad(eloss_generic)(e0(), ids0()))
out_concrete = tbl(grad(eloss_concrete)(e0(), ids0()))
out_inline = tbl(grad(eloss_inline)(e0(), ids0()))
"#
    );
    let stdout = eval_stdout(&source, "record_gather");
    for name in ["out", "out_concrete", "out_inline"] {
        assert_eq!(parse_tensor_data(&stdout, name), TABLE_GRADIENT, "{name}");
    }
}

/// #3462 without the record: `grad`'s activation of `tloss` bound none of
/// its checked binders, so the helper's result carried the checker's name
/// for `d` while its table operand carried `d` (`gather ... output dims
/// [.., Named("d57", None)], expected [.., Named("d", None)]`).
#[test]
fn generic_gather_helper_over_a_bare_table_differentiates_on_both_lanes() {
    let source = format!(
        r#"
def lookup_t[a, s, v, d](ids: &tensor[a, s, i64], t: &tensor[v, d, f32]) -> tensor[a, s, d, f32] = gather(t, ids, 0i32)
def tloss[v, d](t: &tensor[v, d, f32], ids: &tensor[2, 3, i64]) -> f32 = {{
  x = lookup_t(ids, t)
  tensor_to_scalar(sum(sum(sum(mul(&x, &x), 2i32), 1i32), 0i32))
}}
{TABLE_AND_IDS}
def dir() -> tensor[4, 3, f32] = reshape(to_tensor([1.0f32, 0.0f32, -1.0f32, 0.5f32, 0.5f32, 0.5f32, -1.0f32, 2.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32]), [4i64, 3i64])
g = grad(tloss)(t0(), ids0())
out = reshape(g, [12i64])
fd = mul(sub(tloss(add(t0(), dir()), ids0()), tloss(sub(t0(), dir()), ids0())), 0.5f32)
dot = tensor_to_scalar(sum(reshape(mul(g, dir()), [12i64]), 0i32))
"#
    );
    let lanes = [
        ("eval", eval_stdout(&source, "bare_gather")),
        ("C", build_and_run(&source, "bare_gather")),
    ];
    for (lane, stdout) in lanes {
        assert_eq!(parse_tensor_data(&stdout, "out"), TABLE_GRADIENT, "{lane}");
        assert_central_difference(lane, &stdout);
    }
}

const KERNEL: [[f64; 2]; 2] = [[1.0, 0.5], [0.25, 1.0]];
const BIAS: [f64; 2] = [0.5, -0.25];

/// `sum_j (k[c][j] + k[j][c]) z_j` at every pixel of channel `c`, for a
/// two-channel image `z` stored channel by channel.
fn symmetric_mix(z: &[f64]) -> Vec<f64> {
    let n = z.len() / 2;
    (0..2)
        .flat_map(|c| {
            (0..n).map(move |p| {
                (0..2)
                    .map(|j| (KERNEL[c][j] + KERNEL[j][c]) * z[j * n + p])
                    .sum::<f64>()
            })
        })
        .collect()
}

/// #3415 on both lanes, reduced: one spatially generic helper at 2x2 and at
/// 1x1 inside one `grad`. The checker gives the `conv` result in `sub_block`
/// extents no binder constrains, and `y` carries them to `twice`. Both
/// activations stamped the checker's one name for that height on their
/// results, so the backward graph checked the 1x1 use against the 2x2 use's
/// extent (``extent `d86`: claimed = 2, add axis 2 = 1``). `sub_block(x)` is
/// `2 (k x + b)`, so the gradient of `wsum(sub_block(x), x)` at channel `c`
/// is `2 sum_j (k[c][j] + k[j][c]) x_j + 2 b_c`.
#[test]
fn generic_helper_at_two_sizes_differentiates_on_both_lanes() {
    let source = r#"
def channel_bias[a, o, h, w](y: tensor[a, o, h, w, f32], b: &tensor[o, f32]) -> tensor[a, o, h, w, f32] = {
  bias = insert(insert(insert(b, 0i32, shape(&y, 0i32)), 2i32, shape(&y, 2i32)), 3i32, shape(&y, 3i32))
  add(y, bias)
}
def twice[a, c, h, w](x: tensor[a, c, h, w, f32]) -> tensor[a, c, h, w, f32] = add(x, copy(&x))
def wsum[a, c, h, w](y: tensor[a, c, h, w, f32], wt: tensor[a, c, h, w, f32]) -> f32 = tensor_to_scalar(sum(sum(sum(sum(mul(y, wt), 3i32), 2i32), 1i32), 0i32))
def sub_block[a, i, o, h, w](x: tensor[a, i, h, w, f32], k: &tensor[o, i, 1, 1, f32], b: &tensor[o, f32]) -> tensor[a, o, h, w, f32] = {
  y = channel_bias(conv(x, k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), b)
  twice(y)
}
def kern() -> tensor[2, 2, 1, 1, f32] = reshape(to_tensor([1.0f32, 0.5f32, 0.25f32, 1.0f32]), [2i64, 2i64, 1i64, 1i64])
def bias() -> tensor[2, f32] = to_tensor([0.5f32, -0.25f32])
def two_sub(a: tensor[1, 2, 2, 2, f32], b: tensor[1, 2, 1, 1, f32]) -> f32 = {
  k = kern()
  c = bias()
  add(wsum(sub_block(copy(&a), &k, &c), a), wsum(sub_block(copy(&b), &k, &c), b))
}
g = grad(two_sub)(reshape(to_tensor([1.0f32, 2.0f32, 4.0f32, 8.0f32, 1.0f32, 3.0f32, 5.0f32, 9.0f32]), [1i64, 2i64, 2i64, 2i64]), reshape(to_tensor([1.0f32, 3.0f32]), [1i64, 2i64, 1i64, 1i64]))
out2 = reshape(g.0, [8i64])
out1 = reshape(g.1, [2i64])
"#;
    let expected = |x: &[f64]| {
        let n = x.len() / 2;
        symmetric_mix(x)
            .into_iter()
            .enumerate()
            .map(|(index, mixed)| 2.0 * mixed + 2.0 * BIAS[index / n])
            .collect::<Vec<_>>()
    };
    for (lane, stdout) in [
        ("eval", eval_stdout(source, "two_sizes")),
        ("C", build_and_run(source, "two_sizes")),
    ] {
        assert_eq!(
            parse_tensor_data(&stdout, "out2"),
            expected(&[1.0, 2.0, 4.0, 8.0, 1.0, 3.0, 5.0, 9.0]),
            "{lane}: 2x2 use"
        );
        assert_eq!(
            parse_tensor_data(&stdout, "out1"),
            expected(&[1.0, 3.0]),
            "{lane}: 1x1 use"
        );
    }
}

const X8: [f64; 32] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 9.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 9.0,
];
const X4: [f64; 8] = [1.0, 2.0, 4.0, 8.0, 1.0, 3.0, 5.0, 9.0];

/// #3415's reproducer, with a bias exact in binary: the helper matches a
/// borrowed record and subtracts each channel's spatial sum, at 4x4 and 2x2.
/// With `C x = x - sum(x)` per channel, the loss is
/// `sum_{o,i} k[o][i] <x_o, C x_i> + (1 - n) sum_o b_o sum(x_o)`, whose
/// gradient at channel `c` is `sum_j (k[c][j] + k[j][c]) C x_j + (1 - n) b_c`.
/// The 4x4 part also equals the gradient of the helper used once, and a
/// central difference agrees. Evaluator only: the C lane does not lower the
/// borrowed local record `&c` in host position, on the release before this
/// change too.
#[test]
fn generic_record_helper_at_two_sizes_differentiates() {
    let source = r#"
type Conv[o, i, kh, kw] =
  | Conv { k: tensor[o, i, kh, kw, f32], b: tensor[o, f32] }
def channel_bias[a, o, h, w](y: tensor[a, o, h, w, f32], b: &tensor[o, f32]) -> tensor[a, o, h, w, f32] = {
  bias = insert(insert(insert(b, 0i32, shape(&y, 0i32)), 2i32, shape(&y, 2i32)), 3i32, shape(&y, 3i32))
  add(y, bias)
}
def group_center[a, c, h, w](x: tensor[a, c, h, w, f32], groups: i64, group_size: i64, eps: f32) -> tensor[a, c, h, w, f32] = {
  n_batch = shape(&x, 0i32)
  rest = mul(group_size, mul(shape(&x, 2i32), shape(&x, 3i32)))
  mean = insert(insert(reshape(insert(sum(reshape(copy(&x), [n_batch, groups, rest]), 2i32), 2i32, group_size), [shape(&x, 0i32), shape(&x, 1i32)]), 2i32, shape(&x, 2i32)), 3i32, shape(&x, 3i32))
  sub(x, mean)
}
def wsum[a, c, h, w](y: tensor[a, c, h, w, f32], wt: tensor[a, c, h, w, f32]) -> f32 = tensor_to_scalar(sum(sum(sum(sum(mul(y, wt), 3i32), 2i32), 1i32), 0i32))
def x8() -> tensor[1, 2, 4, 4, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 1.0f32, 3.0f32, 5.0f32, 7.0f32, 2.0f32, 4.0f32, 6.0f32, 9.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 1.0f32, 3.0f32, 5.0f32, 7.0f32, 2.0f32, 4.0f32, 6.0f32, 9.0f32]), [1i64, 2i64, 4i64, 4i64])
def x4() -> tensor[1, 2, 2, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32, 4.0f32, 8.0f32, 1.0f32, 3.0f32, 5.0f32, 9.0f32]), [1i64, 2i64, 2i64, 2i64])
def cv() -> Conv[2, 2, 1, 1] = Conv { k: reshape(to_tensor([1.0f32, 0.5f32, 0.25f32, 1.0f32]), [2i64, 2i64, 1i64, 1i64]), b: to_tensor([0.5f32, -0.25f32]) }
def sub_block[a, i, o, h, w](x: tensor[a, i, h, w, f32], c: &Conv[o, i, 1, 1]) -> tensor[a, o, h, w, f32] =
  match c with {
    | Conv { k, b } => {
    y = channel_bias(conv(x, k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), b)
    group_center(y, 2i64, 1i64, 0.00001f32)
  }
  }
def two_sub(a: tensor[1, 2, 4, 4, f32], b: tensor[1, 2, 2, 2, f32]) -> f32 = {
  c = cv()
  ya = sub_block(copy(&a), &c)
  yb = sub_block(copy(&b), &c)
  add(wsum(ya, a), wsum(yb, b))
}
def one_sub(a: tensor[1, 2, 4, 4, f32]) -> f32 = {
  c = cv()
  wsum(sub_block(copy(&a), &c), a)
}
g = grad(two_sub)(x8(), x4())
out8 = reshape(g.0, [32i64])
out4 = reshape(g.1, [8i64])
once8 = reshape(grad(one_sub)(x8()), [32i64])
fd = mul(sub(two_sub(add(x8(), x8()), add(x4(), x4())), two_sub(sub(x8(), x8()), sub(x4(), x4()))), 0.5f32)
dot = add(tensor_to_scalar(sum(reshape(mul(g.0, x8()), [32i64]), 0i32)), tensor_to_scalar(sum(reshape(mul(g.1, x4()), [8i64]), 0i32)))
"#;
    let expected = |x: &[f64]| {
        let n = x.len() / 2;
        let centered = x
            .chunks(n)
            .flat_map(|channel| {
                let total: f64 = channel.iter().sum();
                channel.iter().map(move |v| v - total)
            })
            .collect::<Vec<_>>();
        symmetric_mix(&centered)
            .into_iter()
            .enumerate()
            .map(|(index, mixed)| mixed + (1.0 - n as f64) * BIAS[index / n])
            .collect::<Vec<_>>()
    };
    let stdout = eval_stdout(source, "record_two_sizes");
    assert_eq!(parse_tensor_data(&stdout, "out8"), expected(&X8), "4x4 use");
    assert_eq!(parse_tensor_data(&stdout, "out4"), expected(&X4), "2x2 use");
    assert_eq!(
        parse_tensor_data(&stdout, "once8"),
        parse_tensor_data(&stdout, "out8"),
        "the 4x4 part is the single use's gradient"
    );
    // The loss is quadratic, so the central difference along `x` itself,
    // `(L(2x) - L(0)) / 2`, is the gradient's projection on `x` exactly.
    assert_central_difference("eval", &stdout);
}

/// Negative twin: each use of `tie` is fresh, but the two actuals of one use
/// still share its `n`. The second use's actuals disagree at run time, so
/// `grad` traps on both lanes; the twin whose second use agrees
/// differentiates, `d/da sum(a * a) = 2a`.
#[test]
fn a_use_whose_own_actuals_disagree_still_traps_under_grad() {
    let program = |c: &str| {
        format!(
            r#"
def tie[n](x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = mul(x, y)
def loss(a: tensor[*, f32], b: tensor[*, f32], c: tensor[*, f32]) -> f32 = tensor_to_scalar(add(sum(tie(copy(&a), copy(&a)), 0i32), sum(tie(copy(&b), c), 0i32)))
out = grad(loss, wrt=a)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32, 2.0f32]), to_tensor({c}))
"#
        )
    };
    let agreeing = program("[5.0f32, 7.0f32]");
    for (lane, stdout) in [
        ("eval", eval_stdout(&agreeing, "tie_agrees")),
        ("C", build_and_run(&agreeing, "tie_agrees")),
    ] {
        assert_eq!(parse_tensor_data(&stdout, "out"), [2.0, 4.0, 6.0], "{lane}");
    }

    let disagreeing = program("[1.0f32, 2.0f32, 3.0f32]");
    let needle = "extent `n`: x axis 0 = 2, y axis 0 = 3";
    assert_refused_on_eval(&disagreeing, needle, "tie_disagrees");
    let dir = tempdir().unwrap();
    let path = dir.path().join("tie_disagrees.ch");
    let out = dir.path().join("out");
    write_file(&path, &disagreeing);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    let run = std::process::Command::new(out.join("tie_disagrees"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "C must trap: {stderr}");
    assert!(stderr.contains(needle), "{stderr}");
    assert!(
        !String::from_utf8_lossy(&run.stdout)
            .lines()
            .any(|line| line.starts_with("out = ")),
        "no gradient may be printed"
    );
}

/// Negative twin of #3462: the generic `gather` helper also scales by a
/// vector its signature ties to the table's `d`. That vector is runtime-sized
/// and one element short, so `grad` of the generic loss traps on the
/// evaluator. The C lane sees both extents as literals once the call is
/// inlined and refuses to build.
#[test]
fn generic_gather_helper_with_disagreeing_extents_still_refuses_under_grad() {
    let source = format!(
        r#"
def lookup_scaled[a, s, v, d](ids: &tensor[a, s, i64], t: &tensor[v, d, f32], w: &tensor[d, f32]) -> tensor[a, s, d, f32] = mul(gather(t, ids, 0i32), insert(insert(copy(w), 0i32, shape(ids, 1i32)), 0i32, shape(ids, 0i32)))
def tloss[v, d](t: &tensor[v, d, f32], ids: &tensor[2, 3, i64], w: &tensor[*, f32]) -> f32 = {{
  x = lookup_scaled(ids, t, w)
  tensor_to_scalar(sum(sum(sum(mul(&x, &x), 2i32), 1i32), 0i32))
}}
{TABLE_AND_IDS}
out = reshape(grad(tloss, wrt=t)(t0(), ids0(), to_tensor([1.0f32, 2.0f32])), [12i64])
"#
    );
    assert_refused_on_eval(
        &source,
        "extent `d`: t axis 1 = 3, w axis 0 = 2",
        "gather_disagrees",
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("gather_disagrees.ch");
    write_file(&path, &source);
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(!build.status.success(), "C must refuse: {stderr}");
    assert!(
        stderr.contains("`mul` argument 2, axis 2: expected 3, got 2"),
        "{stderr}"
    );
}
