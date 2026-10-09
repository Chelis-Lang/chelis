//! chelis#3379 / chelis#3380 / [05-OP-51]: `conv` and `layer_norm` whose
//! operand extents, strides, or padding are known only at run time.
//!
//! A dimension-generic def with a declared return type, a `grad` through a
//! generic def, and runtime `i64` strides and padding all reach IR lowering.
//! Forward values and both adjoints must agree bit for bit between evaluation
//! and generated C, and must equal an independent direct cross-correlation
//! and its finite differences. Invalid runtime metadata traps with the same
//! diagnostic in both lanes, as [05-OP-51]'s failure rule requires of
//! runtime-dependent obligations.
mod common;
use assert_cmd::Command;
use common::{build_and_run, parse_tensor_data};
use std::process::Command as StdCommand;

struct Case {
    input: Vec<usize>,
    kernel: Vec<usize>,
    strides: Vec<usize>,
    padding: Vec<(usize, usize)>,
}

fn case(input: &[usize], kernel: &[usize], strides: &[usize], padding: &[(usize, usize)]) -> Case {
    Case {
        input: input.to_vec(),
        kernel: kernel.to_vec(),
        strides: strides.to_vec(),
        padding: padding.to_vec(),
    }
}

impl Case {
    fn rank(&self) -> usize {
        self.input.len() - 2
    }

    fn output(&self) -> Vec<usize> {
        let mut out = vec![self.input[0], self.kernel[0]];
        for axis in 0..self.rank() {
            let padded = self.input[axis + 2] + self.padding[axis].0 + self.padding[axis].1;
            out.push((padded - self.kernel[axis + 2]) / self.strides[axis] + 1);
        }
        out
    }

    /// The runtime metadata arguments: strides, then (low, high) per axis.
    fn metadata_args(&self) -> String {
        let mut args: Vec<String> = self.strides.iter().map(|s| format!("{s}i64")).collect();
        for (low, high) in &self.padding {
            args.push(format!("{low}i64"));
            args.push(format!("{high}i64"));
        }
        args.join(", ")
    }

    fn metadata_params(&self) -> String {
        let mut params: Vec<String> = (0..self.rank()).map(|a| format!("st{a}: i64")).collect();
        for axis in 0..self.rank() {
            params.push(format!("lo{axis}: i64"));
            params.push(format!("hi{axis}: i64"));
        }
        params.join(", ")
    }

    fn metadata_names(&self) -> String {
        let mut names: Vec<String> = (0..self.rank()).map(|a| format!("st{a}")).collect();
        for axis in 0..self.rank() {
            names.push(format!("lo{axis}"));
            names.push(format!("hi{axis}"));
        }
        names.join(", ")
    }

    fn conv_call(&self, x: &str, k: &str) -> String {
        let strides = (0..self.rank())
            .map(|a| format!("st{a}"))
            .collect::<Vec<_>>()
            .join(", ");
        let padding = (0..self.rank())
            .map(|a| format!("(lo{a}, hi{a})"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("conv({x}, {k}, [{strides}], [{padding}])")
    }

    fn generic_params(&self) -> (String, String, String) {
        let spatial: Vec<String> = (0..self.rank()).map(|a| format!("s{a}")).collect();
        let window: Vec<String> = (0..self.rank()).map(|a| format!("w{a}")).collect();
        let mut binders = vec!["n".to_string(), "c".to_string(), "o".to_string()];
        binders.extend(spatial.iter().cloned());
        binders.extend(window.iter().cloned());
        let x = format!("n, c, {}", spatial.join(", "));
        let k = format!("o, c, {}", window.join(", "));
        (binders.join(", "), x, k)
    }
}

fn dims(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn integer_values(count: usize, seed: usize) -> Vec<f64> {
    (0..count)
        .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

fn tensor_literal(shape: &[usize], values: &[f64]) -> String {
    let items = values
        .iter()
        .map(|v| format!("{v:.1}f32"))
        .collect::<Vec<_>>()
        .join(", ");
    let shape = shape
        .iter()
        .map(|n| format!("{n}i64"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("reshape(to_tensor([{items}]), [{shape}])")
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn multi_index(extents: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    for &extent in extents {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                (0..extent).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    all
}

/// Direct cross-correlation. Integer operands keep every partial sum exact.
fn reference(c: &Case, x: &[f64], k: &[f64]) -> Vec<f64> {
    let rank = c.rank();
    let out = c.output();
    let in_strides = row_major_strides(&c.input);
    let k_strides = row_major_strides(&c.kernel);
    let mut result = Vec::new();
    for n in 0..out[0] {
        for o in 0..out[1] {
            for position in multi_index(&out[2..]) {
                let mut acc = 0.0;
                for channel in 0..c.input[1] {
                    for offset in multi_index(&c.kernel[2..]) {
                        let mut flat = n * in_strides[0] + channel * in_strides[1];
                        let mut inside = true;
                        for axis in 0..rank {
                            let p = position[axis] * c.strides[axis] + offset[axis];
                            let low = c.padding[axis].0;
                            if p < low || p - low >= c.input[axis + 2] {
                                inside = false;
                                break;
                            }
                            flat += (p - low) * in_strides[axis + 2];
                        }
                        if inside {
                            let mut kf = o * k_strides[0] + channel * k_strides[1];
                            for axis in 0..rank {
                                kf += offset[axis] * k_strides[axis + 2];
                            }
                            acc += x[flat] * k[kf];
                        }
                    }
                }
                result.push(acc);
            }
        }
    }
    result
}

fn weighted_loss(c: &Case, x: &[f64], k: &[f64], weights: &[f64]) -> f64 {
    reference(c, x, k)
        .iter()
        .zip(weights)
        .map(|(y, w)| y * w)
        .sum()
}

/// Forward differences with unit steps. The loss is linear in each operand
/// and every value is a small integer, so each difference is exact.
fn finite_differences(values: &[f64], loss: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let base = loss(values);
    (0..values.len())
        .map(|i| {
            let mut stepped = values.to_vec();
            stepped[i] += 1.0;
            loss(&stepped) - base
        })
        .collect()
}

fn grid() -> Vec<Case> {
    vec![
        case(&[1, 1, 5], &[1, 1, 2], &[2], &[(1, 0)]),
        case(&[2, 2, 4, 5], &[3, 2, 2, 3], &[2, 1], &[(1, 0), (0, 2)]),
        case(
            &[1, 2, 3, 4, 3],
            &[2, 2, 2, 1, 2],
            &[1, 2, 1],
            &[(0, 1), (1, 0), (0, 0)],
        ),
    ]
}

fn evaluate(source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conv.ch");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap()
}

fn evaluate_ok(source: &str) -> String {
    let output = evaluate(source);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn result_line<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{stdout}"))
}

/// The renderer elides tensors past 32 elements, so every compared result is
/// also bound as consecutive flat 32-element slices.
fn chunked(source: &mut String, name: &str, count: usize) -> Vec<String> {
    (0..count.div_ceil(32))
        .map(|j| {
            let chunk = format!("{name}_c{j}");
            source.push_str(&format!(
                "{chunk} = shrink(reshape({name}, [{count}i64]), [[{}i64, {}i64]])\n",
                j * 32,
                (j * 32 + 32).min(count)
            ));
            chunk
        })
        .collect()
}

#[test]
fn runtime_metadata_conv_agrees_across_lanes_with_reference_and_finite_differences() {
    let mut source = String::new();
    let mut expectations: Vec<(Vec<String>, Vec<f64>)> = Vec::new();
    for (index, c) in grid().iter().enumerate() {
        let x_count: usize = c.input.iter().product();
        let k_count: usize = c.kernel.iter().product();
        let out_count: usize = c.output().iter().product();
        let x = integer_values(x_count, index);
        let k = integer_values(k_count, index + 3);
        let weights = integer_values(out_count, index + 5);
        let (binders, x_dims, k_dims) = c.generic_params();
        let spatial_out = vec!["*"; c.rank()].join(", ");
        // The declared generic def is the chelis#3379 shape; its metadata
        // parameters are the chelis#3380 shape.
        source.push_str(&format!(
            "def cv{index}[{binders}](x: tensor[{x_dims}, f32], k: tensor[{k_dims}, f32], {}) -> tensor[n, o, {spatial_out}, f32] = {}\n",
            c.metadata_params(),
            c.conv_call("x", "k"),
        ));
        source.push_str(&format!(
            "def cu{index}[{binders}](x: tensor[{x_dims}, f32], k: tensor[{k_dims}, f32], {}) = {}\n",
            c.metadata_params(),
            c.conv_call("x", "k"),
        ));
        source.push_str(&format!(
            "def x{index}() -> tensor[{}, f32] = {}\n",
            dims(&c.input),
            tensor_literal(&c.input, &x)
        ));
        source.push_str(&format!(
            "def k{index}() -> tensor[{}, f32] = {}\n",
            dims(&c.kernel),
            tensor_literal(&c.kernel, &k)
        ));
        source.push_str(&format!(
            "def w{index}() -> tensor[{}, f32] = {}\n",
            dims(&c.output()),
            tensor_literal(&c.output(), &weights)
        ));
        source.push_str(&format!(
            "def loss{index}(x: tensor[{}, f32], k: tensor[{}, f32], {}) -> f32 = tensor_to_scalar(sum(reshape(mul(cu{index}(x, k, {}), w{index}()), [{out_count}i64]), 0i32))\n",
            dims(&c.input),
            dims(&c.kernel),
            c.metadata_params(),
            c.metadata_names(),
        ));
        let args = format!("x{index}(), k{index}(), {}", c.metadata_args());
        source.push_str(&format!("f{index} = cv{index}({args})\n"));
        let chunks = chunked(&mut source, &format!("f{index}"), out_count);
        expectations.push((chunks, reference(c, &x, &k)));
        source.push_str(&format!("gk{index} = grad(loss{index}, wrt=k)({args})\n"));
        let chunks = chunked(&mut source, &format!("gk{index}"), k_count);
        expectations.push((
            chunks,
            finite_differences(&k, |k| weighted_loss(c, &x, k, &weights)),
        ));
        source.push_str(&format!("gx{index} = grad(loss{index}, wrt=x)({args})\n"));
        let chunks = chunked(&mut source, &format!("gx{index}"), x_count);
        expectations.push((
            chunks,
            finite_differences(&x, |x| weighted_loss(c, x, &k, &weights)),
        ));
    }

    let interpreted = evaluate_ok(&source);
    let compiled = build_and_run(&source, "conv_runtime_grid");
    for (chunks, expected) in &expectations {
        let mut values = Vec::new();
        for name in chunks {
            assert_eq!(
                result_line(&interpreted, name),
                result_line(&compiled, name),
                "eval and C disagree on {name}"
            );
            values.extend(parse_tensor_data(&interpreted, name));
        }
        assert_eq!(
            &values, expected,
            "{} differs from the direct reference",
            chunks[0]
        );
    }
}

/// The issue reproducers: a generic conv with a declared return type, a
/// generic layer_norm, and a conv whose input takes its extents from a
/// shape-derived broadcast under `grad`.
const REPRODUCERS: &str = "\
def conv_declared[a, i, h, w, o](x: tensor[a, i, h, w, f32], k: tensor[o, i, 1, 1, f32], b: tensor[o, f32]) -> tensor[a, o, h, w, f32] = {
  y = conv(x, k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  add(y, insert(insert(insert(b, 0i32, shape(&y, 0i32)), 2i32, shape(&y, 2i32)), 3i32, shape(&y, 3i32)))
}
def project[a, i, h, w, o](x: tensor[a, i, h, w, f32], k: tensor[o, i, 1, 1, f32], s: i64) -> tensor[a, o, h, w, f32] = conv(x, k, [s, s], [(0i64, 0i64), (0i64, 0i64)])
def norm2[a, h](x: tensor[a, h, f32], g: tensor[h, f32], b: tensor[h, f32]) -> tensor[a, h, f32] = layer_norm(x, g, b, 0.00001f32)
def bc[a, h, w](v: &tensor[2, f32], like: &tensor[a, 2, h, w, f32]) -> tensor[a, 2, h, w, f32] = insert(insert(insert(v, 0i32, shape(like, 0i32)), 2i32, shape(like, 2i32)), 3i32, shape(like, 3i32))
def scale_first(v: tensor[2, f32], x: tensor[1, 2, 3, 3, f32]) -> f32 = tensor_to_scalar(sum(reshape(conv(mul(bc(&v, &x), x), reshape(to_tensor([1.0f32, 0.0f32, 0.0f32, 1.0f32]), [2i64, 2i64, 1i64, 1i64]), [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), [18i64]), 0i32))
def xs() -> tensor[1, 2, 3, 3, f32] = reshape(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32]), [1i64, 2i64, 3i64, 3i64])
def k1() -> tensor[2, 1, 1, 1, f32] = reshape(to_tensor([1.0f32, 2.0f32]), [2i64, 1i64, 1i64, 1i64])
def x1() -> tensor[1, 1, 2, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [1i64, 1i64, 2i64, 2i64])
def stride_loss(x: tensor[1, 1, 2, 2, f32], k: tensor[2, 1, 1, 1, f32], s: i64) -> f32 = tensor_to_scalar(sum(reshape(project(x, k, s), [8i64]), 0i32))
biased = reshape(conv_declared(x1(), k1(), to_tensor([0.0f32, 1.0f32])), [8i64])
projected = reshape(project(x1(), k1(), 1i64), [8i64])
stride_grad = reshape(grad(stride_loss, wrt=k)(x1(), k1(), 1i64), [2i64])
normed = norm2(to_tensor([[1.0f32, 3.0f32]]), to_tensor([1.0f32, 1.0f32]), to_tensor([0.0f32, 0.0f32]))
order_grad = grad(scale_first)(to_tensor([1.0f32, 1.0f32]), xs()).0
";

#[test]
fn issue_reproducers_evaluate_and_compile_to_their_expected_values() {
    let interpreted = evaluate_ok(REPRODUCERS);
    let compiled = build_and_run(REPRODUCERS, "conv_runtime_reproducers");
    let expected: [(&str, &[f64]); 5] = [
        ("biased", &[1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 7.0, 9.0]),
        ("projected", &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]),
        ("stride_grad", &[10.0, 10.0]),
        ("normed", &[-0.999995, 0.999995]),
        // d/dv_c of sum over channel c of v_c * x_c: 9 * 1 and 9 * 2.
        ("order_grad", &[9.0, 18.0]),
    ];
    for (name, values) in expected {
        assert_eq!(
            result_line(&interpreted, name),
            result_line(&compiled, name),
            "eval and C disagree on {name}"
        );
        let actual = parse_tensor_data(&interpreted, name);
        assert_eq!(actual.len(), values.len(), "{name}: {actual:?}");
        for (a, e) in actual.iter().zip(values) {
            assert!((a - e).abs() <= 1e-5, "{name}: {actual:?} != {values:?}");
        }
    }
}

/// layer_norm's hidden extent read at run time: both lanes agree on the
/// forward value and on the x and gamma adjoints, and those adjoints match
/// central differences of an f64 reference.
#[test]
fn generic_layer_norm_gradients_agree_across_lanes_and_match_finite_differences() {
    let x = [1.0, 3.0, 2.0, 0.5, -1.0, 4.0];
    let gamma = [1.0, 2.0, 0.5];
    let beta = [0.0, 1.0, -1.0];
    let weights = [1.0, -2.0, 0.5, 3.0, 1.0, -1.0];
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|v| format!("{v:?}f64"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let source = format!(
        "def norm[a, h](x: tensor[a, h, f64], g: tensor[h, f64], b: tensor[h, f64]) -> tensor[a, h, f64] = layer_norm(x, g, b, 0.00001f64)
def xs() -> tensor[2, 3, f64] = reshape(to_tensor([{}]), [2i64, 3i64])
def gs() -> tensor[3, f64] = to_tensor([{}])
def bs() -> tensor[3, f64] = to_tensor([{}])
def ws() -> tensor[2, 3, f64] = reshape(to_tensor([{}]), [2i64, 3i64])
def loss(x: tensor[2, 3, f64], g: tensor[3, f64], b: tensor[3, f64]) -> f64 = tensor_to_scalar(sum(reshape(mul(norm(x, g, b), ws()), [6i64]), 0i32))
y = norm(xs(), gs(), bs())
gx = reshape(grad(loss, wrt=x)(xs(), gs(), bs()), [6i64])
gg = grad(loss, wrt=g)(xs(), gs(), bs())
",
        list(&x),
        list(&gamma),
        list(&beta),
        list(&weights)
    );
    let forward = |x: &[f64], gamma: &[f64]| -> Vec<f64> {
        x.chunks(3)
            .flat_map(|row| {
                let mean = row.iter().sum::<f64>() / 3.0;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 3.0;
                let denom = (var + 0.00001).sqrt();
                row.iter()
                    .enumerate()
                    .map(move |(j, v)| (v - mean) / denom * gamma[j] + beta[j])
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let loss = |x: &[f64], gamma: &[f64]| -> f64 {
        forward(x, gamma)
            .iter()
            .zip(&weights)
            .map(|(y, w)| y * w)
            .sum()
    };
    let central = |values: &[f64], f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
        let h = 1e-6;
        (0..values.len())
            .map(|i| {
                let mut up = values.to_vec();
                let mut down = values.to_vec();
                up[i] += h;
                down[i] -= h;
                (f(&up) - f(&down)) / (2.0 * h)
            })
            .collect()
    };
    let interpreted = evaluate_ok(&source);
    let compiled = build_and_run(&source, "layer_norm_runtime_hidden");
    let expected = [
        ("y", forward(&x, &gamma), 1e-12),
        ("gx", central(&x, &|x| loss(x, &gamma)), 1e-6),
        ("gg", central(&gamma, &|g| loss(&x, g)), 1e-6),
    ];
    for (name, values, tolerance) in expected {
        assert_eq!(
            result_line(&interpreted, name),
            result_line(&compiled, name),
            "eval and C disagree on {name}"
        );
        let actual = parse_tensor_data(&interpreted, name);
        assert_eq!(actual.len(), values.len(), "{name}");
        for (a, e) in actual.iter().zip(&values) {
            assert!(
                (a - e).abs() <= tolerance,
                "{name}: {actual:?} != {values:?}"
            );
        }
    }
}

fn compiled_failure(source: &str, name: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join("out");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--target", "c", "--output"])
        .arg(&out_dir)
        .arg(&path)
        .assert()
        .success();
    let run = StdCommand::new(out_dir.join(name)).output().unwrap();
    assert!(
        !run.status.success(),
        "invalid conv metadata must trap in generated C; stdout: {}",
        String::from_utf8_lossy(&run.stdout)
    );
    String::from_utf8_lossy(&run.stderr).into_owned()
}

/// Negative twins: each runtime [05-OP-51] metadata rule is a guard, in the
/// forward pass and under `grad`, in both lanes. Evaluation covers every
/// (rule, form) pair; generated C covers each rule once.
#[test]
fn invalid_runtime_conv_metadata_traps_in_both_lanes() {
    let prelude = "\
def cv[n, c, h, w, o, kh, kw](x: tensor[n, c, h, w, f32], k: tensor[o, c, kh, kw, f32], s: i64, lo: i64, hi: i64) -> tensor[n, o, *, *, f32] = conv(x, k, [s, 1i64], [(lo, hi), (0i64, 0i64)])
def cu[n, c, h, w, o, kh, kw](x: tensor[n, c, h, w, f32], k: tensor[o, c, kh, kw, f32], s: i64, lo: i64, hi: i64) = conv(x, k, [s, 1i64], [(lo, hi), (0i64, 0i64)])
def ks() -> tensor[1, 1, 2, 2, f32] = reshape(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), [1i64, 1i64, 2i64, 2i64])
def xs() -> tensor[1, 1, 2, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [1i64, 1i64, 2i64, 2i64])
def short() -> tensor[1, 1, 1, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32]), [1i64, 1i64, 1i64, 2i64])
def loss(x: tensor[1, 1, 2, 2, f32], k: tensor[1, 1, 2, 2, f32], s: i64, lo: i64, hi: i64) -> f32 = tensor_to_scalar(sum(reshape(cu(x, k, s, lo, hi), [1i64]), 0i32))
def short_loss(x: tensor[1, 1, 1, 2, f32], k: tensor[1, 1, 2, 2, f32]) -> f32 = tensor_to_scalar(sum(reshape(cu(x, k, 1i64, 0i64, 0i64), [1i64]), 0i32))
";
    let invalid = "conv invalid kernel/stride/padding at spatial axis 0";
    let negative = "conv padding must be non-negative";
    // (forward call, grad call, message, which form generated C runs)
    let cases = [
        (
            "cv(xs(), ks(), 0i64, 0i64, 0i64)",
            "grad(loss, wrt=k)(xs(), ks(), 0i64, 0i64, 0i64)",
            invalid,
            "grad",
        ),
        (
            "cv(xs(), ks(), 1i64, neg(1i64), 1i64)",
            "grad(loss, wrt=x)(xs(), ks(), 1i64, neg(1i64), 1i64)",
            negative,
            "forward",
        ),
        (
            "cv(xs(), ks(), 1i64, 0i64, neg(1i64))",
            "grad(loss, wrt=k)(xs(), ks(), 1i64, 0i64, neg(1i64))",
            negative,
            "grad",
        ),
        // A kernel taller than its unpadded runtime input height.
        (
            "cv(short(), ks(), 1i64, 0i64, 0i64)",
            "grad(short_loss, wrt=k)(short(), ks())",
            invalid,
            "forward",
        ),
    ];
    for (index, (forward, gradient, message, compiled_form)) in cases.iter().enumerate() {
        for (form, call) in [("forward", forward), ("grad", gradient)] {
            let source = format!("{prelude}y = {call}\n");
            let output = evaluate(&source);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success() && stderr.contains(message),
                "eval {form} `{call}` must fail with `{message}`: {stderr}"
            );
            if form == *compiled_form {
                let stderr = compiled_failure(&source, &format!("conv_invalid_{index}"));
                assert!(
                    stderr.contains(message),
                    "C {form} `{call}` must fail with `{message}`: {stderr}"
                );
            }
        }
    }
}

/// A stride larger than its padded extent on an outer spatial axis gives
/// that axis one output coordinate. Every final window index fits in i64, so
/// the runtime index arithmetic must not form a larger partial product
/// (`stride * row length`) than the literal path does: both forms return
/// the same value in both lanes instead of trapping on overflow.
#[test]
fn huge_runtime_stride_on_an_outer_axis_matches_the_literal_path() {
    let source = "\
def cv[n, c, h, w, o, kh, kw](x: tensor[n, c, h, w, f32], k: tensor[o, c, kh, kw, f32], s0: i64, s1: i64) -> tensor[n, o, *, *, f32] = conv(x, k, [s0, s1], [(0i64, 0i64), (0i64, 0i64)])
def xs() -> tensor[1, 1, 2, 2, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), [1i64, 1i64, 2i64, 2i64])
def ks() -> tensor[1, 1, 2, 2, f32] = reshape(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), [1i64, 1i64, 2i64, 2i64])
runtime_max = reshape(cv(xs(), ks(), 9223372036854775807i64, 1i64), [1i64])
runtime_half = reshape(cv(xs(), ks(), 4611686018427387904i64, 1i64), [1i64])
literal_max = reshape(conv(xs(), ks(), [9223372036854775807i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]), [1i64])
";
    let interpreted = evaluate_ok(source);
    let compiled = build_and_run(source, "conv_huge_runtime_stride");
    for name in ["runtime_max", "runtime_half", "literal_max"] {
        assert_eq!(
            result_line(&interpreted, name),
            result_line(&compiled, name),
            "eval and C disagree on {name}"
        );
        assert_eq!(parse_tensor_data(&interpreted, name), [10.0], "{name}");
    }
}
