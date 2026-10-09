//! chelis#3352 / [05-OP-51]: convolution through evaluation and generated C.
//!
//! The evaluator and the C backend must agree bit for bit on `conv` over a
//! grid of spatial ranks, strides and asymmetric padding, and on grouped and
//! depthwise convolutions composed from channel slices. Integer-valued cases
//! also equal an independent direct cross-correlation. The generated C for a
//! model-scale layer must not contain a table with one entry per
//! (window row, output column) pair.
mod common;
use assert_cmd::Command;
use common::{build_and_run, parse_tensor_data};

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
}

fn dims(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn i64_list(values: &[usize]) -> String {
    let items = values.iter().map(|v| format!("{v}i64")).collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn padding_list(padding: &[(usize, usize)]) -> String {
    let items = padding
        .iter()
        .map(|(lo, hi)| format!("({lo}i64,{hi}i64)"))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn integer_values(count: usize, seed: usize) -> Vec<f64> {
    (0..count)
        .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

fn fractional_literals(count: usize, seed: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("{}.{}f32", (i * 5 + seed) % 7, (i * 37 + seed * 11) % 1000))
        .collect()
}

fn tensor_literal(shape: &[usize], literals: &[String]) -> String {
    format!(
        "reshape(to_tensor([{}]), {})",
        literals.join(","),
        i64_list(shape)
    )
}

fn integer_literals(values: &[f64]) -> Vec<String> {
    values.iter().map(|v| format!("{v:.1}f32")).collect()
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

/// Direct cross-correlation with `groups` disjoint channel groups. Integer
/// operands keep every partial sum exact in f32.
fn reference(c: &Case, x: &[f64], k: &[f64], groups: usize) -> Vec<f64> {
    let rank = c.rank();
    let out = c.output();
    let in_strides = row_major_strides(&c.input);
    let k_strides = row_major_strides(&c.kernel);
    let group_in = c.kernel[1];
    let group_out = c.kernel[0] / groups;
    let mut result = Vec::new();
    for n in 0..out[0] {
        for o in 0..out[1] {
            let group = o / group_out;
            for position in multi_index(&out[2..]) {
                let mut acc = 0.0;
                for local in 0..group_in {
                    let channel = group * group_in + local;
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
                            let mut kf = o * k_strides[0] + local * k_strides[1];
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

fn grid() -> Vec<Case> {
    vec![
        case(&[1, 1, 4], &[1, 1, 2], &[1], &[(0, 0)]),
        case(&[2, 3, 7], &[2, 3, 3], &[2], &[(1, 2)]),
        case(&[1, 1, 5], &[1, 1, 2], &[3], &[(0, 0)]),
        case(&[1, 3, 8, 8], &[4, 3, 3, 3], &[1, 1], &[(0, 0), (0, 0)]),
        case(&[1, 2, 5, 6], &[3, 2, 3, 2], &[2, 1], &[(1, 0), (0, 2)]),
        case(&[2, 2, 4, 4], &[2, 2, 4, 4], &[1, 1], &[(0, 0), (0, 0)]),
        case(&[1, 3, 7, 7], &[4, 3, 3, 3], &[3, 2], &[(1, 1), (2, 0)]),
        case(
            &[1, 2, 3, 4, 3],
            &[2, 2, 2, 1, 2],
            &[1, 2, 1],
            &[(0, 1), (1, 0), (0, 0)],
        ),
    ]
}

/// Grouped convolution composed from per-group channel slices: each group
/// convolves its `shrink`ed input channels with its kernel rows, is padded back
/// to the full output-channel axis, and the disjoint results are added.
fn grouped_expr(c: &Case, groups: usize, x: &str, k: &str) -> String {
    let rank = c.rank();
    let group_in = c.kernel[1];
    let group_out = c.kernel[0] / groups;
    let mut terms = Vec::new();
    for g in 0..groups {
        let mut x_bounds = vec![(0, c.input[0]), (g * group_in, (g + 1) * group_in)];
        let mut k_bounds = vec![(g * group_out, (g + 1) * group_out), (0, group_in)];
        for axis in 0..rank {
            x_bounds.push((0, c.input[axis + 2]));
            k_bounds.push((0, c.kernel[axis + 2]));
        }
        let bounds = |b: &[(usize, usize)]| {
            let items = b
                .iter()
                .map(|(lo, hi)| format!("[{lo}i64,{hi}i64]"))
                .collect::<Vec<_>>();
            format!("[{}]", items.join(","))
        };
        let mut pads = vec![(0, 0), (g * group_out, c.kernel[0] - (g + 1) * group_out)];
        pads.extend(std::iter::repeat_n((0, 0), rank));
        terms.push(format!(
            "pad(conv(shrink({x},{}),shrink({k},{}),{},{}),{},0.0f32)",
            bounds(&x_bounds),
            bounds(&k_bounds),
            i64_list(&c.strides),
            padding_list(&c.padding),
            bounds(&pads),
        ));
    }
    terms
        .into_iter()
        .reduce(|acc, term| format!("add({acc},{term})"))
        .expect("at least one group")
}

fn evaluate(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conv.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The renderer elides tensors past 32 elements, so every compared result is
/// also bound as consecutive flat 32-element slices.
fn chunked(source: &mut String, name: &str, count: usize) -> Vec<String> {
    (0..count.div_ceil(32))
        .map(|j| {
            let chunk = format!("{name}_c{j}");
            source.push_str(&format!(
                "{chunk} = shrink(reshape({name}, [{count}i64]), [[{}i64,{}i64]])\n",
                j * 32,
                (j * 32 + 32).min(count)
            ));
            chunk
        })
        .collect()
}

fn result_line<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = tensor(");
    stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{stdout}"))
}

#[test]
fn evaluation_and_generated_c_agree_and_match_direct_cross_correlation() {
    let mut source = String::new();
    // (result chunk names, exact expected values when integer-valued)
    let mut expectations: Vec<(Vec<String>, Option<Vec<f64>>)> = Vec::new();
    for (index, c) in grid().iter().enumerate() {
        let x_count: usize = c.input.iter().product();
        let k_count: usize = c.kernel.iter().product();
        let signature = format!(
            "def conv{index}(x: tensor[{},f32], k: tensor[{},f32]) -> tensor[{},f32] = conv(x,k,{},{})\n",
            dims(&c.input),
            dims(&c.kernel),
            dims(&c.output()),
            i64_list(&c.strides),
            padding_list(&c.padding)
        );
        source.push_str(&signature);
        let xi = integer_values(x_count, index);
        let ki = integer_values(k_count, index + 3);
        source.push_str(&format!(
            "i{index} = conv{index}({}, {})\n",
            tensor_literal(&c.input, &integer_literals(&xi)),
            tensor_literal(&c.kernel, &integer_literals(&ki))
        ));
        let out_count: usize = c.output().iter().product();
        let chunks = chunked(&mut source, &format!("i{index}"), out_count);
        expectations.push((chunks, Some(reference(c, &xi, &ki, 1))));
        source.push_str(&format!(
            "f{index} = conv{index}({}, {})\n",
            tensor_literal(&c.input, &fractional_literals(x_count, index)),
            tensor_literal(&c.kernel, &fractional_literals(k_count, index + 3))
        ));
        let chunks = chunked(&mut source, &format!("f{index}"), out_count);
        expectations.push((chunks, None));
    }
    // Grouped (two groups) and depthwise (one channel per group).
    let composed = [
        (
            case(&[1, 4, 5, 5], &[4, 2, 3, 3], &[1, 1], &[(1, 0), (0, 1)]),
            2,
        ),
        (
            case(&[2, 3, 6, 5], &[3, 1, 3, 3], &[2, 1], &[(1, 1), (1, 1)]),
            3,
        ),
    ];
    for (index, (c, groups)) in composed.iter().enumerate() {
        let x_count: usize = c.input.iter().product();
        let k_count: usize = c.kernel.iter().product();
        let xi = integer_values(x_count, index + 20);
        let ki = integer_values(k_count, index + 23);
        source.push_str(&format!(
            "gx{index}: tensor[{},f32] = {}\ngk{index}: tensor[{},f32] = {}\n",
            dims(&c.input),
            tensor_literal(&c.input, &integer_literals(&xi)),
            dims(&c.kernel),
            tensor_literal(&c.kernel, &integer_literals(&ki)),
        ));
        source.push_str(&format!(
            "g{index}: tensor[{},f32] = {}\n",
            dims(&c.output()),
            grouped_expr(c, *groups, &format!("gx{index}"), &format!("gk{index}"))
        ));
        let out_count: usize = c.output().iter().product();
        let chunks = chunked(&mut source, &format!("g{index}"), out_count);
        expectations.push((chunks, Some(reference(c, &xi, &ki, *groups))));
    }

    let interpreted = evaluate(&source);
    let compiled = build_and_run(&source, "conv_index_grid");
    for (chunks, exact) in &expectations {
        let mut values = Vec::new();
        for name in chunks {
            assert_eq!(
                result_line(&interpreted, name),
                result_line(&compiled, name),
                "eval and C disagree on {name}"
            );
            values.extend(parse_tensor_data(&interpreted, name));
        }
        if let Some(expected) = exact {
            assert_eq!(
                &values, expected,
                "{} differs from direct cross-correlation",
                chunks[0]
            );
        }
    }
}

fn emitted_c_bytes(source: &str, name: &str) -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join("out");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--output"])
        .arg(&out_dir)
        .arg(&path)
        .assert()
        .success();
    std::fs::metadata(out_dir.join(format!("{name}.c")))
        .expect("generated C source")
        .len()
}

#[test]
fn generated_c_does_not_tabulate_every_window_index() {
    // The chelis#3352 probe: 3->8 channels, 3x3 kernel, 32x32 input. Its
    // direct index table has 27 * 900 = 24,300 entries; at about 87 bytes of
    // C per tagged entry it alone produced 2,114,386 bytes.
    let probe = "def forward(x: tensor[1,3,32,32,f32], w: tensor[8,3,3,3,f32]) -> tensor[1,8,30,30,f32] = conv(x,w,[1i64,1i64],[(0i64,0i64),(0i64,0i64)])\n";
    let probe_bytes = emitted_c_bytes(probe, "convprobe");
    assert!(
        probe_bytes < 400_000,
        "convprobe generated {probe_bytes} bytes of C"
    );
    // A ResNet-scale layer: 64->64 channels, 3x3, 56x56 with unit padding. Its
    // direct table has 576 * 3,136 = 1,806,336 entries (about 157 MB of C).
    let layer = "def forward(x: tensor[1,64,56,56,f32], w: tensor[64,64,3,3,f32]) -> tensor[1,64,56,56,f32] = conv(x,w,[1i64,1i64],[(1i64,1i64),(1i64,1i64)])\n";
    let layer_bytes = emitted_c_bytes(layer, "resnetlayer");
    assert!(
        layer_bytes < 2_000_000,
        "ResNet-scale layer generated {layer_bytes} bytes of C"
    );
}
