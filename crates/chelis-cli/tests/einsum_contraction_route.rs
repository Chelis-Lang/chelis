//! chelis#3736 / chelis#3703 / chelis#3783: a compiled float
//! `einsum` whose [05-OP-33] contraction is spec/05 §4.1's product sum over
//! one reduction label runs the emitted product-sum kernel `matmul` uses,
//! with the bits `chelis eval` computes; every other einsum keeps the runtime
//! kernel.
//!
//! The fixture covers both sides. Its routed definitions span f32 and f64,
//! f16 and bf16 at their f32 default, f16 and f32 at an f64 accumulator,
//! permuted and swapped operands, batch and runtime row extents (including
//! zero rows), flattened row labels, a 257-leaf tree, inf, NaN and
//! signed-zero products, an f16 total that overflows to inf, and an f16
//! double-rounding witness. Its retained definitions are a diagonal, two and
//! zero reduction labels, a matrix-vector product, a runtime contraction
//! extent, an integer contraction, a lead label only the column operand
//! carries, a runtime row beside a static row above one and two runtime rows
//! (chelis#3781; including empty ones whose row product would overflow i64
//! before a zero factor), and empty batches whose expansion strides would
//! exceed i64, with static or runtime rows. Its f16 and bf16 `matmul`s share the sum's narrowing.

mod common;

use common::{authored_host_body_definition, gcc_available};
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

use assert_cmd::Command;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/einsum_contraction/route.ch"
);

const ROUTED: &[&str] = &[
    "rows",
    "transposed",
    "swapped",
    "batched",
    "runtime_rows",
    "deep",
    "flattened",
    "interleaved",
    "sequence",
    "special",
    "half",
    "brain",
    "wide",
    "widened",
    "narrowed",
    "batched_brain",
    "special_half",
    "special_wide",
];

const RETAINED: &[&str] = &[
    "diagonal_scale",
    "frobenius",
    "outer_product",
    "matvec",
    "runtime_depth",
    "integer",
    "broadcast_columns",
    "product_rows",
    "narrow_rows",
    "paired_rows",
    "empty_pair",
    "empty_mixed",
    "huge_empty",
    "zero_batch_rows",
];

fn stdout(output: &std::process::Output, lane: &str) -> String {
    assert!(
        output.status.success(),
        "{lane} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).expect("utf-8 output")
}

/// One definition's owned body: from its definition to the first closing
/// brace in column zero, which ends a top-level C function.
fn body<'a>(emitted: &'a str, name: &str) -> &'a str {
    let rest = authored_host_body_definition(emitted, name);
    &rest[..rest.find("\n}\n").expect("the body closes")]
}

fn build(out_dir: &Path) -> String {
    build_source(Path::new(FIXTURE), out_dir, "route")
}

fn build_source(source: &Path, out_dir: &Path, stem: &str) -> String {
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    stdout(&built, "build");
    std::fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("generated C")
}

/// The element count and dtype of every statically shaped `chelis_alloc`
/// in `emitted`.
fn static_allocations_with_dtype(emitted: &str) -> Vec<(u64, &str)> {
    emitted
        .match_indices("chelis_alloc(")
        .filter_map(|(at, _)| {
            let rest = &emitted[at..];
            let dims = &rest[rest.find("(int64_t[]){")? + "(int64_t[]){".len()..];
            let (dims, rest) = dims.split_at(dims.find('}')?);
            let dtype = rest[1..rest.find(')')?].trim_start_matches(',').trim();
            let count = dims
                .split(',')
                .map(|dim| dim.trim().parse::<u64>().ok())
                .product::<Option<u64>>()?;
            Some((count, dtype))
        })
        .collect()
}

/// The element counts of every statically shaped `chelis_alloc` in `emitted`.
fn static_allocations(emitted: &str) -> Vec<u64> {
    static_allocations_with_dtype(emitted)
        .into_iter()
        .map(|(count, _)| count)
        .collect()
}

/// Each routed definition's body calls no runtime einsum and the program
/// has the blocked product-sum kernel; each retained one still calls it.
#[test]
fn only_contraction_form_einsums_leave_the_runtime_kernel() {
    let dir = tempdir().expect("tempdir");
    let emitted = build(&dir.path().join("out"));
    for name in ROUTED {
        let body = body(&emitted, name);
        assert!(
            !body.contains("chelis_tensor_einsum("),
            "`{name}` still calls the runtime einsum kernel:\n{body}"
        );
    }
    for name in RETAINED {
        let body = body(&emitted, name);
        assert!(
            body.contains("chelis_tensor_einsum("),
            "`{name}` left the runtime einsum kernel:\n{body}"
        );
    }
    assert_eq!(
        emitted.matches("chelis_tensor_einsum(").count(),
        RETAINED.len(),
        "runtime einsum calls in the generated C"
    );
    assert!(
        emitted.contains("__sum_columns_"),
        "no blocked product-sum kernel in the generated C"
    );
}

/// The compiled program prints exactly what `chelis eval` prints. Printed
/// floats are shortest round-trip, so equal text is equal finite bits, equal
/// infinities and signed zeros, and NaN in the same positions.
#[test]
fn routed_and_retained_einsums_match_eval_bit_for_bit() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("out");
    build(&out_dir);
    let compiled = StdCommand::new(out_dir.join("route"))
        .output()
        .expect("compiled binary should run");
    let compiled = stdout(&compiled, "compiled");
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--timeout", "300", "--file", FIXTURE])
        .output()
        .expect("eval");
    let eval = stdout(&eval, "eval");
    let bindings = eval.lines().filter(|line| line.starts_with("r_")).count();
    let declared = std::fs::read_to_string(FIXTURE)
        .expect("fixture")
        .lines()
        .filter(|line| line.starts_with("r_"))
        .count();
    assert_eq!(bindings, declared, "{eval}");
    assert_eq!(compiled, eval);
    // 1 + 2^-11 + 2^-40 rounds once from f64 to the f16 above 1, and to 1
    // through an intermediate f32; at the f32 accumulator the 2^-40 term is
    // already gone and the tie rounds to even.
    // An f16 total above 65504 narrows to inf; at f64 it is exact.
    for line in [
        "r_widened = tensor(shape=[1, 1], data=[1.001])",
        "r_narrowed = tensor(shape=[1, 1], data=[1.0])",
        "r_special_half = tensor(shape=[3, 2], data=[NaN, NaN, 0.0, 0.0, inf, 2.5])",
        "r_special_wide = tensor(shape=[3, 2], data=[NaN, NaN, 0.0, 0.0, 90000.0, 2.5])",
    ] {
        assert!(
            eval.lines().any(|printed| printed == line),
            "{line}\n{eval}"
        );
    }
}

/// A lead label only the row operand carries flattens into rows, so the
/// routed contraction stores nothing larger than an operand or the result.
/// Copying the weight along `a` would allocate `256 * 512 * 512` elements
/// here.
#[test]
fn a_row_only_lead_label_allocates_no_copy_of_the_other_operand() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("rows.ch");
    std::fs::write(
        &source,
        "module EinsumRows\n\
         def g32(n: i64, lo: f32) -> tensor[*, f32] = cumsum(expand(to_tensor([lo]), 0i32, n), 0i32)\n\
         def project(x: tensor[256, 1, 512, f32], w: tensor[512, 512, f32]) -> tensor[256, 1, 512, f32] = einsum(\"amk,kn->amn\", x, w)\n\
         out = project(reshape(g32(131072i64, 0.001f32), [256i64, 1i64, 512i64]), reshape(g32(262144i64, 0.002f32), [512i64, 512i64]))\n",
    )
    .expect("write program");
    let emitted = build_source(&source, &dir.path().join("out"), "rows");
    assert!(
        !body(&emitted, "project").contains("chelis_tensor_einsum("),
        "`project` still calls the runtime einsum kernel"
    );
    let allocations = static_allocations(&emitted);
    assert!(!allocations.is_empty(), "no static allocation was read");
    let largest = allocations.iter().copied().max().unwrap();
    assert!(
        largest <= 512 * 512,
        "a {largest}-element allocation exceeds the largest operand: {allocations:?}"
    );
}

/// chelis#3783: a reduced-float or widened contraction converts each operand
/// as the kernel reads it and narrows each total as it stores it, so no
/// allocation holds more bytes than an operand or the result. Storing the
/// widened operands would allocate the f64 product, 128 MB for `wide`, and
/// storing the f32 sum would allocate 16 MB for `outer`, twice its f16
/// result.
#[test]
fn converted_contractions_allocate_no_accumulator_width_copy() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("converted.ch");
    std::fs::write(
        &source,
        "module EinsumConverted\n\
         def outer(x: tensor[2048, 1, f16], w: tensor[1, 2048, f16]) -> tensor[2048, 2048, f16] = einsum(\"mk,kn->mn\", x, w)\n\
         def wide(x: tensor[64, 4096, bf16], w: tensor[4096, 64, bf16]) -> tensor[64, 64, bf16] = einsum(\"mk,kn->mn\", x, w, accumulator=f64)\n\
         r_outer = outer(reshape(expand(cast(to_tensor([0.5f32]), f16), 0i32, 2048i64), [2048i64, 1i64]), reshape(expand(cast(to_tensor([0.25f32]), f16), 0i32, 2048i64), [1i64, 2048i64]))\n\
         r_wide = wide(reshape(expand(cast(to_tensor([0.5f32]), bf16), 0i32, 262144i64), [64i64, 4096i64]), reshape(expand(cast(to_tensor([0.25f32]), bf16), 0i32, 262144i64), [4096i64, 64i64]))\n",
    )
    .expect("write program");
    let emitted = build_source(&source, &dir.path().join("out"), "converted");
    for name in ["outer", "wide"] {
        assert!(
            !body(&emitted, name).contains("chelis_tensor_einsum("),
            "`{name}` still calls the runtime einsum kernel"
        );
    }
    let allocations = static_allocations_with_dtype(&emitted);
    assert!(!allocations.is_empty(), "no static allocation was read");
    let largest = allocations
        .iter()
        .map(|&(count, dtype)| {
            let width = match dtype {
                "CHELIS_DTYPE_F16" | "CHELIS_DTYPE_BF16" => 2,
                "CHELIS_DTYPE_F32" => 4,
                "CHELIS_DTYPE_F64" | "CHELIS_DTYPE_I64" => 8,
                other => panic!("an allocation of {other}"),
            };
            count * width
        })
        .max()
        .unwrap();
    assert!(
        largest <= 2048 * 2048 * 2,
        "a {largest}-byte allocation exceeds the largest operand or result: {allocations:?}"
    );
}
