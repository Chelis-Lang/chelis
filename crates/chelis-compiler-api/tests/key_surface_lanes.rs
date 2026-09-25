//! chelis#2413 step 3: random draws take an explicit key from source, and
//! keys flow through the host interpreter (`chelis eval`) and native C.
//!
//! Every expected value is a bit pattern computed by
//! `briefs/keys-b-slice2-probes/slice2_ref.py`, which extends `key_ref.py`'s
//! independent transcription of [05-RNG-2] with the [05-OP-8] and [05-OP-37]
//! text (f32 bounds; one fused multiply-add at f32 for f32, f16 and bf16 and
//! at f64 for f64; one narrowing for f16 and bf16). No expected value is
//! computed by compiler code; the formatter only spells the reference bits.
//! Each program runs in eval and in C, and both must print the reference.
mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_types::types::Prim;
use chelis_types::{ElementRef, format_element};
use std::collections::BTreeMap;

// ---- the reference (slice2_ref.py) ----

/// `uniform_like(key_from_seed(7), t, 0, 1)` over 4 elements.
fn seed7_unit(prim: Prim) -> Vec<u64> {
    match prim {
        Prim::F16 => vec![0x300d, 0x3ab3, 0x3963, 0x38d0],
        Prim::Bf16 => vec![0x3e02, 0x3f56, 0x3f2c, 0x3f1a],
        Prim::F32 => vec![0x3e019516, 0x3f56526f, 0x3f2c55af, 0x3f1a0770],
        Prim::F64 => vec![
            0x3fc032a2c47e6928,
            0x3feaca4def0c7f5a,
            0x3fe58ab5e15533e8,
            0x3fe340ee07cd3a66,
        ],
        other => panic!("no reference for {other:?}"),
    }
}

/// `uniform_like(a, t, -2, 3)` with `(a, _) = split_key(key_from_seed(-3))`.
fn left_half_bounded(prim: Prim) -> Vec<u64> {
    match prim {
        Prim::F16 => vec![0x406f, 0x40ff, 0x34a8, 0xb778],
        Prim::Bf16 => vec![0x400e, 0x4020, 0x3e95, 0xbeef],
        Prim::F32 => vec![0x400dd313, 0x401fd2df, 0x3e950f7a, 0xbeeef60a],
        Prim::F64 => vec![
            0x4001ba6249b12534,
            0x4003fa5beb9447ef,
            0x3fd2a1ef273c9e84,
            0xbfdddec12f33ac04,
        ],
        other => panic!("no reference for {other:?}"),
    }
}

/// `dropout(fold_in(c, -5), [1, 2, 3, 4], 0.5)` with `(_, b) = split_key(key(-3))`
/// and `(c, _) = split_key(b)`.
fn folded_dropout(prim: Prim) -> Vec<u64> {
    match prim {
        Prim::F16 => vec![0x4000, 0x4400, 0x4600, 0x0000],
        Prim::Bf16 => vec![0x4000, 0x4080, 0x40c0, 0x0000],
        Prim::F32 => vec![0x40000000, 0x40800000, 0x40c00000, 0x00000000],
        Prim::F64 => vec![
            0x4000000000000000,
            0x4010000000000000,
            0x4018000000000000,
            0x0000000000000000,
        ],
        other => panic!("no reference for {other:?}"),
    }
}

/// `uniform_like(row, t, 0, 1)` for each row of `split_keys(d, 3)`, where
/// `(_, d) = split_key(b)`; rows are `fold_in(d, j)`.
fn split_rows_unit(prim: Prim) -> Vec<u64> {
    match prim {
        Prim::F16 => vec![
            0x3083, 0x3984, 0x38a9, 0x3597, 0x3918, 0x3818, 0x3b83, 0x359e, 0x39af, 0x39bb, 0x358a,
            0x3852,
        ],
        Prim::Bf16 => vec![
            0x3e10, 0x3f30, 0x3f15, 0x3eb3, 0x3f23, 0x3f03, 0x3f70, 0x3eb4, 0x3f36, 0x3f37, 0x3eb1,
            0x3f0a,
        ],
        Prim::F32 => vec![
            0x3e106927, 0x3f307a7a, 0x3f1518c7, 0x3eb2edaa, 0x3f2301c9, 0x3f02fd86, 0x3f706137,
            0x3eb3c1d2, 0x3f35d2f0, 0x3f3756bc, 0x3eb13d4f, 0x3f0a3dda,
        ],
        Prim::F64 => vec![
            0x3fc20d24de77a4c8,
            0x3fe60f4f3a4ab507,
            0x3fe2a318db90e98c,
            0x3fd65db532c7fa2c,
            0x3fe4603919ad8e1a,
            0x3fe05fb0bf9300b0,
            0x3fee0c26dd070094,
            0x3fd6783a32482246,
            0x3fe6ba5dfddb6350,
            0x3fe6ead78a1eb8fc,
            0x3fd627a9d03d7ff0,
            0x3fe147bb3b0c92d7,
        ],
        other => panic!("no reference for {other:?}"),
    }
}

const DROP_KEY7: [u64; 4] = [0x00000000, 0x40800000, 0x40c00000, 0x41000000];
const ROWS5_DROP: [u64; 12] = [
    0x00000000, 0x40800000, 0x40c00000, 0x41000000, 0x00000000, 0x41400000, 0x00000000, 0x00000000,
    0x41900000, 0x41a00000, 0x00000000, 0x00000000,
];
const U_KEY9_2: [u64; 2] = [0x3f5a1b1e, 0x3e2763a0];
const U_KEY7_2: [u64; 2] = [0x3e019516, 0x3f56526f];
const U_KEY8_2: [u64; 2] = [0x3ed25d8d, 0x3f20c2f1];
const U_FOLD3_1: [u64; 4] = [0x3f33bd78, 0x3f68a379, 0x3f7c3266, 0x3f1c6a92];
const CHAIN3: [u64; 2] = [0x3fdef512, 0x3fcff383];

const FLOATS: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

// ---- rendering the reference and running both lanes ----

fn element(prim: Prim, bits: u64) -> String {
    let value = match prim {
        Prim::F16 => ElementRef::F16(half::f16::from_bits(bits as u16)),
        Prim::Bf16 => ElementRef::Bf16(half::bf16::from_bits(bits as u16)),
        Prim::F32 => ElementRef::F32(f32::from_bits(bits as u32)),
        Prim::F64 => ElementRef::F64(f64::from_bits(bits)),
        other => panic!("no element for {other:?}"),
    };
    format_element(prim, value)
}

fn tensor(prim: Prim, shape: &[usize], bits: &[u64]) -> String {
    let data = bits
        .iter()
        .map(|bits| element(prim, *bits))
        .collect::<Vec<_>>()
        .join(", ");
    format!("tensor(shape={shape:?}, data=[{data}])")
}

/// `name = display` for every root `chelis eval` reports, in order.
fn eval_lines(source: &str) -> Vec<String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .unwrap_or_else(|error| panic!("eval failed: {error:?}\n{source}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().expect("a named root"),
                root.display.as_deref().expect("an in-process display")
            )
        })
        .collect()
}

/// The lines the native C program prints for its roots, with the ownership
/// ledger balanced: every key tensor a host key crosses into is released.
fn c_lines(source: &str, label: &str) -> Vec<String> {
    let generated = ownership_support::emit(source, label);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    stdout.lines().map(str::to_string).collect()
}

/// Both lanes print exactly `expected`, so they also agree bit for bit.
fn both_lanes(source: &str, label: &str, expected: &[String]) {
    let eval = eval_lines(source);
    assert_eq!(eval, expected, "eval: {label}\n{source}");
    let native = c_lines(source, label);
    assert_eq!(native, expected, "C: {label}\n{source}");
}

fn main_lines(values: &[String]) -> Vec<String> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| format!("main.{index} = {value}"))
        .collect()
}

// ---- tests ----

/// (a) and (b): `key_from_seed`, both `split_key` halves, `fold_in` and the
/// `split_keys` rows (through `vmap`) key `uniform_like` and `dropout` at
/// every active float dtype, bit for bit in eval and C.
#[test]
fn every_key_source_draws_the_reference_bits_in_eval_and_c() {
    for prim in FLOATS {
        let p = prim.name();
        let zeros = format!("[cast(0.0, {p}), cast(0.0, {p}), cast(0.0, {p}), cast(0.0, {p})]");
        let source = format!(
            "def row(k: key, t: tensor[4, {p}]) -> tensor[4, {p}] = uniform_like(k, t, 0.0f32, 1.0f32)\n\
             def main() = {{\n\
             \x20 t = to_tensor({zeros})\n\
             \x20 x = to_tensor([cast(1.0, {p}), cast(2.0, {p}), cast(3.0, {p}), cast(4.0, {p})])\n\
             \x20 ts = to_tensor([{zeros}, {zeros}, {zeros}])\n\
             \x20 (a, b) = split_key(key_from_seed(-3i64))\n\
             \x20 (c, d) = split_key(b)\n\
             \x20 (uniform_like(key_from_seed(7i64), t, 0.0f32, 1.0f32), uniform_like(a, t, -2.0f32, 3.0f32), \
             dropout(fold_in(c, -5i64), x, cast(0.5, {p})), vmap(row)(split_keys(d, 3i64), ts))\n\
             }}\n"
        );
        let expected = main_lines(&[
            tensor(prim, &[4], &seed7_unit(prim)),
            tensor(prim, &[4], &left_half_bounded(prim)),
            tensor(prim, &[4], &folded_dropout(prim)),
            tensor(prim, &[3, 4], &split_rows_unit(prim)),
        ]);
        both_lanes(&source, p, &expected);
    }
}

/// (c) chelis#2409's oracle: `vmap(f)(split_keys(k, n), xs)` equals the
/// stacked per-row applications `f(fold_in(k, j), xs[j])`, and a draw after
/// the `vmap` is unaffected by it.
#[test]
fn vmap_over_key_rows_is_the_stack_of_per_row_draws() {
    let noisy = "def noisy(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, 0.5f32)\n";
    let rows = [
        "[1.0f32, 2.0f32, 3.0f32, 4.0f32]",
        "[5.0f32, 6.0f32, 7.0f32, 8.0f32]",
        "[9.0f32, 10.0f32, 11.0f32, 12.0f32]",
    ];
    let after = "uniform_like(key_from_seed(9i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)";
    let batched = format!(
        "{noisy}def main() = {{\n  xs = to_tensor([{}])\n  (vmap(noisy)(split_keys(key_from_seed(5i64), 3i64), xs), {after})\n}}\n",
        rows.join(", ")
    );
    let stacked = format!(
        "{noisy}def main() = ({}, {after})\n",
        rows.iter()
            .enumerate()
            .map(|(j, row)| format!(
                "noisy(fold_in(key_from_seed(5i64), {j}i64), to_tensor({row}))"
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let expected_batched = main_lines(&[
        tensor(Prim::F32, &[3, 4], &ROWS5_DROP),
        tensor(Prim::F32, &[2], &U_KEY9_2),
    ]);
    let expected_stacked = main_lines(&[
        tensor(Prim::F32, &[4], &ROWS5_DROP[0..4]),
        tensor(Prim::F32, &[4], &ROWS5_DROP[4..8]),
        tensor(Prim::F32, &[4], &ROWS5_DROP[8..12]),
        tensor(Prim::F32, &[2], &U_KEY9_2),
    ]);
    both_lanes(&batched, "vmap over key rows", &expected_batched);
    both_lanes(&stacked, "stacked rows", &expected_stacked);
    // The draw after the vmap is the draw with no vmap at all.
    both_lanes(
        &format!("def main() = {after}\n"),
        "no vmap",
        &[format!("main = {}", tensor(Prim::F32, &[2], &U_KEY9_2))],
    );
}

/// (d) Rule V3: a draw under a runtime branch inside a kernel takes effect
/// only in its selected arm, and both arms may consume one key.
#[test]
fn a_branch_draws_only_in_its_selected_arm() {
    let source = "def gate(k: key, x: tensor[4, f32], c: bool) -> tensor[4, f32] = if c then dropout(k, x, 0.5f32) else x\n\
                  def either(k: key, x: tensor[4, f32], c: bool) -> tensor[4, f32] = if c then dropout(k, x, 0.5f32) else uniform_like(k, x, 0.0f32, 1.0f32)\n\
                  def main() = {\n\
                  \x20 x = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n\
                  \x20 c = gt(tensor_to_scalar(sum(copy(x), 0i32)), 5.0f32)\n\
                  \x20 (gate(key_from_seed(7i64), x, c), gate(key_from_seed(7i64), x, not(c)), either(key_from_seed(7i64), x, c), either(key_from_seed(7i64), x, not(c)))\n\
                  }\n";
    let ones_to_four = [0x3f800000, 0x40000000, 0x40400000, 0x40800000];
    let unit = seed7_unit(Prim::F32);
    let expected = main_lines(&[
        tensor(Prim::F32, &[4], &DROP_KEY7),
        tensor(Prim::F32, &[4], &ones_to_four),
        tensor(Prim::F32, &[4], &DROP_KEY7),
        tensor(Prim::F32, &[4], &unit),
    ]);
    both_lanes(source, "branch arms", &expected);
}

/// (e) A key made by a host function reaches a kernel; a key parameter is
/// threaded through recursion, split at each step; keys held in a tuple and
/// in a data type are destructured before use.
#[test]
fn keys_cross_host_functions_recursion_tuples_and_data_types() {
    let source = "type Holder =\n  | Holder { k: key, t: tensor[2, f32] }\n\
                  def make_key(s: i64) -> key = fold_in(key_from_seed(s), 1i64)\n\
                  def sample(k: key, t: tensor[4, f32]) -> tensor[4, f32] = uniform_like(k, t, 0.0f32, 1.0f32)\n\
                  def chain(k: key, x: tensor[2, f32], n: i64) -> tensor[2, f32] =\n\
                  \x20 if lte(n, 0i64) then x else {\n\
                  \x20   (now, later) = split_key(k)\n\
                  \x20   chain(later, add(x, uniform_like(now, x, 0.0f32, 1.0f32)), sub(n, 1i64))\n\
                  \x20 }\n\
                  def use_holder(h: Holder) -> tensor[2, f32] = match h with {\n\
                  \x20 | Holder { k, t } => uniform_like(k, t, 0.0f32, 1.0f32)\n\
                  }\n\
                  def main() = {\n\
                  \x20 pair = (key_from_seed(7i64), to_tensor([0.0f32, 0.0f32]))\n\
                  \x20 (k, t) = pair\n\
                  \x20 (sample(make_key(3i64), to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])), chain(key_from_seed(7i64), to_tensor([0.0f32, 0.0f32]), 3i64), uniform_like(k, t, 0.0f32, 1.0f32), use_holder(Holder { k: key_from_seed(8i64), t: to_tensor([0.0f32, 0.0f32]) }))\n\
                  }\n";
    let expected = main_lines(&[
        tensor(Prim::F32, &[4], &U_FOLD3_1),
        tensor(Prim::F32, &[2], &CHAIN3),
        tensor(Prim::F32, &[2], &U_KEY7_2),
        tensor(Prim::F32, &[2], &U_KEY8_2),
    ]);
    both_lanes(source, "host keys", &expected);
}

/// `grad` treats the key as a discrete input: the gradient of a keyed
/// dropout reuses its forward mask (the kept elements' `1 / (1 - rate)`),
/// and a draw keyed by the other half is unaffected (spec/06 section 2.11).
#[test]
fn grad_through_a_keyed_draw_replays_its_mask() {
    let source = "def loss(k: key, x: tensor[4, f32]) -> tensor[f32] = sum(dropout(k, x, 0.5f32), 0i32)\n\
                  def main() = {\n\
                  \x20 x = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n\
                  \x20 (a, b) = split_key(key_from_seed(7i64))\n\
                  \x20 (grad(loss, wrt=x)(a, copy(x)), uniform_like(b, x, 0.0f32, 1.0f32))\n\
                  }\n";
    let expected = main_lines(&[
        tensor(
            Prim::F32,
            &[4],
            &[0x40000000, 0x40000000, 0x00000000, 0x00000000],
        ),
        tensor(
            Prim::F32,
            &[4],
            &[0x3ef80128, 0x3d52c14c, 0x3d44057c, 0x3ec22639],
        ),
    ]);
    both_lanes(source, "grad", &expected);
}
