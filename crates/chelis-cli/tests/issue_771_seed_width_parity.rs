//! chelis#771 - a seed literal `>= 2^31` must derive the SAME u64 seed in
//! `chelis eval` as in the compiled C lane, so both lanes sample the same
//! random stream. Under explicit keys (chelis#2413) the seed reaches the draw
//! through `key_from_seed(Ni64)`, whose key is the seed's two's-complement
//! bits ([05-RNG-2]); the parity rows also compare eval with the
//! `common::key_ref` reference, so both lanes agreeing on a wrong key fails.
//!
//! ## The bug (verified against origin/main 7412d44f, pre-fix)
//!
//! The evaluator read the seed by routing the literal through `eval_lit`,
//! which narrows an unsuffixed integer to the spec/04-type-system.md §5.3
//! i32 default: `4294967295 as i32` = `-1`, sign-extended to the u64 seed
//! `0xFFFF_FFFF_FFFF_FFFF`. The C host lane reads the raw i64 atom and seeds
//! `(uint64_t)4294967295`. Same source seed, completely unrelated streams
//! (0/8 elements agreed). The fix reads a *literal* seed at full i64 width in
//! the eval random handle-effect branch, mirroring host lowering.
//!
//! ## Oracle: element-wise f32-BIT equality on `[0, 1)` probes
//!
//! The lanes agree on the sample *bit-for-bit in f32*; the oracle asserts
//! exactly that, never a float tolerance. A tolerant compare is the
//! divergence-laundering anti-pattern chelis#687 exists to kill - both prior
//! cross-lane RNG oracles stayed green through rounds of this bug because they
//! compared with a tolerance.
//!
//! - **f32 bits, not stdout bytes.** `eval` renders tensor data in f64 and the
//!   C lane renders f32, so the two printed decimals never match verbatim (that
//!   rendering divergence is chelis#732). Both printed decimals are parsed and
//!   cast to f32; casting recovers the exact f32 each lane computed (the C lane
//!   prints enough digits to round-trip its f32), so bit-equality tests the
//!   sample, not the text.
//! - **`[0, 1)` range, bare-literal bounds.** On `[0, 1)` the sampler's affine
//!   is `0.0 + 1.0 * unit = unit` with no rounding, so `f32(eval) == C`
//!   bit-exact (the #735 sweep measured 0/200000 disagreements). Non-`[0, 1)`
//!   ranges reintroduce an affine rounding difference (chelis#770); cast-wrapped
//!   bounds are a separate divergence (chelis#776). Bare bounds on `[0, 1)`
//!   keep this test isolated to #771's seed-agreement question.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, parse_tensor_data, write_file};

/// An 8-element f32 template sampled through `key_from_seed(seed)` on a `[0, 1)`
/// uniform with bare-literal bounds. No explicit `print`: both lanes dump every
/// top-level binding, so the `sampled = tensor(...)` line is present either way.
fn seeded_uniform(seed: u64) -> String {
    let zeros = ["cast(0.0, f32)"; 8].join(", ");
    format!(
        "template = to_tensor([{zeros}])\n\
         sampled = uniform_like(key_from_seed({seed}i64), copy(template), 0.0, 1.0)\n"
    )
}

/// The `sampled` stream from `chelis eval` (f64 data, rendered f64).
fn eval_stream(seed: u64) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("seeded.ch");
    write_file(&path, &seeded_uniform(seed));
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed for seed {seed}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse_tensor_data(&String::from_utf8_lossy(&out.stdout), "sampled")
}

/// The `sampled` stream from the compiled C lane (f32 data, rendered f32).
fn c_stream(seed: u64) -> Vec<f64> {
    parse_tensor_data(&build_and_run(&seeded_uniform(seed), "seeded"), "sampled")
}

/// The f32 bit pattern of a printed sample. Both lanes' decimals are parsed to
/// f64 (via `parse_tensor_data`) and cast to f32 here, recovering the exact f32
/// each lane computed regardless of how many digits it rendered.
fn f32_bits(sample: f64) -> u32 {
    (sample as f32).to_bits()
}

/// Whether two streams are element-wise bit-identical in f32.
fn streams_bit_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| f32_bits(*x) == f32_bits(*y))
}

/// The reference `[0, 1)` f32 stream of `key_from_seed(seed)` from
/// `common::key_ref`'s [05-OP-8] transcription.
fn reference_stream(seed: u64) -> Vec<f64> {
    common::key_ref::uniform_f32(common::key_ref::key_from_seed(seed as i64), 8, 0.0, 1.0)
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Assert two sampled streams are element-wise **bit-identical in f32**.
fn assert_f32_bit_equal(label: &str, eval: &[f64], c: &[f64]) {
    assert_eq!(
        eval.len(),
        c.len(),
        "{label}: length mismatch\n eval={eval:?}\n c   ={c:?}"
    );
    for (i, (e, cc)) in eval.iter().zip(c).enumerate() {
        assert_eq!(
            f32_bits(*e),
            f32_bits(*cc),
            "{label}: element {i} differs in f32 bits (eval {e:?} vs C {cc:?})\n \
             eval={eval:?}\n c   ={c:?}",
        );
    }
}

/// The three seeds in chelis#771: the `2^31 - 1` control (fits i32, so it
/// samples the same stream both lanes even pre-fix), the exact `2^31` boundary,
/// and the max-u32 case. A matching seed produces a bit-identical f32 stream in
/// both lanes. Pre-fix the boundary and max cases sampled unrelated streams
/// (eval seeded `0xFFFF_FFFF_8000_0000` / `0xFFFF_FFFF_FFFF_FFFF`); all three
/// are bit-exact post-fix.
#[test]
fn seed_literal_parity_across_lanes() {
    for &seed in &[2_147_483_647_u64, 2_147_483_648, 4_294_967_295] {
        let eval = eval_stream(seed);
        assert_f32_bit_equal(&format!("seed {seed}"), &eval, &c_stream(seed));
        assert_f32_bit_equal(
            &format!("seed {seed} reference"),
            &eval,
            &reference_stream(seed),
        );
    }
}

/// Degenerate-fix guard: distinct large seeds must still produce DISTINCT
/// streams. A fix that dropped the seed (e.g. always seeding 0, or clamping
/// every large seed to one truncated value) would make these collide while
/// still passing `seed_literal_parity_across_lanes`. Checked in BOTH lanes.
#[test]
fn distinct_large_seeds_differ() {
    let a = 2_147_483_648_u64;
    let b = 4_294_967_295_u64;
    assert!(
        !streams_bit_equal(&eval_stream(a), &eval_stream(b)),
        "eval: distinct large seeds {a} and {b} collided to the same stream",
    );
    assert!(
        !streams_bit_equal(&c_stream(a), &c_stream(b)),
        "C: distinct large seeds {a} and {b} collided to the same stream",
    );
}

/// In-range regression: a small seed (`< 2^31`, never affected by #771) still
/// agrees bit-for-bit across lanes. Pins the unchanged path so a future
/// refactor of the literal peel cannot silently break small seeds.
#[test]
fn small_seed_42_parity_across_lanes() {
    let eval = eval_stream(42);
    assert_f32_bit_equal("seed 42", &eval, &c_stream(42));
    assert_f32_bit_equal("seed 42 reference", &eval, &reference_stream(42));
}
