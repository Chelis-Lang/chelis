//! chelis#2120 — `chelis build` rejected `uniform_like` whenever the template
//! tensor's CONTENTS were runtime-derived, while `chelis check` scored 1.0 and
//! `chelis eval` returned correct values. Only C emission failed:
//!
//! ```text
//! error: unsupported: builtin `uniform_like` on `chelis build` host emission
//! (codegen:c); deliberate [04-TOT-2]: the checked builtin vocabulary and C
//! expression vocabulary disagree; no fallback expression is permitted
//! ```
//!
//! ## What was actually wrong
//!
//! Not "the emitter requires a foldable template". The same runtime-derived
//! template consumed by `mul` emitted fine; only `uniform_like` failed. The
//! tensor-DAG lane has a correct `uniform_like` arm
//! (chelis-backend-c/src/emit.rs `emit_uniform_like`), but the C HOST lane's
//! `CExpressionBuiltin::decode` (chelis-backend-c/src/host_emit.rs) had none,
//! so it fell through to the `[04-TOT-2]` catch-all. Template foldability only
//! decided ROUTING: a non-foldable template makes the binding non-DAG-lowerable
//! (`def_is_lowered` -> `skip_for_lowered` in chelis-ir/src/host.rs), which
//! lands it in the host lane, where the missing arm then bit.
//!
//! ## Spec authority
//!
//! `spec/05-risc-primitives.md` [05-OP-8]: `uniform_like` "returns `tensor[D,
//! p]` with the template's dimensions" and "does not observe the template's
//! element values". Only shape and float dtype participate, so a template whose
//! elements are computed at runtime supplies exactly what a literal one does.
//! `template_values_do_not_affect_the_draw` asserts that claim directly, in
//! both lanes, rather than trusting it.
//!
//! ## Oracle: f32-bit parity, not stdout byte-equality
//!
//! Element-wise f32-bit equality between `chelis eval` and the compiled C
//! binary, matching chelis#770's comparator and for the same reason: the two
//! lanes render floats through different printers, so raw stdout can differ
//! while the underlying f32 values are bit-identical. `parse_tensor_data` fails
//! loudly on a missing `sampled` line, so a skipped element cannot mask a
//! divergence.
//!
//! Bounds are bare positive literals over the non-unit range `[2, 5)`. That is
//! deliberate: `[0, 1)` is also the range a silently-collapsed bound would
//! produce (chelis#776), so a unit-range oracle could pass while the bounds
//! were being dropped. A non-unit range cannot pass that way.
//!
//! ## Evidentiary status, per assertion
//!
//! * REGRESSION TESTS — `chelis build` exits 1 on the base sha (122b55ab8)
//!   with the message above, verified by running this file against a
//!   base-sha `chelis`:
//!   `runtime_derived_template_builds_and_matches_eval`,
//!   `expand_inside_parameterised_def_matches_eval`,
//!   `template_values_do_not_affect_the_draw`,
//!   `two_draws_from_split_keys_differ_and_match_eval` (then an ordinal test),
//!   `host_and_dag_draws_from_split_keys_agree` (then a shared-scope test), and
//!   `every_active_float_dtype_matches_eval_with_a_runtime_template`.
//!   The last one is red twice over: on the base sha the build is refused,
//!   and with only f32/f64 arms its f16 and bf16 rows built successfully and
//!   then aborted at run time.
//! * `constant_foldable_template_still_matches_eval` is a DISPOSITION LOCK:
//!   green before and after. It proves the fix did not perturb the path that
//!   already worked.
//! * `different_keys_produce_different_draws_with_runtime_template` is an
//!   EVALUATOR-ONLY non-triviality control; it does not drive the C lane.
//! * `runtime_computed_bounds_sample_in_both_lanes` pins chelis#2411: computed
//!   bounds are ordinary operands in both lanes.
//!
//! chelis#2413 moved every program here to explicit keys: a draw reads
//! `key_from_seed(seed)` or a half of `split_key`, and the headline and
//! literal-template rows also compare the f32 draw with `common::key_ref`'s
//! [05-OP-8] reference, so both lanes being wrong in the same way fails.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// An 8-element template whose elements are RUNTIME-DERIVED: `bc`'s parameter
/// `c` reaches `to_tensor`, so the template cannot be constant-folded. This is
/// the shape chelis#2120 rejected.
fn runtime_template_program(low: &str, high: &str, seed: u64) -> String {
    format!(
        "def bc(c: f32) -> tensor[8, f32] = \
         to_tensor([c, c, c, c, c, c, c, c])\n\
         sampled = uniform_like(key_from_seed({seed}i64), \
         bc(cast(0.5, f32)), {low}, {high})\n"
    )
}

/// The same draw from a template the compiler CAN fold. This path built before
/// the fix and must keep building, with the same values.
fn literal_template_program(low: &str, high: &str, seed: u64) -> String {
    format!(
        "template = to_tensor([cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), \
         cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), \
         cast(0.5, f32)])\n\
         sampled = uniform_like(key_from_seed({seed}i64), copy(template), \
         {low}, {high})\n"
    )
}

/// A runtime-derived template of the same shape and dtype, but with wildly
/// different element VALUES. [05-OP-8] says those values are not observed, so
/// this must draw exactly what `runtime_template_program` draws.
fn runtime_template_varied_values_program(low: &str, high: &str, seed: u64) -> String {
    format!(
        "def bc(c: f32) -> tensor[8, f32] = \
         to_tensor([c, mul(c, 3.0f32), -7.5f32, 1000.0f32, \
         neg(c), 0.0f32, mul(c, -2.0f32), 12345.0f32])\n\
         sampled = uniform_like(key_from_seed({seed}i64), \
         bc(cast(0.5, f32)), {low}, {high})\n"
    )
}

/// The field program from the issue: `expand` + `reshape` inside a
/// parameterised def. What this row adds over
/// `runtime_derived_template_builds_and_matches_eval` is the movement-op
/// shape, not the size, so the extent drops from the reported 100000 to 16.
/// The issue's own bisection establishes that extent is not the
/// discriminator, and the tensor renderer prints only the first 32 elements
/// before eliding with `...`, which `parse_tensor_data` cannot read — a
/// larger extent would make this a test of the printer, not of the draw.
fn expand_in_def_program(seed: u64) -> String {
    format!(
        "def bc(c: f32) -> tensor[16, f32] = \
         reshape(expand(to_tensor([c]), 0, 16i64), [16i64])\n\
         sampled = uniform_like(key_from_seed({seed}i64), \
         bc(cast(0.5, f32)), 2.0f32, 5.0f32)\n"
    )
}

/// Run `chelis eval --file` and return the printed `sampled` tensor data.
fn eval_sampled(program: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "chelis eval failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&out.stdout), "sampled")
}

/// Build the program to C, link it, run it, and return the printed `sampled`
/// tensor data.
fn c_sampled(program: &str, name: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let status = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary exited non-zero: {}",
        String::from_utf8_lossy(&run.stderr),
    );
    parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled")
}

/// Assert both lanes produce the same number of elements and that every
/// element is bit-identical once the host f64 value is narrowed to f32.
fn assert_f32_bit_parity(eval: &[f64], c: &[f64], context: &str) {
    assert_eq!(
        eval.len(),
        c.len(),
        "[{context}] element count mismatch: eval {} vs C {}",
        eval.len(),
        c.len(),
    );
    assert!(!eval.is_empty(), "[{context}] no elements parsed");
    for (i, (e, cv)) in eval.iter().zip(c.iter()).enumerate() {
        assert_eq!(
            (*e as f32).to_bits(),
            (*cv as f32).to_bits(),
            "[{context}] elem[{i}] f32 bits diverge: eval {} (bits {:#010x}) vs C {} (bits {:#010x})",
            e,
            (*e as f32).to_bits(),
            cv,
            (*cv as f32).to_bits(),
        );
    }
}

/// `uniform_like(key, template, low, high)` over `count` f32 elements, from
/// `common::key_ref`'s [05-OP-8] transcription.
fn reference_f32(key: u64, count: usize, low: f32, high: f32) -> Vec<f64> {
    common::key_ref::uniform_f32(key, count, low, high)
        .into_iter()
        .map(f64::from)
        .collect()
}

/// The headline chelis#2120 regression: a runtime-derived template must build
/// and must agree with `eval` element-for-element.
#[test]
fn runtime_derived_template_builds_and_matches_eval() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = runtime_template_program("2.0f32", "5.0f32", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "u2120_runtime");
    assert_f32_bit_parity(&eval, &c, "runtime-derived template, [2,5)");
    let reference = reference_f32(common::key_ref::key_from_seed(42), 8, 2.0, 5.0);
    assert_f32_bit_parity(&reference, &c, "reference vs C, [2,5)");
}

/// The issue's original field program shape: `expand` + `reshape` behind a
/// parameterised def.
#[test]
fn expand_inside_parameterised_def_matches_eval() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = expand_in_def_program(42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "u2120_expand");
    assert_f32_bit_parity(&eval, &c, "expand/reshape inside a def, [2,5)");
}

/// [05-OP-8]: "does not observe the template's element values". Two programs
/// differing ONLY in the template's runtime values must draw identically — and
/// must do so in both lanes, which is what distinguishes a real fix from both
/// lanes being wrong in the same direction.
#[test]
fn template_values_do_not_affect_the_draw() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let plain = runtime_template_program("2.0f32", "5.0f32", 42);
    let varied = runtime_template_varied_values_program("2.0f32", "5.0f32", 42);

    let eval_plain = eval_sampled(&plain);
    let eval_varied = eval_sampled(&varied);
    assert_f32_bit_parity(&eval_plain, &eval_varied, "eval lane, values not observed");

    let c_plain = c_sampled(&plain, "u2120_vals_plain");
    let c_varied = c_sampled(&varied, "u2120_vals_varied");
    assert_f32_bit_parity(&c_plain, &c_varied, "C lane, values not observed");

    // and the two lanes still agree with each other
    assert_f32_bit_parity(
        &eval_varied,
        &c_varied,
        "cross-lane, varied template values",
    );
}

/// Disposition lock: the constant-foldable template already built before this
/// fix. It must still build, and still match `eval`.
#[test]
fn constant_foldable_template_still_matches_eval() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = literal_template_program("2.0f32", "5.0f32", 42);
    let eval = eval_sampled(&src);
    let c = c_sampled(&src, "u2120_literal");
    assert_f32_bit_parity(&eval, &c, "constant-foldable template, [2,5)");
    let reference = reference_f32(common::key_ref::key_from_seed(42), 8, 2.0, 5.0);
    assert_f32_bit_parity(&reference, &c, "reference vs C, literal template");
}

/// Two host-lane draws from the two halves of one `split_key`. A host-lane
/// `uniform_like` must read the key it is given: were both draws to read the
/// same key, every element of their difference would be exactly 0.0. Both
/// lanes agree with eval and with the reference difference.
#[test]
fn two_draws_from_split_keys_differ_and_match_eval() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
               sampled = {\n\
               \x20 (a, b) = split_key(key_from_seed(42i64))\n\
               \x20 sub(uniform_like(a, bc(cast(0.5, f32)), 2.0f32, 5.0f32), \
               uniform_like(b, bc(cast(0.5, f32)), 2.0f32, 5.0f32))\n\
               }\n";
    let eval = eval_sampled(src);
    let c = c_sampled(src, "u2120_two_draws");
    assert_f32_bit_parity(&eval, &c, "two host-lane draws from split keys");
    assert!(
        eval.iter().any(|v| *v != 0.0),
        "two draws from split keys produced an all-zero difference, so both \
         read one key: {eval:?}",
    );
    let (a, b) = common::key_ref::split(common::key_ref::key_from_seed(42));
    let reference = common::key_ref::uniform_f32(a, 4, 2.0, 5.0)
        .into_iter()
        .zip(common::key_ref::uniform_f32(b, 4, 2.0, 5.0))
        .map(|(left, right)| f64::from(left - right))
        .collect::<Vec<_>>();
    assert_f32_bit_parity(&reference, &c, "reference vs C, split-key difference");
}

/// One draw whose template is runtime-derived (host lane) beside one whose
/// template folds (tensor-DAG lane), keyed by the two halves of one
/// `split_key`. Each lane must read its own key, so the difference is
/// non-zero and agrees with eval.
#[test]
fn host_and_dag_draws_from_split_keys_agree() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = "template = to_tensor([cast(0.5, f32), cast(0.5, f32), \
               cast(0.5, f32), cast(0.5, f32)])\n\
               def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
               sampled = {\n\
               \x20 (a, b) = split_key(key_from_seed(42i64))\n\
               \x20 sub(uniform_like(a, bc(cast(0.5, f32)), 2.0f32, 5.0f32), \
               uniform_like(b, copy(template), 2.0f32, 5.0f32))\n\
               }\n";
    let eval = eval_sampled(src);
    let c = c_sampled(src, "u2120_mixed_lanes");
    assert_f32_bit_parity(&eval, &c, "host-lane and DAG-lane draws from split keys");
    assert!(
        eval.iter().any(|v| *v != 0.0),
        "mixed-lane draws produced an all-zero difference, so the two lanes \
         read one key: {eval:?}",
    );
}

/// [05-OP-8] "admits every active float template dtype `p`". A runtime-derived
/// template is precisely the case the DAG lane does not serve, so each dtype
/// must be served here or the program builds and then aborts. f16/bf16 sample
/// in f32 and narrow exactly once at the store, per the atom.
///
/// Before the f16/bf16 arms existed, `chelis build` exited 0 on these and the
/// compiled binary died with `uniform_like unsupported dtype 6` (f16) / `5`
/// (bf16) — strictly worse than the build-time refusal on the base sha.
#[test]
fn every_active_float_dtype_matches_eval_with_a_runtime_template() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    for (dtype, label) in [
        ("f32", "f32"),
        ("f64", "f64"),
        ("f16", "f16"),
        ("bf16", "bf16"),
    ] {
        let src = format!(
            "def bc(c: {dtype}) -> tensor[4, {dtype}] = to_tensor([c, c, c, c])\n\
             sampled = uniform_like(key_from_seed(42i64), \
             bc(cast(0.5, {dtype})), 2.0f32, 5.0f32)\n"
        );
        let eval = eval_sampled(&src);
        let c = c_sampled(&src, &format!("u2120_dtype_{label}"));
        assert_f32_bit_parity(&eval, &c, &format!("runtime template at {label}"));
    }
}

/// Non-triviality control, on the EVALUATOR lane only: two keys must give
/// different draws, so a parity oracle cannot pass by every value being equal.
/// The cross-lane non-vacuity proof is elsewhere: the two all-zero assertions
/// in the split-key tests, and the reference comparisons above.
/// Mirrors chelis#770's `different_keys_produce_different_draws`.
#[test]
fn different_keys_produce_different_draws_with_runtime_template() {
    let a = eval_sampled(&runtime_template_program("2.0f32", "5.0f32", 42));
    let b = eval_sampled(&runtime_template_program("2.0f32", "5.0f32", 43));
    assert_eq!(a.len(), b.len(), "both draws should have 8 elements");
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| x != y),
        "seeds 42 and 43 produced identical draws: {a:?} vs {b:?}",
    );
}

/// chelis#2411/#2413: runtime-computed bounds are ordinary [05-OP-8] scalar
/// operands. The host and DAG lanes both draw from the computed range, and
/// agree with eval bit for bit.
#[test]
fn runtime_computed_bounds_sample_in_both_lanes() {
    let src = "def bc(c: f32) -> tensor[8, f32] = \
               to_tensor([c, c, c, c, c, c, c, c])\n\
               def lo(x: f32) -> f32 = mul(x, 2.0f32)\n\
               sampled = uniform_like(key_from_seed(42i64), bc(cast(0.5, f32)), \
               lo(cast(1.0, f32)), 5.0f32)\n";
    let eval = eval_sampled(src);
    assert!(
        eval.iter().all(|sample| (2.0..5.0).contains(sample)),
        "{eval:?}"
    );
    if !gcc_available() {
        eprintln!("skipping the C lane: no host C compiler");
        return;
    }
    let c = c_sampled(src, "runtime_bounds");
    assert_f32_bit_parity(&eval, &c, "runtime bounds");
}
