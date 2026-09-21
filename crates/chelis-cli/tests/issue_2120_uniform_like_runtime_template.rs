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
//! * `runtime_derived_template_builds_and_matches_eval`,
//!   `expand_inside_parameterised_def_matches_eval` and
//!   `template_values_do_not_affect_the_draw` are REGRESSION TESTS: `chelis
//!   build` exits 1 on the base sha (122b55ab8) with the message above.
//! * `constant_foldable_template_still_matches_eval` is a DISPOSITION LOCK:
//!   green before and after. It proves the fix did not perturb the path that
//!   already worked.
//! * `different_seeds_produce_different_draws_with_runtime_template` is a
//!   non-triviality control, so the parity oracle cannot pass vacuously.
//! * `runtime_computed_bounds_remain_rejected_in_both_lanes` is NEGATIVE
//!   PARITY: the separate chelis#776 bounds gate is a CHECKER rejection and
//!   must stay closed in both lanes. This fix widens the template contract
//!   only; it must not open the bounds contract as a side effect.

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
         sampled = with seed({seed}i64) \
         {{ uniform_like(bc(cast(0.5, f32)), {low}, {high}) }}\n"
    )
}

/// The same draw from a template the compiler CAN fold. This path built before
/// the fix and must keep building, with the same values.
fn literal_template_program(low: &str, high: &str, seed: u64) -> String {
    format!(
        "template = to_tensor([cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), \
         cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), cast(0.5, f32), \
         cast(0.5, f32)])\n\
         sampled = with seed({seed}i64) \
         {{ uniform_like(copy(template), {low}, {high}) }}\n"
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
         sampled = with seed({seed}i64) \
         {{ uniform_like(bc(cast(0.5, f32)), {low}, {high}) }}\n"
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
         sampled = with seed({seed}i64) \
         {{ uniform_like(bc(cast(0.5, f32)), 2.0f32, 5.0f32) }}\n"
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
}

/// Ordinal consumption, the highest-risk property of this fix. A host-lane
/// `uniform_like` is the FIRST host-lane consumer of a Random call ordinal;
/// every earlier host draw site was a carrier that handed `__chelis_rng` to a
/// DAG kernel. Two draws in one `with seed` scope must therefore consume two
/// distinct ordinals, exactly as the evaluator's `random_counter` does.
///
/// `sub` of the two draws makes the failure mode visible: if the second draw
/// reused the first's ordinal, every element would be exactly 0.0.
#[test]
fn two_draws_in_one_scope_consume_distinct_ordinals() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
               sampled = with seed(42i64) \
               { sub(uniform_like(bc(cast(0.5, f32)), 2.0f32, 5.0f32), \
               uniform_like(bc(cast(0.5, f32)), 2.0f32, 5.0f32)) }\n";
    let eval = eval_sampled(src);
    let c = c_sampled(src, "u2120_two_draws");
    assert_f32_bit_parity(&eval, &c, "two host-lane draws in one seed scope");
    assert!(
        eval.iter().any(|v| *v != 0.0),
        "two draws in one scope produced an all-zero difference, so the second \
         draw reused the first draw's ordinal: {eval:?}",
    );
}

/// The desync case the shared RNG state makes possible: one draw whose
/// template is runtime-derived (host lane) beside one whose template folds
/// (tensor-DAG lane), inside a single `with seed` scope. Both mutate the same
/// `__chelis_rng`, so a host arm that advanced the counter at a different
/// point than the evaluator would shift every later draw in the scope.
#[test]
fn host_and_dag_draws_share_one_rng_scope_consistently() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let src = "template = to_tensor([cast(0.5, f32), cast(0.5, f32), \
               cast(0.5, f32), cast(0.5, f32)])\n\
               def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
               sampled = with seed(42i64) \
               { sub(uniform_like(bc(cast(0.5, f32)), 2.0f32, 5.0f32), \
               uniform_like(copy(template), 2.0f32, 5.0f32)) }\n";
    let eval = eval_sampled(src);
    let c = c_sampled(src, "u2120_mixed_lanes");
    assert_f32_bit_parity(&eval, &c, "host-lane and DAG-lane draws in one scope");
    assert!(
        eval.iter().any(|v| *v != 0.0),
        "mixed-lane draws produced an all-zero difference, so the two lanes \
         drew the same ordinal: {eval:?}",
    );
}

/// Non-triviality control: the parity oracle must not be trivially always-equal.
/// Mirrors chelis#770's `different_seeds_produce_different_draws`.
#[test]
fn different_seeds_produce_different_draws_with_runtime_template() {
    let a = eval_sampled(&runtime_template_program("2.0f32", "5.0f32", 42));
    let b = eval_sampled(&runtime_template_program("2.0f32", "5.0f32", 43));
    assert_eq!(a.len(), b.len(), "both draws should have 8 elements");
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| x != y),
        "seeds 42 and 43 produced identical draws: {a:?} vs {b:?}",
    );
}

/// Negative parity. chelis#2120 widens only the TEMPLATE contract. The separate
/// chelis#776 bounds gate is a checker rejection and must stay closed, with the
/// same message on both lanes — a C emission arm must not become a back door to
/// runtime-computed bounds.
#[test]
fn runtime_computed_bounds_remain_rejected_in_both_lanes() {
    const EXPECTED: &str = "uniform_like currently requires literal low/high bounds";
    let src = "def bc(c: f32) -> tensor[8, f32] = \
               to_tensor([c, c, c, c, c, c, c, c])\n\
               def lo(x: f32) -> f32 = mul(x, 2.0f32)\n\
               sampled = with seed(42i64) \
               { uniform_like(bc(cast(0.5, f32)), lo(cast(1.0, f32)), 5.0f32) }\n";

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg.ch");
    let out_dir = dir.path().join("neg-out");
    write_file(&path, src);

    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(
        !build.status.success(),
        "runtime-computed bounds must still be rejected by `chelis build`",
    );
    let build_err = String::from_utf8_lossy(&build.stderr).to_string();
    assert!(
        build_err.contains(EXPECTED),
        "build rejection should cite the chelis#776 bounds gate, got: {build_err}",
    );

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        !eval.status.success(),
        "runtime-computed bounds must still be rejected by `chelis eval`",
    );
    let eval_err = String::from_utf8_lossy(&eval.stderr).to_string();
    assert!(
        eval_err.contains(EXPECTED),
        "eval rejection should cite the chelis#776 bounds gate, got: {eval_err}",
    );
}
