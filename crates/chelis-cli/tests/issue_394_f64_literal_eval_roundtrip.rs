//! Issue #394: f64 literal precision survives parse -> eval end to end.
//!
//! #394 (filed at 0.8.0) reported that float literals lex at f32
//! precision, so an f64 source constant is unrecoverable. This does not
//! reproduce at HEAD: the lexer parses unsuffixed float literals at f64
//! (`crates/chelis-surf/src/lexer.rs`, `let val: f64 = clean.parse()`),
//! and `cast(<lit>, f64)` binds the literal AT f64 rather than narrowing
//! to the §5.3 f32 default and widening -- the fix landed in #372
//! (`crates/chelis-surf/src/desugar.rs`, `scalar_literal_adopts_cast_target`)
//! and the guarantee is stated in `spec/04-type-system.md` §5.6 position 4
//! / §5.3 (added under #308). The desugar-level unit tests in
//! `desugar.rs` (`cast_of_float_literal_adopts_target_precision`, etc) pin
//! the lowering; this file adds the one missing lock -- an end-to-end
//! `chelis eval` round-trip through the JSON channel -- so a regression at
//! any later stage (eval, runtime tensor storage, JSON serialization)
//! would be caught, not just a desugar regression.
//!
//! Discriminator value: `1.0000000000000002` is `1.0 + 2^-52`, the
//! smallest f64 above 1.0. It is NOT representable in f32 (it rounds to
//! exactly `1.0`). So `cast(1.0000000000000002, f64) - cast(1.0, f64)`
//! is `2.220446049250313e-16` (f64 machine epsilon) iff full f64
//! precision is retained, and exactly `0.0` if the literal was ever
//! quantized to f32.

use assert_cmd::Command;
use serde_json::Value;

/// f64 machine epsilon -- `1.0000000000000002 - 1.0` at full f64 width.
const F64_EPSILON: f64 = 2.220446049250313e-16;

fn eval_json(expr: &str) -> Value {
    // `chelis eval EXPR` (inline) has no on-disk source, so the style
    // gate does not apply; stdout carries JSON only.
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--json", expr])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(
        output.status.success(),
        "chelis eval must succeed; stdout={stdout}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("chelis eval stdout must be valid JSON: {err}\n{stdout}"))
}

fn scalar_result(json: &Value) -> f64 {
    assert_eq!(json["schema_version"], 4);
    let value = json
        .get("roots")
        .and_then(Value::as_array)
        .and_then(|roots| roots.first())
        .and_then(|root| root.get("value"))
        .unwrap_or_else(|| panic!("expected a scalar root value; json={json}"));
    assert_eq!(value["type"], "scalar");
    assert_eq!(value["value"]["dtype"], "f64");
    let bits = value["value"]["bits"].as_str().expect("stored f64 bits");
    assert_eq!(bits.len(), 16, "f64 carries exactly 64 bits");
    assert!(
        bits.bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    f64::from_bits(u64::from_str_radix(bits, 16).expect("canonical f64 bits"))
}

#[test]
fn f64_literal_above_one_ulp_survives_eval() {
    // The literal itself round-trips bit-for-bit through parse -> eval.
    // If the lexer or any later stage quantized to f32 this would read
    // back as exactly 1.0.
    let v = scalar_result(&eval_json("cast(1.0000000000000002, f64)"));
    assert_eq!(
        v, 1.0000000000000002,
        "f64 literal must retain its >f32-ULP precision through eval"
    );
    assert_ne!(v, 1.0, "an f32-quantized literal would read back as 1.0");
}

#[test]
fn f64_literal_subtraction_yields_f64_epsilon_not_zero() {
    // The decisive discriminator: f64 retains the sub-ULP difference, f32
    // does not. `0.0` here would mean the literal was quantized to f32.
    let diff = scalar_result(&eval_json("cast(1.0000000000000002, f64) - cast(1.0, f64)"));
    assert_eq!(
        diff, F64_EPSILON,
        "f64 literal precision must survive parse->eval; got {diff} (0.0 would mean f32 quantization)"
    );
    assert_ne!(diff, 0.0, "an f32-quantized literal would cancel to 0.0");
}

// The f32-rounded value of 0.1 is written out to full f32 precision on
// purpose: the assert below must compare against exactly that value, so
// clippy::excessive_precision (which would have us drop the "meaningless"
// digits) is intentionally allowed here.
#[allow(clippy::excessive_precision)]
#[test]
fn f64_tenth_literal_is_exact_dyadic_not_f32_rounded() {
    // `0.1` cast to f64 is the f64-nearest value to one tenth. The f32
    // round of 0.1 (0.10000000149011612) widened to f64 would be a
    // different, larger value; binding at f64 directly avoids that.
    let v = scalar_result(&eval_json("cast(0.1, f64)"));
    assert_eq!(
        v, 0.1,
        "cast(0.1, f64) must bind at f64, not the f32-widened 0.1"
    );
    assert_ne!(
        v, 0.10000000149011612_f32 as f64,
        "must not be the f32-rounded-then-widened value"
    );
}
