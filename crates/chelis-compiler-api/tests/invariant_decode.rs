//! Conformance suite for the opaque-type decode chokepoint
//! (RFC `opaque_invariants_rfc.md` D-DECODE, spec/10 section 4).
//!
//! This suite is the acceptance ORACLE for W5: it exercises the public
//! `decode_adt_value` chokepoint against valid, boundary, boundary-violating,
//! NaN, Inf, structurally-corrupt, and nested payloads, and pins the
//! fail-closed and structural-vs-invariant distinctions. It is the only
//! caller of the chokepoint in V1 (no production codec materializes a typed
//! ADT value from an external payload; survey section 7).
//!
//! Run: `cargo nextest run -p chelis-compiler-api --test invariant_decode`.

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use chelis_compiler_api::{DecodeError, RuntimeValue, decode_adt_value, try_decode_adt_value};
use chelis_deep::ast::Expr;

/// The flagship probability invariant: `p.value >= 0.0 && p.value <= 1.0`.
/// The defining module exports `make`, an in-module constructor, so the
/// evaluator can build oracle values from text (in-module construction is
/// legal under opacity).
const PROBABILITY_SRC: &str = r#"
module Stats.Prob

@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability = | Probability { value: f32 }

def make(x: f32) -> Probability = Probability { value: x }
"#;

/// A non-opaque wrapper carrying an opaque field, to exercise nested-record
/// decode (valid inner accepted, inner-violating rejected naming the inner
/// type). `wrap` builds it in-module.
const WRAPPER_SRC: &str = r#"
module Stats.Prob

@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability = | Probability { value: f32 }

type Pair = | Pair { prob: Probability, weight: f32 }

def make(x: f32) -> Probability = Probability { value: x }
def wrap(x: f32, w: f32) -> Pair = Pair { prob: make(x), weight: w }
"#;

/// Desugar Surf to the Deep program exprs the chokepoint needs (deftype
/// declarations supply the field tables and invariants).
fn program_exprs(source: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    chelis_surf::desugar::desugar_program(&decls)
}

/// Evaluate `source` and return the `ExecutionValue` bound to top-level
/// `result`. This is the ORACLE: the value the evaluator builds from text.
fn eval_result_value(source: &str) -> ExecutionValue {
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: Default::default(),
    })
    .expect("program evaluates");
    outcome
        .roots
        .into_iter()
        .find(|root| root.name.as_deref() == Some("result"))
        .expect("a `result` root")
        .value
}

/// Structural equality over `ExecutionValue` with EXACT scalar comparison
/// (bit-identical floats; no tolerance). `ExecutionValue` does not derive
/// `PartialEq` because it carries `f64`, so the comparison is explicit and
/// the float equality is deliberately strict (this is a round-trip check,
/// not a numeric-agreement check).
fn execution_values_identical(a: &ExecutionValue, b: &ExecutionValue) -> bool {
    use ExecutionValue::*;
    match (a, b) {
        (Int64 { value: x }, Int64 { value: y }) => x == y,
        // Bit-identical float compare: `to_bits` so a NaN payload would
        // compare equal to itself, and -0.0 is distinguished from 0.0.
        (Float64 { value: x }, Float64 { value: y }) => x.to_bits() == y.to_bits(),
        (Bool { value: x }, Bool { value: y }) => x == y,
        (String { value: x }, String { value: y }) => x == y,
        (Unit, Unit) => true,
        (Tensor { value: x }, Tensor { value: y }) => {
            x.shape == y.shape
                && x.data.len() == y.data.len()
                && x.data
                    .iter()
                    .zip(y.data.iter())
                    .all(|(l, r)| l.to_bits() == r.to_bits())
        }
        (List { value: x }, List { value: y }) | (Tuple { value: x }, Tuple { value: y }) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(l, r)| execution_values_identical(l, r))
        }
        (
            Adt {
                ctor: cx,
                fields: fx,
            },
            Adt {
                ctor: cy,
                fields: fy,
            },
        ) => {
            cx == cy
                && fx.len() == fy.len()
                && fx
                    .iter()
                    .zip(fy.iter())
                    .all(|(l, r)| execution_values_identical(l, r))
        }
        _ => false,
    }
}

/// Pull the single f32 field out of a decoded `Probability` value.
fn probability_field(value: &RuntimeValue) -> f64 {
    let (ctor, fields) = value.as_adt().expect("decoded an ADT");
    assert_eq!(ctor, "Probability", "constructor preserved");
    assert_eq!(fields.len(), 1, "Probability has exactly one field");
    fields[0].as_f64().expect("value field is a float scalar")
}

// ── Oracle agreement + never-repair (accepted decode is bit-identical) ────

#[test]
fn valid_payload_decodes_to_evaluator_value_oracle_agreement() {
    // ORACLE: evaluate `make(0.3)` from text, then decode that exact
    // ExecutionValue payload. The decoded value, re-encoded, must be
    // bit-identical to the evaluator's output. This proves decode is the
    // structural inverse of the evaluator's encode on its own output.
    let src = format!("{PROBABILITY_SRC}\nresult = make(cast(0.3, f32))\n");
    let oracle = eval_result_value(&src);
    assert!(
        matches!(&oracle, ExecutionValue::Adt { ctor, .. } if ctor == "Probability"),
        "oracle is a Probability ADT, got {oracle:?}"
    );

    let exprs = program_exprs(PROBABILITY_SRC);
    let decoded = decode_adt_value(&exprs, &oracle).expect("0.3 is admissible");

    // never-repair, accepted side: re-encoding the decoded value is
    // bit-identical to the original payload (no clamping, no rounding).
    let re_encoded = decoded
        .to_execution_value()
        .expect("decoded value re-encodes");
    assert!(
        execution_values_identical(&oracle, &re_encoded),
        "accepted decode must round-trip bit-identically:\n oracle={oracle:?}\n re={re_encoded:?}"
    );

    // And the field value matches the evaluator's stored f32.
    assert_eq!(probability_field(&decoded), 0.3_f32 as f64);
}

#[test]
fn interior_non_trivial_value_accepted() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = prob_payload(0.7);
    let decoded = decode_adt_value(&exprs, &payload).expect("0.7 is admissible");
    assert_eq!(probability_field(&decoded), 0.7_f32 as f64);
}

// ── Boundary acceptance (closed [0, 1] invariant) ─────────────────────────

#[test]
fn boundary_equal_payloads_accepted() {
    let exprs = program_exprs(PROBABILITY_SRC);
    for v in [0.0_f64, 1.0_f64] {
        let decoded = decode_adt_value(&exprs, &prob_payload(v))
            .unwrap_or_else(|e| panic!("boundary value {v} must be accepted: {e}"));
        assert_eq!(probability_field(&decoded), v);
    }
}

#[test]
fn negative_zero_accepted_and_round_trips() {
    // -0.0 satisfies `>= 0.0` (IEEE: -0.0 == 0.0) and must round-trip with
    // its sign bit intact through the f32 field.
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = prob_payload(-0.0);
    let decoded = decode_adt_value(&exprs, &payload).expect("-0.0 satisfies >= 0.0");
    let field = probability_field(&decoded);
    assert_eq!(field, 0.0);
    assert!(
        field.is_sign_negative(),
        "the sign bit of -0.0 must survive decode (no normalization), got {field}"
    );
}

// ── Boundary-violating rejection (naming type + invariant + value) ────────

#[test]
fn just_past_upper_bound_rejected_naming_type_invariant_value() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let err = try_decode_adt_value(&exprs, &prob_payload(1.0000001))
        .expect_err("value just past 1.0 must be rejected");
    let msg = match &err {
        DecodeError::Invariant(msg) => msg.clone(),
        other => panic!("out-of-band value is an invariant violation, got {other:?}"),
    };
    assert!(msg.contains("Probability"), "names the type: {msg}");
    assert!(
        msg.contains("lte") && msg.contains("value"),
        "names the declared invariant: {msg}"
    );
    // The value appears in the message (the offending representation).
    assert!(
        msg.contains("Probability(") || msg.contains("1.000000"),
        "names the offending value: {msg}"
    );
}

#[test]
fn below_lower_bound_rejected() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let err =
        try_decode_adt_value(&exprs, &prob_payload(-0.5)).expect_err("-0.5 violates >= 0.0");
    assert!(matches!(err, DecodeError::Invariant(_)), "got {err:?}");
}

// ── NaN / Inf fail-closed (pinned; rejected pre-predicate) ────────────────

#[test]
fn nan_payload_rejected_fail_closed() {
    // PINNED: NaN must be rejected BEFORE predicate evaluation. The closed
    // [0, 1] predicate `p.value <= 1.0` is false on NaN, but an invariant
    // written as `not (p.value > 1.0)` would be true on NaN, so fail-closed
    // cannot depend on predicate structure. The representation pre-check is
    // what makes NaN rejection unconditional.
    let exprs = program_exprs(PROBABILITY_SRC);
    let err = try_decode_adt_value(&exprs, &prob_payload(f64::NAN))
        .expect_err("NaN representation must be rejected");
    match err {
        DecodeError::Invariant(msg) => {
            assert!(msg.contains("NaN"), "names NaN in the message: {msg}");
            assert!(
                msg.contains("non-finite") || msg.contains("Probability"),
                "names the representation rejection: {msg}"
            );
        }
        other => panic!("NaN is an invariant-class failure (pre-check), got {other:?}"),
    }
}

#[test]
fn positive_and_negative_inf_rejected_fail_closed() {
    let exprs = program_exprs(PROBABILITY_SRC);
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let err = try_decode_adt_value(&exprs, &prob_payload(v))
            .expect_err("infinite payload must be rejected");
        assert!(
            matches!(err, DecodeError::Invariant(_)),
            "Inf is rejected by the representation pre-check: {err:?}"
        );
    }
}

// ── Structural errors DISTINCT from invariant violations ──────────────────

#[test]
fn wrong_constructor_is_structural_not_invariant() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "NotAProbability".to_string(),
        fields: vec![ExecutionValue::Float64 { value: 0.3 }],
    };
    let err = try_decode_adt_value(&exprs, &payload).expect_err("unknown ctor rejected");
    assert!(
        matches!(err, DecodeError::Structural(_)),
        "unknown constructor is structural, got {err:?}"
    );
}

#[test]
fn missing_field_is_structural_not_invariant() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "Probability".to_string(),
        fields: vec![],
    };
    let err = try_decode_adt_value(&exprs, &payload).expect_err("missing field rejected");
    assert!(matches!(err, DecodeError::Structural(_)), "got {err:?}");
}

#[test]
fn extra_field_is_structural_not_invariant() {
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "Probability".to_string(),
        fields: vec![
            ExecutionValue::Float64 { value: 0.3 },
            ExecutionValue::Float64 { value: 0.4 },
        ],
    };
    let err = try_decode_adt_value(&exprs, &payload).expect_err("extra field rejected");
    assert!(matches!(err, DecodeError::Structural(_)), "got {err:?}");
}

#[test]
fn wrong_scalar_type_is_structural_not_invariant() {
    // An int payload into an f32 field is a structural type mismatch, NOT
    // an invariant violation (the wire keeps ints and floats distinct).
    let exprs = program_exprs(PROBABILITY_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "Probability".to_string(),
        fields: vec![ExecutionValue::Int64 { value: 0 }],
    };
    let err = try_decode_adt_value(&exprs, &payload).expect_err("int into float field rejected");
    assert!(matches!(err, DecodeError::Structural(_)), "got {err:?}");
}

#[test]
fn structural_and_invariant_message_prefixes_distinguish() {
    // The String-returning chokepoint distinguishes the two classes by a
    // stable message prefix, so a caller that only has the String can tell
    // a mis-shaped payload from an inadmissible one.
    let exprs = program_exprs(PROBABILITY_SRC);
    let structural = decode_adt_value(
        &exprs,
        &ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![ExecutionValue::Int64 { value: 0 }],
        },
    )
    .expect_err("structural");
    let invariant = decode_adt_value(&exprs, &prob_payload(2.0)).expect_err("invariant");
    assert!(
        structural.starts_with("structural decode error:"),
        "structural prefix: {structural}"
    );
    assert!(
        invariant.starts_with("decode rejected for opaque type"),
        "invariant prefix: {invariant}"
    );
    assert_ne!(structural, invariant, "the two classes are distinct messages");
}

// ── Nested record with opaque field ───────────────────────────────────────

#[test]
fn nested_valid_opaque_field_accepted_oracle_agreement() {
    let src = format!("{WRAPPER_SRC}\nresult = wrap(cast(0.7, f32), cast(2.0, f32))\n");
    let oracle = eval_result_value(&src);
    let exprs = program_exprs(WRAPPER_SRC);
    let decoded = decode_adt_value(&exprs, &oracle).expect("nested valid value accepted");

    let re_encoded = decoded.to_execution_value().expect("re-encodes");
    assert!(
        execution_values_identical(&oracle, &re_encoded),
        "nested accepted decode round-trips bit-identically"
    );
    let (ctor, fields) = decoded.as_adt().expect("Pair ADT");
    assert_eq!(ctor, "Pair");
    assert_eq!(fields.len(), 2);
    // Inner opaque field is the first field.
    assert_eq!(probability_field(&fields[0]), 0.7_f32 as f64);
}

#[test]
fn nested_inner_violating_rejected_naming_inner_type() {
    // The wrapper is fine; the inner Probability is out of band. The
    // violation must name the INNER opaque type, not the wrapper.
    let exprs = program_exprs(WRAPPER_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "Pair".to_string(),
        fields: vec![
            ExecutionValue::Adt {
                ctor: "Probability".to_string(),
                fields: vec![ExecutionValue::Float64 { value: 1.5 }],
            },
            ExecutionValue::Float64 { value: 2.0 },
        ],
    };
    let err = try_decode_adt_value(&exprs, &payload)
        .expect_err("inner out-of-band value must be rejected");
    match err {
        DecodeError::Invariant(msg) => {
            assert!(
                msg.contains("Probability"),
                "names the inner opaque type, not the wrapper: {msg}"
            );
        }
        other => panic!("inner violation is an invariant failure, got {other:?}"),
    }
}

#[test]
fn nested_inner_nan_rejected_fail_closed() {
    // A NaN in the inner opaque field is rejected pre-predicate, naming the
    // inner type.
    let exprs = program_exprs(WRAPPER_SRC);
    let payload = ExecutionValue::Adt {
        ctor: "Pair".to_string(),
        fields: vec![
            ExecutionValue::Adt {
                ctor: "Probability".to_string(),
                fields: vec![ExecutionValue::Float64 { value: f64::NAN }],
            },
            ExecutionValue::Float64 { value: 2.0 },
        ],
    };
    let err = try_decode_adt_value(&exprs, &payload).expect_err("inner NaN rejected");
    match err {
        DecodeError::Invariant(msg) => {
            assert!(msg.contains("NaN"), "names NaN: {msg}");
            assert!(msg.contains("Probability"), "names the inner type: {msg}");
        }
        other => panic!("inner NaN is an invariant-class failure, got {other:?}"),
    }
}

// ── Never-repair: a rejected decode yields no value ───────────────────────

#[test]
fn rejected_decode_yields_no_value() {
    // Exhaustive: every rejecting payload returns Err (no value), and the
    // typed API never returns Ok for a violating or mis-shaped payload.
    let exprs = program_exprs(PROBABILITY_SRC);
    let rejecting: Vec<ExecutionValue> = vec![
        prob_payload(1.0000001),
        prob_payload(-0.5),
        prob_payload(f64::NAN),
        prob_payload(f64::INFINITY),
        prob_payload(f64::NEG_INFINITY),
        ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![],
        },
        ExecutionValue::Adt {
            ctor: "Nope".to_string(),
            fields: vec![ExecutionValue::Float64 { value: 0.3 }],
        },
    ];
    for payload in &rejecting {
        assert!(
            try_decode_adt_value(&exprs, payload).is_err(),
            "rejecting payload must yield no value: {payload:?}"
        );
    }
}

/// A `Probability` wire payload carrying `value`.
fn prob_payload(value: f64) -> ExecutionValue {
    ExecutionValue::Adt {
        ctor: "Probability".to_string(),
        fields: vec![ExecutionValue::Float64 { value }],
    }
}
