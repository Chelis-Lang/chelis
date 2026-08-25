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
        (Int8 { value: x }, Int8 { value: y }) => x == y,
        (Int16 { value: x }, Int16 { value: y }) => x == y,
        (Int32 { value: x }, Int32 { value: y }) => x == y,
        (Int64 { value: x }, Int64 { value: y }) => x == y,
        (Float16 { value: x }, Float16 { value: y })
        | (Bfloat16 { value: x }, Bfloat16 { value: y }) => x.to_bits() == y.to_bits(),
        (Float32 { value: x }, Float32 { value: y }) => x.to_bits() == y.to_bits(),
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
                    .to_f64_lossy_vec()
                    .iter()
                    .zip(y.data.to_f64_lossy_vec())
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
    let err = try_decode_adt_value(&exprs, &prob_payload(-0.5)).expect_err("-0.5 violates >= 0.0");
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
        fields: vec![ExecutionValue::Float32 { value: 0.3 }],
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
            ExecutionValue::Float32 { value: 0.3 },
            ExecutionValue::Float32 { value: 0.4 },
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
    assert_ne!(
        structural, invariant,
        "the two classes are distinct messages"
    );
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
                fields: vec![ExecutionValue::Float32 { value: 1.5 }],
            },
            ExecutionValue::Float32 { value: 2.0 },
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
                fields: vec![ExecutionValue::Float32 { value: f32::NAN }],
            },
            ExecutionValue::Float32 { value: 2.0 },
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
            fields: vec![ExecutionValue::Float32 { value: 0.3 }],
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
        fields: vec![ExecutionValue::Float32 {
            value: value as f32,
        }],
    }
}

// ── In-module constant references in the invariant (CR-3) ─────────────────

/// An opaque type whose invariant references an in-module zero-argument
/// constant def, which RFC D-WF explicitly permits (predicate free vars may
/// be the binder OR in-module constants). The tolerance band `[-eps, 1+eps]`
/// is the documented float-aggregate idiom (RFC D-STARVE). The constant is
/// load-bearing: a payload of `1.0005` is admissible only because `eps`
/// widens the upper bound to `1.001`, and decode must resolve `eps` to its
/// value to evaluate the predicate at all.
const CONST_INVARIANT_SRC: &str = r#"
module Stats.Tol

def eps() -> f32 = 0.001

@opaque
@invariant(p) p.value >= 0.0 - eps && p.value <= 1.0 + eps
type Tol = | Tol { value: f32 }

def make(x: f32) -> Tol = Tol { value: x }
"#;

/// A `Tol` wire payload carrying `value`.
fn tol_payload(value: f64) -> ExecutionValue {
    ExecutionValue::Adt {
        ctor: "Tol".to_string(),
        fields: vec![ExecutionValue::Float32 {
            value: value as f32,
        }],
    }
}

#[test]
fn constant_referencing_invariant_accepts_valid_payload() {
    // CR-3 (red test): the invariant references `eps`, an in-module
    // constant. A payload of `0.5` is plainly inside the band. Before the
    // fix this WRONGLY REJECTS (the decode eval context has empty
    // top_level_defs, so `eps` is an unknown runtime name and the predicate
    // errors out, failing the decode).
    let exprs = program_exprs(CONST_INVARIANT_SRC);
    let decoded = decode_adt_value(&exprs, &tol_payload(0.5))
        .expect("a valid payload of a constant-referencing invariant must decode");
    let (ctor, fields) = decoded.as_adt().expect("decoded an ADT");
    assert_eq!(ctor, "Tol");
    assert_eq!(fields[0].as_f64(), Some(0.5_f32 as f64));
}

#[test]
fn constant_widens_bound_so_just_past_one_is_accepted() {
    // The constant is load-bearing: `1.0005` is outside the bare `[0, 1]`
    // band but inside `[-eps, 1 + eps]` = `[-0.001, 1.001]`, so it is only
    // admissible because `eps` resolves. This pins that the constant value
    // actually flows into the bound (not just that the predicate runs).
    let exprs = program_exprs(CONST_INVARIANT_SRC);
    let decoded = decode_adt_value(&exprs, &tol_payload(1.0005))
        .expect("1.0005 is inside the eps-widened upper bound");
    assert_eq!(
        decoded.as_adt().expect("ADT").0,
        "Tol",
        "value within the eps band decodes"
    );
}

#[test]
fn constant_referencing_invariant_rejects_violating_payload() {
    // CR-3 (negative parity): a payload outside even the eps-widened band
    // is still rejected as an invariant violation. The constant resolving
    // must not silently turn the predicate into a pass-through.
    let exprs = program_exprs(CONST_INVARIANT_SRC);
    let err = try_decode_adt_value(&exprs, &tol_payload(1.5))
        .expect_err("1.5 is outside [-eps, 1 + eps] and must be rejected");
    assert!(
        matches!(err, DecodeError::Invariant(_)),
        "out-of-band value is an invariant violation, got {err:?}"
    );

    let below = try_decode_adt_value(&exprs, &tol_payload(-0.5))
        .expect_err("-0.5 is below the lower band and must be rejected");
    assert!(matches!(below, DecodeError::Invariant(_)), "got {below:?}");
}

// ── Value-binding constant references in the invariant (CR2-6) ────────────

/// Parse a raw Deep (`.dp`) program into the exprs the chokepoint consumes.
/// The decode chokepoint accepts hand-authored Deep, so this exercises the
/// surface a Surf desugar cannot produce.
fn program_exprs_deep(source: &str) -> Vec<Expr> {
    chelis_deep::parser::parse_str(source).expect("deep parse")
}

/// A hand-authored Deep program declaring the in-module constant `eps` as a
/// BARE VALUE-BINDING def -- `(def {} eps (lit ...))`, where the body is the
/// value directly rather than a `(fn {} (params {}) <inner>)` wrapper. This
/// form is legal Deep (`validate --deep` accepts it) but is never produced
/// by the Surf desugarer, which always wraps a def body in `fn`. CR2-6: the
/// CR-3 constant collector only matched the fn-wrapped form, so a `.dp`
/// invariant referencing this value-binding constant could not resolve it.
///
/// The invariant is `p.value >= 0.0 - eps && p.value <= 1.0 + eps`, the same
/// eps-widened band the fn-form test uses, so `eps` is load-bearing.
const TOL_DEEP_BARE_VALUE_CONST: &str = r#"
(module {}
  stats.tol
  (def {} eps (lit {type: (t-prim {} f32)} 0.001))
  (deftype {opaque: true,
            invariant_amenability: "linear",
            invariant: (fn {}
                          (params {} p)
                          (app {}
                            (var {} and)
                            (app {}
                              (var {} gte)
                              (access {} (var {} p) value)
                              (app {}
                                (var {} sub)
                                (lit {type: (t-prim {} f32)} 0.0)
                                (var {} eps)))
                            (app {}
                              (var {} lte)
                              (access {} (var {} p) value)
                              (app {}
                                (var {} add)
                                (lit {type: (t-prim {} f32)} 1.0)
                                (var {} eps)))))}
    Tol
    ()
    (variant {} Tol (field {} value (t-prim {} f32)))))
"#;

#[test]
fn value_binding_constant_invariant_accepts_valid_payload() {
    // CR2-6 (red test): `eps` is declared as a bare value-binding def, the
    // form the CR-3 fn-only collector skipped. Before the CR2-6 fix this
    // WRONGLY REJECTS (the constant is unresolved, so the predicate errors
    // with `unknown runtime name eps`).
    let exprs = program_exprs_deep(TOL_DEEP_BARE_VALUE_CONST);
    let decoded = decode_adt_value(&exprs, &tol_payload(0.5))
        .expect("a valid payload of a value-binding-constant invariant must decode");
    let (ctor, fields) = decoded.as_adt().expect("decoded an ADT");
    assert_eq!(ctor, "Tol");
    assert_eq!(fields[0].as_f64(), Some(0.5_f32 as f64));
}

#[test]
fn value_binding_constant_widens_bound_so_just_past_one_is_accepted() {
    // The value-binding constant is load-bearing: `1.0005` is outside the
    // bare `[0, 1]` band but inside `[-eps, 1 + eps]`, so it is admissible
    // only because `eps` resolves through the value-binding form.
    let exprs = program_exprs_deep(TOL_DEEP_BARE_VALUE_CONST);
    let decoded = decode_adt_value(&exprs, &tol_payload(1.0005))
        .expect("1.0005 is inside the eps-widened upper bound");
    assert_eq!(decoded.as_adt().expect("ADT").0, "Tol");
}

#[test]
fn value_binding_constant_invariant_rejects_violating_payload() {
    // CR2-6 (negative parity): a payload outside the eps-widened band is
    // still rejected. Resolving the value-binding constant must not turn the
    // predicate into a pass-through.
    let exprs = program_exprs_deep(TOL_DEEP_BARE_VALUE_CONST);
    let err = try_decode_adt_value(&exprs, &tol_payload(1.5))
        .expect_err("1.5 is outside [-eps, 1 + eps] and must be rejected");
    assert!(
        matches!(err, DecodeError::Invariant(_)),
        "out-of-band value is an invariant violation, got {err:?}"
    );

    let below = try_decode_adt_value(&exprs, &tol_payload(-0.5))
        .expect_err("-0.5 is below the lower band and must be rejected");
    assert!(matches!(below, DecodeError::Invariant(_)), "got {below:?}");
}

/// Same Tol program as `TOL_DEEP_BARE_VALUE_CONST`, but with an EXTRA
/// non-constant value-binding `(def {} a (var {} foo))` referencing an
/// undefined name. Review-3: the constant collector must NOT register `a`
/// (it is not a constant), so this garbage binding cannot pollute the decode
/// table or affect resolution of the genuine `eps` constant.
const TOL_DEEP_WITH_NON_CONSTANT_BINDING: &str = r#"
(module {}
  stats.tol
  (def {} a (var {} foo))
  (def {} eps (lit {type: (t-prim {} f32)} 0.001))
  (deftype {opaque: true,
            invariant_amenability: "linear",
            invariant: (fn {}
                          (params {} p)
                          (app {}
                            (var {} and)
                            (app {}
                              (var {} gte)
                              (access {} (var {} p) value)
                              (app {}
                                (var {} sub)
                                (lit {type: (t-prim {} f32)} 0.0)
                                (var {} eps)))
                            (app {}
                              (var {} lte)
                              (access {} (var {} p) value)
                              (app {}
                                (var {} add)
                                (lit {type: (t-prim {} f32)} 1.0)
                                (var {} eps)))))}
    Tol
    ()
    (variant {} Tol (field {} value (t-prim {} f32)))))
"#;

// ── Malformed invariant metadata fails CLOSED at decode (FINDING 2) ───────

/// A hand-authored Deep `deftype` that DECLARES an `invariant` whose
/// metadata is MALFORMED: the value is a bare literal, not the well-formed
/// `(fn {} (params {} <binder>) <body>)` shape the predicate parser expects.
///
/// D-WF rejects this at declaration time, so it is unreachable through a
/// chelis-compiled module. But `decode_adt_value` is the contract chokepoint
/// for the next external/hand-built codec, so it must fail CLOSED: a
/// structurally-valid payload for this constructor must be REJECTED, not
/// materialized with zero invariant check. The earlier collector skipped a
/// malformed invariant, leaving no table entry, so the constructor looked
/// invariant-free and a structurally-valid payload decoded with no check
/// (a fail-OPEN). The fix records a `Malformed` entry so revalidation
/// rejects the value.
///
/// The type is named `Probability` with a single `value: f32` field so the
/// `prob_payload` helper produces a structurally-valid payload for it.
const PROBABILITY_DEEP_MALFORMED_INVARIANT: &str = r#"
(module {}
  stats.prob
  (deftype {opaque: true,
            invariant_amenability: "linear",
            invariant: (lit {type: (t-prim {} f32)} 1.0)}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32)))))
"#;

/// Companion to the malformed case: the SAME `Probability` type declared
/// with a WELL-FORMED invariant `p.value >= 0.0 && p.value <= 1.0` in
/// hand-authored Deep, to confirm the well-formed path still decodes/
/// validates normally (and to show the malformed rejection is about the
/// metadata shape, not about Deep authoring per se).
const PROBABILITY_DEEP_WELL_FORMED_INVARIANT: &str = r#"
(module {}
  stats.prob
  (deftype {opaque: true,
            invariant_amenability: "linear",
            invariant: (fn {}
                          (params {} p)
                          (app {}
                            (var {} and)
                            (app {}
                              (var {} gte)
                              (access {} (var {} p) value)
                              (lit {type: (t-prim {} f32)} 0.0))
                            (app {}
                              (var {} lte)
                              (access {} (var {} p) value)
                              (lit {type: (t-prim {} f32)} 1.0))))}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32)))))
"#;

#[test]
fn malformed_invariant_metadata_fails_closed_at_decode() {
    // FINDING 2 (red test): the deftype declares an `invariant` whose
    // metadata is a bare literal, NOT the `(fn {} (params {} <binder>)
    // <body>)` shape. A structurally-valid payload (`value: 0.3`, which a
    // well-formed `[0, 1]` invariant would happily accept) must STILL be
    // rejected -- the predicate cannot be evaluated, so the value cannot be
    // safely materialized (spec/10 §4.1, fail-closed).
    //
    // Before the fix this WRONGLY returns Ok: the collector skipped the
    // malformed metadata, leaving `Probability` with no table entry, so
    // `revalidate_adt_value` treated it as invariant-free and decoded the
    // payload with zero check (a fail-OPEN soundness hole).
    let exprs = program_exprs_deep(PROBABILITY_DEEP_MALFORMED_INVARIANT);
    let err = try_decode_adt_value(&exprs, &prob_payload(0.3)).expect_err(
        "a malformed declared invariant must fail closed: a structurally-valid \
         payload cannot be safely materialized without a usable predicate",
    );
    match err {
        DecodeError::Invariant(msg) => {
            assert!(
                msg.contains("malformed"),
                "names the malformed-metadata rejection: {msg}"
            );
            assert!(
                msg.contains("Probability"),
                "names the declaring opaque type: {msg}"
            );
        }
        other => panic!(
            "a malformed declared invariant is an invariant-class (fail-closed) \
             decode failure, not a structural one, got {other:?}"
        ),
    }
}

#[test]
fn malformed_invariant_rejects_every_payload_no_pass_through() {
    // Parity: the malformed invariant must reject across the board (no value
    // can be materialized), not just one probe. This pins that the fix is a
    // categorical fail-closed, not a value-dependent fluke.
    let exprs = program_exprs_deep(PROBABILITY_DEEP_MALFORMED_INVARIANT);
    for v in [0.0_f64, 0.5_f64, 1.0_f64, -0.5_f64, 1.5_f64] {
        assert!(
            try_decode_adt_value(&exprs, &prob_payload(v)).is_err(),
            "malformed invariant must reject payload {v} (fail-closed, never a \
             pass-through decode)"
        );
    }
}

#[test]
fn well_formed_deep_invariant_still_decodes_after_fix() {
    // Positive companion: the SAME hand-authored-Deep `Probability` with a
    // WELL-FORMED invariant still accepts an in-band value and rejects an
    // out-of-band one. The Malformed fix must not regress the everyday
    // predicate path.
    let exprs = program_exprs_deep(PROBABILITY_DEEP_WELL_FORMED_INVARIANT);

    let decoded = decode_adt_value(&exprs, &prob_payload(0.3))
        .expect("a well-formed invariant accepts an in-band value");
    let (ctor, fields) = decoded.as_adt().expect("decoded an ADT");
    assert_eq!(ctor, "Probability");
    assert_eq!(fields[0].as_f64(), Some(0.3_f32 as f64));

    let err = try_decode_adt_value(&exprs, &prob_payload(1.5))
        .expect_err("a well-formed invariant still rejects an out-of-band value");
    assert!(matches!(err, DecodeError::Invariant(_)), "got {err:?}");
}

#[test]
fn non_constant_value_binding_does_not_break_genuine_constant_decode() {
    // Review-3: a `(def a (var foo))` non-constant binding sits next to the
    // genuine `eps` literal constant. The genuine constant still resolves so
    // a valid payload decodes; the garbage binding is not registered and
    // does not interfere. (The earlier catch-all would have registered `a`
    // mapped to an unresolved `(var foo)`.)
    let exprs = program_exprs_deep(TOL_DEEP_WITH_NON_CONSTANT_BINDING);
    let decoded = decode_adt_value(&exprs, &tol_payload(0.5))
        .expect("genuine eps constant still resolves; valid payload decodes");
    assert_eq!(decoded.as_adt().expect("ADT").0, "Tol");

    // And a violating payload is still rejected (no pass-through).
    let err = try_decode_adt_value(&exprs, &tol_payload(1.5))
        .expect_err("1.5 is outside the eps band and must be rejected");
    assert!(matches!(err, DecodeError::Invariant(_)), "got {err:?}");
}

// ── chelis#1305: unreadable deftype name must fail closed ─────────────────

/// An invariant-declaring `deftype` whose type-name child is NOT a readable
/// symbol (a `lit` node stands where the name belongs). The invariant
/// metadata itself is the well-formed `[0, 1]` predicate — the unreadable
/// declarer, not the predicate shape, is what must force the rejection.
const DEEP_UNREADABLE_TYPE_NAME_WITH_INVARIANT: &str = r#"
(module {}
  stats.prob
  (deftype {opaque: true,
            invariant: (fn {}
                          (params {} p)
                          (app {}
                            (var {} and)
                            (app {}
                              (var {} gte)
                              (access {} (var {} p) value)
                              (lit {type: (t-prim {} f32)} 0.0))
                            (app {}
                              (var {} lte)
                              (access {} (var {} p) value)
                              (lit {type: (t-prim {} f32)} 1.0))))}
    (lit {type: (t-prim {} f32)} 1.0)
    ()
    (variant {} Probability (field {} value (t-prim {} f32)))))
"#;

/// Negative control: the same unreadable type-name child WITHOUT an
/// `invariant` metadata entry. No invariant is declared, so there is
/// nothing to fail closed over and the payload decodes cleanly.
const DEEP_UNREADABLE_TYPE_NAME_NO_INVARIANT: &str = r#"
(module {}
  stats.prob
  (deftype {opaque: true}
    (lit {type: (t-prim {} f32)} 1.0)
    ()
    (variant {} Probability (field {} value (t-prim {} f32)))))
"#;

/// A readable, invariant-declaring `deftype` whose VARIANT ctor-name child
/// is unreadable (a `lit` node stands where the ctor name belongs). The
/// invariant table cannot key that variant; the backstop is `decode_adt`'s
/// unknown-constructor guard.
const DEEP_UNREADABLE_CTOR_NAME: &str = r#"
(module {}
  stats.prob
  (deftype {opaque: true,
            invariant: (fn {}
                          (params {} p)
                          (app {}
                            (var {} gte)
                            (access {} (var {} p) value)
                            (lit {type: (t-prim {} f32)} 0.0)))}
    Probability
    ()
    (variant {} (lit {type: (t-prim {} f32)} 1.0) (field {} value (t-prim {} f32)))))
"#;

#[test]
fn invariant_declaring_deftype_with_unreadable_name_fails_closed() {
    // chelis#1305 (red test): before the fix, the unreadable name child
    // made `collect_type_invariants` skip the WHOLE deftype, so the
    // constructor looked invariant-free and this structurally-valid payload
    // decoded with zero invariant check — the same fail-open the
    // malformed-metadata door already closed, through a different door.
    let exprs = program_exprs_deep(DEEP_UNREADABLE_TYPE_NAME_WITH_INVARIANT);
    let err = try_decode_adt_value(&exprs, &prob_payload(0.3)).expect_err(
        "an invariant-declaring deftype with an unreadable name child must \
         fail closed: the declaration cannot be attributed, so no payload of \
         its constructors can be safely materialized",
    );
    match err {
        DecodeError::Invariant(msg) => {
            assert!(
                msg.contains("malformed"),
                "names the malformed-declaration rejection: {msg}"
            );
            assert!(
                msg.contains("unreadable deftype name"),
                "carries the placeholder for the unreadable declarer: {msg}"
            );
        }
        other => panic!(
            "the unreadable-name fail-closed is an invariant-class decode \
             failure, not a structural one, got {other:?}"
        ),
    }
}

#[test]
fn unreadable_name_without_invariant_still_decodes() {
    // chelis#1305 negative control: with no `invariant` entry declared there
    // is nothing to fail closed over. The unreadable name alone must not
    // reject a structurally-valid payload.
    let exprs = program_exprs_deep(DEEP_UNREADABLE_TYPE_NAME_NO_INVARIANT);
    let decoded = decode_adt_value(&exprs, &prob_payload(0.3))
        .expect("no declared invariant: the payload decodes on structure alone");
    assert_eq!(decoded.as_adt().expect("ADT").0, "Probability");
}

#[test]
fn skipped_variant_ctor_rejects_at_structural_decode() {
    // chelis#1305: a variant whose ctor-name child is unreadable cannot be
    // keyed by the invariant table (there is no key to insert under). That
    // skip is not a fail-open only because the same unreadable name is also
    // absent from the field-type table, so the unknown-constructor guard in
    // `decode_adt` rejects every payload at the STRUCTURAL layer. This test
    // pins the backstop the in-code comment cites.
    let exprs = program_exprs_deep(DEEP_UNREADABLE_CTOR_NAME);
    let err = try_decode_adt_value(&exprs, &prob_payload(0.3))
        .expect_err("no readable constructor is declared, so no payload resolves");
    match err {
        DecodeError::Structural(msg) => {
            assert!(
                msg.contains("unknown constructor"),
                "the unknown-constructor guard is the backstop: {msg}"
            );
        }
        other => panic!(
            "an unresolvable constructor is a structural decode failure, got {other:?}"
        ),
    }
}
