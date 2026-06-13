//! Public decode chokepoint for opaque-type invariant revalidation
//! (RFC `opaque_invariants_rfc.md` D-DECODE, spec/10 "Invariant
//! revalidation at decode boundaries").
//!
//! # Status: experimental
//!
//! As of V1 **no production codec consumes this entry point**. The survey
//! (`spec/design/opaque_invariants_survey.md` §7) established — and W5
//! re-confirmed by direct inspection of [`crate::schema`] — that no
//! external-payload path materializes a typed ADT value today:
//! `EvalRequest.bindings` is tensors-only, `ExecutionValue::Adt` is an
//! output-only shape produced by the evaluator from program text, and the
//! cache / reef / `.chb` envelopes serialize compiler state, not domain
//! values. This module therefore ships as the **contract point** that any
//! future codec materializing a value of an invariant-carrying opaque type
//! (tide ADT ingestion, a Python ADT bridge, a wire deserializer) MUST
//! call. Its only caller in V1 is the conformance suite
//! (`tests/invariant_decode.rs`).
//!
//! # The normative rule
//!
//! [`decode_adt_value`] performs two passes, in order:
//!
//! 1. **Structural validation** — the payload's constructor must exist in
//!    the program's declared types, and its fields must match the declared
//!    representation in arity, order, and field type (scalar vs tensor vs
//!    nested ADT). A structural mismatch is a [`DecodeError::Structural`],
//!    *distinct from* an invariant violation: structural means "this
//!    payload is not even shaped like the type."
//! 2. **Invariant revalidation** — [`crate::runtime::revalidate_adt_value`]
//!    runs the declared predicate through the interpreter's own evaluator
//!    (the crate does **not** depend on `chelis-prove`). NaN and Inf in any
//!    numeric representation field are rejected *before* predicate
//!    evaluation (RFC H1 fail-closed). A violation is a
//!    [`DecodeError::Invariant`].
//!
//! Decode of a violating payload is a **failure, never a repair** (no
//! clamping, no normalization). A rejected decode returns `Err` and yields
//! no value; an accepted decode returns a [`RuntimeValue`] that is
//! value-identical to the structurally-valid input.

use std::collections::HashMap;

use chelis_deep::ast::Expr;
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_types::types::Prim;

use crate::runtime::{
    DecodeField, DecodeFieldType, InvariantPredicate, RuntimeTensorValue, RuntimeValue,
    collect_adt_ctor_fields, collect_ctor_field_types, collect_type_invariants,
    revalidate_adt_value,
};
use crate::schema::ExecutionValue;

/// A decode-boundary failure. The two arms are deliberately distinct: a
/// structural error means the payload is mis-shaped (wrong constructor,
/// wrong field count, wrong field type), while an invariant error means a
/// correctly-shaped payload carries an inadmissible value (out-of-band, or
/// NaN/Inf in a representation field).
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeError {
    /// The payload does not match the declared representation of the type.
    Structural(String),
    /// The payload is structurally valid but violates the declared
    /// invariant (or fails the NaN/Inf representation-sanity pre-check).
    Invariant(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Structural(msg) => write!(f, "structural decode error: {msg}"),
            DecodeError::Invariant(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// The decode chokepoint (RFC D-DECODE).
///
/// `program_exprs` is the desugared Deep program (or library) declaring the
/// opaque types: it supplies the constructor field tables and the invariant
/// predicates. `payload` is the external value being materialized.
///
/// Returns the decoded [`RuntimeValue`] on success. On failure the `Err`
/// message distinguishes the two failure classes by prefix: a structural
/// mismatch begins `structural decode error:`, an invariant violation
/// begins `decode rejected for opaque type`. Callers that need to branch on
/// the class can use [`try_decode_adt_value`], which returns the typed
/// [`DecodeError`].
///
/// This is the experimental contract point future codecs MUST call before
/// handing a value of an invariant-carrying opaque type to in-process code.
/// See the module docs for the V1 reality (no production caller yet).
pub fn decode_adt_value(
    program_exprs: &[Expr],
    payload: &ExecutionValue,
) -> Result<RuntimeValue, String> {
    try_decode_adt_value(program_exprs, payload).map_err(|err| err.to_string())
}

/// Typed-error variant of [`decode_adt_value`]: returns the [`DecodeError`]
/// enum so a caller can branch on structural-vs-invariant without parsing
/// the message string.
pub fn try_decode_adt_value(
    program_exprs: &[Expr],
    payload: &ExecutionValue,
) -> Result<RuntimeValue, DecodeError> {
    let field_types = collect_ctor_field_types(program_exprs);
    let adt_fields = collect_adt_ctor_fields(program_exprs);
    let invariants = collect_type_invariants(program_exprs);
    decode_with_tables(payload, &field_types, &adt_fields, &invariants)
}

/// Decode against pre-built tables. Crate-internal so a caller that already
/// holds the field-type / invariant tables for a program (a real codec
/// running many decodes) does not rebuild them per call. The table types
/// (`DecodeField`, `InvariantPredicate`) are crate-private in V1; when a
/// real codec lands they are promoted to the public surface alongside it.
pub(crate) fn decode_with_tables(
    payload: &ExecutionValue,
    field_types: &HashMap<String, Vec<DecodeField>>,
    adt_fields: &HashMap<String, Vec<String>>,
    invariants: &HashMap<String, InvariantPredicate>,
) -> Result<RuntimeValue, DecodeError> {
    // Pass 1: structural conversion (constructor + field arity/order/type).
    let value = structural_decode(payload, field_types)?;
    // Pass 2: invariant revalidation (NaN/Inf pre-check then predicate).
    revalidate_adt_value(&value, invariants, adt_fields)
        .map_err(|violation| DecodeError::Invariant(violation.to_string()))?;
    Ok(value)
}

/// Convert an [`ExecutionValue`] payload to a [`RuntimeValue`], enforcing
/// that any ADT node matches its declared representation. Non-ADT payloads
/// convert structurally (used for nested scalar/tensor fields); a top-level
/// non-ADT payload converts but carries no opaque invariant.
fn structural_decode(
    payload: &ExecutionValue,
    field_types: &HashMap<String, Vec<DecodeField>>,
) -> Result<RuntimeValue, DecodeError> {
    match payload {
        ExecutionValue::Adt { ctor, fields } => decode_adt(ctor, fields, field_types),
        ExecutionValue::Int64 { value } => Ok(RuntimeValue::int64(*value)),
        ExecutionValue::Float64 { value } => Ok(RuntimeValue::float64(*value)),
        ExecutionValue::Bool { value } => Ok(RuntimeValue::Bool(*value)),
        ExecutionValue::String { value } => Ok(RuntimeValue::String(value.clone())),
        ExecutionValue::Unit => Ok(RuntimeValue::Unit),
        ExecutionValue::Tensor { value } => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(value.shape.clone(), value.data.clone()),
            // No declared element precision at the top level; default to
            // f64 (the wire tensor dtype). Field-position decode overrides
            // this with the declared field precision.
            precision: Prim::F64,
        })),
        ExecutionValue::List { value } => {
            let items = value
                .iter()
                .map(|item| structural_decode(item, field_types))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(RuntimeValue::List(items))
        }
        ExecutionValue::Tuple { value } => {
            let items = value
                .iter()
                .map(|item| structural_decode(item, field_types))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(RuntimeValue::Tuple(items))
        }
        ExecutionValue::Dict { entries } => {
            let decoded = entries
                .iter()
                .map(|entry| {
                    Ok((
                        structural_decode(&entry.key, field_types)?,
                        structural_decode(&entry.value, field_types)?,
                    ))
                })
                .collect::<Result<Vec<_>, DecodeError>>()?;
            Ok(RuntimeValue::Dict(decoded))
        }
    }
}

fn decode_adt(
    ctor: &str,
    payload_fields: &[ExecutionValue],
    field_types: &HashMap<String, Vec<DecodeField>>,
) -> Result<RuntimeValue, DecodeError> {
    let declared = field_types.get(ctor).ok_or_else(|| {
        DecodeError::Structural(format!(
            "unknown constructor `{ctor}` (not declared by any in-scope deftype)"
        ))
    })?;

    if payload_fields.len() != declared.len() {
        return Err(DecodeError::Structural(format!(
            "constructor `{ctor}` expects {} field(s), payload supplied {}",
            declared.len(),
            payload_fields.len()
        )));
    }

    let mut fields = Vec::with_capacity(declared.len());
    let mut field_names = Vec::with_capacity(declared.len());
    for (spec, payload_field) in declared.iter().zip(payload_fields.iter()) {
        let decoded = decode_field(ctor, spec, payload_field, field_types)?;
        fields.push(decoded);
        field_names.push(spec.name.clone());
    }

    Ok(RuntimeValue::Adt {
        ctor: ctor.to_string(),
        fields,
        field_names: Some(field_names),
    })
}

fn decode_field(
    ctor: &str,
    spec: &DecodeField,
    payload: &ExecutionValue,
    field_types: &HashMap<String, Vec<DecodeField>>,
) -> Result<RuntimeValue, DecodeError> {
    match &spec.ty {
        DecodeFieldType::Prim(prim) => decode_scalar_field(ctor, &spec.name, *prim, payload),
        DecodeFieldType::Tensor(prim) => decode_tensor_field(ctor, &spec.name, *prim, payload),
        DecodeFieldType::Adt(type_name) => {
            let ExecutionValue::Adt {
                ctor: inner_ctor,
                fields,
            } = payload
            else {
                return Err(DecodeError::Structural(format!(
                    "constructor `{ctor}` field `{}` expects a nested ADT value of type \
                     `{type_name}`, payload supplied {}",
                    spec.name,
                    describe_payload(payload)
                )));
            };
            decode_adt(inner_ctor, fields, field_types)
        }
    }
}

fn decode_scalar_field(
    ctor: &str,
    field: &str,
    prim: Prim,
    payload: &ExecutionValue,
) -> Result<RuntimeValue, DecodeError> {
    let mismatch = |got: &str| {
        DecodeError::Structural(format!(
            "constructor `{ctor}` field `{field}` expects scalar `{}`, payload supplied {got}",
            prim.name()
        ))
    };

    if prim == Prim::Bool {
        return match payload {
            ExecutionValue::Bool { value } => Ok(RuntimeValue::Bool(*value)),
            other => Err(mismatch(&describe_payload(other))),
        };
    }

    if prim.is_integer() {
        return match payload {
            ExecutionValue::Int64 { value } => {
                RuntimeValue::scalar_like_int(prim, *value).map_err(DecodeError::Structural)
            }
            other => Err(mismatch(&describe_payload(other))),
        };
    }

    // Float-typed field. Accept a wire float; reject ints (the wire keeps
    // ints and floats distinct, so a float field fed an int is a real
    // type mismatch, not a silent widening).
    match payload {
        ExecutionValue::Float64 { value } => {
            RuntimeValue::scalar_like_float(prim, *value).map_err(DecodeError::Structural)
        }
        other => Err(mismatch(&describe_payload(other))),
    }
}

fn decode_tensor_field(
    ctor: &str,
    field: &str,
    prim: Prim,
    payload: &ExecutionValue,
) -> Result<RuntimeValue, DecodeError> {
    match payload {
        ExecutionValue::Tensor { value } => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(value.shape.clone(), value.data.clone()),
            precision: prim,
        })),
        other => Err(DecodeError::Structural(format!(
            "constructor `{ctor}` field `{field}` expects a tensor of `{}`, payload supplied {}",
            prim.name(),
            describe_payload(other)
        ))),
    }
}

fn describe_payload(payload: &ExecutionValue) -> String {
    match payload {
        ExecutionValue::Tensor { .. } => "a tensor".to_string(),
        ExecutionValue::Int64 { .. } => "an int".to_string(),
        ExecutionValue::Float64 { .. } => "a float".to_string(),
        ExecutionValue::Bool { .. } => "a bool".to_string(),
        ExecutionValue::String { .. } => "a string".to_string(),
        ExecutionValue::List { .. } => "a list".to_string(),
        ExecutionValue::Dict { .. } => "a dict".to_string(),
        ExecutionValue::Tuple { .. } => "a tuple".to_string(),
        ExecutionValue::Adt { ctor, .. } => format!("an ADT `{ctor}`"),
        ExecutionValue::Unit => "unit".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::TensorValue;

    const PROBABILITY_SRC: &str = r#"
module Stats.Prob

@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability = | Probability { value: f32 }
"#;

    fn program_exprs(source: &str) -> Vec<Expr> {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        chelis_surf::desugar::desugar_program(&decls)
    }

    fn prob_payload(value: f64) -> ExecutionValue {
        ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![ExecutionValue::Float64 { value }],
        }
    }

    #[test]
    fn decodes_interior_value() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let value = decode_adt_value(&exprs, &prob_payload(0.3)).expect("0.3 is admissible");
        let (ctor, fields) = value.as_adt().expect("decoded an ADT");
        assert_eq!(ctor, "Probability");
        // f32 round-trip: 0.3 stored at f32 reads back as f32-of-0.3.
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].as_f64(), Some(0.3_f32 as f64));
    }

    #[test]
    fn unknown_constructor_is_structural_error() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let payload = ExecutionValue::Adt {
            ctor: "Nope".to_string(),
            fields: vec![ExecutionValue::Float64 { value: 0.3 }],
        };
        let err = try_decode_adt_value(&exprs, &payload).expect_err("unknown ctor rejected");
        assert!(
            matches!(err, DecodeError::Structural(_)),
            "unknown ctor is structural, got {err:?}"
        );
    }

    #[test]
    fn wrong_arity_is_structural_error() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let payload = ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![
                ExecutionValue::Float64 { value: 0.3 },
                ExecutionValue::Float64 { value: 0.4 },
            ],
        };
        let err = try_decode_adt_value(&exprs, &payload).expect_err("wrong arity rejected");
        assert!(matches!(err, DecodeError::Structural(_)), "got {err:?}");
    }

    #[test]
    fn wrong_scalar_type_is_structural_error() {
        // A float field fed an int payload is a structural mismatch, NOT an
        // invariant violation — the wire keeps ints and floats distinct.
        let exprs = program_exprs(PROBABILITY_SRC);
        let payload = ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![ExecutionValue::Int64 { value: 0 }],
        };
        let err = try_decode_adt_value(&exprs, &payload).expect_err("int into float field");
        assert!(matches!(err, DecodeError::Structural(_)), "got {err:?}");
    }

    #[test]
    fn out_of_band_value_is_invariant_error() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let err = try_decode_adt_value(&exprs, &prob_payload(1.5))
            .expect_err("1.5 violates [0, 1] invariant");
        assert!(
            matches!(err, DecodeError::Invariant(_)),
            "out-of-band value is an invariant violation, got {err:?}"
        );
    }

    #[test]
    fn nan_is_invariant_error_distinct_from_structural() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let err = try_decode_adt_value(&exprs, &prob_payload(f64::NAN))
            .expect_err("NaN rejected pre-predicate");
        match err {
            DecodeError::Invariant(msg) => assert!(msg.contains("NaN"), "names NaN: {msg}"),
            other => panic!("NaN is an invariant-class failure, got {other:?}"),
        }
    }

    #[test]
    fn structural_and_invariant_messages_are_distinguishable() {
        let exprs = program_exprs(PROBABILITY_SRC);
        let structural = decode_adt_value(
            &exprs,
            &ExecutionValue::Adt {
                ctor: "Probability".to_string(),
                fields: vec![],
            },
        )
        .expect_err("zero fields rejected");
        let invariant = decode_adt_value(&exprs, &prob_payload(2.0)).expect_err("2.0 rejected");
        assert!(
            structural.starts_with("structural decode error:"),
            "structural message prefix: {structural}"
        );
        assert!(
            invariant.starts_with("decode rejected for opaque type"),
            "invariant message prefix: {invariant}"
        );
    }

    #[test]
    fn tensor_field_decodes() {
        // A tensor representation field decodes a wire tensor at the
        // declared element precision.
        let src = r#"
module M

type Holder = | Holder { weights: tensor[3, f32] }
"#;
        let exprs = program_exprs(src);
        let payload = ExecutionValue::Adt {
            ctor: "Holder".to_string(),
            fields: vec![ExecutionValue::Tensor {
                value: TensorValue {
                    shape: vec![3],
                    data: vec![0.1, 0.2, 0.7],
                },
            }],
        };
        let value = decode_adt_value(&exprs, &payload).expect("tensor field decodes");
        let (ctor, fields) = value.as_adt().expect("decoded ADT");
        assert_eq!(ctor, "Holder");
        match &fields[0] {
            RuntimeValue::Tensor(t) => {
                assert_eq!(t.value.shape, vec![3]);
                assert_eq!(t.value.data, vec![0.1, 0.2, 0.7]);
            }
            other => panic!("expected tensor field, got {other:?}"),
        }
    }
}
