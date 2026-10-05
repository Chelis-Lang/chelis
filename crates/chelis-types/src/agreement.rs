//! Shared cross-lane numerical agreement policy (chelis#732 Phase 3).
//!
//! Byte equality is always sufficient. A byte difference is accepted only
//! when both strings are canonical [05-OBS] renderings of finite float
//! values and the exact operation identity has a row in [`OP_TOLERANCES`].
//! Every absent or misspelled operation therefore defaults to exactness; the
//! table can never become a blanket float fallback like the chelis#687
//! harnesses it replaces.

use std::error::Error;
use std::fmt;

use crate::observation::{ElementRef, format_element};
use crate::types::Prim;

/// Closed operation identity used by the [05-OBS-3] comparator.
///
/// Callers cannot manufacture a tolerant identity from a string. An IR
/// consumer maps its actual operation into this enum with an exhaustive
/// match, so adding an IR operation cannot silently inherit a tolerance.
/// Every operation is exact: [04-NUM-2] and [04-NUM-8] arithmetic is exact
/// by construction and [05-OP-46] makes the transcendentals and `sqrt`
/// correctly rounded, so no operation carries an identity of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementOp {
    /// Every operation; [05-OBS-3]'s table grants no row.
    Exact,
}

impl AgreementOp {
    /// Canonical operation spelling used by the spec table and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
        }
    }
}

/// One explicit [05-OBS-3] tolerance grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpTolerance {
    /// Closed operation identity.
    pub op: AgreementOp,
    /// Maximum difference at [04-NUM-8]'s declared arithmetic width.
    pub max_ulps: u64,
}

/// The authoritative machine form of spec/05's [05-OBS-3] tolerance table.
///
/// The table is empty: every operation has a zero-ULP bound, ordinary
/// arithmetic, comparisons, reductions, compound builtins, and the
/// correctly rounded transcendentals alike.
pub const OP_TOLERANCES: &[OpTolerance] = &[];

/// Return the operation's maximum ULP difference. Absence means exactness.
pub fn tolerance_for(op: AgreementOp) -> u64 {
    OP_TOLERANCES
        .iter()
        .find(|row| row.op == op)
        .map_or(0, |row| row.max_ulps)
}

/// Successful comparison disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementOutcome {
    /// The complete rendered elements were byte-identical.
    ByteExact,
    /// Canonical renderings differed because the values differed within the
    /// exact operation's authored tolerance row.
    WithinTolerance { distance_ulps: u64, max_ulps: u64 },
}

/// Whether both compared lanes compute this operation at [04-NUM-8]'s
/// declared arithmetic width.
///
/// A value tolerance describes implementation variance *at one width*. It
/// must never launder a structural width violation. Callers therefore have
/// to state this precondition explicitly. The evaluator is currently
/// [`Nonconforming`](Self::Nonconforming) for float arithmetic under
/// chelis#897; byte-identical results remain acceptable, but a mismatch may
/// not consult [`OP_TOLERANCES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticWidthStatus {
    /// The stored values are themselves at [04-NUM-8]'s arithmetic width.
    /// This is the evidence shape for f32 and f64.
    StoredAtArithmeticWidth,
    /// f16/bf16 finalize once from f32. Differing stored values need the
    /// actual pre-final f32 bits from both lanes so the arithmetic-width
    /// distance is observable rather than guessed from rounding bins.
    ReducedFloatPreFinal { eval_bits: u32, compiled_bits: u32 },
    /// At least one lane is known not to do so; the issue owns the repair.
    Nonconforming { issue: u32 },
}

/// Why two rendered elements do not satisfy [05-OBS-3].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgreementError {
    ExactMismatch {
        context: String,
        reference: String,
        candidate: String,
    },
    Parse {
        lane: &'static str,
        prim: Prim,
        text: String,
    },
    NonCanonical {
        lane: &'static str,
        prim: Prim,
        text: String,
        canonical: String,
    },
    FormattingMismatch {
        prim: Prim,
        eval: String,
        compiled: String,
    },
    NonFiniteMismatch {
        op: String,
        prim: Prim,
        eval: String,
        compiled: String,
    },
    SignedZeroMismatch {
        op: String,
        prim: Prim,
        eval: String,
        compiled: String,
    },
    ArithmeticWidthNonconforming {
        op: String,
        prim: Prim,
        issue: u32,
        eval: String,
        compiled: String,
    },
    MissingReducedFloatEvidence {
        op: String,
        prim: Prim,
        eval: String,
        compiled: String,
    },
    ReducedFloatEvidenceMismatch {
        op: String,
        prim: Prim,
        detail: String,
    },
    UlpExceeded {
        op: String,
        prim: Prim,
        distance_ulps: u64,
        max_ulps: u64,
        eval: String,
        compiled: String,
    },
}

impl fmt::Display for AgreementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactMismatch {
                context,
                reference,
                candidate,
            } => write!(
                f,
                "{context} requires byte-exact agreement: reference `{reference}`, candidate `{candidate}`"
            ),
            Self::Parse { lane, prim, text } => write!(
                f,
                "{lane} emitted an invalid {} element `{text}`",
                prim.name()
            ),
            Self::NonCanonical {
                lane,
                prim,
                text,
                canonical,
            } => write!(
                f,
                "{lane} emitted non-canonical {} text `{text}`; [05-OBS] requires `{canonical}`",
                prim.name()
            ),
            Self::FormattingMismatch {
                prim,
                eval,
                compiled,
            } => write!(
                f,
                "{} formatting differs for identical bits: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
            Self::NonFiniteMismatch {
                op,
                prim,
                eval,
                compiled,
            } => write!(
                f,
                "{op}[{}] has a non-finite lane mismatch: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
            Self::SignedZeroMismatch {
                op,
                prim,
                eval,
                compiled,
            } => write!(
                f,
                "{op}[{}] differs only in the sign of zero, which [05-OBS-3] does not tolerance: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
            Self::ArithmeticWidthNonconforming {
                op,
                prim,
                issue,
                eval,
                compiled,
            } => write!(
                f,
                "{op}[{}] differs, but chelis#{issue} records a lane outside [04-NUM-8]'s declared arithmetic width; tolerance is ineligible: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
            Self::MissingReducedFloatEvidence {
                op,
                prim,
                eval,
                compiled,
            } => write!(
                f,
                "{op}[{}] differs after f32-to-storage finalization, but no pre-final f32 evidence was supplied: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
            Self::ReducedFloatEvidenceMismatch { op, prim, detail } => write!(
                f,
                "{op}[{}] supplied inconsistent reduced-float arithmetic evidence: {detail}",
                prim.name()
            ),
            Self::UlpExceeded {
                op,
                prim,
                distance_ulps,
                max_ulps,
                eval,
                compiled,
            } => write!(
                f,
                "{op}[{}] differs by {distance_ulps} ULP, above the authored {max_ulps}-ULP bound: eval `{eval}`, compiled `{compiled}`",
                prim.name()
            ),
        }
    }
}

impl Error for AgreementError {}

/// Compare arbitrary observations under the exact branch of the one Phase 3
/// comparator. Diagnostics, shapes, envelopes, integer payloads, and any
/// value without an authored tolerance row all use this path.
pub fn compare_exact_observations(
    context: &str,
    reference: &str,
    candidate: &str,
) -> Result<AgreementOutcome, AgreementError> {
    if reference.as_bytes() == candidate.as_bytes() {
        Ok(AgreementOutcome::ByteExact)
    } else {
        Err(AgreementError::ExactMismatch {
            context: context.to_string(),
            reference: reference.to_string(),
            candidate: candidate.to_string(),
        })
    }
}

/// Compare two complete, rendered scalar elements under [05-OBS-3].
///
/// Integer, bool, deferred, and string dtypes never enter float parsing. For
/// floats, both sides must first prove that they use the frozen canonical
/// grammar. Text that parses to the same stored bits but uses another spelling
/// is a formatting error, never a toleranced value difference.
pub fn compare_rendered_elements(
    op: AgreementOp,
    prim: Prim,
    width_status: ArithmeticWidthStatus,
    eval: &str,
    compiled: &str,
) -> Result<AgreementOutcome, AgreementError> {
    if let Ok(outcome) =
        compare_exact_observations(&format!("{}[{}]", op.name(), prim.name()), eval, compiled)
    {
        return Ok(outcome);
    }
    if !prim.is_float() {
        return Err(AgreementError::ExactMismatch {
            context: format!("{}[{}]", op.name(), prim.name()),
            reference: eval.to_string(),
            candidate: compiled.to_string(),
        });
    }

    let eval_value = parse_canonical_float("eval", prim, eval)?;
    let compiled_value = parse_canonical_float("compiled", prim, compiled)?;
    if eval_value.same_bits(compiled_value) {
        return Err(AgreementError::FormattingMismatch {
            prim,
            eval: eval.to_string(),
            compiled: compiled.to_string(),
        });
    }
    if eval_value.is_zero() && compiled_value.is_zero() {
        return Err(AgreementError::SignedZeroMismatch {
            op: op.name().to_string(),
            prim,
            eval: eval.to_string(),
            compiled: compiled.to_string(),
        });
    }
    if !eval_value.is_finite() || !compiled_value.is_finite() {
        return Err(AgreementError::NonFiniteMismatch {
            op: op.name().to_string(),
            prim,
            eval: eval.to_string(),
            compiled: compiled.to_string(),
        });
    }

    let distance_ulps = match width_status {
        ArithmeticWidthStatus::Nonconforming { issue } => {
            return Err(AgreementError::ArithmeticWidthNonconforming {
                op: op.name().to_string(),
                prim,
                issue,
                eval: eval.to_string(),
                compiled: compiled.to_string(),
            });
        }
        ArithmeticWidthStatus::StoredAtArithmeticWidth => match prim {
            Prim::F32 | Prim::F64 => eval_value.stored_distance_ulps(compiled_value),
            Prim::F16 | Prim::Bf16 => {
                return Err(AgreementError::MissingReducedFloatEvidence {
                    op: op.name().to_string(),
                    prim,
                    eval: eval.to_string(),
                    compiled: compiled.to_string(),
                });
            }
            Prim::F8e4m3
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool
            | Prim::String
            | Prim::Key => unreachable!("only active floats reach ULP comparison"),
        },
        ArithmeticWidthStatus::ReducedFloatPreFinal {
            eval_bits,
            compiled_bits,
        } => reduced_float_pre_final_distance(
            op.name(),
            prim,
            eval_value,
            compiled_value,
            eval_bits,
            compiled_bits,
        )?,
    };
    let max_ulps = tolerance_for(op);
    if distance_ulps <= max_ulps {
        Ok(AgreementOutcome::WithinTolerance {
            distance_ulps,
            max_ulps,
        })
    } else {
        Err(AgreementError::UlpExceeded {
            op: op.name().to_string(),
            prim,
            distance_ulps,
            max_ulps,
            eval: eval.to_string(),
            compiled: compiled.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum FloatValue {
    F16(half::f16),
    Bf16(half::bf16),
    F32(f32),
    F64(f64),
}

impl FloatValue {
    fn same_bits(self, other: Self) -> bool {
        match (self, other) {
            (Self::F16(a), Self::F16(b)) => a.to_bits() == b.to_bits(),
            (Self::Bf16(a), Self::Bf16(b)) => a.to_bits() == b.to_bits(),
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            _ => unreachable!("both parsed values use the caller's one Prim"),
        }
    }

    fn is_finite(self) -> bool {
        match self {
            Self::F16(v) => v.is_finite(),
            Self::Bf16(v) => v.is_finite(),
            Self::F32(v) => v.is_finite(),
            Self::F64(v) => v.is_finite(),
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Self::F16(v) => v == half::f16::ZERO,
            Self::Bf16(v) => v == half::bf16::ZERO,
            Self::F32(v) => v == 0.0,
            Self::F64(v) => v == 0.0,
        }
    }

    /// Ordinary ordered-bit distance for values stored at their arithmetic
    /// width. Reduced floats require explicit pre-final evidence instead.
    fn stored_distance_ulps(self, other: Self) -> u64 {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => u64::from(ordered_f32(a).abs_diff(ordered_f32(b))),
            (Self::F64(a), Self::F64(b)) => ordered_f64(a).abs_diff(ordered_f64(b)),
            _ => unreachable!("only f32/f64 are stored at their arithmetic width"),
        }
    }
}

fn reduced_float_pre_final_distance(
    op: &str,
    prim: Prim,
    eval_value: FloatValue,
    compiled_value: FloatValue,
    eval_bits: u32,
    compiled_bits: u32,
) -> Result<u64, AgreementError> {
    let eval_pre = f32::from_bits(eval_bits);
    let compiled_pre = f32::from_bits(compiled_bits);
    let evidence_matches = match (prim, eval_value, compiled_value) {
        (Prim::F16, FloatValue::F16(eval), FloatValue::F16(compiled)) => {
            half::f16::from_f32(eval_pre).to_bits() == eval.to_bits()
                && half::f16::from_f32(compiled_pre).to_bits() == compiled.to_bits()
        }
        (Prim::Bf16, FloatValue::Bf16(eval), FloatValue::Bf16(compiled)) => {
            half::bf16::from_f32(eval_pre).to_bits() == eval.to_bits()
                && half::bf16::from_f32(compiled_pre).to_bits() == compiled.to_bits()
        }
        _ => {
            return Err(AgreementError::ReducedFloatEvidenceMismatch {
                op: op.to_string(),
                prim,
                detail: "pre-final f32 evidence is valid only for f16/bf16 results".to_string(),
            });
        }
    };
    if !evidence_matches {
        return Err(AgreementError::ReducedFloatEvidenceMismatch {
            op: op.to_string(),
            prim,
            detail: format!(
                "eval bits {eval_bits:#010x} and compiled bits {compiled_bits:#010x} do not round to the observed values"
            ),
        });
    }
    if !eval_pre.is_finite() || !compiled_pre.is_finite() {
        return Err(AgreementError::ReducedFloatEvidenceMismatch {
            op: op.to_string(),
            prim,
            detail: "pre-final evidence must be finite for a toleranced comparison".to_string(),
        });
    }
    Ok(u64::from(
        ordered_f32(eval_pre).abs_diff(ordered_f32(compiled_pre)),
    ))
}

fn parse_canonical_float(
    lane: &'static str,
    prim: Prim,
    text: &str,
) -> Result<FloatValue, AgreementError> {
    let parse_error = || AgreementError::Parse {
        lane,
        prim,
        text: text.to_string(),
    };
    let (value, element) = match prim {
        Prim::F16 => {
            let parsed = text.parse::<f64>().map_err(|_| parse_error())?;
            let value = crate::dtype_semantics::f16_from_f64_rne(parsed);
            (FloatValue::F16(value), ElementRef::F16(value))
        }
        Prim::Bf16 => {
            let parsed = text.parse::<f64>().map_err(|_| parse_error())?;
            let value = crate::dtype_semantics::bf16_from_f64_rne(parsed);
            (FloatValue::Bf16(value), ElementRef::Bf16(value))
        }
        Prim::F32 => {
            let value = text.parse::<f32>().map_err(|_| parse_error())?;
            (FloatValue::F32(value), ElementRef::F32(value))
        }
        Prim::F64 => {
            let value = text.parse::<f64>().map_err(|_| parse_error())?;
            (FloatValue::F64(value), ElementRef::F64(value))
        }
        Prim::F8e4m3
        | Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool
        | Prim::String
        | Prim::Key => unreachable!("only active floats reach float parsing"),
    };
    let canonical = format_element(prim, element);
    if canonical != text {
        return Err(AgreementError::NonCanonical {
            lane,
            prim,
            text: text.to_string(),
            canonical,
        });
    }
    Ok(value)
}

fn ordered_f32(value: f32) -> u32 {
    const SIGN_MASK: u32 = 1 << 31;
    let bits = value.to_bits();
    if bits & SIGN_MASK == 0 {
        bits | SIGN_MASK
    } else {
        !bits
    }
}

fn ordered_f64(value: f64) -> u64 {
    const SIGN_MASK: u64 = 1 << 63;
    let bits = value.to_bits();
    if bits & SIGN_MASK == 0 {
        bits | SIGN_MASK
    } else {
        !bits
    }
}

/// Render the exact Markdown table mirrored into spec/05 §8.
pub fn render_spec_tolerance_table() -> String {
    let mut out = String::from(
        "| operation | maximum cross-lane value difference | authority |\n\
         |---|---:|---|\n",
    );
    for row in OP_TOLERANCES {
        out.push_str(&format!(
            "| `{}` | {} ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |\n",
            row.op.name(),
            row.max_ulps
        ));
    }
    out
}
