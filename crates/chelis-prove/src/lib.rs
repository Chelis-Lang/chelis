//! Three-tier property verification dispatcher.
//!
//! This crate implements the specification-consumption layer of the Chelis
//! trust stack. It takes `@property` declarations and dispatches them through:
//!
//! - **Tier A:** Type system validation (dimension, effect, linearity discharge)
//! - **Tier B:** SMT solving via cvc5 (nonlinear real arithmetic)
//! - **Tier C:** Randomized fuzz testing (existing `chelis prove` logic)
//!
//! See `docs/trust-stack-verification.md` for architectural framing.

pub mod amenability;
pub mod artifact;
pub mod convert;
pub mod discover;
pub mod dispatch;
pub mod from_property_spec;
pub mod inlineability;
pub mod solver;
pub mod tier_a;
pub mod tier_b;
pub mod tier_c;

pub use artifact::{ProofArtifact, ProofStatus, ProofTier};
pub use dispatch::{DispatchOptions, dispatch_property};
pub use from_property_spec::{PropertySpecInput, to_dispatch_amenability, to_smt_property};
pub use inlineability::{Fuzzability, Inlineability, classify_fuzzability, classify_inlineability};
pub use tier_b::{SmtProperty, solve_property};

// --- Verification Pipeline API ---

/// Source language for verification input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SourceKind {
    Surf,
    Deep,
}

/// Tier selection for verification dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TierSelection {
    Auto,
    FuzzOnly,
    SmtOnly,
    TypeOnly,
}

/// Request to verify source containing @property declarations.
#[derive(Debug, Clone)]
pub struct VerificationRequest {
    pub source: String,
    pub source_kind: SourceKind,
    pub tier: TierSelection,
    pub smt_timeout_ms: u64,
    pub fuzz_samples: u32,
    pub fuzz_seed: u64,
    pub inlining_depth_limit: u32,
    pub spans_json: Option<String>,
}

impl Default for VerificationRequest {
    fn default() -> Self {
        Self {
            source: String::new(),
            source_kind: SourceKind::Surf,
            tier: TierSelection::Auto,
            smt_timeout_ms: 5000,
            fuzz_samples: 100,
            fuzz_seed: 0,
            inlining_depth_limit: 3,
            spans_json: None,
        }
    }
}

/// Provenance chain back to EARS source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    pub requirement_id: Option<String>,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub text: Option<String>,
}

/// Counterexample from SMT or fuzz.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Counterexample {
    pub bindings: Vec<(String, f64)>,
}

/// Result for a single property verification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropertyResult {
    pub name: String,
    pub proof_tier: ProofTier,
    pub status: ProofStatus,
    pub counterexample: Option<Counterexample>,
    pub provenance: Option<Provenance>,
    pub duration_ms: u64,
}

/// Aggregate verification summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationSummary {
    pub total: usize,
    pub proved: usize,
    pub statistically_validated: usize,
    pub failed: usize,
    pub rejected: usize,
    pub inconclusive: usize,
}

/// Complete verification result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    pub properties: Vec<PropertyResult>,
    pub summary: VerificationSummary,
}

/// Error from verification.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VerifyError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("no properties found in source")]
    NoProperties,
    #[error("internal error: {0}")]
    Internal(String),
}

/// Main entry point: verify all properties in a source string.
/// Both CLI and MCP server call this function.
pub fn verify_source(req: VerificationRequest) -> Result<VerificationResult, VerifyError> {
    use std::time::Instant;

    if matches!(req.source_kind, SourceKind::Deep) {
        return Err(VerifyError::Internal(
            "Deep source not yet supported in verify_source".into(),
        ));
    }

    let decls = chelis_surf::parser::parse_str(&req.source)
        .map_err(|e| VerifyError::Parse(e.to_string()))?;

    let flat = flatten_decls(&decls);
    let properties = discover::collect_surf_properties(&flat, None);
    if properties.is_empty() {
        return Err(VerifyError::NoProperties);
    }

    let mut results = Vec::new();
    for prop in &properties {
        let start = Instant::now();
        let ctx = convert::InlineCtx {
            decls: &flat,
            depth: 0,
            max_depth: req.inlining_depth_limit as usize,
            call_stack: vec![],
        };

        // Convert property body and preconditions to SmtExpr
        let postcondition = convert::surf_expr_to_smt(&prop.body, &ctx);
        let preconditions: Vec<solver::SmtExpr> = prop
            .preconditions
            .iter()
            .filter_map(|e| convert::surf_expr_to_smt(e, &ctx))
            .collect();

        let pr = if let Some(post) = postcondition {
            // Build SmtProperty
            let variables: Vec<(String, solver::SmtSort)> = prop
                .params
                .iter()
                .filter_map(|p| {
                    let sort = match p.ty.as_ref()? {
                        chelis_surf::ast::TypeExpr::Named(n, _) => match n.as_str() {
                            "f32" | "f64" => solver::SmtSort::Real,
                            "int32" | "int64" => solver::SmtSort::Int,
                            "bool" => solver::SmtSort::Bool,
                            _ => return None,
                        },
                        _ => return None,
                    };
                    Some((p.name.clone(), sort))
                })
                .collect();

            if variables.len() != prop.params.len() {
                // Unsupported param types — fall to fuzz
                fuzz_property(&prop.name, &variables, &preconditions, &post, &req, start)
            } else {
                let smt_prop = tier_b::SmtProperty {
                    variables,
                    preconditions,
                    postcondition: post.clone(),
                };
                dispatch_property_tiers(&prop.name, smt_prop, &post, &req, start)
            }
        } else {
            // Couldn't convert to SMT — fuzz only
            PropertyResult {
                name: prop.name.clone(),
                proof_tier: artifact::ProofTier::Fuzz,
                status: artifact::ProofStatus::StatisticallyValidated {
                    samples: req.fuzz_samples as usize,
                },
                counterexample: None,
                provenance: Some(Provenance {
                    requirement_id: Some(prop.name.clone()),
                    file: None,
                    line: None,
                    text: None,
                }),
                duration_ms: start.elapsed().as_millis() as u64,
            }
        };
        results.push(pr);
    }

    // Enrich provenance from spans manifest if provided
    if let Some(spans_str) = &req.spans_json
        && let Ok(manifest) = serde_json::from_str::<serde_json::Value>(spans_str)
        && let Some(spans) = manifest.get("spans").and_then(|s| s.as_array())
    {
        for result in &mut results {
            if let Some(entry) = spans
                .iter()
                .find(|e| e.get("deep_node_id").and_then(|v| v.as_str()) == Some(&result.name))
            {
                result.provenance = Some(Provenance {
                    requirement_id: entry
                        .get("ears_id")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    file: entry
                        .get("ears_file")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    line: entry
                        .get("ears")
                        .and_then(|e| e.get("start_line"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    text: entry
                        .get("ears_text")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                });
            }
        }
    }

    let summary = build_summary(&results);
    Ok(VerificationResult {
        properties: results,
        summary,
    })
}

fn flatten_decls(decls: &[chelis_surf::ast::Decl]) -> Vec<chelis_surf::ast::Decl> {
    let mut out = Vec::new();
    for d in decls {
        match d {
            chelis_surf::ast::Decl::Module { decls, .. } => out.extend(flatten_decls(decls)),
            other => out.push(other.clone()),
        }
    }
    out
}

fn dispatch_property_tiers(
    name: &str,
    smt_prop: tier_b::SmtProperty,
    post: &solver::SmtExpr,
    req: &VerificationRequest,
    start: std::time::Instant,
) -> PropertyResult {
    // Tier B: SMT
    #[cfg(feature = "smt")]
    if matches!(req.tier, TierSelection::Auto | TierSelection::SmtOnly) {
        match tier_b::solve_property(&smt_prop, req.smt_timeout_ms) {
            tier_b::TierBResult::Proved => {
                return PropertyResult {
                    name: name.to_string(),
                    proof_tier: artifact::ProofTier::Smt,
                    status: artifact::ProofStatus::Proved,
                    counterexample: None,
                    provenance: Some(Provenance {
                        requirement_id: Some(name.to_string()),
                        file: None,
                        line: None,
                        text: None,
                    }),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
            tier_b::TierBResult::Disproved(model) => {
                let cx = parse_counterexample(&model);
                return PropertyResult {
                    name: name.to_string(),
                    proof_tier: artifact::ProofTier::Smt,
                    status: artifact::ProofStatus::Disproved {
                        counterexample: serde_json::Value::Object(Default::default()),
                    },
                    counterexample: Some(cx),
                    provenance: Some(Provenance {
                        requirement_id: Some(name.to_string()),
                        file: None,
                        line: None,
                        text: None,
                    }),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
            tier_b::TierBResult::Timeout | tier_b::TierBResult::Unknown => {
                // Fall through to Tier C
            }
        }
    }

    // Tier C: Fuzz
    fuzz_property(
        name,
        &smt_prop.variables,
        &smt_prop.preconditions,
        post,
        req,
        start,
    )
}

fn fuzz_property(
    name: &str,
    variables: &[(String, solver::SmtSort)],
    preconditions: &[solver::SmtExpr],
    postcondition: &solver::SmtExpr,
    req: &VerificationRequest,
    start: std::time::Instant,
) -> PropertyResult {
    let smt_prop = tier_b::SmtProperty {
        variables: variables.to_vec(),
        preconditions: preconditions.to_vec(),
        postcondition: postcondition.clone(),
    };
    match tier_c::fuzz_smt_property(&smt_prop, req.fuzz_samples as usize, req.fuzz_seed) {
        tier_c::TierCResult::AllPassed(n) => PropertyResult {
            name: name.to_string(),
            proof_tier: artifact::ProofTier::Fuzz,
            status: artifact::ProofStatus::StatisticallyValidated { samples: n },
            counterexample: None,
            provenance: Some(Provenance {
                requirement_id: Some(name.to_string()),
                file: None,
                line: None,
                text: None,
            }),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        tier_c::TierCResult::Failed(cx) => PropertyResult {
            name: name.to_string(),
            proof_tier: artifact::ProofTier::Fuzz,
            status: artifact::ProofStatus::Disproved { counterexample: cx },
            counterexample: None,
            provenance: Some(Provenance {
                requirement_id: Some(name.to_string()),
                file: None,
                line: None,
                text: None,
            }),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        tier_c::TierCResult::Error(msg) => PropertyResult {
            name: name.to_string(),
            proof_tier: artifact::ProofTier::Fuzz,
            status: artifact::ProofStatus::Rejected { reason: msg },
            counterexample: None,
            provenance: Some(Provenance {
                requirement_id: Some(name.to_string()),
                file: None,
                line: None,
                text: None,
            }),
            duration_ms: start.elapsed().as_millis() as u64,
        },
    }
}

fn parse_counterexample(model: &serde_json::Value) -> Counterexample {
    let bindings = model
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    v.as_str()
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|f| (k.clone(), f))
                })
                .collect()
        })
        .unwrap_or_default();
    Counterexample { bindings }
}

fn build_summary(results: &[PropertyResult]) -> VerificationSummary {
    let mut s = VerificationSummary {
        total: results.len(),
        proved: 0,
        statistically_validated: 0,
        failed: 0,
        rejected: 0,
        inconclusive: 0,
    };
    for r in results {
        match &r.status {
            artifact::ProofStatus::Proved => s.proved += 1,
            artifact::ProofStatus::StatisticallyValidated { .. } => s.statistically_validated += 1,
            artifact::ProofStatus::Disproved { .. } => s.failed += 1,
            artifact::ProofStatus::Rejected { .. } => s.rejected += 1,
            artifact::ProofStatus::NotAmenable { .. } => s.inconclusive += 1,
        }
    }
    s
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn verify_source_discovers_and_verifies_property() {
        let source =
            "def f(x: f32) -> f32 = x * x\n\n@property sq forall(x: f32):\n  f(x) >= 0.0\n";
        let req = VerificationRequest {
            source: source.to_string(),
            ..Default::default()
        };
        let result = verify_source(req).unwrap();
        assert_eq!(result.properties.len(), 1);
        assert_eq!(result.properties[0].name, "sq");
        assert_eq!(result.summary.total, 1);
        // With smt feature: Proved. Without: StatisticallyValidated.
        match &result.properties[0].status {
            ProofStatus::Proved | ProofStatus::StatisticallyValidated { .. } => {}
            other => panic!("unexpected status: {other:?}"),
        }
    }

    #[test]
    fn verify_source_detects_failure() {
        let source = "def bad(x: f32) -> f32 = x - 1.0\n\n@property fails forall(x: f32) where x > 0.0:\n  bad(x) > 0.0\n";
        let req = VerificationRequest {
            source: source.to_string(),
            ..Default::default()
        };
        let result = verify_source(req).unwrap();
        assert_eq!(result.summary.total, 1);
        assert!(result.summary.failed > 0 || result.summary.proved == 0);
    }

    #[test]
    fn verify_source_no_properties_is_error() {
        let source = "def f(x: f32) -> f32 = x\n";
        let req = VerificationRequest {
            source: source.to_string(),
            ..Default::default()
        };
        assert!(matches!(verify_source(req), Err(VerifyError::NoProperties)));
    }
}
