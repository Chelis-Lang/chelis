//! Fitness scoring for Chelis programs.
//!
//! Produces a 0.0-1.0 fitness score measuring "how close to valid" a program
//! is. This is the training signal for AI agents (see spec section 6).

use crate::errors::{CheckError, CheckErrorKind};
use crate::infer::{InferResult, infer_ir_program};

/// Weights for each fitness component (spec section 6.1).
const W_PARSE: f64 = 0.1;
const W_STRUCTURE: f64 = 0.1;
const W_NAMES: f64 = 0.2;
const W_TYPES: f64 = 0.6;

/// Full fitness report for a compilation attempt.
#[derive(Debug, Clone)]
pub struct FitnessReport {
    /// Weighted fitness score in [0.0, 1.0].
    pub score: f64,
    /// Individual component scores.
    pub components: FitnessComponents,
    /// All type-check errors encountered.
    pub errors: Vec<CheckError>,
    /// Number of sub-expressions that typed successfully.
    pub typed_nodes: usize,
    /// Number of sub-expressions that failed to type.
    pub untyped_nodes: usize,
    /// Total number of sub-expressions visited.
    pub total_nodes: usize,
    /// Names that could not be resolved.
    pub unresolved_names: Vec<String>,
}

/// Individual component scores, each in [0.0, 1.0].
#[derive(Debug, Clone)]
pub struct FitnessComponents {
    /// 1.0 if the program parsed successfully.
    pub parse: f64,
    /// Fraction of nodes with valid tags and arity (1.0 for now; validator runs separately).
    pub structure: f64,
    /// Fraction of variable references that resolve.
    pub names: f64,
    /// Fraction of sub-expressions that unify successfully (typed_nodes / total_nodes).
    pub types: f64,
}

impl FitnessReport {
    /// Compute fitness from an inference result.
    ///
    /// `parse` is 1.0 (we only reach the checker if parsing succeeded).
    /// `structure` is 1.0 by default; use `with_structure` to override
    /// if the Deep tag validator was run upstream.
    pub fn from_infer_result(result: &InferResult) -> FitnessReport {
        Self::from_infer_result_with_structure(result, 1.0)
    }

    /// Compute fitness with an explicit structure score (from tag validator).
    pub fn from_infer_result_with_structure(result: &InferResult, structure: f64) -> FitnessReport {
        let parse = 1.0;
        // structure comes from parameter (tag validator score)

        // Names: count UnboundVariable errors relative to total nodes.
        let unbound_count = result
            .errors
            .iter()
            .filter(|e| matches!(e.kind, CheckErrorKind::UnboundVariable))
            .count();
        let total = result.total_nodes.max(1);
        let names = if unbound_count == 0 {
            1.0
        } else {
            1.0 - (unbound_count as f64 / total as f64).min(1.0)
        };

        // Types: fraction of nodes that typed successfully.
        let types = if result.total_nodes == 0 {
            1.0
        } else {
            result.typed_nodes as f64 / result.total_nodes as f64
        };

        let score = W_PARSE * parse + W_STRUCTURE * structure + W_NAMES * names + W_TYPES * types;

        // Collect unresolved names from UnboundVariable errors
        let unresolved_names: Vec<String> = result
            .errors
            .iter()
            .filter(|e| matches!(e.kind, CheckErrorKind::UnboundVariable))
            .filter_map(|e| {
                e.message
                    .strip_prefix("unbound variable: ")
                    .map(|s| s.to_string())
            })
            .collect();

        FitnessReport {
            score,
            components: FitnessComponents {
                parse,
                structure,
                names,
                types,
            },
            errors: result.errors.clone(),
            typed_nodes: result.typed_nodes,
            untyped_nodes: result.total_nodes.saturating_sub(result.typed_nodes),
            total_nodes: result.total_nodes,
            unresolved_names,
        }
    }
}

/// Type-check a Deep program and produce a fitness report.
/// Runs tag validation to compute the structure component.
pub fn check_program(exprs: &[chelis_deep::Expr]) -> FitnessReport {
    let structure = structure_score(exprs);
    let result = crate::infer::infer_program(exprs);
    FitnessReport::from_infer_result_with_structure(&result, structure)
}

/// IR/0h-aware fitness report that treats typed self-loads as valid
/// program inputs, matching the executable compiler pipeline.
pub fn check_ir_program(exprs: &[chelis_deep::Expr]) -> FitnessReport {
    let structure = structure_score(exprs);
    let result = infer_ir_program(exprs);
    if result.errors.is_empty() {
        let total_nodes = count_nodes(exprs);
        return FitnessReport {
            score: 1.0,
            components: FitnessComponents {
                parse: 1.0,
                structure,
                names: 1.0,
                types: 1.0,
            },
            errors: Vec::new(),
            typed_nodes: total_nodes,
            untyped_nodes: 0,
            total_nodes,
            unresolved_names: Vec::new(),
        };
    }

    let mut report = FitnessReport::from_infer_result_with_structure(&result, structure);

    // IR adds executable-pipeline validation after type inference. Ensure
    // those failures reduce the fitness score as well, so score 1.0 always means
    // error-free on the executable Phase 0 path.
    let min_untyped = result.errors.len().min(report.total_nodes);
    let effective_untyped = report.untyped_nodes.max(min_untyped);
    let effective_typed = report.total_nodes.saturating_sub(effective_untyped);

    report.typed_nodes = effective_typed;
    report.untyped_nodes = effective_untyped;
    report.components.types = if report.total_nodes == 0 {
        1.0
    } else {
        effective_typed as f64 / report.total_nodes as f64
    };
    report.score = W_PARSE * report.components.parse
        + W_STRUCTURE * report.components.structure
        + W_NAMES * report.components.names
        + W_TYPES * report.components.types;

    report
}

fn structure_score(exprs: &[chelis_deep::Expr]) -> f64 {
    let warnings = chelis_deep::validate::validate(exprs);
    if exprs.is_empty() {
        1.0
    } else {
        let node_count = count_nodes(exprs).max(1);
        let invalid_nodes = warnings
            .iter()
            .map(|warning| warning.offset)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let valid = node_count.saturating_sub(invalid_nodes);
        valid as f64 / node_count as f64
    }
}

/// Count total AST nodes for structure scoring.
fn count_nodes(exprs: &[chelis_deep::Expr]) -> usize {
    exprs.iter().map(count_node).sum()
}

fn count_node(expr: &chelis_deep::Expr) -> usize {
    match expr {
        chelis_deep::Expr::Atom(_, _) => 1,
        chelis_deep::Expr::List(list, _) => 1 + list.elements.iter().map(count_node).sum::<usize>(),
        chelis_deep::Expr::Map(map, _) => {
            1 + map
                .entries
                .iter()
                .map(|(_, v)| count_node(v))
                .sum::<usize>()
        }
        chelis_deep::Expr::MetaExpr(meta, _) => {
            1 + count_node(&meta.expr)
                + meta
                    .entries
                    .iter()
                    .map(|(_, v)| count_node(v))
                    .sum::<usize>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(src: &str) -> FitnessReport {
        let exprs = chelis_deep::parser::parse_str(src).unwrap();
        FitnessReport::from_infer_result(&crate::infer::infer_program(&exprs))
    }

    #[test]
    fn correct_program_scores_1() {
        let r = score("(def {} x (lit {type: (t-prim {} int32)} 42))");
        assert!(
            (r.score - 1.0).abs() < 1e-9,
            "expected score 1.0, got {}",
            r.score
        );
        assert!(r.errors.is_empty());
    }

    #[test]
    fn unbound_variable_reduces_score() {
        let r = score("(def {} x (var {} unknown))");
        assert!(r.score < 1.0, "expected score < 1.0, got {}", r.score);
        assert!(r.components.names < 1.0);
    }

    #[test]
    fn empty_program_scores_1() {
        let r = score("");
        assert!(
            (r.score - 1.0).abs() < 0.01,
            "expected score ~1.0 for empty program, got {}",
            r.score
        );
    }

    #[test]
    fn parse_component_always_1() {
        let r = score("(def {} x (lit {type: (t-prim {} f32)} 3.0))");
        assert!((r.components.parse - 1.0).abs() < 1e-9);
        assert!((r.components.structure - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_errors_give_intermediate_score() {
        let r = score(
            "(def {} good (lit {type: (t-prim {} int32)} 42)) \
             (def {} bad (var {} nope))",
        );
        assert!(
            r.score > 0.0 && r.score < 1.0,
            "expected intermediate score, got {}",
            r.score
        );
    }

    #[test]
    fn multiple_unbound_vars_lower_names_further() {
        let r1 = score("(def {} a (var {} x))");
        let r2 = score("(def {} a (var {} x)) (def {} b (var {} y))");
        // More unbound variables should mean lower (or equal) names component
        assert!(r2.components.names <= r1.components.names + 1e-9);
    }

    #[test]
    fn typed_and_total_nodes_tracked() {
        let r = score("(def {} x (lit {type: (t-prim {} int32)} 42))");
        assert!(r.total_nodes > 0, "should have visited some nodes");
        assert_eq!(
            r.typed_nodes, r.total_nodes,
            "all nodes should type-check in a correct program"
        );
    }

    #[test]
    fn check_program_fn_works() {
        let exprs = chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} int32)} 42))")
            .unwrap();
        let r = check_program(&exprs);
        assert!((r.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ir_check_accepts_typed_self_loads() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))} x))",
        )
        .unwrap();
        let r = check_ir_program(&exprs);
        assert!((r.score - 1.0).abs() < 1e-9, "{}", r.score);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn ir_check_never_reports_perfect_score_with_errors() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} f (app {} (var {} add) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} bool)} true)))",
        )
        .unwrap();
        let r = check_ir_program(&exprs);
        assert!(!r.errors.is_empty(), "{r:?}");
        assert!(r.score < 1.0, "{r:?}");
        assert!(r.untyped_nodes > 0, "{r:?}");
    }
}
