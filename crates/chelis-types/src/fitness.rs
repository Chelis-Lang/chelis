//! Fitness scoring for Chelis programs.
//!
//! Produces a 0.0-1.0 fitness score measuring "how close to valid" a program
//! is. This is the training signal for AI agents (see spec section 6).

use crate::errors::CheckError;
use crate::infer::{CheckedProgram, InferResult, InferStats};

/// Weights for each fitness component (spec section 6.1).
const W_PARSE: f64 = 0.1;
const W_STRUCTURE: f64 = 0.1;
const W_NAMES: f64 = 0.2;
const W_TYPES: f64 = 0.6;

/// Backstop upper bound for the §C4.4 fitness-honesty invariant: the
/// strict ceiling a score is forced to if a diagnostic was reported but
/// the weighted components still sum to a perfect 1.0. See
/// [`weighted_score`]. The primary honesty signal is the `types`
/// component, which scores 0.0 for a runtime-nodeless failed unit
/// (chelis#833); this ceiling only guards the residual case where every
/// runtime node typed individually yet a declaration-level diagnostic was
/// still pushed (reachable through the non-IR `check_program` path).
const HONEST_ERROR_CEILING: f64 = 0.99;

/// Weighted component sum, subject to the §C4.4 fitness-honesty invariant
/// (spec/design/checker_totality.md, open question 4 / chelis#731;
/// regression chelis#833): a non-empty error vector forces the score
/// strictly below 1.0, independent of node counts.
///
/// A declaration-level error on a program with no runtime nodes -- a
/// duplicate `deftype` / `defsig` / `typealias` yields `total_nodes == 0`
/// -- is scored honestly upstream by driving the `types` component to 0.0
/// (see [`FitnessReport::from_infer_result_with_structure`] and
/// [`check_ir_program`]), so such a unit lands near 0.4 rather than a
/// vacuous 1.0. This function is the final backstop: if some residual path
/// still sums to 1.0 with a reported error, cap it strictly below 1.0.
fn weighted_score(components: &FitnessComponents, has_errors: bool) -> f64 {
    let raw = W_PARSE * components.parse
        + W_STRUCTURE * components.structure
        + W_NAMES * components.names
        + W_TYPES * components.types;
    if has_errors && raw >= 1.0 {
        HONEST_ERROR_CEILING
    } else {
        raw
    }
}

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

        // Names: both unresolved value references and unresolved
        // constructors are name failures. The kind carries the exact source
        // identifier, so this one structured extraction drives the score and
        // the public unresolved-name list without parsing rendered prose.
        let unresolved_names = result
            .errors
            .iter()
            .filter_map(|error| error.kind.unresolved_identifier().map(str::to_string))
            .collect::<Vec<_>>();
        let unresolved_count = unresolved_names.len();
        let total = result.total_nodes.max(1);
        let names = if unresolved_count == 0 {
            1.0
        } else {
            1.0 - (unresolved_count as f64 / total as f64).min(1.0)
        };

        // Types: fraction of nodes that typed successfully. With no runtime
        // nodes there is no coverage fraction: a clean program is vacuously
        // fully typed (1.0), but a reported declaration-level error (a
        // duplicate `deftype` / `defsig` / `typealias`) means type checking
        // failed with nothing salvageable, so it scores 0.0 rather than a
        // vacuous 1.0 (chelis#833 / §C4.4).
        let types = if result.total_nodes == 0 {
            if result.errors.is_empty() { 1.0 } else { 0.0 }
        } else {
            result.typed_nodes as f64 / result.total_nodes as f64
        };

        let components = FitnessComponents {
            parse,
            structure,
            names,
            types,
        };
        // §C4.4 (chelis#833): a reported error forces score < 1.0 even when
        // the coverage components sum to a perfect 1.0 (declaration-only
        // failures have no runtime node to lower `types`).
        let score = weighted_score(&components, !result.errors.is_empty());

        FitnessReport {
            score,
            components,
            errors: result.errors.clone(),
            typed_nodes: result.typed_nodes,
            untyped_nodes: result.total_nodes.saturating_sub(result.typed_nodes),
            total_nodes: result.total_nodes,
            unresolved_names,
        }
    }
}

/// A single IR type-analysis product.
///
/// Rejected analysis carries only its fitness report. Accepted analysis carries
/// the report and the checked program produced by the same inference session.
#[derive(Debug)]
pub enum TypeAnalysisOutcome {
    Rejected {
        fitness: FitnessReport,
    },
    Accepted {
        fitness: FitnessReport,
        program: Box<CheckedProgram>,
    },
}

/// Type-check a Deep program and produce a fitness report.
/// Runs tag validation to compute the structure component.
pub fn check_program(exprs: &[chelis_deep::Expr]) -> FitnessReport {
    let structure = structure_score(exprs);
    let result = crate::infer::infer_program(exprs);
    FitnessReport::from_infer_result_with_structure(&result, structure)
}

/// Run one IR-aware inference session and return its fitness and checked product.
pub fn analyze_ir_program(exprs: &[chelis_deep::Expr]) -> TypeAnalysisOutcome {
    let structural = structural_stats(exprs);
    let structure = structure_score_from_stats(structural);
    match crate::infer::check_ir_program(exprs) {
        Ok(program) => TypeAnalysisOutcome::Accepted {
            fitness: clean_fitness_from_stats(structural, program.infer_stats()),
            program: Box::new(program),
        },
        Err(result) => TypeAnalysisOutcome::Rejected {
            fitness: rejected_ir_fitness(&result, structure),
        },
    }
}

/// Build the canonical clean fitness report from structural and inference data.
///
/// Contextual and layered callers use this function instead of copying fitness
/// weights or clean-report formulas.
pub fn clean_fitness_from_stats(structural: StructuralStats, infer: InferStats) -> FitnessReport {
    let structure = structure_score_from_stats(structural);
    let components = FitnessComponents {
        parse: 1.0,
        structure,
        names: 1.0,
        types: 1.0,
    };
    FitnessReport {
        score: weighted_score(&components, false),
        components,
        errors: Vec::new(),
        typed_nodes: infer.typed_nodes,
        untyped_nodes: infer.total_nodes.saturating_sub(infer.typed_nodes),
        total_nodes: infer.total_nodes,
        unresolved_names: Vec::new(),
    }
}

/// IR-aware fitness compatibility entry.
///
/// The report now comes from the same inference product as
/// [`analyze_ir_program`].
pub fn check_ir_program(exprs: &[chelis_deep::Expr]) -> FitnessReport {
    match analyze_ir_program(exprs) {
        TypeAnalysisOutcome::Rejected { fitness }
        | TypeAnalysisOutcome::Accepted { fitness, .. } => fitness,
    }
}

fn rejected_ir_fitness(result: &InferResult, structure: f64) -> FitnessReport {
    let mut report = FitnessReport::from_infer_result_with_structure(result, structure);

    // IR adds executable-pipeline validation after type inference. Ensure
    // those failures reduce the fitness score as well, so score 1.0 always means
    // error-free on the executable Phase 0 path.
    let min_untyped = result.errors.len().min(report.total_nodes);
    let effective_untyped = report.untyped_nodes.max(min_untyped);
    let effective_typed = report.total_nodes.saturating_sub(effective_untyped);

    report.typed_nodes = effective_typed;
    report.untyped_nodes = effective_untyped;
    // A runtime-nodeless unit here failed at declaration level, so its
    // `types` component is 0.0 instead of a vacuous 1.0.
    report.components.types = if report.total_nodes == 0 {
        0.0
    } else {
        effective_typed as f64 / report.total_nodes as f64
    };
    report.score = weighted_score(&report.components, true);

    report
}

fn structure_score(exprs: &[chelis_deep::Expr]) -> f64 {
    structure_score_from_stats(structural_stats(exprs))
}

fn structure_score_from_stats(stats: StructuralStats) -> f64 {
    if stats.total_nodes == 0 {
        1.0
    } else {
        let valid = stats.total_nodes.saturating_sub(stats.invalid_nodes);
        valid as f64 / stats.total_nodes as f64
    }
}

/// Structural statistics for a Deep expr list: the total AST node count
/// and the number of nodes the Deep tag validator flagged.
///
/// Both fields are computed by pure structural walks
/// (`chelis_deep::validate::validate` is per-expr; `count_nodes` is a
/// recursive sum) with no cross-expr-list interaction. They are the
/// inputs the cross-process chelis-std typecheck cache stores so the
/// in-context fitness report can reconstitute the whole-program `structure`
/// component without re-walking the chelis-std library decls. Fitness
/// `typed_nodes` / `total_nodes` are the inference product's distinct
/// checker-visit counters (chelis#973).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StructuralStats {
    /// Total nodes in the structural AST walk.
    pub total_nodes: usize,
    /// Count of distinct node offsets the Deep tag validator flagged.
    pub invalid_nodes: usize,
}

/// Compute [`StructuralStats`] for `exprs`. See the type docs for why
/// the result is suitable for partition-and-recombine.
pub fn structural_stats(exprs: &[chelis_deep::Expr]) -> StructuralStats {
    let warnings = chelis_deep::validate::validate(exprs);
    let invalid_nodes = warnings
        .iter()
        .map(|warning| warning.offset)
        .collect::<chelis_unord::UnordSet<_>>()
        .len();
    StructuralStats {
        total_nodes: count_nodes(exprs),
        invalid_nodes,
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
        chelis_deep::Expr::Map(map, _) => 1 + count_metadata(map),
        chelis_deep::Expr::MetaExpr(meta, _) => {
            1 + count_node(&meta.expr) + count_metadata(&meta.metadata)
        }
        // Bridge: reconstruct List so all children (including meta) are counted (#908)
        chelis_deep::Expr::Node(node, span) => {
            let bridged = chelis_deep::Expr::List(node.to_list(*span), *span);
            count_node(&bridged)
        }
        chelis_deep::Expr::BareList(elems, _) => 1 + elems.iter().map(count_node).sum::<usize>(),
        chelis_deep::Expr::UnknownForm(data) => {
            1 + data.children.iter().map(count_node).sum::<usize>()
        }
    }
}

fn count_metadata(meta: &chelis_deep::Metadata) -> usize {
    fn raw_count(expr: &chelis_deep::RawExpr) -> usize {
        use chelis_deep::RawExpr;
        match expr {
            RawExpr::ExtensionData(_) => 1,
            RawExpr::Atom(..) => 1,
            RawExpr::List(values, _) => 1 + values.iter().map(raw_count).sum::<usize>(),
            RawExpr::Map(values, _) => 1 + values.iter().map(|(_, v)| raw_count(v)).sum::<usize>(),
            RawExpr::MetaExpr { entries, expr, .. } => {
                1 + raw_count(expr) + entries.iter().map(|(_, v)| raw_count(v)).sum::<usize>()
            }
        }
    }
    let mut count = meta.values().count() + meta.extensions().iter().count();
    meta.visit_syntax(&mut |_, value| count += count_node(value) - 1);
    if let Some(source) = meta.source() {
        count += 1 + source.arguments().iter().map(raw_count).sum::<usize>();
    }
    if let Some(bounds) = meta.dtype_bounds() {
        count += bounds.bounds().count();
    }
    if meta.loc().is_some() {
        count += 4;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(src: &str) -> FitnessReport {
        let exprs = chelis_deep::parser::parse_str(src).unwrap();
        FitnessReport::from_infer_result(&crate::infer::infer_program(&exprs))
    }

    #[test]
    fn accepted_analysis_uses_one_type_session() {
        let exprs =
            chelis_deep::parser::parse_str("(def {} answer (lit {type: (t-prim {} i32)} 42))")
                .unwrap();
        crate::session::reset_type_analysis_session_count();

        assert!(matches!(
            analyze_ir_program(&exprs),
            TypeAnalysisOutcome::Accepted { .. }
        ));
        assert_eq!(crate::session::type_analysis_session_count(), 1);
    }

    #[test]
    fn rejected_analysis_uses_one_type_session() {
        let exprs = chelis_deep::parser::parse_str("(def {} answer (var {} missing))").unwrap();
        crate::session::reset_type_analysis_session_count();

        assert!(matches!(
            analyze_ir_program(&exprs),
            TypeAnalysisOutcome::Rejected { .. }
        ));
        assert_eq!(crate::session::type_analysis_session_count(), 1);
    }

    #[test]
    fn correct_program_scores_1() {
        let r = score("(def {} x (lit {type: (t-prim {} i32)} 42))");
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
    fn both_name_diagnostic_kinds_share_structured_fitness_accounting() {
        let result = InferResult {
            errors: vec![
                CheckError::new(
                    crate::errors::CheckErrorKind::UnboundVariable {
                        identifier: "missing_value".to_string(),
                    },
                    "rendered value message may change".to_string(),
                    vec![],
                ),
                CheckError::new(
                    crate::errors::CheckErrorKind::UnknownConstructor {
                        identifier: "MissingCtor".to_string(),
                    },
                    "rendered constructor message may change".to_string(),
                    vec![],
                ),
            ],
            typed_nodes: 8,
            total_nodes: 10,
        };

        let report = FitnessReport::from_infer_result(&result);

        assert_eq!(report.components.names, 0.8);
        assert_eq!(
            report.unresolved_names,
            ["missing_value", "MissingCtor"],
            "diagnostic order and identifiers come from structured kinds, not messages"
        );
        assert_eq!(report.errors.len(), 2);
        assert!(report.score < 1.0);
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
            "(def {} good (lit {type: (t-prim {} i32)} 42)) \
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
        let r = score("(def {} x (lit {type: (t-prim {} i32)} 42))");
        assert!(r.total_nodes > 0, "should have visited some nodes");
        assert_eq!(
            r.typed_nodes, r.total_nodes,
            "all nodes should type-check in a correct program"
        );
    }

    #[test]
    fn check_program_fn_works() {
        let exprs =
            chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} i32)} 42))").unwrap();
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
            "(def {} f (app {} (var {} add) (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} bool)} true)))",
        )
        .unwrap();
        let r = check_ir_program(&exprs);
        assert!(!r.errors.is_empty(), "{r:?}");
        assert!(r.score < 1.0, "{r:?}");
        assert!(r.untyped_nodes > 0, "{r:?}");
    }

    /// chelis#833 / §C4.4: a duplicate `deftype` is a declaration-level error
    /// with NO runtime nodes (`total_nodes == 0`). Every coverage component is
    /// vacuously 1.0, so before the honesty cap the weighted sum read a
    /// perfect 1.0 while the `DuplicateDefinition` error was in the vector.
    #[test]
    fn declaration_only_error_forces_score_below_one_ir() {
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Foo () (variant {} Foo)) (deftype {} Foo () (variant {} Foo))",
        )
        .unwrap();
        let r = check_ir_program(&exprs);
        assert!(!r.errors.is_empty(), "duplicate deftype must report: {r:?}");
        assert_eq!(
            r.total_nodes, 0,
            "a declaration-only program has no runtime nodes: {r:?}"
        );
        // types is 0.0 (type-check failed), so the score is the residual
        // parse+structure+names weight only: at most 0.1+0.1+0.2 = 0.4, well
        // below both 1.0 and the 0.99 backstop ceiling.
        assert!(
            r.score < 0.5,
            "a declaration-level failure must score near 0.4, not near 1.0, got {}",
            r.score
        );
    }

    /// The non-IR `check_program` path must honor the same invariant.
    #[test]
    fn declaration_only_error_forces_score_below_one_non_ir() {
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Foo () (variant {} Foo)) (deftype {} Foo () (variant {} Foo))",
        )
        .unwrap();
        let r = check_program(&exprs);
        assert!(!r.errors.is_empty(), "{r:?}");
        assert!(r.score < 0.5, "got {}", r.score);
    }

    /// Positive parity: a well-formed declaration-only program (empty error
    /// vector) is unaffected by the honesty fix and still scores a clean 1.0.
    #[test]
    fn clean_declaration_only_program_still_scores_one() {
        let exprs = chelis_deep::parser::parse_str("(deftype {} Foo () (variant {} Foo))").unwrap();
        let r = check_ir_program(&exprs);
        assert!(r.errors.is_empty(), "{r:?}");
        assert!((r.score - 1.0).abs() < 1e-9, "got {}", r.score);
    }

    /// The cap must not flatten an already-honest sub-1.0 score: a program
    /// with runtime nodes and a type error keeps its coverage-based score
    /// (which is below the ceiling), proving the cap only rescues the
    /// degenerate perfect-sum case rather than overwriting calibration.
    #[test]
    fn honesty_cap_leaves_coverage_scored_failures_untouched() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} f (app {} (var {} add) (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} bool)} true)))",
        )
        .unwrap();
        let r = check_ir_program(&exprs);
        assert!(!r.errors.is_empty(), "{r:?}");
        assert!(r.total_nodes > 0, "{r:?}");
        assert!(
            r.score < HONEST_ERROR_CEILING,
            "coverage score for this program is below the ceiling, so the cap \
             did not manufacture it: got {}",
            r.score
        );
    }
}
