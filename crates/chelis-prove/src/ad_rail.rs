//! WI-10 AD-as-verification-target rail: make the gradient (adjoint) graph
//! dispatchable like any other property.
//!
//! This is the last item of verification-stack Wave 2. It builds on the WI-3
//! graph-extraction producer ([`crate::graph_extract`]) and the WI-9
//! fitness-based dispatcher ([`crate::engine_registry`]). Where WI-3 turns a
//! FORWARD program into a box/range goal over one of its outputs, this rail
//! turns the GRADIENT of a program into a fan-out of box/range goals -- one per
//! gradient target -- so a "verified-bounded-sensitivities" property (Greeks
//! over an input region) discharges through the exact same dispatch path as any
//! other goal.
//!
//! ## What a gradient target is, and why it fans out
//!
//! `chelis_compiler_api::compiler::grad` lowers a program's combined
//! forward+backward DAG and returns
//! [`GradResult::grad_nodes_by_name`](chelis_compiler_api::schema::GradResult::grad_nodes_by_name):
//! a map from each requested `wrt` input NAME to the ROOT INDEX of that input's
//! gradient (a Greek = d output / d input) in the gradient DAG. That map is
//! exactly the `&BTreeMap<String, usize>` named-roots shape
//! [`crate::graph_extract::box_range_goal_from_wire_dag`] resolves an output
//! against, so the gradient case reuses the WI-3 core verbatim.
//!
//! A gradient over N targets becomes N SEPARATE box/range goals, NOT one goal
//! packing a vector of sensitivities. Each goal is a single scalar output --
//! one Greek -- addressed by that target's NAME -> root_index, with its own
//! requested output range. This is the Beacon agent's hard requirement: an
//! interval engine bounds ONE scalar output per goal, so a vector gradient must
//! be decomposed into per-target scalar goals. The gradient DAG is lowered
//! ONCE: every goal in the fan-out shares the SAME gradient-DAG content hash and
//! the same name-keyed input box, differing only in root index and output
//! range. This mirrors [`crate::graph_extract::box_range_goals_from_source`],
//! which fans a multi-output FORWARD program out the same way.
//!
//! ## Equal gradients still have named canonical roots
//!
//! Numerical equality does not imply root-index equality. The canonical
//! accumulation tree in `spec/06` §2.4 gives each forward value's adjoint its
//! own exact positive-zero base leaf. For example, d/dx and d/dy of
//! `mean(x + y)` are both `1/4`, and they share the same contribution tail, but
//! their final `add(+0, contribution)` nodes are distinct roots associated with
//! `x` and `y`. A later semantics-preserving pass may share structure where its
//! own contract permits that, so consumers must key goals by target NAME and
//! follow `grad_nodes_by_name`; they must not infer either equality or
//! distinctness from target count.
//!
//! ## The no-in-tree-fit path (what this wave actually lands)
//!
//! There is NO `BoxRange` engine in-tree: cvc5 fits only [`GoalShape::Smt`], and
//! Beacon's interval engine is out-of-tree. So each gradient goal, routed
//! through [`crate::engine_registry::DischargeRegistry::with_builtin_engines`],
//! is NO-FIT and takes the canonical no-fit path: [`Soundness::Untrusted`] with
//! an empty [`crate::discharge::QualifierSet`], which the WI-6 algebra renders
//! [`crate::composition::CompositeVerdict::Unsupported`] -- never green, never a
//! silent pass. This wave lands the PLUMBING (grad -> per-target goal ->
//! dispatch) plus the honest no-in-tree-fit outcome. The real verified-Greeks
//! discharge is Beacon's, out of tree; an in-tree mock engine proves the full
//! AD -> producer -> dispatch -> engine path without it (see the tests).
//!
//! ## Crate-boundary note
//!
//! Like WI-3, this reaches a real gradient DAG through the PUBLIC
//! [`chelis_compiler_api::compiler::grad`] API (source + output + wrt names in,
//! `WireDag` + `grad_nodes_by_name` out). It pulls no `chelis-ir` dependency
//! into this crate and changes no `chelis-compiler-api` API surface.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{GradRequest, GradResult, SourceKind};

use crate::discharge::{Discharge, IntervalBox, OutputRange};
use crate::engine_registry::DischargeRegistry;
use crate::graph_extract::{ExtractedGoal, GraphExtractError, box_range_goal_from_wire_dag};

/// The box/range output range for one gradient target, naming the `wrt` input
/// whose gradient (Greek) is being bounded plus the asserted `[lo, hi]`.
///
/// The `target` here is a `wrt` INPUT name (a key of
/// [`GradResult::grad_nodes_by_name`]), NOT a forward output name: the goal
/// bounds d(output) / d(target) over the input box.
#[derive(Debug, Clone, PartialEq)]
pub struct GradTargetRange {
    /// The `wrt` input name whose gradient this range bounds. Must be one of
    /// the `wrt_names` the gradient was taken with respect to.
    pub target: String,
    /// The asserted lower bound on the gradient at `target`.
    pub lo: f64,
    /// The asserted upper bound on the gradient at `target`.
    pub hi: f64,
}

/// A request to build the AD verification rail for one program: differentiate
/// `output_name` w.r.t. `wrt_names`, then bound each named gradient target over
/// `input_box`.
///
/// `target_ranges` carries one [`GradTargetRange`] per gradient target to fan
/// out into a goal. Every `target` named here must be one of `wrt_names`; a
/// `target` that is not a gradient target fails the whole fan-out with
/// [`AdRailError::UnknownGradTarget`] (no partial / wrong-root goal set).
#[derive(Debug, Clone, PartialEq)]
pub struct AdRailRequest {
    /// Surf or Deep source.
    pub source: String,
    /// Which surface `source` is written in.
    pub source_kind: SourceKind,
    /// The scalar output to differentiate (a forward output name).
    pub output_name: String,
    /// The inputs to differentiate with respect to (forward input names). Each
    /// becomes a gradient target = one Greek.
    pub wrt_names: Vec<String>,
    /// The shared input region every gradient goal is bounded over. Emitted
    /// name-sorted, like WI-3.
    pub input_box: IntervalBox,
    /// One output range per gradient target to fan out. Each `target` must be a
    /// `wrt` name.
    pub target_ranges: Vec<GradTargetRange>,
}

/// An error building the AD verification rail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdRailError {
    /// `chelis_compiler_api::compiler::grad` failed (an unknown output/wrt
    /// name, a non-differentiable op, a non-scalar output, a lowering error).
    /// Carries the joined diagnostic messages, so a non-differentiable-op
    /// rejection from the checked AD path surfaces verbatim rather than as a
    /// fabricated goal.
    #[error("gradient construction failed: {0}")]
    GradFailed(String),

    /// A requested `target` is not a gradient target of this program (not one
    /// of the `wrt_names`, so no gradient root index addresses it). Carries the
    /// missing target and the available gradient-target names (sorted) for
    /// diagnosis. The whole fan-out fails: a caller never gets a goal set with a
    /// silently-dropped or wrong-root target.
    #[error(
        "gradient target `{target}` is not a differentiated input; available targets: {available:?}"
    )]
    UnknownGradTarget {
        target: String,
        available: Vec<String>,
    },

    /// Building the per-target box/range goal failed at the WI-3 producer
    /// boundary: a non-current gradient `WireDag`, a non-finite float in the
    /// gradient DAG, or an ill-formed (inverted/NaN) output range. The gradient
    /// DAG flows through the same fail-closed boundary checks as a forward DAG.
    #[error("box/range goal construction failed: {0}")]
    GoalConstruction(#[from] GraphExtractError),
}

/// One gradient target's box/range goal, plus the gradient-DAG artifact it
/// addresses. A thin wrapper over [`ExtractedGoal`] that also records WHICH
/// gradient target (`wrt` name) the goal bounds, so the fan-out is
/// self-describing.
#[derive(Debug, Clone, PartialEq)]
pub struct GradGoal {
    /// The `wrt` input name this goal's gradient corresponds to.
    pub target: String,
    /// The box/range goal (with a populated [`crate::discharge::IrHandle`]
    /// addressing the gradient DAG by content hash + this target's root index)
    /// and the serialized exact-version gradient `WireDag` v6 bytes.
    pub extracted: ExtractedGoal,
}

/// Build the gradient DAG for `request` and fan it out into one box/range
/// [`Goal`](crate::discharge::Goal) per gradient target, each addressed by that
/// target's OWN root index.
///
/// Calls [`chelis_compiler_api::compiler::grad`] ONCE to lower the combined
/// forward+backward DAG, then -- for each [`GradTargetRange`] -- resolves the
/// target's gradient root by NAME via
/// [`GradResult::grad_nodes_by_name`](chelis_compiler_api::schema::GradResult::grad_nodes_by_name)
/// and delegates to [`box_range_goal_from_wire_dag`]. Every returned goal shares
/// the one gradient-DAG content hash (the DAG was lowered once) but carries a
/// distinct root index and its own output range. Goals are returned in the same
/// order as `request.target_ranges`.
///
/// A `target` that is not a gradient target fails the WHOLE fan-out with
/// [`AdRailError::UnknownGradTarget`]: no partial goal set, no goal addressed by
/// a wrong root.
pub fn grad_goals_from_request(request: &AdRailRequest) -> Result<Vec<GradGoal>, AdRailError> {
    let grad = compiler::grad(GradRequest {
        source_kind: request.source_kind,
        source: request.source.clone(),
        output_name: request.output_name.clone(),
        wrt_names: request.wrt_names.clone(),
        // The combined forward+backward DAG is what is content-addressed and
        // bounded; fuse is the gradient's own concern and does not change the
        // named gradient roots, so keep the API default.
        fuse: true,
    })
    .map_err(|err| {
        let messages: Vec<String> = err.errors.iter().map(|d| d.message.clone()).collect();
        AdRailError::GradFailed(messages.join("; "))
    })?;

    fan_out_grad_goals(&grad, &request.input_box, &request.target_ranges)
}

/// The pure fan-out core: given an already-lowered [`GradResult`], the shared
/// input box, and the per-target ranges, build one [`GradGoal`] per range.
///
/// Separated from [`grad_goals_from_request`] so the fan-out logic (and its
/// unknown-target / shared-hash / distinct-root invariants) can be exercised
/// over a constructed [`GradResult`] without re-lowering.
fn fan_out_grad_goals(
    grad: &GradResult,
    input_box: &IntervalBox,
    target_ranges: &[GradTargetRange],
) -> Result<Vec<GradGoal>, AdRailError> {
    let mut goals = Vec::with_capacity(target_ranges.len());
    for range in target_ranges {
        // Resolve the target to a gradient root by NAME before building the
        // goal, so an unknown target fails with the AD-rail-specific diagnostic
        // (listing the gradient targets) rather than the generic
        // `UnknownOutput` (which would list gradient ROOT names, not wrt names).
        if !grad.grad_nodes_by_name.contains_key(&range.target) {
            let mut available: Vec<String> = grad.grad_nodes_by_name.keys().cloned().collect();
            available.sort();
            return Err(AdRailError::UnknownGradTarget {
                target: range.target.clone(),
                available,
            });
        }

        // Each goal addresses ONE scalar output -- this target's gradient root
        // -- by name, over the shared input box. box_range_goal_from_wire_dag
        // resolves `output.output` against grad_nodes_by_name, validates the
        // gradient DAG's schema version + finiteness at the boundary, and
        // rejects an inverted/NaN range as IllFormedGoal.
        let extracted = box_range_goal_from_wire_dag(
            &grad.dag,
            &grad.grad_nodes_by_name,
            input_box.clone(),
            OutputRange {
                output: range.target.clone(),
                lo: range.lo,
                hi: range.hi,
            },
        )?;

        goals.push(GradGoal {
            target: range.target.clone(),
            extracted,
        });
    }
    Ok(goals)
}

/// Route every gradient goal in `goals` through `registry`, pairing each with
/// its dispatch [`Discharge`].
///
/// This is the end of the rail: each per-target box/range
/// [`Goal`](crate::discharge::Goal) dispatches through the WI-9
/// [`DischargeRegistry`] exactly like any other goal. With only
/// the in-tree engines registered ([`DischargeRegistry::with_builtin_engines`])
/// there is no `BoxRange` fit, so each discharge is the canonical no-fit
/// [`Soundness::Untrusted`]/empty result (rendering
/// [`crate::composition::CompositeVerdict::Unsupported`]). Registering an
/// out-of-tree [`crate::discharge::DischargeEngine`] (Beacon's interval engine)
/// routes each goal to it instead. Results are returned in the same order as
/// `goals`.
pub fn dispatch_grad_goals(
    registry: &DischargeRegistry,
    goals: &[GradGoal],
    timeout_ms: u64,
) -> Vec<(String, Discharge)> {
    goals
        .iter()
        .map(|g| {
            (
                g.target.clone(),
                registry.dispatch(&g.extracted.goal, timeout_ms),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;
