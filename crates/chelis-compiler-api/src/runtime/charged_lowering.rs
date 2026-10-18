//! The one place host evaluation lowers a program fragment to a tensor DAG
//! that it then runs as a single unit (spec/08-backends.md section 1.1).
//!
//! A unit runs every function it inlines inside one graph, so the
//! applications it performs never enter an evaluator frame. Each lowering
//! here records the deepest nesting of inlined applications it built
//! ([`chelis_ir::lower::recording_inline_depth`]) and hands it back as a
//! [`UnitDepth`], which the caller charges to the recursion budget for as
//! long as the unit runs. `runtime::tests::every_unit_lowering_is_charged`
//! scans the runtime sources outside this module and rejects the lowering
//! spellings it recognizes: a `chelis_ir` `lower` or `host` path or import
//! naming a function with the word `lower`, a glob or module import of
//! either, and `host_def_kernel` or `recording_inline_depth`.

use chelis_deep::ast::Expr;
use chelis_ir::dag::{Dag, TensorType};
use chelis_ir::host::{HostDefKernel, HostLoweringSession};
use chelis_ir::lower::{
    LowerDiagnostic, LoweredSubexprWithControls, SubexprLoweringContext, recording_inline_depth,
};
use chelis_unord::UnordMap;

use super::eval::EvalCallStackFrame;

/// The deepest nesting of applications a unit's lowering inlined: the levels
/// the unit holds while it runs.
#[must_use = "a unit's depth is charged for as long as the unit runs"]
#[derive(Clone, Copy, Debug)]
pub(super) struct UnitDepth(usize);

impl UnitDepth {
    /// Charge the unit's levels until the returned frame drops, or report
    /// why the unit cannot run within the budget.
    pub(super) fn charge(self) -> Result<EvalCallStackFrame, String> {
        EvalCallStackFrame::enter_weighted(self.0)
    }
}

/// The kernel the C lane emits for def `name`, or `None` for the host lane
/// (`chelis_ir::host::host_def_kernel`).
pub(super) fn def_kernel(
    session: &HostLoweringSession<'_>,
    name: &str,
) -> (Result<Option<HostDefKernel>, LowerDiagnostic>, UnitDepth) {
    let (kernel, depth) =
        recording_inline_depth(|| chelis_ir::host::host_def_kernel(session, name));
    (kernel, UnitDepth(depth))
}

/// `expr` lowered to a DAG
/// (`chelis_ir::lower::try_lower_subexpr_program_with_context`).
pub(super) fn subexpr(
    expr: &Expr,
    scoped_types: UnordMap<String, TensorType>,
    context: &SubexprLoweringContext,
) -> (Result<Dag, LowerDiagnostic>, UnitDepth) {
    let (dag, depth) = recording_inline_depth(|| {
        chelis_ir::lower::try_lower_subexpr_program_with_context(expr, scoped_types, context)
    });
    (dag, UnitDepth(depth))
}

/// `expr` lowered to a DAG with its runtime list checks
/// (`chelis_ir::lower::try_lower_subexpr_program_with_context_and_controls`).
pub(super) fn subexpr_with_controls(
    expr: &Expr,
    scoped_types: UnordMap<String, TensorType>,
    context: &SubexprLoweringContext,
) -> (
    Result<LoweredSubexprWithControls, LowerDiagnostic>,
    UnitDepth,
) {
    let (lowered, depth) = recording_inline_depth(|| {
        chelis_ir::lower::try_lower_subexpr_program_with_context_and_controls(
            expr,
            scoped_types,
            context,
        )
    });
    (lowered, UnitDepth(depth))
}
