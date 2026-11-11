//! Native checked-program adapter for optional library traces.

use crate::lower::LoweredLibrary;
use chelis_ir_core::lower::LowerDiagnostic;
use chelis_ir_core::lowering_trace::LoweringTrace;
use chelis_types::CheckedProgram;

/// Run ordinary library lowering with explicit snapshot collection.
/// Failure returns the ordinary diagnostic, never a partial successful trace.
pub fn try_lower_program_to_library_with_trace(
    program: &CheckedProgram,
) -> Result<(LoweredLibrary, LoweringTrace), LowerDiagnostic> {
    crate::lower::try_lower_program_to_library_with_trace(program)
}
