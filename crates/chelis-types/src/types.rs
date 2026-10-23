//! Hosted compilation targets and lane policy.

use serde::{Deserialize, Serialize};

/// Compilation/evaluation target. Determines which backend's capability set
/// is used for realizability inference and manifest computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Target {
    /// The Rust tensor evaluator (stores f64, supports all prims).
    Eval,
    /// The C backend (f32/bool/bf16/f16/i32/i64; rejects f64).
    C,
    /// The HIP/ROCm backend.
    Hip,
    /// The Metal backend.
    Metal,
}

/// Lane assignment for a top-level root. Determined by realizability inference
/// from per-builtin declarations, per-tag declarations, and def-level
/// precision checks against backend capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lane {
    /// Realized through the tensor DAG path (lowering → codegen).
    Tensor,
    /// Realized through the host runtime path (interpreter or host C emission).
    Host,
}
