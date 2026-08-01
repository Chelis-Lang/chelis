//! Root manifest types for issue #912.
//!
//! The manifest is the single source of truth for what roots a program has,
//! which lane realizes each, and what inputs each needs. Computed after
//! realizability inference, attached via `ManifestedProgram`.

use std::collections::BTreeSet;

use chelis_deep::ast::Expr;

use crate::CheckedProgram;
use crate::types::{Lane, Target};

/// A single root entry in the manifest.
#[derive(Debug, Clone)]
pub struct RootEntry {
    /// The rendered name (dotted for tuple/ADT components, e.g. "result.0").
    pub name: String,
    /// The originating def name (for realizability lookup when expanded).
    pub def_name: String,
    /// The declared type expression.
    pub ty: Expr,
    /// Which lane realizes this root.
    pub lane: Lane,
    /// Free tensor-typed variables this root needs at runtime.
    pub required_inputs: BTreeSet<String>,
    /// Why this def routes to its lane (empty for Tensor-lane defs).
    pub reasons: Vec<crate::manifest::HostReason>,
}

/// Why a def routes to the Host lane. Carried on each entry for #883
/// diagnostic emission without reaching back into the inference result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostReason {
    HostOnlyBuiltin { name: String },
    ScalarTypedOp { builtin: String },
    PrecisionExceedsCapability { prim: crate::types::Prim },
    StructuralForm { tag: String },
    TransitiveCaller { callee: String },
    UnrecognizedTag { tag: String },
}

/// The root manifest for a checked program against a specific target.
#[derive(Debug, Clone)]
pub struct RootManifest {
    pub entries: Vec<RootEntry>,
}

impl RootManifest {
    /// Does this program have an observation boundary?
    /// True iff entries is non-empty. An all-[05-UNS-1] program still
    /// requires main (prints diagnostics). Empty = object.
    pub fn requires_main(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Names of roots that should be realized through the tensor-DAG path.
    pub fn tensor_root_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.lane == Lane::Tensor)
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Names of roots that should be realized through the host path.
    pub fn host_root_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.lane == Lane::Host)
            .map(|e| e.name.as_str())
            .collect()
    }
}

/// A checked program with its root manifest attached. Enforces the phase
/// boundary: you cannot observe roots until the manifest is computed.
/// Carries the target it was computed for.
#[derive(Debug, Clone)]
pub struct ManifestedProgram {
    pub checked: CheckedProgram,
    pub manifest: RootManifest,
    pub target: Target,
}

impl ManifestedProgram {
    pub fn new(checked: CheckedProgram, manifest: RootManifest, target: Target) -> Self {
        Self {
            checked,
            manifest,
            target,
        }
    }
}
