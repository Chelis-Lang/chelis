//! Portable ownership graphs and calculations.

pub mod classify;
pub mod dag;
pub mod error;
pub mod ir;
pub mod last_use;
pub mod locations;
pub mod lower;
pub mod manifest;
pub mod render;
pub mod storage;
pub mod verify;

pub use error::OwnershipError;
pub use ir::{HostSiteId, HostSiteKind};
#[cfg(test)]
pub use locations::counting_site_lookup_entries;

/// Sealed verifier-derived live-byte upper bound.
/// Verification composes local owner costs through the direct-call graph and classifies runtime-dependent or positively recursive storage explicitly.
/// `Unbounded` also covers a statically known total beyond u64: no finite bound is representable.
/// And each allocation keeps its own checked admission, so an oversized summary never refuses a program.
/// The oracle handoff remains an integration step with the reuse proof because physical reused-slot capacity can exceed a tensor's logical shape.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveByteBound {
    Exact(u64),
    Unknown,
    Unbounded,
}

impl LiveByteBound {
    fn maximum(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unbounded, _) | (_, Self::Unbounded) => Self::Unbounded,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Exact(lhs), Self::Exact(rhs)) => Self::Exact(lhs.max(rhs)),
        }
    }
}

#[derive(Debug)]
pub struct LiveSetBound {
    pub max_live_heap_owners: usize,
    pub max_live_bytes: LiveByteBound,
}

impl LiveSetBound {
    pub fn max_live_heap_owners(&self) -> usize {
        self.max_live_heap_owners
    }
}

#[cfg(test)]
mod tests;
