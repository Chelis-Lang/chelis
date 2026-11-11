//! Native scope identity for exact storage capacity calculations.

use chelis_ir_core::capacity_key::CapacityScope;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CAPACITY_SCOPE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn fresh_capacity_scope() -> CapacityScope {
    let scope = NEXT_CAPACITY_SCOPE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("capacity-scope identity space exhausted");
    CapacityScope::from_identity(scope)
}
