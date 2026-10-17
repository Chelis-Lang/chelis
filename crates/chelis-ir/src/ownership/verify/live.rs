//! The live owner set one verification path carries.
//!
//! The per-operation live bounds read the heap-owner count and the live-byte
//! bound of the whole set, and every move asks whether a live borrow still
//! names the moved owner. Recomputing those from the set at each operation
//! makes verification quadratic in a body whose live set grows with it
//! (chelis#2330). [`LiveSet`] keeps them as running totals and a borrow index,
//! updated on each insert and remove, so each read is constant or logarithmic.
//! Its owner set is private to this module: every whole-set traversal goes
//! through a method here, and the test build counts each entry it visits.

use std::collections::{BTreeMap, BTreeSet};

use super::super::LiveByteBound;
use super::super::error::OwnershipError;
use super::super::ir::{OwnerId, OwnerOrigin, Unit};

#[cfg(test)]
thread_local! {
    static LIVE_ENTRY_VISITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static REFERENCE_SCANS_COUNTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Live-set entries visited on this thread by whole-set traversals.
#[cfg(test)]
pub(in crate::ownership) fn live_entry_visits() -> u64 {
    LIVE_ENTRY_VISITS.with(std::cell::Cell::get)
}

/// Count the reference full-scan cross-checks as live-set visits while `run`
/// executes: the negative control that a per-operation rescan is visible to
/// [`live_entry_visits`].
#[cfg(test)]
pub(in crate::ownership) fn with_reference_scans_counted<T>(run: impl FnOnce() -> T) -> T {
    REFERENCE_SCANS_COUNTED.with(|flag| flag.set(true));
    let result = run();
    REFERENCE_SCANS_COUNTED.with(|flag| flag.set(false));
    result
}

#[cfg(test)]
fn visited(entries: usize) {
    LIVE_ENTRY_VISITS.with(|cell| cell.set(cell.get() + entries as u64));
}

#[cfg(not(test))]
fn visited(_entries: usize) {}

/// One owner's contribution to the live aggregates.
#[derive(Debug)]
struct OwnerContribution {
    /// Owned heap owners are the only ones counted and costed.
    heap_owned: bool,
    /// The owner's byte cost, or `None` when computing it is an error.
    /// Evaluated only for owned heap owners, the only ones a live-byte read
    /// costs.
    bytes: Option<LiveByteBound>,
    borrowed_from: Option<OwnerId>,
}

/// Every owner's contribution in one unit, derived once per verification.
#[derive(Debug)]
pub(super) struct UnitOwnerFacts<'a> {
    unit: &'a Unit,
    owners: BTreeMap<OwnerId, OwnerContribution>,
}

impl<'a> UnitOwnerFacts<'a> {
    pub(super) fn new(unit: &'a Unit) -> Self {
        let owners = unit
            .owners
            .iter()
            .map(|(owner, info)| {
                let heap_owned = info.origin == OwnerOrigin::Owned && info.class.is_heap();
                let bytes = if heap_owned {
                    super::owner_byte_cost(unit, *owner).ok()
                } else {
                    Some(LiveByteBound::Exact(0))
                };
                let borrowed_from = match info.origin {
                    OwnerOrigin::BorrowedFrom(source) => Some(source),
                    OwnerOrigin::Owned | OwnerOrigin::ExternalBorrow => None,
                };
                (
                    *owner,
                    OwnerContribution {
                        heap_owned,
                        bytes,
                        borrowed_from,
                    },
                )
            })
            .collect();
        Self { unit, owners }
    }
}

/// A live owner set with its aggregates kept as running totals.
#[derive(Debug)]
pub(super) struct LiveSet<'a> {
    facts: &'a UnitOwnerFacts<'a>,
    owners: BTreeSet<OwnerId>,
    heap_owned: usize,
    /// Sum of the live exact byte costs. It cannot overflow: it would take
    /// more than 2^64 live owners.
    exact_bytes: u128,
    unknown_bytes: usize,
    unbounded_bytes: usize,
    /// Live owned heap owners whose byte cost is an error, in owner order.
    errored_bytes: BTreeSet<OwnerId>,
    /// Live borrows by the owner they borrow from.
    borrows: BTreeMap<OwnerId, BTreeSet<OwnerId>>,
}

impl Clone for LiveSet<'_> {
    fn clone(&self) -> Self {
        visited(self.owners.len());
        Self {
            facts: self.facts,
            owners: self.owners.clone(),
            heap_owned: self.heap_owned,
            exact_bytes: self.exact_bytes,
            unknown_bytes: self.unknown_bytes,
            unbounded_bytes: self.unbounded_bytes,
            errored_bytes: self.errored_bytes.clone(),
            borrows: self.borrows.clone(),
        }
    }
}

/// Two paths agree when they carry the same owners; the aggregates are a
/// function of the owners.
impl PartialEq for LiveSet<'_> {
    fn eq(&self, other: &Self) -> bool {
        visited(self.owners.len().min(other.owners.len()));
        self.owners == other.owners
    }
}

impl<'a> LiveSet<'a> {
    pub(super) fn new(facts: &'a UnitOwnerFacts<'a>) -> Self {
        Self {
            facts,
            owners: BTreeSet::new(),
            heap_owned: 0,
            exact_bytes: 0,
            unknown_bytes: 0,
            unbounded_bytes: 0,
            errored_bytes: BTreeSet::new(),
            borrows: BTreeMap::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.owners.len()
    }

    pub(super) fn contains(&self, owner: &OwnerId) -> bool {
        self.owners.contains(owner)
    }

    /// The live owners in ascending order.
    pub(super) fn iter(&self) -> impl Iterator<Item = OwnerId> + '_ {
        self.owners.iter().map(|owner| {
            visited(1);
            *owner
        })
    }

    pub(super) fn insert(&mut self, owner: OwnerId) -> bool {
        if !self.owners.insert(owner) {
            return false;
        }
        let contribution = &self.facts.owners[&owner];
        if contribution.heap_owned {
            self.heap_owned += 1;
            match &contribution.bytes {
                Some(LiveByteBound::Exact(bytes)) => self.exact_bytes += u128::from(*bytes),
                Some(LiveByteBound::Unknown) => self.unknown_bytes += 1,
                Some(LiveByteBound::Unbounded) => self.unbounded_bytes += 1,
                None => {
                    self.errored_bytes.insert(owner);
                }
            }
        }
        if let Some(source) = contribution.borrowed_from {
            self.borrows.entry(source).or_default().insert(owner);
        }
        true
    }

    pub(super) fn remove(&mut self, owner: &OwnerId) -> bool {
        if !self.owners.remove(owner) {
            return false;
        }
        let contribution = &self.facts.owners[owner];
        if contribution.heap_owned {
            self.heap_owned -= 1;
            match &contribution.bytes {
                Some(LiveByteBound::Exact(bytes)) => self.exact_bytes -= u128::from(*bytes),
                Some(LiveByteBound::Unknown) => self.unknown_bytes -= 1,
                Some(LiveByteBound::Unbounded) => self.unbounded_bytes -= 1,
                None => {
                    self.errored_bytes.remove(owner);
                }
            }
        }
        if let Some(source) = contribution.borrowed_from
            && let Some(borrows) = self.borrows.get_mut(&source)
        {
            borrows.remove(owner);
            if borrows.is_empty() {
                self.borrows.remove(&source);
            }
        }
        true
    }

    /// The lowest live owner borrowed from `source`.
    pub(super) fn first_borrow_of(&self, source: OwnerId) -> Option<OwnerId> {
        let borrow = self
            .borrows
            .get(&source)
            .and_then(|borrows| borrows.first().copied());
        #[cfg(test)]
        assert_eq!(
            borrow,
            self.reference_scan()
                .find(|candidate| {
                    self.facts.unit.owners[candidate].origin == OwnerOrigin::BorrowedFrom(source)
                })
                .copied(),
            "borrow index diverged from the live set"
        );
        borrow
    }

    /// Live owned heap owners.
    pub(super) fn heap_owned_count(&self) -> usize {
        #[cfg(test)]
        assert_eq!(
            self.heap_owned,
            self.reference_scan()
                .filter(|owner| self.facts.owners[owner].heap_owned)
                .count(),
            "running heap-owner count diverged from the live set"
        );
        self.heap_owned
    }

    /// The live-byte bound of the whole set.
    pub(super) fn byte_cost(&self) -> Result<LiveByteBound, OwnershipError> {
        self.byte_cost_without(&BTreeSet::new())
    }

    /// The live-byte bound of the set without `excluded`, which need not be
    /// live.
    ///
    /// The result equals summing each live owned heap owner's cost in owner
    /// order with [`super::add_live_byte_bounds`]: the first owner whose cost
    /// is an error is the error; any unbounded cost is unbounded; otherwise
    /// an exact total beyond u64 is unbounded and any unknown cost is unknown.
    /// The one order-dependent case, an unknown cost together with an exact
    /// total beyond u64, keeps that sum's order by folding the set.
    pub(super) fn byte_cost_without(
        &self,
        excluded: &BTreeSet<OwnerId>,
    ) -> Result<LiveByteBound, OwnershipError> {
        let result = self.summed_byte_cost(excluded);
        #[cfg(test)]
        assert_eq!(
            result,
            self.reference_byte_cost(excluded),
            "running live-byte bound diverged from the live set"
        );
        result
    }

    fn summed_byte_cost(
        &self,
        excluded: &BTreeSet<OwnerId>,
    ) -> Result<LiveByteBound, OwnershipError> {
        if let Some(owner) = self
            .errored_bytes
            .iter()
            .find(|owner| !excluded.contains(owner))
        {
            // The same computation that failed when the facts were derived.
            return super::owner_byte_cost(self.facts.unit, *owner);
        }
        let mut exact_bytes = self.exact_bytes;
        let mut unknown_bytes = self.unknown_bytes;
        let mut unbounded_bytes = self.unbounded_bytes;
        for owner in excluded.iter().filter(|owner| self.owners.contains(owner)) {
            let contribution = &self.facts.owners[owner];
            if !contribution.heap_owned {
                continue;
            }
            match &contribution.bytes {
                Some(LiveByteBound::Exact(bytes)) => exact_bytes -= u128::from(*bytes),
                Some(LiveByteBound::Unknown) => unknown_bytes -= 1,
                Some(LiveByteBound::Unbounded) => unbounded_bytes -= 1,
                // An errored owner is held in `errored_bytes`, never in the
                // totals, and the error search above already passed over an
                // excluded one: like the rescan, it contributes nothing.
                None => {}
            }
        }
        if unbounded_bytes > 0 {
            return Ok(LiveByteBound::Unbounded);
        }
        match u64::try_from(exact_bytes) {
            Ok(exact) if unknown_bytes == 0 => Ok(LiveByteBound::Exact(exact)),
            Ok(_) => Ok(LiveByteBound::Unknown),
            Err(_) if unknown_bytes == 0 => Ok(LiveByteBound::Unbounded),
            Err(_) => Ok(self.ordered_byte_sum(excluded)),
        }
    }

    /// Sum the live owned heap owners' costs in owner order. Only reached
    /// when no cost is an error or unbounded.
    fn ordered_byte_sum(&self, excluded: &BTreeSet<OwnerId>) -> LiveByteBound {
        self.iter()
            .filter(|owner| !excluded.contains(owner))
            .filter_map(|owner| {
                let contribution = &self.facts.owners[&owner];
                contribution
                    .heap_owned
                    .then(|| contribution.bytes.expect("no errored cost"))
            })
            .fold(LiveByteBound::Exact(0), super::add_live_byte_bounds)
    }

    /// The owners, for a test-build cross-check that the running totals
    /// match a full scan. The scan counts as visits only inside
    /// [`with_reference_scans_counted`].
    #[cfg(test)]
    fn reference_scan(&self) -> impl Iterator<Item = &OwnerId> + '_ {
        let counted = REFERENCE_SCANS_COUNTED.with(std::cell::Cell::get);
        self.owners.iter().inspect(move |_| {
            if counted {
                visited(1);
            }
        })
    }

    /// The live-byte bound by a full scan, as verification computed it before
    /// the running totals.
    #[cfg(test)]
    fn reference_byte_cost(
        &self,
        excluded: &BTreeSet<OwnerId>,
    ) -> Result<LiveByteBound, OwnershipError> {
        let unit = self.facts.unit;
        let mut result = LiveByteBound::Exact(0);
        for owner in self
            .reference_scan()
            .filter(|owner| !excluded.contains(owner))
        {
            let info = unit
                .owners
                .get(owner)
                .ok_or_else(|| super::incomplete(unit, *owner, "metadata"))?;
            if info.origin != OwnerOrigin::Owned || !info.class.is_heap() {
                continue;
            }
            result = super::add_live_byte_bounds(result, super::owner_byte_cost(unit, *owner)?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use chelis_types::types::Prim;

    use super::{LiveSet, UnitOwnerFacts};
    use crate::dag::{DimInfo, TensorType};
    use crate::host_type_state::ConcreteHostType;
    use crate::ownership::LiveByteBound;
    use crate::ownership::classify::{Placement, classify};
    use crate::ownership::error::OwnershipError;
    use crate::ownership::ir::{
        BlockId, OwnerId, OwnerInfo, OwnerOrigin, ScheduleState, Unit, UnitId, UnitKind,
    };

    fn tensor(elements: usize, precision: Prim) -> OwnerInfo {
        let ty = ConcreteHostType::Tensor(TensorType {
            dims: vec![DimInfo::Lit(elements)],
            precision,
        });
        OwnerInfo {
            class: classify(&ty, Placement::Value).unwrap(),
            ty,
            placement: Placement::Value,
            origin: OwnerOrigin::Owned,
            names: Vec::new(),
            span_id: None,
        }
    }

    #[test]
    fn excluding_an_owner_whose_byte_cost_errors_drops_its_error() {
        // Verification never costs a carry while an errored owner is live:
        // the full-set read before it fails first. The running totals still
        // match the rescan when a caller excludes one.
        let unit = Unit {
            id: UnitId(0),
            name: "roots".into(),
            kind: UnitKind::Roots,
            schedule: ScheduleState::Phase2ScopeExit,
            callable_body: None,
            entry: BlockId(0),
            blocks: Vec::new(),
            owners: BTreeMap::from([
                (OwnerId(0), tensor(4, Prim::F8e4m3)),
                (OwnerId(1), tensor(4, Prim::F32)),
            ]),
            captured_globals: BTreeMap::new(),
        };
        let facts = UnitOwnerFacts::new(&unit);
        let mut live = LiveSet::new(&facts);
        live.insert(OwnerId(0));
        live.insert(OwnerId(1));
        assert!(matches!(
            live.byte_cost(),
            Err(OwnershipError::LiveByteBoundDType { owner: 0, .. })
        ));
        assert_eq!(
            live.byte_cost_without(&BTreeSet::from([OwnerId(0)])),
            Ok(LiveByteBound::Exact(16))
        );
    }
}
