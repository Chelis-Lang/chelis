//! The `List` heap representation, and the one view every reader goes
//! through.
//!
//! `chelis_list` lives in its own module so that its backing store is
//! unreachable from `lib.rs`. Rust privacy is by module subtree, and the
//! crate root is not a descendant of this module, so a reader in
//! `lib.rs` that tries to touch the `Vec` directly does not compile.
//! That is the whole reason this module exists: the compiler, rather
//! than a reviewer, is what keeps every reader on [`chelis_list::live`].
//!
//! [05-OP-44] publishes the type as the opaque
//! `typedef struct chelis_list chelis_list;`, so the layout is the
//! runtime's alone. Emitted C reaches a list only through the
//! `chelis_list_*` entry points, and the interpreter has its own
//! `RuntimeValue::List(Vec<RuntimeValue>)`; neither observes this
//! struct.
//!
//! Since chelis#2334 the store carries an exclusive offset, so `live()`
//! is `items[head..]` rather than the whole `Vec`. That is the reason
//! the hiding matters rather than a tidiness: one un-migrated read
//! would report the wrong length, index the wrong element, and copy
//! released slots into handles that double-release at finalize, all
//! silently in a release build and loudly only under the
//! `ownership-ledger` feature.

use crate::ownership_ledger;
use crate::{chelis_value, HeapHeader};

#[repr(C)]
pub struct chelis_list {
    /// Deliberately reachable from `lib.rs`: retain, release and the
    /// exclusivity checks of the in-place mutators read the strong
    /// count. The backing store below is the field this module hides,
    /// and hiding it is unaffected by exposing the header.
    pub(crate) header: HeapHeader,
    /// The allocation. `items[..head]` has been retired by a skip and
    /// holds handles this list released; `items[head..]` is the list.
    items: Vec<chelis_value>,
    /// The first live slot (chelis#2334).
    ///
    /// An exclusive offset inside the one allocation, never a window
    /// onto a buffer a second handle can see: it moves only in
    /// `chelis_list_drop_owned`, only at strong-owner count one, and
    /// only under an ownership `Move` the verifier proved. Any other
    /// holder forces that entry point's cloning arm, which builds a
    /// fresh list at head zero. [05-OP-44]'s heap-kind universe is
    /// therefore untouched: there is no second kind and no shared
    /// buffer, and this list still owns each of its children exactly
    /// once.
    head: usize,
}

impl chelis_list {
    /// Build a list owning `items`, with one strong owner.
    ///
    /// The ledger byte count is the caller's, because it is recorded
    /// against the boxed pointer this value has not got yet.
    pub(crate) fn new(items: Vec<chelis_value>) -> Self {
        Self {
            header: HeapHeader::new(ownership_ledger::Kind::List),
            items,
            head: 0,
        }
    }

    /// Every element the list holds, in order.
    ///
    /// The single view of the backing store, and the reason the store is
    /// private. A reader that bypassed it would be reading slots this
    /// list does not own.
    pub(crate) fn live(&self) -> &[chelis_value] {
        &self.items[self.head..]
    }

    /// Slots the allocation holds, occupied or not.
    ///
    /// This is the quantity the ownership ledger records, so it is
    /// deliberately the buffer's capacity rather than `live().len()`.
    pub(crate) fn buffer_capacity(&self) -> usize {
        self.items.capacity()
    }

    /// Append one value the list takes ownership of.
    ///
    /// Callers own the retain: this stores the value as handed over.
    pub(crate) fn push(&mut self, value: chelis_value) {
        self.items.push(value);
    }

    /// Advance the live window past `count` elements.
    ///
    /// The caller has already released the first `count` elements of
    /// `live()`, so the retired slots hold handles this list no longer
    /// owns, and nothing reads them again: `live()` excludes them, the
    /// finalizer walks `live()`, and a compaction discards them
    /// without looking.
    ///
    /// Releasing before the advance rather than after it is what keeps
    /// every read inside `live()`, so this module exposes no accessor
    /// onto a retired slot and the one-view invariant holds without an
    /// exception. The window in which the live view still names an
    /// already-released handle is the one `finalize_heap` has held
    /// since [05-OP-44] was written, because it too releases each
    /// child while the list still lists it.
    ///
    /// A count above the live length advances to the end, which is how
    /// [05-OP-32]'s "a count above length yields the empty List"
    /// arrives here.
    pub(crate) fn advance_head(&mut self, count: usize) {
        debug_assert!(self.head <= self.items.len(), "head past the allocation");
        self.head += count.min(self.items.len() - self.head);
    }

    /// Reclaim the retired prefix once it is larger than the live
    /// window.
    ///
    /// This is what bounds the waste an offset would otherwise leave.
    /// Without it, `skip(xs, n - 1)` kept alive holds `n` slots to
    /// serve one element, which the cloning path it replaces did not:
    /// that path allocated the suffix and freed the operand.
    ///
    /// The bound is over the **allocation**, not merely the length,
    /// which is why this rebuilds the buffer at the live length rather
    /// than draining in place. `Vec::drain` keeps capacity, so a
    /// compaction that drained would leave `buffer_capacity()` at the
    /// original figure for ever, and that is the exact quantity the
    /// ownership ledger multiplies by `LEDGER_VALUE_SLOT_BYTES`. The
    /// claim would then be false of the number the instrument reports:
    /// a one-element list would hold a million slots and the ledger
    /// would say so.
    ///
    /// The cost is a reallocation where a drain would memmove, both of
    /// the live length, and the amortisation is unchanged. A
    /// compaction copies `live_len` slots and resets `head` to zero,
    /// and `head` must then grow past the *new* live length before
    /// another can run, so a cursor walking to the end copies O(n)
    /// slots in total rather than O(n) per step.
    ///
    /// Returns whether it moved anything, which is what makes the
    /// amortisation above testable rather than argued.
    pub(crate) fn compact_retired_prefix(&mut self) -> bool {
        debug_assert!(self.head <= self.items.len(), "head past the allocation");
        let live_len = self.items.len() - self.head;
        if self.head <= live_len {
            return false;
        }
        // Exactly the live length, matching the exact-capacity
        // discipline `chelis_list_append` already keeps.
        self.items = self.items[self.head..].to_vec();
        self.head = 0;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::chelis_list;
    use crate::{chelis_value, CHELIS_VALUE_UNIT};

    /// A unit value carries no heap handle, so these tests exercise the
    /// offset arithmetic without owing a release.
    fn unit() -> chelis_value {
        chelis_value {
            tag: CHELIS_VALUE_UNIT,
            reserved: [0; 7],
            payload: unsafe { std::mem::zeroed() },
        }
    }

    fn list_of(len: usize) -> chelis_list {
        chelis_list::new(vec![unit(); len])
    }

    /// A list whose allocation is larger than its contents, which is
    /// what every push-built list looks like.
    ///
    /// `vec![v; n]` allocates exactly, so a fixture built that way
    /// cannot tell a spurious reallocation from a no-op: shrinking a
    /// `Vec` whose capacity already equals its length does nothing.
    fn list_with_slack(capacity: usize, len: usize) -> chelis_list {
        let mut items = Vec::with_capacity(capacity);
        items.resize(len, unit());
        chelis_list::new(items)
    }

    #[test]
    fn live_excludes_the_retired_prefix_and_advance_clamps_at_the_end() {
        let mut list = list_of(4);
        assert_eq!(list.live().len(), 4);
        list.advance_head(1);
        assert_eq!(list.live().len(), 3);
        // [05-OP-32]: a count above the length yields the empty List
        // rather than reading past the end.
        list.advance_head(usize::MAX);
        assert!(list.live().is_empty());
    }

    /// The compaction fires on exactly the stated condition, and not one
    /// step earlier. Without the negative half a rule that compacted on
    /// every skip would pass the positive one.
    #[test]
    fn compaction_fires_only_once_the_dead_prefix_exceeds_the_live_window() {
        let mut list = list_of(4);
        list.advance_head(2);
        assert!(
            !list.compact_retired_prefix(),
            "an equal split is not yet wasteful"
        );
        assert_eq!(list.head, 2, "nothing moved");
        list.advance_head(1);
        assert!(list.compact_retired_prefix(), "three dead to one live is");
        assert_eq!(list.head, 0);
        assert_eq!(list.live().len(), 1);
    }

    /// A compaction gives the retired capacity back, so the bound this
    /// module claims is true of the allocation and not merely of the
    /// length.
    ///
    /// Evidentiary status: REGRESSION TEST, and it replaces an earlier
    /// assertion that pinned the defect. That one asserted the capacity
    /// was **unchanged** across a compaction, which is what
    /// `Vec::drain` does, so it would have gone green on the wrong
    /// behaviour and red on this one. Red-team round 1 measured what it
    /// was protecting: `skip(n - 1)` on a 1,000,000-element list left a
    /// one-element list holding a 1,000,000-slot buffer, 24 MB by the
    /// ledger's own formula, where the cloning path freed it.
    #[test]
    fn compaction_releases_the_retired_capacity() {
        let mut list = list_of(8);
        assert!(list.buffer_capacity() >= 8);
        list.advance_head(7);
        assert!(list.compact_retired_prefix());
        assert_eq!(list.live().len(), 1);
        assert_eq!(
            list.buffer_capacity(),
            1,
            "the allocation is the live length, not the original"
        );
    }

    /// The negative half: a compaction that does not fire must not
    /// reallocate, or every skip below the threshold would pay for one.
    ///
    /// The fixture carries slack deliberately. Round 1 found the
    /// earlier version blind: it used an exactly-sized list, and
    /// shrinking a `Vec` whose capacity already equals its length is a
    /// no-op, so an unconditional `shrink_to_fit` ahead of the
    /// threshold check -- the most likely over-correction anyone would
    /// write -- passed it. The first assertion guards the fixture
    /// itself, so this cannot quietly become vacuous again.
    #[test]
    fn a_compaction_that_does_not_fire_keeps_the_allocation() {
        let mut list = list_with_slack(8, 6);
        let before = list.buffer_capacity();
        assert!(
            before > list.live().len(),
            "the fixture must carry slack or the assertion below is vacuous"
        );
        list.advance_head(2);
        assert!(!list.compact_retired_prefix());
        assert_eq!(list.buffer_capacity(), before);
    }

    /// The amortisation claim, executed rather than argued: a cursor that
    /// advances one element at a time over `n` moves O(n) slots in total,
    /// not O(n) per step, and the dead prefix never exceeds the live
    /// window.
    ///
    /// Evidentiary status: REGRESSION TEST for the compaction rule.
    /// Compacting on every non-zero head instead makes `moved` quadratic
    /// (523,776 at n = 1024 against the 2n bound of 2,048), and dropping
    /// the compaction entirely fails the waste bound below.
    #[test]
    fn a_single_step_walk_moves_a_linear_number_of_slots() {
        const N: usize = 1024;
        let mut list = list_of(N);
        let mut moved = 0usize;
        let mut compactions = 0usize;
        while !list.live().is_empty() {
            list.advance_head(1);
            let live_before = list.items.len() - list.head;
            if list.compact_retired_prefix() {
                moved += live_before;
                compactions += 1;
            }
            assert!(
                list.head <= list.items.len() - list.head,
                "the dead prefix never exceeds the live window"
            );
        }
        assert!(
            moved <= 2 * N,
            "a full walk moved {moved} slots, above the 2n bound of {}",
            2 * N
        );
        assert!(
            compactions <= N.ilog2() as usize + 2,
            "a full walk compacted {compactions} times"
        );
    }
}
