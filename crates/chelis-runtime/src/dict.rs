//! The `Dict` heap representation: insertion-ordered entries with a key
//! index beside them.
//!
//! `chelis_dict` lives in its own module for the reason `chelis_list` does:
//! Rust privacy is by module subtree, so a reader in `lib.rs` cannot reach
//! the slot store and has to go through [`chelis_dict::entries`] and the
//! position handles [`chelis_dict::find`] returns. The store holds holes and
//! an index that a direct read would misinterpret, and the compiler rather
//! than a reviewer is what keeps every reader on the view.
//!
//! [05-OP-32] fixes what a dictionary is: keys are `string`, `bool`, or a
//! signed-integer scalar; equality is the exact kind, dtype and stored value;
//! a later duplicate replaces the value at the existing insertion position and
//! a new key appends. That rule says nothing about how a key is found, and
//! finding it by scanning every entry made `dict_of`, and every `dict_insert`
//! or `dict_remove` accumulator, quadratic (chelis#2301). The entries stay
//! the one thing ever iterated, in insertion order; the index only maps a key
//! to its entry's position, so no hash order can reach observation, and it
//! holds positions rather than owned children, so [05-OP-44]'s finalizer
//! still releases each child exactly once through the entries.
//!
//! Every key stored here passed `validate_dict_key` when it entered, and the
//! dictionary owns one strong reference to it until it leaves. A comparison
//! inside the store therefore reads the raw identity, `(dtype, bits)` or the
//! string bytes, and does not validate either side again: that repeated
//! validation was most of the cost of each comparison (chelis#3733). A probe
//! key is validated once by the entry point that received it.
//!
//! [05-OP-44] publishes the type as the opaque
//! `typedef struct chelis_dict chelis_dict;`, so the layout is the
//! runtime's alone. Emitted C reaches a dictionary only through the
//! `chelis_dict_*` entry points.

use crate::ownership_ledger;
use crate::{
    chelis_dict_entry, chelis_value, HeapHeader, CHELIS_VALUE_SCALAR, CHELIS_VALUE_STRING,
    CHELIS_VALUE_UNIT,
};

/// A store with at most this many slots is searched by a linear scan and
/// carries no index, so a small dictionary (a CSV row, a record) pays no
/// index allocation.
const LINEAR_SCAN_SLOTS: usize = 8;

/// The fewest buckets an index has: more than twice the largest store that
/// is scanned instead.
const MIN_BUCKETS: usize = 2 * LINEAR_SCAN_SLOTS;

/// A free index bucket.
const EMPTY: usize = usize::MAX;

/// The slot a removal leaves. Its key is unit, which `validate_dict_key`
/// never admits, so a hole can neither equal a probe nor be mistaken for an
/// entry; it holds no reference.
fn hole() -> chelis_dict_entry {
    // SAFETY: every field of the carrier is plain bytes, and all-zero bytes
    // are the canonical unit value in both halves.
    unsafe { std::mem::zeroed() }
}

fn is_live(slot: &chelis_dict_entry) -> bool {
    slot.key.tag != CHELIS_VALUE_UNIT
}

#[repr(C)]
pub struct chelis_dict {
    /// Reachable from `lib.rs` for retain, release, and the strong-count
    /// test of the in-place mutators, as `chelis_list`'s header is.
    pub(crate) header: HeapHeader,
    /// The entries in insertion order, and the [`hole`]s removals leave. A
    /// hole keeps every later entry at its position, so a removal touches one
    /// slot instead of shifting the tail and every index bucket after it.
    /// [`chelis_dict::remove_at`] compacts once holes outnumber entries.
    slots: Vec<chelis_dict_entry>,
    /// The number of live (non-hole) slots.
    live: usize,
    /// Open-addressed buckets holding slot positions, probed linearly from
    /// the key's hash. Empty while `slots.len() <= LINEAR_SCAN_SLOTS`;
    /// otherwise a power of two at least twice `live`, holding exactly the
    /// position of every live slot.
    ///
    /// Positions are `usize` rather than `u32`: the language puts no bound
    /// on a dictionary's length below the address space, and a narrower
    /// position would add one. At a load between one quarter and one half
    /// the index costs 16 to 32 bytes per entry beside the 48-byte entry. Only the mutators below change it, and
    /// each runs with the dictionary exclusively owned, so a shared reader
    /// never races a lazy build.
    index: Vec<usize>,
}

impl chelis_dict {
    /// An empty dictionary with one strong owner and room for `capacity`
    /// entries. The ledger byte count is the caller's, as for
    /// `chelis_list::new`.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            header: HeapHeader::new(ownership_ledger::Kind::Dict),
            slots: Vec::with_capacity(capacity),
            live: 0,
            index: Vec::new(),
        }
    }

    /// A dense copy of this dictionary whose entries are `clone(entry)` in
    /// order, with room for `extra` more. A store without holes keeps every
    /// position, so its index is copied rather than rebuilt.
    pub(crate) unsafe fn cloned_with_extra(
        &self,
        extra: usize,
        mut clone: impl FnMut(&chelis_dict_entry) -> chelis_dict_entry,
    ) -> Self {
        let mut out = Self::with_capacity(self.live + extra);
        out.slots.extend(self.entries().map(&mut clone));
        out.live = self.live;
        if self.live == self.slots.len() {
            out.index = self.index.clone();
        } else {
            out.rebuild_index();
        }
        out
    }

    pub(crate) fn len(&self) -> usize {
        self.live
    }

    /// The allocation the ownership ledger accounts for.
    pub(crate) fn buffer_capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// The live entries in insertion order.
    pub(crate) fn entries(&self) -> Entries<'_> {
        Entries {
            slots: self.slots.iter(),
            remaining: self.live,
        }
    }

    /// The position of the entry whose key equals `key`, if any. `key` must
    /// have passed `validate_dict_key`; the position is valid until the next
    /// mutation.
    pub(crate) unsafe fn find(&self, key: chelis_value) -> Option<usize> {
        if self.index.is_empty() {
            return self
                .slots
                .iter()
                .position(|slot| is_live(slot) && key_eq(slot.key, key));
        }
        let mask = self.index.len() - 1;
        let mut bucket = key_hash(key) as usize & mask;
        loop {
            let position = self.index[bucket];
            if position == EMPTY {
                return None;
            }
            if key_eq(self.entry(position).key, key) {
                return Some(position);
            }
            bucket = (bucket + 1) & mask;
        }
    }

    /// The live entry at a position [`chelis_dict::find`] returned.
    pub(crate) fn entry(&self, position: usize) -> &chelis_dict_entry {
        let slot = &self.slots[position];
        assert!(is_live(slot), "a dictionary position names a live entry");
        slot
    }

    /// The value of the live entry at a position [`chelis_dict::find`]
    /// returned. The key stays fixed, so the index stays valid.
    pub(crate) fn value_mut(&mut self, position: usize) -> &mut chelis_value {
        let slot = &mut self.slots[position];
        assert!(is_live(slot), "a dictionary position names a live entry");
        &mut slot.value
    }

    /// Append `entry`, whose key the caller has shown is absent with
    /// [`chelis_dict::find`] and which passed `validate_dict_key`. The
    /// dictionary takes the entry's references as they are.
    pub(crate) unsafe fn push_absent(&mut self, entry: chelis_dict_entry) {
        let position = self.slots.len();
        self.slots.push(entry);
        self.live += 1;
        if self.slots.len() <= LINEAR_SCAN_SLOTS {
            return;
        }
        if self.index.is_empty() || self.live * 2 > self.index.len() {
            self.rebuild_index();
        } else {
            self.index_insert(position, key_hash(entry.key));
        }
    }

    /// Remove and return the live entry at a position
    /// [`chelis_dict::find`] returned; the caller owns its references.
    pub(crate) unsafe fn remove_at(&mut self, position: usize) -> chelis_dict_entry {
        let entry = std::mem::replace(&mut self.slots[position], hole());
        assert!(is_live(&entry), "a dictionary position names a live entry");
        self.live -= 1;
        if !self.index.is_empty() {
            self.index_delete(position, key_hash(entry.key));
        }
        if self.slots.len() - self.live > self.live {
            // Holes outnumber entries: close them up. Each compaction is paid
            // for by the removals since the last one, so a removal stays O(1)
            // amortised. `retain` keeps the allocation, as the scan-and-retain
            // removal this replaces did.
            self.slots.retain(is_live);
            self.rebuild_index();
        }
        entry
    }

    /// Rebuild the index from the slots, or drop it when the store is small
    /// enough to scan.
    unsafe fn rebuild_index(&mut self) {
        self.index.clear();
        if self.slots.len() <= LINEAR_SCAN_SLOTS {
            self.index.shrink_to_fit();
            return;
        }
        self.index.resize(Self::buckets_for(self.live), EMPTY);
        for position in 0..self.slots.len() {
            let slot = self.slots[position];
            if is_live(&slot) {
                self.index_insert(position, key_hash(slot.key));
            }
        }
    }

    /// The smallest power of two holding at least two buckets per live
    /// entry. The load after a rebuild is then above one quarter and at most
    /// one half, and [`chelis_dict::push_absent`] rebuilds once it would
    /// pass one half, which is after the live count has doubled.
    fn buckets_for(live: usize) -> usize {
        (live * 2).next_power_of_two().max(MIN_BUCKETS)
    }

    fn index_insert(&mut self, position: usize, hash: u64) {
        let mask = self.index.len() - 1;
        let mut bucket = hash as usize & mask;
        while self.index[bucket] != EMPTY {
            bucket = (bucket + 1) & mask;
        }
        self.index[bucket] = position;
    }

    /// Remove `position`, whose key hashes to `hash`, from the index, and
    /// shift the rest of its probe run back so no lookup stops early at the
    /// freed bucket (backward-shift deletion, which needs no index
    /// tombstones).
    unsafe fn index_delete(&mut self, position: usize, hash: u64) {
        let mask = self.index.len() - 1;
        let mut hole = hash as usize & mask;
        while self.index[hole] != position {
            hole = (hole + 1) & mask;
        }
        let mut next = hole;
        loop {
            next = (next + 1) & mask;
            let moved = self.index[next];
            if moved == EMPTY {
                break;
            }
            let home = key_hash(self.entry(moved).key) as usize & mask;
            // `moved` may fill the hole only if its home bucket does not lie
            // cyclically in (hole, next]; otherwise a probe from its home
            // would never pass the hole to reach it.
            let stays = if hole <= next {
                hole < home && home <= next
            } else {
                hole < home || home <= next
            };
            if !stays {
                self.index[hole] = moved;
                hole = next;
            }
        }
        self.index[hole] = EMPTY;
    }
}

/// The live entries of a dictionary in insertion order, skipping holes.
pub(crate) struct Entries<'a> {
    slots: std::slice::Iter<'a, chelis_dict_entry>,
    remaining: usize,
}

impl<'a> Iterator for Entries<'a> {
    type Item = &'a chelis_dict_entry;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.slots.by_ref().find(|slot| is_live(slot))?;
        self.remaining -= 1;
        Some(entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for Entries<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let entry = self.slots.by_ref().rev().find(|slot| is_live(slot))?;
        self.remaining -= 1;
        Some(entry)
    }
}

impl ExactSizeIterator for Entries<'_> {}

/// [05-OP-32]'s key equality over two validated keys: the same kind, then
/// the same dtype and stored bits, or the same string bytes.
pub(crate) unsafe fn key_eq(lhs: chelis_value, rhs: chelis_value) -> bool {
    #[cfg(test)]
    KEY_COMPARISONS.with(|count| count.set(count.get() + 1));
    if lhs.tag != rhs.tag {
        return false;
    }
    match lhs.tag {
        CHELIS_VALUE_SCALAR => {
            let (lhs, rhs) = (lhs.payload.scalar, rhs.payload.scalar);
            lhs.dtype == rhs.dtype && lhs.bits == rhs.bits
        }
        CHELIS_VALUE_STRING => {
            let (lhs, rhs) = (lhs.payload.string.handle, rhs.payload.string.handle);
            std::ptr::eq(lhs, rhs) || (*lhs).value == (*rhs).value
        }
        _ => false,
    }
}

/// A fixed hash of a validated key's identity. It is a pure function of the
/// key, so a build's index layout, and therefore its probe counts, are the
/// same on every run; nothing observable depends on it.
unsafe fn key_hash(key: chelis_value) -> u64 {
    const MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut state = u64::from(key.tag.0).wrapping_mul(MULTIPLIER);
    match key.tag {
        CHELIS_VALUE_SCALAR => {
            let scalar = key.payload.scalar;
            state = (state ^ u64::from(scalar.dtype)).wrapping_mul(MULTIPLIER);
            state ^= scalar.bits;
        }
        CHELIS_VALUE_STRING => {
            let bytes = (*key.payload.string.handle).value.as_bytes();
            state = (state ^ bytes.len() as u64).wrapping_mul(MULTIPLIER);
            for chunk in bytes.chunks(8) {
                let mut word = [0_u8; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                state = (state ^ u64::from_le_bytes(word))
                    .wrapping_mul(MULTIPLIER)
                    .rotate_left(29);
            }
        }
        _ => {}
    }
    // MurmurHash3's 64-bit finalizer, so the low bits that pick a bucket
    // depend on every input bit and sequential integer keys do not cluster.
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    state = state.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    state ^ (state >> 33)
}

#[cfg(test)]
thread_local! {
    /// Key comparisons made on this thread; tests read it to bound the
    /// comparisons per dictionary operation (chelis#2301).
    pub(crate) static KEY_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal_value_from_i64;

    /// The structural invariants the index owes the slots: it holds exactly
    /// the live positions, each once; every live key is reachable from its
    /// home bucket without crossing an empty bucket; and lookups agree with
    /// the stored insertion order `model`.
    unsafe fn check(dict: &chelis_dict, model: &[i64], context: &str) {
        let stored: Vec<i64> = dict
            .entries()
            .map(|entry| i64::from_ne_bytes(entry.key.payload.scalar.bits.to_ne_bytes()))
            .collect();
        assert_eq!(stored, model, "{context}: insertion order");
        assert_eq!(dict.len(), model.len(), "{context}: live count");
        assert!(
            dict.slots.len() - dict.live <= dict.live.max(1),
            "{context}: holes"
        );
        let live: Vec<usize> = (0..dict.slots.len())
            .filter(|position| is_live(&dict.slots[*position]))
            .collect();
        if dict.slots.len() <= LINEAR_SCAN_SLOTS {
            assert!(
                dict.index.is_empty(),
                "{context}: small store carries an index"
            );
        } else {
            assert!(
                dict.index.len().is_power_of_two(),
                "{context}: bucket count"
            );
            assert!(
                dict.index.len() >= chelis_dict::buckets_for(dict.live),
                "{context}: load"
            );
            let mut held: Vec<usize> = dict.index.iter().copied().filter(|p| *p != EMPTY).collect();
            held.sort_unstable();
            assert_eq!(held, live, "{context}: index holds exactly the live slots");
            let mask = dict.index.len() - 1;
            for &position in &live {
                let mut bucket = key_hash(dict.slots[position].key) as usize & mask;
                while dict.index[bucket] != position {
                    assert_ne!(
                        dict.index[bucket], EMPTY,
                        "{context}: slot {position} unreachable"
                    );
                    bucket = (bucket + 1) & mask;
                }
            }
        }
        for (rank, key) in model.iter().enumerate() {
            let position = dict.find(internal_value_from_i64(*key));
            assert_eq!(position, Some(live[rank]), "{context}: find {key}");
        }
    }

    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Keys whose home buckets straddle the end of a 32-bucket table, so
    /// their probe runs wrap past bucket 31 into bucket 0 and backward-shift
    /// deletion has to take its wrap-around arm. Fourteen of them keep the
    /// indexed store at exactly 32 buckets; removals pass through holes,
    /// compaction back to a scanned store, and a rebuilt index.
    #[test]
    fn deletion_keeps_wrapping_probe_runs_reachable() {
        let mut keys = Vec::new();
        let mut candidate = 0_i64;
        while keys.len() < 14 {
            let home = unsafe { key_hash(internal_value_from_i64(candidate)) } as usize & 31;
            if home >= 29 || home <= 1 {
                keys.push(candidate);
            }
            candidate += 1;
        }
        let mut state = 7_u64;
        let mut wrapped_deletions = 0;
        for trial in 0..400 {
            let mut order = keys.clone();
            for i in (1..order.len()).rev() {
                order.swap(i, next(&mut state) as usize % (i + 1));
            }
            unsafe {
                let mut dict = chelis_dict::with_capacity(0);
                let mut model = Vec::new();
                for key in order {
                    let entry = chelis_dict_entry {
                        key: internal_value_from_i64(key),
                        value: internal_value_from_i64(-key),
                    };
                    assert!(dict.find(entry.key).is_none());
                    dict.push_absent(entry);
                    model.push(key);
                    check(&dict, &model, &format!("trial {trial} insert {key}"));
                }
                assert_eq!(
                    dict.index.len(),
                    32,
                    "the fourteen keys share one table size"
                );
                for step in 0..40 {
                    if !model.is_empty() && !next(&mut state).is_multiple_of(3) {
                        let key = model.remove(next(&mut state) as usize % model.len());
                        let position = dict.find(internal_value_from_i64(key)).unwrap();
                        if !dict.index.is_empty() {
                            wrapped_deletions += 1;
                        }
                        let entry = dict.remove_at(position);
                        assert_eq!(entry.value.payload.scalar.bits, (-key) as u64);
                    } else if let Some(&key) = keys
                        .iter()
                        .find(|key| !model.contains(key) && next(&mut state).is_multiple_of(2))
                    {
                        dict.push_absent(chelis_dict_entry {
                            key: internal_value_from_i64(key),
                            value: internal_value_from_i64(-key),
                        });
                        model.push(key);
                    }
                    let context = format!("trial {trial} step {step}");
                    check(&dict, &model, &context);
                    let copy = dict.cloned_with_extra(0, |entry| *entry);
                    check(&copy, &model, &format!("{context} copy"));
                }
            }
        }
        assert!(
            wrapped_deletions > 4000,
            "{wrapped_deletions} indexed deletions"
        );
    }
}
