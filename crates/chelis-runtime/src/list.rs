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

use crate::ownership_ledger;
use crate::{chelis_value, HeapHeader};

#[repr(C)]
pub struct chelis_list {
    /// Deliberately reachable from `lib.rs`: retain, release and the
    /// exclusivity checks of the in-place mutators read the strong
    /// count. The backing store below is the field this module hides,
    /// and hiding it is unaffected by exposing the header.
    pub(crate) header: HeapHeader,
    items: Vec<chelis_value>,
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
        }
    }

    /// Every element the list holds, in order.
    ///
    /// The single view of the backing store, and the reason the store is
    /// private. A reader that bypassed it would be reading slots this
    /// list does not own.
    pub(crate) fn live(&self) -> &[chelis_value] {
        &self.items
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
}
