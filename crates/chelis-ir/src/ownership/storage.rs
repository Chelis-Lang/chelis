#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_candidate(provenance: StorageProvenance) -> ReuseCandidateFacts {
        ReuseCandidateFacts::complete_for_test(provenance)
    }

    #[test]
    fn complete_runtime_owned_proof_mints_one_linear_token() {
        let token = ReusableOwnedStorage::mint_for_test(
            NodeId(3),
            StorageId::from_index_for_test(7),
            NodeId(5),
            synthetic_candidate(StorageProvenance::RuntimeOwned),
        )
        .expect("all six conditions should mint the synthetic token");

        assert_eq!(token.source(), NodeId(3));
        assert_eq!(token.storage_for_test(), StorageId::from_index_for_test(7));
        assert_eq!(token.consumer(), NodeId(5));
    }

    #[test]
    fn every_missing_proof_condition_rejects() {
        for condition in ReuseProofCondition::ALL {
            let facts = synthetic_candidate(StorageProvenance::RuntimeOwned)
                .without_condition_for_test(condition);
            assert_eq!(
                ReusableOwnedStorage::mint_for_test(
                    NodeId(3),
                    StorageId::from_index_for_test(7),
                    NodeId(5),
                    facts,
                ),
                Err(ReuseProofRejection::Missing(condition)),
                "removing {} must reject reuse",
                condition.name()
            );
        }
    }

    #[test]
    fn entry_borrow_cannot_mint_reusable_storage() {
        assert_eq!(
            ReusableOwnedStorage::mint_for_test(
                NodeId(3),
                StorageId::from_index_for_test(7),
                NodeId(5),
                synthetic_candidate(StorageProvenance::EntryBorrow),
            ),
            Err(ReuseProofRejection::Missing(
                ReuseProofCondition::RuntimeOwnedAndWritable
            ))
        );
    }

    #[test]
    fn shared_view_cannot_mint_reusable_storage() {
        assert_eq!(
            ReusableOwnedStorage::mint_for_test(
                NodeId(3),
                StorageId::from_index_for_test(7),
                NodeId(5),
                synthetic_candidate(StorageProvenance::SharedView),
            ),
            Err(ReuseProofRejection::Missing(
                ReuseProofCondition::RuntimeOwnedAndWritable
            ))
        );
    }
}

use crate::dag::NodeId;

/// The only storage origins the shared ownership planner may classify.
///
/// This vocabulary stays inside `chelis-ir`: backends receive a verified
/// reuse token, not provenance facts from which they could rebuild an
/// ownership decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageProvenance {
    RuntimeOwned,
    EntryBorrow,
    SharedView,
}

/// Stable identity for one allocation in the eventual shared storage graph.
/// Its integer representation is private and has no production constructor
/// until the exact-capacity planner is wired.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct StorageId {
    key: u32,
}

#[cfg(test)]
impl StorageId {
    fn from_index_for_test(key: u32) -> Self {
        Self { key }
    }
}

/// The six independent conditions in
/// `spec/design/compiled_value_ownership.md` C6. Their order mirrors that
/// design and fixes diagnostic/test mutation order; it is not an invitation
/// for a backend to evaluate them independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReuseProofCondition {
    OneLiveProgramOwner,
    UniqueDescriptorAndStorage,
    RuntimeOwnedAndWritable,
    NoLiveViewOrBorrow,
    ExactShapeDtypeCapacityAndOperation,
    TerminalUse,
}

impl ReuseProofCondition {
    pub(crate) const ALL: [Self; 6] = [
        Self::OneLiveProgramOwner,
        Self::UniqueDescriptorAndStorage,
        Self::RuntimeOwnedAndWritable,
        Self::NoLiveViewOrBorrow,
        Self::ExactShapeDtypeCapacityAndOperation,
        Self::TerminalUse,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::OneLiveProgramOwner => "one live program owner",
            Self::UniqueDescriptorAndStorage => "unique descriptor and storage",
            Self::RuntimeOwnedAndWritable => "runtime-owned and writable storage",
            Self::NoLiveViewOrBorrow => "no live view or borrow",
            Self::ExactShapeDtypeCapacityAndOperation => {
                "exact shape, dtype, capacity, and operation requirements"
            }
            Self::TerminalUse => "terminal source use",
        }
    }
}

/// Private facts from which the future shared planner will mint a token.
///
/// Condition 3 is deliberately the conjunction of a closed provenance and a
/// writable fact. A caller cannot mark an EntryBorrow or SharedView reusable
/// merely by setting the writable half. Condition 5 remains a boolean test
/// seam only in this source milestone; production construction stays absent
/// until chelis#893 supplies the exact `CapacityKey` witness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReuseCandidateFacts {
    provenance: StorageProvenance,
    one_live_program_owner: bool,
    unique_descriptor_and_storage: bool,
    writable: bool,
    no_live_view_or_borrow: bool,
    exact_shape_dtype_capacity_and_operation: bool,
    terminal_use: bool,
}

impl ReuseCandidateFacts {
    fn first_missing_condition(self) -> Option<ReuseProofCondition> {
        if !self.one_live_program_owner {
            return Some(ReuseProofCondition::OneLiveProgramOwner);
        }
        if !self.unique_descriptor_and_storage {
            return Some(ReuseProofCondition::UniqueDescriptorAndStorage);
        }
        if self.provenance != StorageProvenance::RuntimeOwned || !self.writable {
            return Some(ReuseProofCondition::RuntimeOwnedAndWritable);
        }
        if !self.no_live_view_or_borrow {
            return Some(ReuseProofCondition::NoLiveViewOrBorrow);
        }
        if !self.exact_shape_dtype_capacity_and_operation {
            return Some(ReuseProofCondition::ExactShapeDtypeCapacityAndOperation);
        }
        if !self.terminal_use {
            return Some(ReuseProofCondition::TerminalUse);
        }
        None
    }

    #[cfg(test)]
    fn complete_for_test(provenance: StorageProvenance) -> Self {
        Self {
            provenance,
            one_live_program_owner: true,
            unique_descriptor_and_storage: true,
            writable: true,
            no_live_view_or_borrow: true,
            exact_shape_dtype_capacity_and_operation: true,
            terminal_use: true,
        }
    }

    #[cfg(test)]
    fn without_condition_for_test(mut self, condition: ReuseProofCondition) -> Self {
        match condition {
            ReuseProofCondition::OneLiveProgramOwner => {
                self.one_live_program_owner = false;
            }
            ReuseProofCondition::UniqueDescriptorAndStorage => {
                self.unique_descriptor_and_storage = false;
            }
            ReuseProofCondition::RuntimeOwnedAndWritable => {
                self.writable = false;
            }
            ReuseProofCondition::NoLiveViewOrBorrow => {
                self.no_live_view_or_borrow = false;
            }
            ReuseProofCondition::ExactShapeDtypeCapacityAndOperation => {
                self.exact_shape_dtype_capacity_and_operation = false;
            }
            ReuseProofCondition::TerminalUse => {
                self.terminal_use = false;
            }
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReuseProofRejection {
    Missing(ReuseProofCondition),
}

/// Linear authority for reusing one exact program-owned storage allocation.
///
/// The token names its source owner, allocation identity, and consuming node.
/// It intentionally implements neither [`Clone`] nor [`Copy`]. A backend
/// consumer must take the token by value; it may borrow the token internally
/// only while selecting target mechanics for that one consumption.
///
/// That by-value consumer and read-only identity pair are the complete
/// pre-integration backend seam:
///
/// ```
/// # use chelis_ir::dag::NodeId;
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn selected_nodes(token: ReusableOwnedStorage) -> (NodeId, NodeId) {
///     (token.source(), token.consumer())
/// }
/// ```
///
/// Passing one token to two consumers is a use-after-move even without an
/// explicit `Clone` or `Copy` bound:
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn consume(_: ReusableOwnedStorage) {}
/// fn duplicate(token: ReusableOwnedStorage) {
///     consume(token);
///     consume(token);
/// }
/// ```
///
/// Its fields cannot be inspected or used to forge a sibling ownership model:
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn inspect(token: &ReusableOwnedStorage) {
///     let _ = token.source;
/// }
/// ```
///
/// It cannot be cloned into a second consuming operation:
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn require_clone<T: Clone>() {}
/// fn duplicate() {
///     require_clone::<ReusableOwnedStorage>();
/// }
/// ```
///
/// It cannot be copied into a second consuming operation:
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn require_copy<T: Copy>() {}
/// fn duplicate() {
///     require_copy::<ReusableOwnedStorage>();
/// }
/// ```
///
/// There is no public constructor; struct-literal minting is also sealed:
///
/// ```compile_fail
/// # use chelis_ir::dag::NodeId;
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn forge(source: NodeId, consumer: NodeId) -> ReusableOwnedStorage {
///     ReusableOwnedStorage { source, storage: todo!(), consumer }
/// }
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct ReusableOwnedStorage {
    source: NodeId,
    storage: StorageId,
    consumer: NodeId,
}

impl ReusableOwnedStorage {
    /// The exact source node whose owned storage is consumed by the reuse.
    pub const fn source(&self) -> NodeId {
        self.source
    }

    /// The exact operation that consumes the source and receives its storage.
    pub const fn consumer(&self) -> NodeId {
        self.consumer
    }

    pub(crate) const fn storage_id(&self) -> StorageId {
        self.storage
    }

    /// Synthetic constructor for the spec-derived proof matrix only.
    ///
    /// Production builds contain no constructor. The eventual constructor
    /// must additionally consume chelis#893's exact `CapacityKey`/`Repr`
    /// witness before it can create this token.
    #[cfg(test)]
    fn mint_for_test(
        source: NodeId,
        storage: StorageId,
        consumer: NodeId,
        facts: ReuseCandidateFacts,
    ) -> Result<Self, ReuseProofRejection> {
        if let Some(condition) = facts.first_missing_condition() {
            return Err(ReuseProofRejection::Missing(condition));
        }

        Ok(Self {
            source,
            storage,
            consumer,
        })
    }

    #[cfg(test)]
    const fn storage_for_test(&self) -> StorageId {
        self.storage_id()
    }
}
