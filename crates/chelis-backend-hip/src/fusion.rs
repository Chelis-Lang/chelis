//! HIP mechanics for a shared-planner-authorized fused storage reuse.

use chelis_ir::dag::NodeId;
use chelis_ir::ownership::ReusableOwnedStorage;

/// Backend-private owner of the one linear storage-reuse authority.
#[derive(Debug)]
pub(crate) struct HipFusedReuse {
    token: ReusableOwnedStorage,
    slot_has_later_owner: bool,
}

/// Copyable emission facts derived only while the linear authority remains
/// owned by [`HipFusedReuse`]. This type decides no eligibility.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedReuseMechanics {
    pub reusable_input: NodeId,
    pub slot_has_later_owner: bool,
}

impl HipFusedReuse {
    pub(crate) fn new(token: ReusableOwnedStorage, slot_has_later_owner: bool) -> Self {
        Self {
            token,
            slot_has_later_owner,
        }
    }

    pub(crate) fn mechanics(&self, consumer: NodeId) -> FusedReuseMechanics {
        assert_eq!(self.token.consumer(), consumer);
        FusedReuseMechanics {
            reusable_input: self.token.source(),
            slot_has_later_owner: self.slot_has_later_owner,
        }
    }
}
