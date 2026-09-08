//! Typed Deep AST lane contributions for realizability inference
//! (issue #912, Task 5; issue #1080).
//!
//! Every [`DeepTag`] receives an explicit disposition. The exhaustive match
//! is the drift guard: extending the closed vocabulary without deciding its
//! lane contribution stops compilation.
//!
//! Serialized strings are decoded at the Deep ingress boundary. A raw or
//! otherwise untyped form is not a `DeepTag`; the realizability walker owns
//! its fail-closed Host diagnostic.

use chelis_deep::DeepTag;

/// How a structural expression form contributes to lane assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneContribution {
    /// This tag forces the def to the host lane.
    ForcesHost,
    /// This tag propagates — walk its children for realizability.
    Propagates,
}

/// Return the realizability contribution for a decoded Deep tag.
///
/// Keep this match wildcard-free. The four Host-forcing forms mirror the
/// target-independent lowering classifier; every other recognized form walks
/// its children. Raw strings have no entry here by construction.
pub const fn deep_tag_lane_contribution(tag: DeepTag) -> LaneContribution {
    match tag {
        DeepTag::Match | DeepTag::Record | DeepTag::Access | DeepTag::TupleGet => {
            LaneContribution::ForcesHost
        }
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Fn
        | DeepTag::App
        | DeepTag::Let
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Pipe
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => LaneContribution::Propagates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::DeepTag;

    #[test]
    fn every_deep_tag_has_a_typed_lane_disposition() {
        for tag in DeepTag::ALL {
            let _ = deep_tag_lane_contribution(tag);
        }
    }

    #[test]
    fn host_forcing_tag_set_is_exact() {
        let host_forcing: Vec<DeepTag> = DeepTag::ALL
            .into_iter()
            .filter(|tag| deep_tag_lane_contribution(*tag) == LaneContribution::ForcesHost)
            .collect();

        assert_eq!(
            host_forcing,
            [
                DeepTag::Match,
                DeepTag::Record,
                DeepTag::Access,
                DeepTag::TupleGet,
            ]
        );
    }

    #[test]
    fn tags_omitted_by_the_old_partial_table_propagate() {
        for tag in [
            DeepTag::ImportAll,
            DeepTag::Block,
            DeepTag::RecordUpdate,
            DeepTag::PatCtor,
            DeepTag::Unquote,
            DeepTag::Resource,
        ] {
            assert_eq!(
                deep_tag_lane_contribution(tag),
                LaneContribution::Propagates,
                "DeepTag::{tag:?} must not false-route Host"
            );
        }
    }
}
