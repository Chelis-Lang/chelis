//! Known Deep AST expression tags and their lane contribution for
//! realizability inference (issue #912, Task 5).
//!
//! Both the realizability walker and the lowering walker look up from
//! this table. Unknown tags default to Host + diagnostic (fail-closed
//! for realizability, fail-loud for observation).
//!
//! Lives in `chelis-types` (the shared floor) so both `chelis-effects`
//! (realizability inference) and `chelis-ir` (lowering) can consult it
//! without cross-dependencies.

/// How a structural expression form contributes to lane assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneContribution {
    /// This tag forces the def to the host lane.
    ForcesHost,
    /// This tag propagates — walk its children for realizability.
    Propagates,
}

/// A structural form's realizability declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagDecl {
    pub tag: &'static str,
    pub lane_contribution: LaneContribution,
}

/// The complete known-tag table. Every tag the lowering and realizability
/// walkers handle must appear here. Unknown tags → Host + diagnostic.
///
/// Adding a tag to the lowering walker without adding it here will cause
/// the drift test to fail.
pub const KNOWN_TAGS: &[TagDecl] = &[
    // ─── Host-forcing structural forms ───────────────────────────────
    TagDecl { tag: "match", lane_contribution: LaneContribution::ForcesHost },
    TagDecl { tag: "record", lane_contribution: LaneContribution::ForcesHost },
    TagDecl { tag: "access", lane_contribution: LaneContribution::ForcesHost },
    TagDecl { tag: "tuple-get", lane_contribution: LaneContribution::ForcesHost },
    // ─── Propagating forms (walk children) ───────────────────────────
    TagDecl { tag: "app", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "var", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "lit", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "fn", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "let", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "if", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "def", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "tuple", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "pipe", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "par", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "cast", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "copy", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "drop", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "realize", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "jit", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "grad", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "vmap", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "handle-effect", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "module", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "import", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "type-decl", lane_contribution: LaneContribution::Propagates },
    TagDecl { tag: "adt-decl", lane_contribution: LaneContribution::Propagates },
];

/// Look up a tag's lane contribution. Returns `None` for unknown tags
/// (caller should treat as Host + emit diagnostic).
pub fn tag_lane_contribution(tag: &str) -> Option<LaneContribution> {
    KNOWN_TAGS.iter().find(|t| t.tag == tag).map(|t| t.lane_contribution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tags_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for decl in KNOWN_TAGS {
            assert!(
                seen.insert(decl.tag),
                "duplicate tag in KNOWN_TAGS: `{}`",
                decl.tag
            );
        }
    }

    #[test]
    fn host_forcing_tags_are_classified() {
        assert_eq!(tag_lane_contribution("match"), Some(LaneContribution::ForcesHost));
        assert_eq!(tag_lane_contribution("record"), Some(LaneContribution::ForcesHost));
        assert_eq!(tag_lane_contribution("access"), Some(LaneContribution::ForcesHost));
        assert_eq!(tag_lane_contribution("tuple-get"), Some(LaneContribution::ForcesHost));
    }

    #[test]
    fn propagating_tags_are_classified() {
        assert_eq!(tag_lane_contribution("app"), Some(LaneContribution::Propagates));
        assert_eq!(tag_lane_contribution("if"), Some(LaneContribution::Propagates));
        assert_eq!(tag_lane_contribution("fn"), Some(LaneContribution::Propagates));
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert_eq!(tag_lane_contribution("nonexistent_tag_xyz"), None);
    }
}
