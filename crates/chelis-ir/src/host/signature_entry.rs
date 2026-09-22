//! Signature-owned extent obligations, independent of tensor-helper partitioning.

use super::HostTensorInput;
use crate::axis_sources::EntryExtentGuard;
use crate::{Dag, DimInfo, NodeId, RiscOp};

/// One signature's ordered input observations and equality obligations.
///
/// The plan owns its Load identities. Names associate repeated declarations
/// only while constructing this one signature; plans never merge by spelling.
/// Both an ordinary function entry and a retained inlined invocation consume
/// this representation before executing their bodies.
#[derive(Debug, Clone)]
pub struct SignatureEntryPlan {
    observations: Dag,
    guards: Vec<EntryExtentGuard>,
}

impl SignatureEntryPlan {
    pub fn new(inputs: impl IntoIterator<Item = HostTensorInput>) -> Self {
        let mut observations = Dag::new();
        let mut guards = Vec::new();
        let mut first: Vec<(String, (NodeId, usize))> = Vec::new();
        for input in inputs {
            let load = observations.add_node(
                RiscOp::Load {
                    name: input.name.into(),
                },
                Vec::new(),
                input.ty.clone(),
                None,
            );
            for (axis, dim) in input.ty.dims.iter().enumerate() {
                let observed = (load, axis);
                match dim {
                    DimInfo::Lit(required) => guards.push(EntryExtentGuard::Literal {
                        required: *required,
                        observed,
                    }),
                    DimInfo::Named(name, _) if name != "*" => {
                        if let Some((_, canonical)) = first.iter().find(|(seen, _)| seen == name) {
                            guards.push(EntryExtentGuard::Named {
                                claim: crate::lower::extent_binder_label(name),
                                canonical: *canonical,
                                observed,
                            });
                        } else {
                            first.push((name.clone(), observed));
                        }
                    }
                    _ => {}
                }
            }
        }
        Self {
            observations,
            guards,
        }
    }

    /// Each node is exactly one tensor parameter, in declaration order.
    pub fn observations(&self) -> &Dag {
        &self.observations
    }

    /// Literal and repeated-binder checks already interleaved by signature axis.
    pub fn guards(&self) -> &[EntryExtentGuard] {
        &self.guards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorType;
    use chelis_types::types::Prim;

    fn input(name: &str, dims: Vec<DimInfo>) -> HostTensorInput {
        HostTensorInput {
            name: name.into(),
            ty: TensorType {
                dims,
                precision: Prim::F32,
            },
        }
    }

    #[test]
    fn later_witnesses_and_literals_follow_parameter_axis_order() {
        let n = || DimInfo::Named("seq".into(), None);
        let plan = SignatureEntryPlan::new([
            input("z", vec![n()]),
            input("a", vec![DimInfo::Lit(2), n()]),
            input("unused", vec![n()]),
        ]);
        assert_eq!(
            plan.guards(),
            &[
                EntryExtentGuard::Literal {
                    required: 2,
                    observed: (NodeId(1), 0)
                },
                EntryExtentGuard::Named {
                    claim: "seq".into(),
                    canonical: (NodeId(0), 0),
                    observed: (NodeId(1), 1)
                },
                EntryExtentGuard::Named {
                    claim: "seq".into(),
                    canonical: (NodeId(0), 0),
                    observed: (NodeId(2), 0)
                },
            ]
        );
    }

    #[test]
    fn independent_signatures_and_wildcards_create_no_equality() {
        for name in ["x", "y"] {
            let plan = SignatureEntryPlan::new([input(
                name,
                vec![
                    DimInfo::Named("seq".into(), None),
                    DimInfo::Named("*".into(), None),
                    DimInfo::Named("*".into(), None),
                ],
            )]);
            assert!(plan.guards().is_empty());
        }
    }

    #[test]
    fn internal_rank_axis_identity_has_a_printable_guard_label() {
        let internal = || DimInfo::Named("\0rank-axis:rest:0".into(), None);
        let plan = SignatureEntryPlan::new([
            input("first", vec![internal()]),
            input("second", vec![internal()]),
        ]);
        assert_eq!(
            plan.guards(),
            &[EntryExtentGuard::Named {
                claim: "rest[0]".into(),
                canonical: (NodeId(0), 0),
                observed: (NodeId(1), 0),
            }]
        );
    }
}
