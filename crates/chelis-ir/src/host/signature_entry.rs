//! Signature-owned extent obligations, independent of tensor-helper partitioning.

use super::{HostParam, HostTensorInput};
use crate::axis_sources::EntryExtentGuard;
use crate::host_type_state::{HostShapeSlot, HostShapeTerm, HostTypeTerm, decode_host_type};
use crate::{Dag, DimInfo, NodeId, RiscOp};
use chelis_deep::ast::Expr;

/// The tensor-bearing shape of one authored formal. `Other` retains its
/// position, so a later List cannot become the first formal by filtering.
/// Tuple and ADT admission have separate owners.
#[derive(Debug, Clone, PartialEq)]
pub enum EntryPattern<T> {
    Tensor(T),
    List(Box<EntryPattern<T>>),
    Other,
}

impl<T> EntryPattern<T> {
    fn try_map_tensor<U, E>(
        &self,
        convert: &mut impl FnMut(&T) -> Result<U, E>,
    ) -> Result<EntryPattern<U>, E> {
        Ok(match self {
            Self::Tensor(tensor) => EntryPattern::Tensor(convert(tensor)?),
            Self::List(inner) => EntryPattern::List(Box::new(inner.try_map_tensor(convert)?)),
            Self::Other => EntryPattern::Other,
        })
    }
}

fn entry_tensor_dims(ty: &HostTypeTerm) -> Vec<&DimInfo> {
    match ty {
        HostTypeTerm::Tensor(tensor) => tensor.dims.iter().collect(),
        HostTypeTerm::PolymorphicTensor(tensor) => match &tensor.shape {
            HostShapeTerm::Concrete(dims) => dims.iter().collect(),
            HostShapeTerm::Polymorphic(slots) => slots
                .iter()
                .filter_map(|slot| match slot {
                    HostShapeSlot::Dim(dim) => Some(dim),
                    HostShapeSlot::RankVariable(_) => None,
                })
                .collect(),
        },
        _ => Vec::new(),
    }
}

impl EntryPattern<HostTypeTerm> {
    /// An entry check is needed for a declared literal or named tensor axis.
    pub fn has_extent_claim(&self) -> bool {
        match self {
            Self::Tensor(tensor) => entry_tensor_dims(tensor).into_iter().any(|dim| {
                matches!(dim, DimInfo::Lit(_))
                    || matches!(dim, DimInfo::Named(name, _) if name != "*")
            }),
            Self::List(inner) => inner.has_extent_claim(),
            Self::Other => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryFormal<T> {
    name: String,
    pattern: EntryPattern<T>,
}

impl<T> EntryFormal<T> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn pattern(&self) -> &EntryPattern<T> {
        &self.pattern
    }
}

/// One signature's ordered List and direct-tensor admission contract.
///
/// Binder indices are scoped to this value. Display spellings remain here,
/// while each invocation creates fresh witness state indexed by this roster.
/// The plan below continues to own fixed tensor DAG identities.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryContract<T> {
    formals: Vec<EntryFormal<T>>,
    binders: Vec<String>,
    named_list_binders: Vec<String>,
}

impl<T> Default for EntryContract<T> {
    fn default() -> Self {
        Self {
            formals: Vec::new(),
            binders: Vec::new(),
            named_list_binders: Vec::new(),
        }
    }
}

impl<T> EntryContract<T> {
    pub fn formals(&self) -> &[EntryFormal<T>] {
        &self.formals
    }

    pub fn binders(&self) -> &[String] {
        &self.binders
    }

    pub fn named_list_binders(&self) -> &[String] {
        &self.named_list_binders
    }

    pub fn try_map_tensor<U, E>(
        &self,
        mut convert: impl FnMut(usize, &T) -> Result<U, E>,
    ) -> Result<EntryContract<U>, E> {
        Ok(EntryContract {
            formals: self
                .formals
                .iter()
                .enumerate()
                .map(|(index, formal)| {
                    Ok(EntryFormal {
                        name: formal.name.clone(),
                        pattern: formal
                            .pattern
                            .try_map_tensor(&mut |tensor| convert(index, tensor))?,
                    })
                })
                .collect::<Result<Vec<_>, E>>()?,
            binders: self.binders.clone(),
            named_list_binders: self.named_list_binders.clone(),
        })
    }
}

impl EntryContract<HostTypeTerm> {
    pub fn from_normalized_types(
        authored: &[Option<Expr>],
        checked: Option<&[Expr]>,
        names: &[String],
    ) -> Result<Self, String> {
        fn contains_entry_tensor(ty: &HostTypeTerm) -> bool {
            match ty {
                HostTypeTerm::Tensor(_) | HostTypeTerm::PolymorphicTensor(_) => true,
                HostTypeTerm::List(inner) => contains_entry_tensor(inner),
                _ => false,
            }
        }
        if authored.len() != names.len() || checked.is_some_and(|items| items.len() != names.len())
        {
            return Err("host runtime: entry contract lost formal alignment".into());
        }
        let params = names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let decode = |ty: &Expr| {
                    decode_host_type(ty).map_err(|error| {
                        format!("host runtime: could not decode `{name}` entry type: {error}")
                    })
                };
                let authored_ty = authored[index].as_ref().map(decode).transpose()?;
                let checked_ty = checked.map(|items| decode(&items[index])).transpose()?;
                Ok(HostParam {
                    name: name.clone(),
                    ty: authored_ty
                        .filter(contains_entry_tensor)
                        .or(checked_ty)
                        .unwrap_or(HostTypeTerm::Unit),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::from_params(&params))
    }

    pub fn from_params(params: &[HostParam]) -> Self {
        fn pattern(ty: &HostTypeTerm) -> EntryPattern<HostTypeTerm> {
            match ty {
                HostTypeTerm::Tensor(_) | HostTypeTerm::PolymorphicTensor(_) => {
                    EntryPattern::Tensor(ty.clone())
                }
                HostTypeTerm::List(item) => EntryPattern::List(Box::new(pattern(item))),
                _ => EntryPattern::Other,
            }
        }
        fn named_axes(ty: &HostTypeTerm) -> Vec<&str> {
            entry_tensor_dims(ty)
                .into_iter()
                .filter_map(|dim| match dim {
                    DimInfo::Named(name, _) if name != "*" => Some(name.as_str()),
                    _ => None,
                })
                .collect()
        }
        fn collect(
            pattern: &EntryPattern<HostTypeTerm>,
            inside_list: bool,
            binders: &mut Vec<String>,
            named_list_binders: &mut Vec<String>,
        ) {
            match pattern {
                EntryPattern::Tensor(tensor) => {
                    for name in named_axes(tensor) {
                        if !binders.iter().any(|seen| seen == name) {
                            binders.push(name.to_owned());
                        }
                        if inside_list && !named_list_binders.iter().any(|seen| seen == name) {
                            named_list_binders.push(name.to_owned());
                        }
                    }
                }
                EntryPattern::List(inner) => collect(inner, true, binders, named_list_binders),
                EntryPattern::Other => {}
            }
        }
        let mut formals = Vec::with_capacity(params.len());
        let mut binders = Vec::new();
        let mut named_list_binders = Vec::new();
        for param in params {
            let pattern = pattern(&param.ty);
            collect(&pattern, false, &mut binders, &mut named_list_binders);
            formals.push(EntryFormal {
                name: param.name.clone(),
                pattern,
            });
        }
        Self {
            formals,
            binders,
            named_list_binders,
        }
    }
}

/// One signature's ordered input observations and equality obligations.
///
/// The plan owns its Load identities. Names associate repeated declarations
/// only while constructing this one signature; plans never merge by spelling.
/// Both an ordinary function entry and a retained inlined invocation consume
/// this representation before executing their bodies.
#[derive(Debug, Clone)]
pub struct SignatureEntryPlan {
    observations: Dag,
    labels: Vec<String>,
    guards: Vec<EntryExtentGuard>,
}

impl SignatureEntryPlan {
    pub fn new(inputs: impl IntoIterator<Item = HostTensorInput>) -> Self {
        // The plan observes one signature's inputs, so its graph is that
        // signature's only declaration.
        let mut observations = Dag::new();
        let decl = observations.declare("signature entry");
        let mut guards = Vec::new();
        let mut first: Vec<(String, (NodeId, usize))> = Vec::new();
        let mut labels = Vec::new();
        for input in inputs {
            let label = input.name;
            // A dynamic List observation has a display path such as `xs[1]`,
            // which is not a Load identifier. Keep the path off the DAG name.
            let load_name = crate::LoadStoreName::new(label.clone()).unwrap_or_else(|_| {
                crate::LoadStoreName::must(format!("__entry_observation_{}", labels.len()))
            });
            let load = observations.add_node(
                decl,
                RiscOp::Load { name: load_name },
                Vec::new(),
                input.ty.clone(),
                None,
            );
            labels.push(label);
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
            labels,
            guards,
        }
    }

    /// Each node is exactly one tensor parameter, in declaration order.
    pub fn observations(&self) -> &Dag {
        &self.observations
    }

    pub fn label(&self, observation: NodeId) -> &str {
        &self.labels[observation.0]
    }

    /// Literal and repeated-binder checks already interleaved by signature axis.
    pub fn guards(&self) -> &[EntryExtentGuard] {
        &self.guards
    }
}

#[cfg(test)]
mod tests {
    use super::super::HostParam;
    use super::*;
    use crate::TensorType;
    use crate::host_type_state::HostTypeTerm;
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

    #[test]
    fn recursive_list_contract_keeps_signature_binder_and_empty_slot() {
        let tensor = || {
            HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Named("seq".into(), None)],
                precision: Prim::F32,
            })
        };
        let contract = EntryContract::from_params(&[
            HostParam {
                name: "empty".into(),
                ty: HostTypeTerm::List(Box::new(HostTypeTerm::List(Box::new(tensor())))),
            },
            HostParam {
                name: "later".into(),
                ty: tensor(),
            },
        ]);
        assert_eq!(contract.named_list_binders(), &["seq"]);
        assert_eq!(contract.formals().len(), 2);
        assert!(matches!(
            contract.formals()[0].pattern(),
            EntryPattern::List(inner) if matches!(inner.as_ref(), EntryPattern::List(_))
        ));
        assert!(matches!(
            contract.formals()[1].pattern(),
            EntryPattern::Tensor(_)
        ));
    }

    #[test]
    fn checked_tensor_instantiation_fills_an_authored_type_variable() {
        let span = chelis_deep::Span::new(0, 0);
        let authored =
            super::super::host_type_syntax(&HostTypeTerm::TypeVariable("t".into()), span).unwrap();
        let checked = super::super::host_type_syntax(
            &HostTypeTerm::List(Box::new(HostTypeTerm::Tensor(TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            }))),
            span,
        )
        .unwrap();
        let contract = EntryContract::from_normalized_types(
            &[Some(authored)],
            Some(&[checked]),
            &["xs".into()],
        )
        .unwrap();
        assert!(matches!(
            contract.formals()[0].pattern(),
            EntryPattern::List(inner) if matches!(inner.as_ref(), EntryPattern::Tensor(_))
        ));
    }
}
