//! Lexical scope of deferred result equalities.
//!
//! Value unification is deliberately deferred, but its scope dependency is
//! immediate. A relation cannot be copied with fresh variables on one side
//! and shared inference variables on the other: the copy could settle the
//! shared value using a fresh input that no longer owns its semantic rule.
//! Components below carry that dependency without binding any value type.

use alloc::{collections::BTreeSet, vec::Vec};

use crate::model::*;
use crate::scheme::{free_dvars, free_rvars, free_tvars};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Variable {
    Type(TypeVar),
    Dimension(DimVar),
    Rank(RankVar),
}

fn variables<E>(ty: &Type<E>) -> impl Iterator<Item = Variable> {
    free_tvars(ty)
        .into_iter()
        .map(Variable::Type)
        .chain(free_dvars(ty).into_iter().map(Variable::Dimension))
        .chain(free_rvars(ty).into_iter().map(Variable::Rank))
}

#[derive(Default)]
pub struct ResultScope {
    components: Vec<BTreeSet<Variable>>,
}

impl ResultScope {
    pub fn new<E>(equations: &[ResultConstraint<E>]) -> Self {
        let mut scope = Self::default();
        let mut pairs = Vec::new();
        for equation in equations {
            match equation {
                ResultConstraint::Annotation { actual, declared } => pairs.push((actual, declared)),
                ResultConstraint::Join { inputs, result } => {
                    pairs.extend(inputs.iter().map(|input| (input, result)));
                }
            }
        }
        // Decompose known structures. An unrelated free variable in another
        // field must not make an otherwise independent component monomorphic.
        // An unknown constructor, however, depends on the whole opposing type.
        while let Some((left, right)) = pairs.pop() {
            match (left, right) {
                (Type::Fn(args, result), Type::Fn(other_args, other_result))
                    if args.len() == other_args.len() =>
                {
                    pairs.extend(args.iter().zip(other_args.iter()));
                    pairs.push((result, other_result));
                }
                (Type::Tuple(items), Type::Tuple(other)) if items.len() == other.len() => {
                    pairs.extend(items.iter().zip(other.iter()));
                }
                (Type::Adt(name, items), Type::Adt(other_name, other))
                    if name == other_name && items.len() == other.len() =>
                {
                    pairs.extend(items.iter().zip(other.iter()));
                }
                (Type::KindedAdt(name, items), Type::KindedAdt(other_name, other))
                    if name == other_name && items.len() == other.len() =>
                {
                    for (left, right) in items.iter().zip(other.iter()) {
                        match (left, right) {
                            (NominalArg::Type(left), NominalArg::Type(right)) => {
                                pairs.push((left, right));
                            }
                            (NominalArg::Dimension(left), NominalArg::Dimension(right)) => {
                                scope.link_dimensions(
                                    core::iter::once(left),
                                    core::iter::once(right),
                                );
                            }
                            // Different kinds already fail ordinary equality.
                            _ => {}
                        }
                    }
                }
                (Type::Ref(left), Type::Ref(right)) => pairs.push((left, right)),
                (Type::Tensor(dims, precision), Type::Tensor(other_dims, other_precision)) => {
                    if dims.len() == other_dims.len()
                        && !dims
                            .iter()
                            .chain(other_dims)
                            .any(|dim| matches!(dim, Dim::Rank(_)))
                    {
                        for (left, right) in dims.iter().zip(other_dims) {
                            scope.link_dimensions(core::iter::once(left), core::iter::once(right));
                        }
                    } else {
                        scope.link_dimensions(dims.iter(), other_dims.iter());
                    }
                    scope.link([precision, other_precision].into_iter().filter_map(
                        |slot| match slot {
                            TensorPrec::Var(var) => Some(Variable::Type(*var)),
                            TensorPrec::Concrete(_) => None,
                        },
                    ));
                }
                _ => scope.link(variables(left).chain(variables(right))),
            }
        }
        scope
    }

    fn link(&mut self, variables: impl IntoIterator<Item = Variable>) {
        let mut connected = variables.into_iter().collect::<BTreeSet<_>>();
        if connected.is_empty() {
            return;
        }
        self.components.retain(|component| {
            if component.is_disjoint(&connected) {
                true
            } else {
                connected.extend(component);
                false
            }
        });
        self.components.push(connected);
    }

    fn link_dimensions<'a>(
        &mut self,
        left: impl Iterator<Item = &'a Dim>,
        right: impl Iterator<Item = &'a Dim>,
    ) {
        self.link(left.chain(right).filter_map(|dim| match dim {
            Dim::Var(var) => Some(Variable::Dimension(*var)),
            Dim::Rank(var) => Some(Variable::Rank(*var)),
            Dim::Name(_) | Dim::Lit(_) | Dim::Wildcard => None,
        }));
    }

    /// A declared binder keeps ownership when deferred equalities change the
    /// representative variable in its published signature.
    pub fn shares_type_component(&self, left: TypeVar, right: TypeVar) -> bool {
        left == right
            || self.components.iter().any(|component| {
                component.contains(&Variable::Type(left))
                    && component.contains(&Variable::Type(right))
            })
    }

    pub fn retain_closed_quantifiers<E>(&self, scheme: &mut Scheme<E>) {
        let mut quantified = scheme
            .tvars
            .iter()
            .copied()
            .map(Variable::Type)
            .chain(scheme.dvars.iter().copied().map(Variable::Dimension))
            .chain(scheme.rvars.iter().copied().map(Variable::Rank))
            .collect::<BTreeSet<_>>();
        for component in &self.components {
            if !component.is_subset(&quantified) {
                for variable in component {
                    quantified.remove(variable);
                }
            }
        }
        scheme
            .tvars
            .retain(|var| quantified.contains(&Variable::Type(*var)));
        scheme
            .tvar_restrictions
            .retain(|(var, _)| quantified.contains(&Variable::Type(*var)));
        scheme
            .dvars
            .retain(|var| quantified.contains(&Variable::Dimension(*var)));
        scheme
            .rvars
            .retain(|var| quantified.contains(&Variable::Rank(*var)));
    }
}
