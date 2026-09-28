//! Lexical scope of deferred result equalities.
//!
//! Value unification is deliberately deferred, but its scope dependency is
//! immediate. A relation cannot be copied with fresh variables on one side
//! and shared inference variables on the other: the copy could settle the
//! shared value using a fresh input that no longer owns its semantic rule.
//! Components below carry that dependency without binding any value type.

use std::collections::BTreeSet;

use crate::env::{free_dvars, free_rvars, free_tvars};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Variable {
    Type(TypeVar),
    Dimension(DimVar),
    Rank(RankVar),
}

fn variables(ty: &Type) -> impl Iterator<Item = Variable> {
    free_tvars(ty)
        .into_iter()
        .map(Variable::Type)
        .chain(free_dvars(ty).into_iter().map(Variable::Dimension))
        .chain(free_rvars(ty).into_iter().map(Variable::Rank))
}

#[derive(Default)]
pub(crate) struct ResultScope {
    components: Vec<BTreeSet<Variable>>,
}

impl ResultScope {
    pub(crate) fn new(equations: &[ResultConstraint]) -> Self {
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
                    pairs.extend(args.iter().zip(other_args));
                    pairs.push((result, other_result));
                }
                (Type::Tuple(items), Type::Tuple(other)) if items.len() == other.len() => {
                    pairs.extend(items.iter().zip(other));
                }
                (Type::Adt(name, items), Type::Adt(other_name, other))
                    if name == other_name && items.len() == other.len() =>
                {
                    pairs.extend(items.iter().zip(other));
                }
                (Type::KindedAdt(name, items), Type::KindedAdt(other_name, other))
                    if name == other_name && items.len() == other.len() =>
                {
                    for (left, right) in items.iter().zip(other) {
                        match (left, right) {
                            (NominalArg::Type(left), NominalArg::Type(right)) => {
                                pairs.push((left, right));
                            }
                            (NominalArg::Dimension(left), NominalArg::Dimension(right)) => {
                                scope
                                    .link_dimensions(std::iter::once(left), std::iter::once(right));
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
                            scope.link_dimensions(std::iter::once(left), std::iter::once(right));
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

    pub(crate) fn retain_closed_quantifiers(&self, scheme: &mut Scheme) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn quantified(tvars: &[u32]) -> Scheme {
        let mut scheme = Scheme::mono(Type::Unit);
        scheme.tvars = tvars.iter().copied().map(TypeVar).collect();
        scheme
    }

    #[test]
    fn a_component_is_freshened_whole_or_kept_shared() {
        for offset in [0, 10, 100] {
            let var = |id| Type::Var(TypeVar(offset + id));
            for container in [
                var(1),
                Type::Fn(vec![var(1)], Box::new(var(1))),
                Type::Tuple(vec![var(1)]),
                Type::Adt("List".to_string(), vec![var(1)]),
                Type::KindedAdt("Holder".to_string(), vec![NominalArg::Type(var(1))]),
                Type::Ref(Box::new(var(1))),
            ] {
                let mut equations = vec![
                    ResultConstraint::Join {
                        inputs: vec![container],
                        result: var(0),
                    },
                    ResultConstraint::Annotation {
                        actual: var(1),
                        declared: var(2),
                    },
                ];
                for _ in 0..2 {
                    let graph = ResultScope::new(&equations);
                    let mut shared = quantified(&[offset + 1, offset + 2, offset + 3]);
                    graph.retain_closed_quantifiers(&mut shared);
                    assert_eq!(shared.tvars, [TypeVar(offset + 3)]);
                    assert_eq!(shared.body, Type::Unit, "scope closure binds no value type");
                    let mut fresh = quantified(&[offset, offset + 1, offset + 2, offset + 3]);
                    graph.retain_closed_quantifiers(&mut fresh);
                    assert_eq!(fresh.tvars.len(), 4);
                    equations.reverse();
                }
            }
        }
    }

    #[test]
    fn independent_structural_fields_keep_their_polymorphism() {
        let var = |id| Type::Var(TypeVar(id));
        let function = |id| Type::Fn(vec![var(id)], Box::new(var(id)));
        let equation = ResultConstraint::Annotation {
            actual: Type::Tuple(vec![var(0), function(1)]),
            declared: Type::Tuple(vec![var(2), function(3)]),
        };
        let mut scheme = quantified(&[1, 2, 3]);
        ResultScope::new(&[equation]).retain_closed_quantifiers(&mut scheme);
        assert_eq!(scheme.tvars, [TypeVar(1), TypeVar(3)]);
    }

    #[test]
    fn unknown_constructor_scope_reaches_dimensions_ranks_and_precision() {
        let equation = ResultConstraint::Annotation {
            actual: Type::Var(TypeVar(0)),
            declared: Type::Tensor(
                vec![Dim::Var(DimVar(1)), Dim::Rank(RankVar(2))],
                TensorPrec::Var(TypeVar(3)),
            ),
        };
        let mut scheme = quantified(&[3]);
        scheme.dvars = vec![DimVar(1)];
        scheme.rvars = vec![RankVar(2)];
        ResultScope::new(&[equation]).retain_closed_quantifiers(&mut scheme);
        assert!(scheme.tvars.is_empty() && scheme.dvars.is_empty() && scheme.rvars.is_empty());
    }

    #[test]
    fn known_tensor_fields_have_independent_scope() {
        let equation = ResultConstraint::Annotation {
            actual: Type::Tensor(vec![Dim::Var(DimVar(0))], TensorPrec::Var(TypeVar(1))),
            declared: Type::Tensor(vec![Dim::Var(DimVar(2))], TensorPrec::Var(TypeVar(3))),
        };
        let mut scheme = quantified(&[1, 3]);
        scheme.dvars = vec![DimVar(2)];
        ResultScope::new(&[equation]).retain_closed_quantifiers(&mut scheme);
        assert_eq!(scheme.tvars, [TypeVar(1), TypeVar(3)]);
        assert!(scheme.dvars.is_empty());
    }

    #[test]
    fn the_solved_signature_cannot_freshen_a_shared_origin() {
        let mut vg = VarGen::default();
        let shared = vg.fresh_tvar();
        let mut scheme = quantified(&[shared.0]);
        scheme.body = Type::Var(shared);
        scheme.result_origin = Some(ResultOrigin {
            body: Type::Var(shared),
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            equations: vec![],
        });
        let subst = crate::unify::Subst::new();
        let env = crate::env::Env::new();
        assert_eq!(env.instantiate(&scheme, &mut vg, &subst), Type::Var(shared));
        scheme.result_origin.as_mut().unwrap().tvars.push(shared);
        assert_ne!(env.instantiate(&scheme, &mut vg, &subst), Type::Var(shared));
    }
}
