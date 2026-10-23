//! Host evidence tests for portable result scope.

#[cfg(test)]
mod tests {
    use crate::errors::ErrorWitness;
    use chelis_type_core::*;

    fn quantified(tvars: &[u32]) -> Scheme<ErrorWitness> {
        let mut scheme = Scheme::mono(Type::<ErrorWitness>::Unit);
        scheme.tvars = tvars.iter().copied().map(TypeVar).collect();
        scheme
    }

    #[test]
    fn a_component_is_freshened_whole_or_kept_shared() {
        for offset in [0, 10, 100] {
            let var = |id| Type::<ErrorWitness>::Var(TypeVar(offset + id));
            for container in [
                var(1),
                Type::<ErrorWitness>::function(vec![var(1)], var(1)),
                Type::<ErrorWitness>::tuple(vec![var(1)]),
                Type::<ErrorWitness>::adt("List".to_string(), vec![var(1)]),
                Type::<ErrorWitness>::kinded_adt(
                    "Holder".to_string(),
                    vec![NominalArg::Type(var(1))],
                ),
                Type::<ErrorWitness>::reference(var(1)),
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
                    assert_eq!(
                        shared.body,
                        Type::<ErrorWitness>::Unit,
                        "scope closure binds no value type"
                    );
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
        let var = |id| Type::<ErrorWitness>::Var(TypeVar(id));
        let function = |id| Type::<ErrorWitness>::function(vec![var(id)], var(id));
        let equation = ResultConstraint::Annotation {
            actual: Type::<ErrorWitness>::tuple(vec![var(0), function(1)]),
            declared: Type::<ErrorWitness>::tuple(vec![var(2), function(3)]),
        };
        let mut scheme = quantified(&[1, 2, 3]);
        ResultScope::new(&[equation]).retain_closed_quantifiers(&mut scheme);
        assert_eq!(scheme.tvars, [TypeVar(1), TypeVar(3)]);
    }

    #[test]
    fn unknown_constructor_scope_reaches_dimensions_ranks_and_precision() {
        let equation = ResultConstraint::Annotation {
            actual: Type::<ErrorWitness>::Var(TypeVar(0)),
            declared: Type::<ErrorWitness>::Tensor(
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
            actual: Type::<ErrorWitness>::Tensor(
                vec![Dim::Var(DimVar(0))],
                TensorPrec::Var(TypeVar(1)),
            ),
            declared: Type::<ErrorWitness>::Tensor(
                vec![Dim::Var(DimVar(2))],
                TensorPrec::Var(TypeVar(3)),
            ),
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
        scheme.body = Type::<ErrorWitness>::Var(shared);
        scheme.result_origin = Some(ResultOrigin {
            body: Type::<ErrorWitness>::Var(shared),
            tvar_restrictions: vec![],
            tvars: vec![],
            dvars: vec![],
            rvars: vec![],
            equations: vec![],
        });
        let subst = chelis_type_core::Subst::new();
        assert_eq!(
            instantiate(&scheme, &mut vg, &subst),
            Type::<ErrorWitness>::Var(shared)
        );
        scheme.result_origin.as_mut().unwrap().tvars.push(shared);
        assert_ne!(
            instantiate(&scheme, &mut vg, &subst),
            Type::<ErrorWitness>::Var(shared)
        );
    }
}
