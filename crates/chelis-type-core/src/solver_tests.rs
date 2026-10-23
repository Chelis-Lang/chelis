use alloc::{boxed::Box, string::ToString, vec, vec::Vec};
use chelis_unord::UnordSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct TestEvidence;

#[test]
fn substitution_resolves_variables_inside_a_late_bound_constructor() {
    let outer = Type::<TestEvidence>::Var(TypeVar(100));
    let inner = Type::<TestEvidence>::Var(TypeVar(101));
    let mut subst = Subst::<TestEvidence>::new();
    unify(
        &outer,
        &Type::<TestEvidence>::tuple(vec![inner.clone()]),
        &mut subst,
    )
    .unwrap();
    unify(&inner, &Type::<TestEvidence>::Prim(Prim::F32), &mut subst).unwrap();
    assert_eq!(
        subst.apply(&outer),
        Type::<TestEvidence>::tuple(vec![Type::<TestEvidence>::Prim(Prim::F32)])
    );
    assert!(
        unify(
            &outer,
            &Type::<TestEvidence>::tuple(vec![Type::<TestEvidence>::Prim(Prim::Bool)]),
            &mut subst
        )
        .is_err()
    );
}

#[test]
fn explicit_recursion_exclusions_remain_monomorphic() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let level = subst.enter_level(&vg);
    let pinned = vg.fresh_tvar();
    let free = vg.fresh_tvar();
    let ty = Type::tuple(vec![Type::Var(pinned), Type::Var(free)]);
    subst.leave_level(level, &vg);
    let excluded = UnordSet::from_iter([pinned]);
    let scheme = crate::scheme::generalize(&ty, &subst, &excluded, None, &[]);
    assert_eq!(scheme.tvars, vec![free]);
    assert_eq!(scheme.body, ty);
    let instance = crate::scheme::instantiate(&scheme, &mut vg, &subst);
    let Type::Tuple(elements) = instance else {
        panic!("tuple instance");
    };
    assert_eq!(elements[0], Type::Var(pinned));
    assert_ne!(elements[1], Type::Var(free));
    unify(&elements[1], &Type::Prim(Prim::F32), &mut subst).unwrap();
    assert_eq!(subst.apply(&Type::Var(pinned)), Type::Var(pinned));
}
use super::*;

#[test]
fn protected_label_aliases_keep_identity_after_compression_and_literal_binding() {
    let mut subst = Subst::<TestEvidence>::new();
    let root = DimVar(0);
    subst.protect_dimensions([root]);
    unify_dim(&Dim::Var(root), &Dim::Name("fixed".into()), &mut subst).unwrap();
    for i in 1..5 {
        unify_dim(&Dim::Var(DimVar(i)), &Dim::Var(DimVar(i - 1)), &mut subst).unwrap();
    }
    assert_eq!(subst.apply_dim(&Dim::Var(DimVar(4))), Dim::Var(root));
    unify_dim(&Dim::Var(root), &Dim::Lit(2), &mut subst).unwrap();
    assert_eq!(subst.constraint_dim(&Dim::Var(DimVar(4))), Dim::Lit(2));
    assert_eq!(
        subst.semantic_dim(&Dim::Var(DimVar(4))),
        Dim::Name("fixed".into())
    );
    assert!(matches!(subst.apply_dim(&Dim::Var(DimVar(4))), Dim::Var(_)));
    assert!(unify_dim(&Dim::Var(DimVar(4)), &Dim::Lit(3), &mut subst).is_err());
}

#[test]
fn protected_labels_do_not_leak_from_a_failed_type_unification() {
    let mut subst = Subst::<TestEvidence>::new();
    subst.protect_dimensions([DimVar(0)]);
    let tensor = |dim| Type::<TestEvidence>::Tensor(vec![dim], TensorPrec::Concrete(Prim::F32));
    let left = Type::<TestEvidence>::tuple(vec![
        tensor(Dim::Var(DimVar(0))),
        Type::<TestEvidence>::Unit,
    ]);
    let right =
        Type::<TestEvidence>::tuple(vec![tensor(Dim::Name("fixed".into())), tensor(Dim::Lit(1))]);
    assert!(unify(&left, &right, &mut subst).is_err());
    assert_eq!(subst.dimension_label(DimVar(0)), None);
    unify_dim(&Dim::Var(DimVar(0)), &Dim::Name("other".into()), &mut subst).unwrap();
}

#[test]
fn dimension_label_undo_replays_repeated_copy_writes_and_skips_noops() {
    let mut subst = Subst::<TestEvidence>::new();
    subst.set_dimension_label(DimVar(0), "fixed".into());
    subst.set_dimension_label(DimVar(1), "other".into());
    subst.set_dimension_label(DimVar(2), "before".into());
    subst.label_unification_depth = 1;
    // Exercise the &Subst writer used by scheme instantiation, including.
    //
    // repeated writes to both a pre-existing entry and a fresh entry.
    //
    for target in [DimVar(2), DimVar(3)] {
        subst.copy_dimension_label(DimVar(0), target);
        subst.copy_dimension_label(DimVar(0), target);
        subst.copy_dimension_label(DimVar(1), target);
    }
    assert_eq!(subst.dimension_label_undo.with(|journal| journal.len()), 4);
    subst.label_unification_depth = 0;
    subst.finish_dimension_label_writes(true);
    assert_eq!(subst.dimension_label(DimVar(2)).as_deref(), Some("before"));
    assert_eq!(subst.dimension_label(DimVar(3)), None);
    assert!(
        subst
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
}

#[test]
fn protected_label_nested_alias_failure_preserves_prior_success() {
    let mut subst = Subst::<TestEvidence>::new();
    subst.protect_dimensions([DimVar(0), DimVar(1)]);
    let tensor = |d| Type::<TestEvidence>::Tensor(vec![d], TensorPrec::Concrete(Prim::F32));
    unify(
        &tensor(Dim::Var(DimVar(0))),
        &tensor(Dim::Name("fixed".into())),
        &mut subst,
    )
    .unwrap();
    assert!(
        subst
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
    let lhs = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::tuple(vec![
            tensor(Dim::Var(DimVar(1))),
            tensor(Dim::Var(DimVar(2))),
        ]),
        Type::<TestEvidence>::Unit,
    ]);
    let rhs = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::tuple(vec![
            tensor(Dim::Name("other".into())),
            tensor(Dim::Var(DimVar(1))),
        ]),
        tensor(Dim::Lit(1)),
    ]);
    assert!(unify(&lhs, &rhs, &mut subst).is_err());
    assert_eq!(subst.dimension_label(DimVar(0)).as_deref(), Some("fixed"));
    assert_eq!(subst.dimension_label(DimVar(1)), None);
    assert_eq!(subst.dimension_label(DimVar(2)), None);
    assert_eq!(subst.label_unification_depth, 0);
    assert!(
        subst
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
    // Existing dim aliases are outside this narrow label rollback. retry.
    //
    // must nevertheless be free to install a different semantic label.
    //
    unify(
        &tensor(Dim::Var(DimVar(1))),
        &tensor(Dim::Name("retry".into())),
        &mut subst,
    )
    .unwrap();
    assert_eq!(subst.dimension_label(DimVar(2)).as_deref(), Some("retry"));
    assert!(
        subst
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
}

#[test]
#[cfg(feature = "serde")]
fn dimension_label_journal_is_absent_from_clones_and_wire() {
    let mut subst = Subst::<TestEvidence>::new();
    subst.label_unification_depth = 1;
    subst.set_dimension_label(DimVar(0), "fixed".into());
    assert_eq!(subst.dimension_label_undo.with(|journal| journal.len()), 1);
    let cloned = subst.clone();
    assert_eq!(cloned.label_unification_depth, 0);
    assert!(
        cloned
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
    let bytes = bincode::serialize(&subst).unwrap();
    assert_eq!(bytes, bincode::serialize(&cloned).unwrap());
    let decoded: Subst<TestEvidence> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded.label_unification_depth, 0);
    assert!(
        decoded
            .dimension_label_undo
            .with(|journal| journal.is_empty())
    );
    subst.label_unification_depth = 0;
    subst.finish_dimension_label_writes(true);
    assert_eq!(subst.dimension_label(DimVar(0)), None);
    for independent in [&cloned, &decoded] {
        assert_eq!(
            independent.dimension_label(DimVar(0)).as_deref(),
            Some("fixed")
        );
    }
}

#[test]
#[cfg(feature = "serde")]
fn mapped_axis_annotation_survives_clone_wire_and_compose() {
    let mapped = DimVar(0);
    let mut subst = Subst::<TestEvidence>::new();
    subst.mark_mapped_axis(mapped, 3);
    assert_eq!(subst.mapped_axis(mapped), Some(3));
    assert_eq!(
        subst.dimension_label(mapped),
        None,
        "compiler transform metadata must never become a user dimension name"
    );

    let cloned = subst.clone();
    let decoded: Subst<TestEvidence> =
        bincode::deserialize(&bincode::serialize(&subst).unwrap()).unwrap();
    for carrier in [&cloned, &decoded] {
        assert_eq!(carrier.mapped_axis(mapped), Some(3));
    }

    let mut receiver = Subst::<TestEvidence>::new();
    receiver.compose(&subst).unwrap();
    assert_eq!(receiver.mapped_axis(mapped), Some(3));
    subst.compose(&Subst::<TestEvidence>::new()).unwrap();
    assert_eq!(subst.mapped_axis(mapped), Some(3));
}

#[test]
fn mapped_axis_is_removed_before_every_anchored_spread_split() {
    for axis in 0..=4 {
        let batch = DimVar(0);
        let pre = RankVar(0);
        let post = RankVar(1);
        let mut subst = Subst::<TestEvidence>::new();
        subst.mark_mapped_axis(batch, axis);
        let row = Type::<TestEvidence>::Tensor(
            vec![
                Dim::Rank(pre),
                Dim::Name("seq".into()),
                Dim::Rank(post),
                Dim::Var(batch),
            ],
            TensorPrec::Concrete(Prim::F32),
        );
        let authored = vec![
            Dim::Name("left".into()),
            Dim::Name("inner".into()),
            Dim::Name("seq".into()),
            Dim::Name("right".into()),
        ];
        let mut ground = authored.clone();
        ground.insert(axis, Dim::Name("batch".into()));
        let ground = Type::<TestEvidence>::Tensor(ground, TensorPrec::Concrete(Prim::F32));

        unify(&row, &ground, &mut subst).unwrap_or_else(|error| panic!("axis {axis}: {error:?}"));
        assert_eq!(
            subst.resolve_rvar(pre),
            authored[..2],
            "axis {axis}: mapped batch leaked into the leading spread"
        );
        assert_eq!(
            subst.resolve_rvar(post),
            authored[3..],
            "axis {axis}: mapped batch leaked into the trailing spread"
        );
        assert_eq!(
            subst.resolve_dvar(batch),
            Dim::Name("batch".into()),
            "axis {axis}: mapped batch lost its actual extent identity"
        );
    }
}

#[test]
fn protected_labels_are_freshened_but_captured_ids_are_not() {
    let mut vg = VarGen::default();
    let quantified = vg.fresh_dvar();
    let capture = vg.fresh_dvar();
    let alias = vg.fresh_dvar();
    let alias_chain = vg.fresh_dvar();
    let mut subst = Subst::<TestEvidence>::new();
    subst.protect_dimensions([quantified, capture]);
    for v in [quantified, capture] {
        unify_dim(&Dim::Var(v), &Dim::Name("fixed".into()), &mut subst).unwrap();
    }
    // Exercise both alias operand orders. binding prefers the protected.
    //
    // representative, including after path compression.
    //
    unify_dim(&Dim::Var(capture), &Dim::Var(alias), &mut subst).unwrap();
    unify_dim(&Dim::Var(alias_chain), &Dim::Var(alias), &mut subst).unwrap();
    assert!(subst.is_protected_dimension(alias_chain));
    assert_eq!(
        subst.constraint_dim(&Dim::Var(alias_chain)),
        Dim::Var(capture)
    );
    let scheme = Scheme::<TestEvidence> {
        result_origin: None,
        constraints: vec![],
        tvars: vec![],
        tvar_restrictions: vec![],
        dvars: vec![quantified],
        rvars: vec![],
        body: Type::<TestEvidence>::Tensor(
            vec![Dim::Var(quantified), Dim::Var(alias_chain)],
            TensorPrec::Concrete(Prim::F32),
        ),
    };
    let a = crate::scheme::instantiate(&scheme, &mut vg, &subst);
    let b = crate::scheme::instantiate(&scheme, &mut vg, &subst);
    let (Type::<TestEvidence>::Tensor(a, _), Type::<TestEvidence>::Tensor(b, _)) = (a, b) else {
        panic!("tensor")
    };
    assert_ne!(a[0], b[0]);
    for dims in [&a, &b] {
        let Dim::Var(fresh) = dims[0] else {
            panic!("fresh dimension")
        };
        assert!(!subst.is_protected_dimension(fresh));
        let Dim::Var(captured) = dims[1] else {
            panic!("captured dimension")
        };
        assert!(subst.is_protected_dimension(captured));
        assert_eq!(subst.constraint_dim(&dims[1]), Dim::Var(capture));
    }
    unify_dim(&a[0], &Dim::Lit(2), &mut subst).unwrap();
    unify_dim(&b[0], &Dim::Lit(3), &mut subst).unwrap();
    assert_eq!(subst.semantic_dim(&a[0]), Dim::Name("fixed".into()));
    assert_eq!(subst.semantic_dim(&b[0]), Dim::Name("fixed".into()));
}

#[cfg(feature = "serde")]
#[test]
fn protected_dimension_queries_agree_after_live_and_decoded_resume() {
    let mut vg = VarGen::default();
    let historical = vg.fresh_dvar();
    let alias = vg.fresh_dvar();
    let mut published = Subst::<TestEvidence>::new();
    published.protect_dimensions([historical]);
    unify_dim(
        &Dim::Var(historical),
        &Dim::Name("fixed".into()),
        &mut published,
    )
    .unwrap();
    unify_dim(&Dim::Var(alias), &Dim::Var(historical), &mut published).unwrap();
    let mut live = published.clone();
    let mut decoded: Subst<TestEvidence> =
        bincode::deserialize(&bincode::serialize(&published).unwrap()).unwrap();
    // Raw mid-check serialization intentionally omits active protection.
    //
    // The public context entry point resumes BOTH paths before inference.
    //
    assert!(live.is_protected_dimension(alias));
    assert!(!decoded.is_protected_dimension(alias));
    for subst in [&mut live, &mut decoded] {
        subst.resume_for_new_check(&vg);
        assert!(!subst.is_protected_dimension(alias));
        assert_eq!(
            subst.semantic_dim(&Dim::Var(alias)),
            Dim::Name("fixed".into())
        );
    }
    let current = vg.fresh_dvar();
    let current_alias = vg.fresh_dvar();
    for subst in [&mut live, &mut decoded] {
        subst.protect_dimensions([current]);
        unify_dim(&Dim::Var(current), &Dim::Name("fixed".into()), subst).unwrap();
        unify_dim(&Dim::Var(current), &Dim::Var(current_alias), subst).unwrap();
        assert!(subst.is_protected_dimension(current_alias));
        assert!(!subst.is_protected_dimension(alias));
    }
    assert_eq!(
        bincode::serialize(&live).unwrap(),
        bincode::serialize(&decoded).unwrap()
    );
}

#[test]
fn protected_label_pending_result_stays_monomorphic_in_both_generalizers() {
    let mut vg = VarGen::default();
    let mut subst = Subst::<TestEvidence>::new();
    let level = subst.enter_level(&vg);
    let operand = vg.fresh_tvar();
    let dim = vg.fresh_dvar();
    let rank = vg.fresh_rvar();
    let precision = vg.fresh_tvar();
    subst.protect_dimensions([dim]);
    unify_dim(&Dim::Var(dim), &Dim::Name("fixed".into()), &mut subst).unwrap();
    let result = Type::<TestEvidence>::Tensor(
        vec![Dim::Var(dim), Dim::Rank(rank)],
        TensorPrec::Var(precision),
    );
    subst.record_deferred_tensor_operand(
        operand,
        DeferredOperandGate::<TestEvidence>::Copy {
            result: Box::new(result.clone()),
            location: None,
        },
    );
    subst.leave_level(level, &vg);
    let scheme = crate::scheme::generalize(&result, &subst, &UnordSet::new(), None, &[]);
    assert!(scheme.dvars.is_empty());
    assert!(scheme.rvars.is_empty());
    assert!(scheme.tvars.is_empty());
    let reference = crate::scheme::generalize_over_environment(
        &result,
        &subst,
        core::iter::empty::<&Scheme<TestEvidence>>(),
        &UnordSet::new(),
    );
    assert_eq!(scheme.tvars, reference.tvars);
    assert_eq!(scheme.dvars, reference.dvars);
    assert_eq!(scheme.rvars, reference.rvars);
}

#[test]
fn protected_label_shared_type_refinement_rolls_back_with_failed_retry() {
    let mut subst = Subst::<TestEvidence>::new();
    let tv = TypeVar(0);
    let dv = DimVar(0);
    subst.protect_dimensions([dv]);
    let tensor = |d| Type::<TestEvidence>::Tensor(vec![d], TensorPrec::Concrete(Prim::F32));
    unify(
        &Type::<TestEvidence>::Var(tv),
        &tensor(Dim::Name("fixed".into())),
        &mut subst,
    )
    .unwrap();
    let lhs = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Var(tv),
        Type::<TestEvidence>::Unit,
    ]);
    let rhs = Type::<TestEvidence>::tuple(vec![tensor(Dim::Var(dv)), tensor(Dim::Lit(1))]);
    assert!(unify(&lhs, &rhs, &mut subst).is_err());
    assert_eq!(
        subst.apply(&Type::<TestEvidence>::Var(tv)),
        tensor(Dim::Name("fixed".into()))
    );
    assert_eq!(subst.dimension_label(dv), None);
    unify_dim(&Dim::Var(dv), &Dim::Name("other".into()), &mut subst).unwrap();
}

#[test]
fn protected_label_is_a_rank_anchor_without_erasing_its_identity() {
    let mut subst = Subst::<TestEvidence>::new();
    let dv = DimVar(0);
    subst.protect_dimensions([dv]);
    unify_dim(&Dim::Var(dv), &Dim::Name("fixed".into()), &mut subst).unwrap();
    let row = [Dim::Rank(RankVar(0)), Dim::Name("fixed".into())];
    let ground = [Dim::Name("batch".into()), Dim::Var(dv)];
    unify_row_against_ground(&row, &ground, &mut subst).unwrap();
    assert_eq!(subst.constraint_dim(&Dim::Var(dv)), Dim::Var(dv));
    assert_eq!(
        subst.resolve_rvar(RankVar(0)),
        vec![Dim::Name("batch".into())]
    );
}

#[test]
fn protected_label_in_rank_spread_cannot_collide_with_inserted_name() {
    let mut subst = Subst::<TestEvidence>::new();
    let dv = DimVar(0);
    subst.protect_dimensions([dv]);
    unify_dim(&Dim::Var(dv), &Dim::Name("fixed".into()), &mut subst).unwrap();
    subst.insert_rank(RankVar(0), vec![Dim::Var(dv)]);
    let tensor = |dims| Type::<TestEvidence>::Tensor(dims, TensorPrec::Concrete(Prim::F32));
    let args = [tensor(vec![Dim::Rank(RankVar(0))])];
    let ret = tensor(vec![Dim::Rank(RankVar(0)), Dim::Name("fixed".into())]);
    assert!(check_introduced_name_rank_collision(&args, &ret, &subst).is_err());
}

#[test]
fn protected_label_compose_rejects_conflicting_alias_classes_transactionally() {
    let mut left = Subst::<TestEvidence>::new();
    let mut right = Subst::<TestEvidence>::new();
    left.protect_dimensions([DimVar(0)]);
    right.protect_dimensions([DimVar(1)]);
    unify_dim(&Dim::Var(DimVar(0)), &Dim::Name("fixed".into()), &mut left).unwrap();
    unify_dim(&Dim::Var(DimVar(1)), &Dim::Name("other".into()), &mut right).unwrap();
    right.insert_dim(DimVar(0), Dim::Var(DimVar(1)));
    assert!(left.compose(&right).is_err());
    assert_eq!(
        left.constraint_dim(&Dim::Var(DimVar(0))),
        Dim::Var(DimVar(0))
    );
    assert_eq!(
        left.semantic_dim(&Dim::Var(DimVar(0))),
        Dim::Name("fixed".into())
    );
}

fn var_gen() -> VarGen {
    VarGen::default()
}

#[test]
fn the_transition_log_serializes_as_the_plain_sequence_and_finds_every_mint_level() {
    // The log is stored persistently but must keep the wire form of the.
    //
    // `Vec` it replaced, and its descent must answer what the sorted scan.
    //
    // answered for every identifier.
    //
    let mut log = TransitionLog::default();
    let mut plain = Vec::new();
    for index in 0..200u32 {
        let transition = LevelTransition {
            kind: if index % 2 == 0 {
                LevelTransitionKind::Enter
            } else {
                LevelTransitionKind::Leave
            },
            level: index % 7,
            watermarks: VarWatermarks {
                next_tvar: index * 3,
                next_dvar: index * 2,
                next_rvar: index,
            },
        };
        log.push(transition);
        plain.push(transition);
    }
    #[cfg(feature = "serde")]
    let decoded: TransitionLog = {
        assert_eq!(
            serde_json::to_string(&log).unwrap(),
            serde_json::to_string(&plain).unwrap()
        );
        assert_eq!(
            bincode::serialize(&log).unwrap(),
            bincode::serialize(&plain).unwrap()
        );
        bincode::deserialize(&bincode::serialize(&plain).unwrap()).unwrap()
    };
    for id in 0..700u32 {
        let scanned = plain
            .partition_point(|transition| transition.watermarks.next_tvar <= id)
            .checked_sub(1)
            .map(|index| plain[index]);
        let found = log
            .last_at_or_below(id, |transition| transition.watermarks.next_tvar)
            .copied();
        assert_eq!(found, scanned, "type variable {id}");
        #[cfg(feature = "serde")]
        assert_eq!(
            decoded
                .last_at_or_below(id, |transition| transition.watermarks.next_tvar)
                .copied(),
            scanned,
            "decoded type variable {id}"
        );
    }
}

#[test]
fn level_transitions_restore_parent_mint_levels_for_every_id_class() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();

    let child = subst.enter_level(&vg);
    let child_t = vg.fresh_tvar();
    let child_d = vg.fresh_dvar();
    let child_r = vg.fresh_rvar();
    assert_eq!(subst.level_of_tvar(child_t), 1);
    assert_eq!(subst.level_of_dvar(child_d), 1);
    assert_eq!(subst.level_of_rvar(child_r), 1);

    subst.leave_level(child, &vg);
    let parent_t = vg.fresh_tvar();
    let parent_d = vg.fresh_dvar();
    let parent_r = vg.fresh_rvar();
    assert_eq!(subst.level_of_tvar(parent_t), 0);
    assert_eq!(subst.level_of_dvar(parent_d), 0);
    assert_eq!(subst.level_of_rvar(parent_r), 0);

    let sibling = subst.enter_level(&vg);
    let sibling_t = vg.fresh_tvar();
    let sibling_d = vg.fresh_dvar();
    let sibling_r = vg.fresh_rvar();
    assert_eq!(subst.level_of_tvar(sibling_t), 1);
    assert_eq!(subst.level_of_dvar(sibling_d), 1);
    assert_eq!(subst.level_of_rvar(sibling_r), 1);
    subst.leave_level(sibling, &vg);
    assert_eq!(subst.current_level(), 0);
    assert_eq!(subst.level_of_tvar(child_t), 1);
    assert_eq!(subst.level_of_dvar(child_d), 1);
    assert_eq!(subst.level_of_rvar(child_r), 1);
    assert_eq!(subst.level_of_tvar(parent_t), 0);
    assert_eq!(subst.level_of_dvar(parent_d), 0);
    assert_eq!(subst.level_of_rvar(parent_r), 0);
}

#[test]
fn binding_older_type_to_younger_composite_lowers_every_reachable_class() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let older = vg.fresh_tvar();
    let env = [Scheme::<TestEvidence>::mono(Type::<TestEvidence>::Var(
        older,
    ))];
    let inner = subst.enter_level(&vg);
    let younger_t = vg.fresh_tvar();
    let younger_d = vg.fresh_dvar();
    let younger_r = vg.fresh_rvar();

    let composite = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Var(younger_t),
        Type::<TestEvidence>::Tensor(
            vec![Dim::Var(younger_d), Dim::Rank(younger_r)],
            TensorPrec::Var(younger_t),
        ),
    ]);
    unify(&Type::<TestEvidence>::Var(older), &composite, &mut subst).expect("composite bind");

    assert_eq!(subst.level_of_tvar(younger_t), 0);
    assert_eq!(subst.level_of_dvar(younger_d), 0);
    assert_eq!(subst.level_of_rvar(younger_r), 0);
    subst.leave_level(inner, &vg);
    let escaped = crate::scheme::generalize_over_environment(
        &composite,
        &subst,
        env.iter(),
        &UnordSet::new(),
    );
    assert!(escaped.tvars.is_empty());
    assert!(escaped.dvars.is_empty());
    assert!(escaped.rvars.is_empty());
}

#[test]
fn dimension_rank_and_rank_alias_bindings_lower_younger_variables() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let older_d = vg.fresh_dvar();
    let older_ground_r = vg.fresh_rvar();
    let older_alias_r = vg.fresh_rvar();
    let env = [Scheme::<TestEvidence>::mono(Type::<TestEvidence>::tuple(
        vec![
            Type::<TestEvidence>::Tensor(vec![Dim::Var(older_d)], TensorPrec::Concrete(Prim::F32)),
            Type::<TestEvidence>::Tensor(
                vec![Dim::Rank(older_ground_r)],
                TensorPrec::Concrete(Prim::F32),
            ),
            Type::<TestEvidence>::Tensor(
                vec![Dim::Rank(older_alias_r)],
                TensorPrec::Concrete(Prim::F32),
            ),
        ],
    ))];
    let inner = subst.enter_level(&vg);
    let younger_d_alias = vg.fresh_dvar();
    let younger_d_in_rank = vg.fresh_dvar();
    let younger_r = vg.fresh_rvar();

    unify_dim(&Dim::Var(older_d), &Dim::Var(younger_d_alias), &mut subst).expect("dimension alias");
    bind_rvar(older_ground_r, &[Dim::Var(younger_d_in_rank)], &mut subst)
        .expect("ground rank bind");
    unify_row_against_row(
        &[Dim::Rank(older_alias_r)],
        &[Dim::Rank(younger_r)],
        &mut subst,
    )
    .expect("rank alias");

    assert_eq!(subst.level_of_dvar(younger_d_alias), 0);
    assert_eq!(subst.level_of_dvar(younger_d_in_rank), 0);
    assert_eq!(subst.level_of_rvar(younger_r), 0);
    subst.leave_level(inner, &vg);
    let escaped = crate::scheme::generalize_over_environment(
        &Type::<TestEvidence>::tuple(vec![
            Type::<TestEvidence>::Tensor(
                vec![Dim::Var(younger_d_alias)],
                TensorPrec::Concrete(Prim::F32),
            ),
            Type::<TestEvidence>::Tensor(
                vec![Dim::Var(younger_d_in_rank)],
                TensorPrec::Concrete(Prim::F32),
            ),
            Type::<TestEvidence>::Tensor(
                vec![Dim::Rank(younger_r)],
                TensorPrec::Concrete(Prim::F32),
            ),
        ]),
        &subst,
        env.iter(),
        &UnordSet::new(),
    );
    assert!(escaped.dvars.is_empty());
    assert!(escaped.rvars.is_empty());
}

#[test]
fn expanded_rank_dimensions_and_precision_aliases_are_lowered() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let older_type = vg.fresh_tvar();
    let older_precision = vg.fresh_tvar();
    let env = [Scheme::<TestEvidence>::mono(Type::<TestEvidence>::tuple(
        vec![
            Type::<TestEvidence>::Var(older_type),
            Type::<TestEvidence>::Tensor(vec![], TensorPrec::Var(older_precision)),
        ],
    ))];
    let inner = subst.enter_level(&vg);
    let younger_rank = vg.fresh_rvar();
    let younger_dim = vg.fresh_dvar();
    let younger_precision = vg.fresh_tvar();

    bind_rvar(younger_rank, &[Dim::Var(younger_dim)], &mut subst).expect("inner ground rank bind");
    unify(
        &Type::<TestEvidence>::Var(older_type),
        &Type::<TestEvidence>::Tensor(
            vec![Dim::Rank(younger_rank)],
            TensorPrec::Concrete(Prim::F32),
        ),
        &mut subst,
    )
    .expect("expanded rank reaches the older type");
    unify_tensor_prec(
        &TensorPrec::Var(older_precision),
        &TensorPrec::Var(younger_precision),
        &mut subst,
    )
    .expect("precision alias");

    assert_eq!(subst.level_of_dvar(younger_dim), 0);
    assert_eq!(subst.level_of_tvar(younger_precision), 0);
    subst.leave_level(inner, &vg);
    let escaped = crate::scheme::generalize_over_environment(
        &Type::<TestEvidence>::tuple(vec![
            Type::<TestEvidence>::Tensor(
                vec![Dim::Var(younger_dim)],
                TensorPrec::Concrete(Prim::F32),
            ),
            Type::<TestEvidence>::Tensor(vec![], TensorPrec::Var(younger_precision)),
        ]),
        &subst,
        env.iter(),
        &UnordSet::new(),
    );
    assert!(escaped.tvars.is_empty());
    assert!(escaped.dvars.is_empty());
}

#[test]
fn tensor_precision_variables_generalize_at_an_ordinary_boundary() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let boundary = subst.enter_level(&vg);
    let precision = vg.fresh_tvar();
    let linked = vg.fresh_tvar();
    unify_tensor_prec(
        &TensorPrec::Var(precision),
        &TensorPrec::Var(linked),
        &mut subst,
    )
    .expect("precision variables link");
    subst.leave_level(boundary, &vg);

    let scheme = crate::scheme::generalize(
        &Type::<TestEvidence>::Tensor(vec![], TensorPrec::Var(precision)),
        &subst,
        &UnordSet::new(),
        None,
        &[],
    );
    assert_eq!(scheme.tvars, vec![linked]);

    let instantiated = crate::scheme::instantiate(&scheme, &mut vg, &subst);
    let Type::<TestEvidence>::Tensor(_, TensorPrec::Var(fresh_precision)) = instantiated else {
        panic!("precision instantiation must remain a tensor precision variable");
    };
    unify_tensor_prec(
        &TensorPrec::Var(fresh_precision),
        &TensorPrec::Concrete(Prim::F32),
        &mut subst,
    )
    .expect("precision variable binds to a concrete precision");
    assert_eq!(
        subst.apply_tensor_prec(&TensorPrec::Var(fresh_precision)),
        TensorPrec::Concrete(Prim::F32)
    );
}

/// chelis#1417: identifying a `Numeric`-bounded variable with a.
///
/// `Float`-bounded one keeps the INTERSECTION, in both orders.
///
///
/// This is the direction test, not the conflict test. The two acceptance.
///
/// tests that exercise narrowing through a program detect.
///
/// `merge_tvar_restrictions` returning `Err`, so a reversion that merely.
///
/// widens survives them. this one asserts the surviving family IS `Float`.
///
/// and that `i32` is consequently rejected.
///
///
/// It does not isolate either mechanism, and measurement rather than.
///
/// reasoning says so. `bind_tvar`'s `merged_restriction` and.
///
/// `narrow_tvar_restriction` each independently suffice to produce the.
///
/// narrowing, so reverting either ALONE leaves this green (and leaves all.
///
/// 1460 crate tests green). only reverting BOTH reds it. Neither is dead.
///
/// code — both sit on live paths — but neither is individually necessary.
///
/// for this behavior, so no single test can discriminate them. Five.
///
/// program shapes, including the reversed order, were tried and none.
///
/// discriminates either.
///
#[test]
fn identifying_a_numeric_variable_with_a_float_one_keeps_float() {
    for numeric_first in [true, false] {
        let mut vg = var_gen();
        let mut subst = Subst::<TestEvidence>::new();
        let numeric = vg.fresh_tvar();
        let float = vg.fresh_tvar();
        subst
            .narrow_tvar_restriction(numeric, TypeVarRestriction::ActiveNumeric)
            .expect("fresh variable takes a bound");
        subst
            .narrow_tvar_restriction(float, TypeVarRestriction::ActiveFloat)
            .expect("fresh variable takes a bound");

        let (left, right) = if numeric_first {
            (numeric, float)
        } else {
            (float, numeric)
        };
        unify(
            &Type::<TestEvidence>::Var(left),
            &Type::<TestEvidence>::Var(right),
            &mut subst,
        )
        .expect("Float is a subset of Numeric, so the two are compatible");

        // Whichever variable survives the binding must admit floats only.
        //
        let surviving = match subst.resolve_tvar(left) {
            Type::<TestEvidence>::Var(v) => v,
            other => panic!("expected a variable, got {other}"),
        };
        assert_eq!(
            subst.tvar_restriction(surviving),
            Some(TypeVarRestriction::ActiveFloat),
            "identifying Numeric with Float must narrow to Float, not keep \
                 Numeric (numeric_first = {numeric_first})"
        );

        // And the narrowing is observable: i32 is in Numeric but not in.
        //
        // Float, so it must now be rejected.
        //
        let error = unify(
            &Type::<TestEvidence>::Var(surviving),
            &Type::<TestEvidence>::Prim(Prim::Int32),
            &mut subst,
        )
        .expect_err("a narrowed variable must reject a non-float dtype");
        assert!(
            matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch)
                && error.message.contains("Float"),
            "expected a Float PrecisionMismatch, got: {}",
            error.message
        );
    }
}

#[test]
fn active_float_restrictions_propagate_through_aliases_and_generalization() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let boundary = subst.enter_level(&vg);
    let restricted = vg.fresh_tvar();
    let alias = vg.fresh_tvar();
    subst
        .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    unify(
        &Type::<TestEvidence>::Var(restricted),
        &Type::<TestEvidence>::Var(alias),
        &mut subst,
    )
    .expect("restriction must flow through an ordinary type-variable alias");
    subst.leave_level(boundary, &vg);

    let scheme = crate::scheme::generalize(
        &Type::<TestEvidence>::Var(alias),
        &subst,
        &UnordSet::new(),
        None,
        &[],
    );
    assert_eq!(scheme.tvars.len(), 1);
    assert_eq!(
        scheme.tvar_restrictions,
        vec![(scheme.tvars[0], TypeVarRestriction::ActiveFloat)]
    );

    let mut call_subst = Subst::<TestEvidence>::new();
    let instantiated = crate::scheme::instantiate(&scheme, &mut vg, &call_subst);
    let Type::<TestEvidence>::Var(call_var) = instantiated else {
        panic!("generalized alias must instantiate to a fresh variable");
    };
    let error = unify(
        &Type::<TestEvidence>::Var(call_var),
        &Type::<TestEvidence>::Prim(Prim::Bool),
        &mut call_subst,
    )
    .expect_err("instantiated alias must retain its active-float domain");
    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    assert!(error.message.contains("active float dtype"));
}

#[cfg(feature = "serde")]
#[test]
fn operation_value_restrictions_survive_serialization_and_late_tensor_binding() {
    for restriction in [
        TypeVarRestriction::FloatValue,
        TypeVarRestriction::IntValue,
        TypeVarRestriction::NumericValue,
    ] {
        let mut vg = var_gen();
        let value = vg.fresh_tvar();
        let scheme = Scheme::<TestEvidence> {
            result_origin: None,
            tvars: vec![value],
            tvar_restrictions: vec![(value, restriction)],
            dvars: vec![],
            rvars: vec![],
            constraints: vec![],
            body: Type::<TestEvidence>::Var(value),
        };
        let scheme: Scheme<TestEvidence> =
            bincode::deserialize(&bincode::serialize(&scheme).unwrap()).unwrap();
        assert_eq!(scheme.tvar_restrictions, vec![(value, restriction)]);
        for (dtype, accepts) in [
            (Prim::F32, restriction != TypeVarRestriction::IntValue),
            (Prim::Int32, restriction != TypeVarRestriction::FloatValue),
            (Prim::Bool, false),
        ] {
            let mut subst = Subst::<TestEvidence>::new();
            let instantiated = crate::scheme::instantiate(&scheme, &mut vg, &subst);
            let precision = vg.fresh_tvar();
            unify(
                &instantiated,
                &Type::<TestEvidence>::Tensor(vec![], TensorPrec::Var(precision)),
                &mut subst,
            )
            .unwrap();
            assert_eq!(
                unify(
                    &Type::<TestEvidence>::Var(precision),
                    &Type::<TestEvidence>::Prim(dtype),
                    &mut subst
                )
                .is_ok(),
                accepts,
                "{restriction:?} at {dtype:?}"
            );
        }
    }
}

#[test]
fn authored_dtype_bounds_still_reject_whole_tensor_types() {
    let mut vg = var_gen();
    for restriction in [
        TypeVarRestriction::ActiveFloat,
        TypeVarRestriction::ActiveInt,
        TypeVarRestriction::ActiveNumeric,
    ] {
        let mut subst = Subst::<TestEvidence>::new();
        let value = vg.fresh_tvar();
        subst.narrow_tvar_restriction(value, restriction).unwrap();
        assert!(
            unify(
                &Type::<TestEvidence>::Var(value),
                &Type::<TestEvidence>::Tensor(vec![], TensorPrec::Concrete(Prim::F32)),
                &mut subst
            )
            .is_err()
        );
    }
}

#[test]
fn compose_preserves_active_float_restrictions_from_both_operands() {
    let receiver_var = TypeVar(80_001);
    let other_var = TypeVar(80_002);
    let shared_var = TypeVar(80_003);
    let mut receiver = Subst::<TestEvidence>::new();
    receiver
        .narrow_tvar_restriction(receiver_var, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    receiver
        .narrow_tvar_restriction(shared_var, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    let other = Subst::<TestEvidence>::new();
    other
        .narrow_tvar_restriction(other_var, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    other
        .narrow_tvar_restriction(shared_var, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");

    receiver
        .compose(&other)
        .expect("identical and independent restrictions compose");

    assert_eq!(
        receiver.tvar_restriction(receiver_var),
        Some(TypeVarRestriction::ActiveFloat),
        "compose must retain the receiver's restriction"
    );
    assert_eq!(
        receiver.tvar_restriction(other_var),
        Some(TypeVarRestriction::ActiveFloat),
        "compose must merge the right operand's restriction"
    );
    assert_eq!(
        receiver.tvar_restriction(shared_var),
        Some(TypeVarRestriction::ActiveFloat),
        "the same restriction on a shared key must compose idempotently"
    );
}

#[test]
fn compose_canonicalizes_active_float_restrictions_through_alias_chains() {
    let source = TypeVar(80_004);
    let middle = TypeVar(80_005);
    let terminal = TypeVar(80_006);
    let mut receiver = Subst::<TestEvidence>::new();
    receiver
        .narrow_tvar_restriction(source, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    receiver
        .insert_type(source, Type::<TestEvidence>::Var(middle))
        .expect("restricted source aliases an unresolved variable");
    let mut other = Subst::<TestEvidence>::new();
    other
        .insert_type(middle, Type::<TestEvidence>::Var(terminal))
        .expect("unrestricted alias chain is valid");

    receiver
        .compose(&other)
        .expect("a restriction follows the composed alias chain");
    assert_eq!(
        receiver.apply(&Type::<TestEvidence>::Var(source)),
        Type::<TestEvidence>::Var(terminal)
    );
    assert_eq!(
        receiver.tvar_restriction(terminal),
        Some(TypeVarRestriction::ActiveFloat)
    );

    receiver
        .insert_type(terminal, Type::<TestEvidence>::Prim(Prim::F64))
        .expect("the composed restriction accepts an active float");
    assert_eq!(
        receiver.apply(&Type::<TestEvidence>::Var(source)),
        Type::<TestEvidence>::Prim(Prim::F64)
    );
    assert_eq!(receiver.tvar_restriction(terminal), None);
}

#[test]
fn compose_rejects_forbidden_bindings_transactionally_in_either_operand() {
    let restricted_in_receiver = TypeVar(80_007);
    let mut receiver = Subst::<TestEvidence>::new();
    receiver
        .narrow_tvar_restriction(restricted_in_receiver, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    let mut other = Subst::<TestEvidence>::new();
    other
        .insert_type(
            restricted_in_receiver,
            Type::<TestEvidence>::Prim(Prim::Int64),
        )
        .expect("the independent substitution does not know the restriction");
    let receiver_types_before = receiver.types_snapshot();
    let receiver_restrictions_before = receiver.tvar_restrictions_snapshot();

    let error = receiver
        .compose(&other)
        .expect_err("the merged substitution must enforce the receiver restriction");
    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    assert!(error.message.contains("active float dtype"));
    assert_eq!(receiver.types_snapshot(), receiver_types_before);
    assert_eq!(
        receiver.tvar_restrictions_snapshot(),
        receiver_restrictions_before
    );

    let restricted_in_other = TypeVar(80_008);
    let mut receiver = Subst::<TestEvidence>::new();
    receiver
        .insert_type(restricted_in_other, Type::<TestEvidence>::Prim(Prim::Int16))
        .expect("the receiver does not yet know the restriction");
    let other = Subst::<TestEvidence>::new();
    other
        .narrow_tvar_restriction(restricted_in_other, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    let receiver_types_before = receiver.types_snapshot();
    let receiver_restrictions_before = receiver.tvar_restrictions_snapshot();

    let error = receiver
        .compose(&other)
        .expect_err("the merged substitution must enforce the right restriction");
    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    assert!(error.message.contains("active float dtype"));
    assert_eq!(receiver.types_snapshot(), receiver_types_before);
    assert_eq!(
        receiver.tvar_restrictions_snapshot(),
        receiver_restrictions_before
    );
}

#[test]
fn direct_type_insertion_cannot_bypass_active_float_restrictions() {
    let restricted = TypeVar(80_009);
    let mut subst = Subst::<TestEvidence>::new();
    subst
        .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");

    let error = subst
        .insert_type(restricted, Type::<TestEvidence>::Prim(Prim::Int32))
        .expect_err("a forbidden direct insertion must fail explicitly");

    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    assert!(error.message.contains("active float dtype"));
    assert!(error.message.contains("i32"));
    assert_eq!(
        subst.apply(&Type::<TestEvidence>::Var(restricted)),
        Type::<TestEvidence>::Var(restricted)
    );
    assert_eq!(
        subst.tvar_restriction(restricted),
        Some(TypeVarRestriction::ActiveFloat),
        "a rejected direct binding must preserve the restriction ledger"
    );
}

#[test]
fn direct_type_insertion_transfers_and_resolves_active_float_restrictions() {
    let restricted = TypeVar(80_010);
    let alias = TypeVar(80_011);
    let mut subst = Subst::<TestEvidence>::new();
    subst
        .narrow_tvar_restriction(restricted, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");

    subst
        .insert_type(restricted, Type::<TestEvidence>::Var(alias))
        .expect("an unresolved alias preserves the semantic domain");
    assert_eq!(subst.tvar_restriction(restricted), None);
    assert_eq!(
        subst.tvar_restriction(alias),
        Some(TypeVarRestriction::ActiveFloat)
    );

    subst
        .insert_type(alias, Type::<TestEvidence>::Prim(Prim::F16))
        .expect("an active float satisfies the transferred restriction");
    assert_eq!(
        subst.apply(&Type::<TestEvidence>::Var(restricted)),
        Type::<TestEvidence>::Prim(Prim::F16)
    );
    assert_eq!(subst.tvar_restriction(alias), None);
}

#[test]
fn structural_projection_preserves_domains_across_an_exact_signature_boundary() {
    let inferred_precision = TypeVar(80_012);
    let declared_precision = TypeVar(80_013);
    let mut subst = Subst::<TestEvidence>::new();
    subst
        .narrow_tvar_restriction(inferred_precision, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    let inferred_tensor =
        Type::<TestEvidence>::Tensor(vec![Dim::Lit(2)], TensorPrec::Var(inferred_precision));
    let declared_tensor =
        Type::<TestEvidence>::Tensor(vec![Dim::Lit(2)], TensorPrec::Var(declared_precision));
    let inferred = Type::<TestEvidence>::function(
        vec![
            Type::<TestEvidence>::reference(inferred_tensor),
            Type::<TestEvidence>::Var(inferred_precision),
        ],
        Type::<TestEvidence>::Unit,
    );
    let declared = Type::<TestEvidence>::function(
        vec![
            Type::<TestEvidence>::reference(declared_tensor),
            Type::<TestEvidence>::Var(declared_precision),
        ],
        Type::<TestEvidence>::Unit,
    );

    subst
        .project_tvar_restrictions(&inferred, &declared)
        .expect("matching signature positions carry the inferred domain");
    assert_eq!(
        subst.tvar_restriction(declared_precision),
        Some(TypeVarRestriction::ActiveFloat)
    );
    let error = subst
        .insert_type(declared_precision, Type::<TestEvidence>::Prim(Prim::Bool))
        .expect_err("the projected declaration must reject a non-float");
    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
}

#[test]
fn structural_projection_is_transactional_when_a_later_slot_conflicts() {
    let first_source = TypeVar(80_014);
    let second_source = TypeVar(80_015);
    let first_target = TypeVar(80_016);
    let mut subst = Subst::<TestEvidence>::new();
    subst
        .narrow_tvar_restriction(first_source, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    subst
        .narrow_tvar_restriction(second_source, TypeVarRestriction::ActiveFloat)
        .expect("an unbounded variable accepts any single dtype family");
    let inferred = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Var(first_source),
        Type::<TestEvidence>::Var(second_source),
    ]);
    let declared = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Var(first_target),
        Type::<TestEvidence>::Prim(Prim::Int32),
    ]);
    let restrictions_before = subst.tvar_restrictions_snapshot();

    let error = subst
        .project_tvar_restrictions(&inferred, &declared)
        .expect_err("a forbidden declared slot rejects the whole projection");
    assert!(matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch));
    assert_eq!(subst.tvar_restrictions_snapshot(), restrictions_before);
    assert_eq!(subst.tvar_restriction(first_target), None);
}

#[test]
fn pp1_monomorphic_binding_is_lowered_before_a_sibling_boundary() {
    let mut vg = var_gen();
    let mut subst = Subst::<TestEvidence>::new();
    let pp1_level = subst.enter_level(&vg);
    let pp1_var = vg.fresh_tvar();
    subst.leave_level(pp1_level, &vg);
    subst.lower_type_to_current(&Type::<TestEvidence>::Var(pp1_var));

    let env = [Scheme::<TestEvidence>::mono(Type::<TestEvidence>::Var(
        pp1_var,
    ))];
    let sibling_level = subst.enter_level(&vg);
    let sibling_var = vg.fresh_tvar();
    subst.leave_level(sibling_level, &vg);
    let scheme = crate::scheme::generalize_over_environment(
        &Type::<TestEvidence>::tuple(vec![
            Type::<TestEvidence>::Var(pp1_var),
            Type::<TestEvidence>::Var(sibling_var),
        ]),
        &subst,
        env.iter(),
        &UnordSet::new(),
    );
    assert_eq!(scheme.tvars, vec![sibling_var]);
}

#[test]
fn unify_same_prim() {
    let mut s = Subst::<TestEvidence>::new();
    assert!(
        unify(
            &Type::<TestEvidence>::Prim(Prim::F32),
            &Type::<TestEvidence>::Prim(Prim::F32),
            &mut s
        )
        .is_ok()
    );
}

#[test]
fn unify_different_prim_fails() {
    let mut s = Subst::<TestEvidence>::new();
    let err = unify(
        &Type::<TestEvidence>::Prim(Prim::F32),
        &Type::<TestEvidence>::Prim(Prim::Bf16),
        &mut s,
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
}

#[test]
fn unify_tvar_binds() {
    let mut g = var_gen();
    let v = g.fresh_type();
    let mut s = Subst::<TestEvidence>::new();
    assert!(unify(&v, &Type::<TestEvidence>::Prim(Prim::F32), &mut s).is_ok());
    assert_eq!(s.apply(&v), Type::<TestEvidence>::Prim(Prim::F32));
}

#[test]
fn unify_fn_types() {
    let mut s = Subst::<TestEvidence>::new();
    let f1 = Type::<TestEvidence>::function(
        vec![Type::<TestEvidence>::Prim(Prim::F32)],
        Type::<TestEvidence>::Prim(Prim::F32),
    );
    let f2 = Type::<TestEvidence>::function(
        vec![Type::<TestEvidence>::Prim(Prim::F32)],
        Type::<TestEvidence>::Prim(Prim::F32),
    );
    assert!(unify(&f1, &f2, &mut s).is_ok());
}

#[test]
fn unify_fn_arity_mismatch() {
    let mut s = Subst::<TestEvidence>::new();
    let f1 = Type::<TestEvidence>::function(
        vec![Type::<TestEvidence>::Prim(Prim::F32)],
        Type::<TestEvidence>::Prim(Prim::F32),
    );
    let f2 = Type::<TestEvidence>::function(
        vec![
            Type::<TestEvidence>::Prim(Prim::F32),
            Type::<TestEvidence>::Prim(Prim::F32),
        ],
        Type::<TestEvidence>::Prim(Prim::F32),
    );
    let err = unify(&f1, &f2, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ArityMismatch));
}

fn tprec(p: Prim) -> TensorPrec {
    TensorPrec::Concrete(p)
}

#[test]
fn unify_tensor_same_dims() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(
        vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
        tprec(Prim::F32),
    );
    let t2 = Type::<TestEvidence>::Tensor(
        vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
        tprec(Prim::F32),
    );
    assert!(unify(&t1, &t2, &mut s).is_ok());
}

#[test]
fn unify_tensor_dim_mismatch() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("seq".into())], tprec(Prim::F32));
    let err = unify(&t1, &t2, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
}

#[test]
fn unify_tensor_precision_mismatch() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
    let err = unify(&t1, &t2, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
}

#[test]
fn unify_dim_var_binds() {
    let mut g = var_gen();
    let dv = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Var(dv)], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    assert!(unify(&t1, &t2, &mut s).is_ok());
    assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Name("batch".into()));
}

// chelis#258 Tier-2 rank polymorphism: the rank-variable unification arm.
//

#[test]
fn unify_rank_binds_whole_shape() {
    let mut g = var_gen();
    let rv = g.fresh_rvar();
    let mut s = Subst::<TestEvidence>::new();
    let rank_tensor = Type::<TestEvidence>::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
    let concrete = Type::<TestEvidence>::Tensor(
        vec![Dim::Name("a".into()), Dim::Name("b".into())],
        tprec(Prim::F32),
    );
    assert!(unify(&rank_tensor, &concrete, &mut s).is_ok());
    // The rank var now stands for the whole 2-dim shape. applying expands it.
    //
    assert_eq!(s.apply(&rank_tensor), concrete);
}

#[test]
fn unify_rank_to_rank_aliases() {
    let mut g = var_gen();
    let r1 = g.fresh_rvar();
    let r2 = g.fresh_rvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Rank(r1)], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Rank(r2)], tprec(Prim::F32));
    assert!(unify(&t1, &t2, &mut s).is_ok());
    // Binding one to a concrete shape resolves both (alias chased).
    //
    let concrete = Type::<TestEvidence>::Tensor(vec![Dim::Name("n".into())], tprec(Prim::F32));
    assert!(unify(&t2, &concrete, &mut s).is_ok());
    assert_eq!(s.apply(&t1), concrete);
}

#[test]
fn unify_rank_precision_still_checked() {
    // The precision slot unifies independently of the rank binding.
    //
    let mut g = var_gen();
    let rv = g.fresh_rvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("a".into())], tprec(Prim::Int32));
    assert!(
        unify(&t1, &t2, &mut s).is_err(),
        "f32 vs i32 precision must fail even with a rank var"
    );
}

// chelis#258 Tier-3 rank polymorphism: name-preserving multi-spread.
//
// unification (`tensor[..pre, seq, ..post]`).
//

fn name(n: &str) -> Dim {
    Dim::Name(n.into())
}

#[test]
fn unify_rank_middle_split_preserves_names() {
    // `tensor[..pre, seq, ..post]` vs `tensor[batch, seq, hidden]` splits.
    //
    // at the unique named anchor: pre:=[batch], post:=[hidden]. The output.
    //
    // `tensor[..pre, ..post]` then carries the surviving names through.
    //
    let mut g = var_gen();
    let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
        tprec(Prim::F32),
    );
    let ground = Type::<TestEvidence>::Tensor(
        vec![name("batch"), name("seq"), name("hidden")],
        tprec(Prim::F32),
    );
    assert!(unify(&row, &ground, &mut s).is_ok());
    let out = Type::<TestEvidence>::Tensor(vec![Dim::Rank(pre), Dim::Rank(post)], tprec(Prim::F32));
    assert_eq!(
        s.apply(&out),
        Type::<TestEvidence>::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32)),
        "output drops `seq`, keeps batch/hidden by name"
    );
}

#[test]
fn unify_rank_multi_anchor_split() {
    // Two named anchors, three spreads: `[..a, seq, ..b, head, ..c]`.
    //
    let mut g = var_gen();
    let (a, b, c) = (g.fresh_rvar(), g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![
            Dim::Rank(a),
            name("seq"),
            Dim::Rank(b),
            name("head"),
            Dim::Rank(c),
        ],
        tprec(Prim::F32),
    );
    let ground = Type::<TestEvidence>::Tensor(
        vec![
            name("batch"),
            name("seq"),
            name("kv"),
            name("head"),
            name("dim"),
        ],
        tprec(Prim::F32),
    );
    assert!(unify(&row, &ground, &mut s).is_ok());
    let out = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(a), Dim::Rank(b), Dim::Rank(c)],
        tprec(Prim::F32),
    );
    assert_eq!(
        s.apply(&out),
        Type::<TestEvidence>::Tensor(
            vec![name("batch"), name("kv"), name("dim")],
            tprec(Prim::F32)
        ),
    );
}

#[test]
fn unify_rank_anchor_absent_rejected() {
    let mut g = var_gen();
    let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
        tprec(Prim::F32),
    );
    let ground =
        Type::<TestEvidence>::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32));
    let err = unify(&row, &ground, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    assert!(
        err.message.contains("no named `seq` axis"),
        "{}",
        err.message
    );
}

#[test]
fn unify_rank_anchor_ambiguous_rejected() {
    let mut g = var_gen();
    let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
        tprec(Prim::F32),
    );
    let ground =
        Type::<TestEvidence>::Tensor(vec![name("seq"), name("x"), name("seq")], tprec(Prim::F32));
    let err = unify(&row, &ground, &mut s).unwrap_err();
    assert!(err.message.contains("ambiguous"), "{}", err.message);
}

#[test]
fn unify_rank_literal_operand_rejected() {
    // Name↔Lit hazard: a fully-literal operand carries no name to find.
    //
    // the anchor — hard reject, never a guessed reduction.
    //
    let mut g = var_gen();
    let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
        tprec(Prim::F32),
    );
    let ground = Type::<TestEvidence>::Tensor(
        vec![Dim::Lit(2), Dim::Lit(768), Dim::Lit(4)],
        tprec(Prim::F32),
    );
    let err = unify(&row, &ground, &mut s).unwrap_err();
    assert!(
        err.message.contains("no named `seq` axis"),
        "{}",
        err.message
    );
}

#[test]
fn unify_rank_too_small_rejected() {
    // Two leading anchors but a rank-1 operand: not enough dims.
    //
    let mut g = var_gen();
    let r = g.fresh_rvar();
    let mut s = Subst::<TestEvidence>::new();
    let row =
        Type::<TestEvidence>::Tensor(vec![name("a"), name("b"), Dim::Rank(r)], tprec(Prim::F32));
    let ground = Type::<TestEvidence>::Tensor(vec![name("a")], tprec(Prim::F32));
    let err = unify(&row, &ground, &mut s).unwrap_err();
    assert!(err.message.contains("rank too small"), "{}", err.message);
}

#[test]
fn unify_two_adjacent_spreads_rejected() {
    // Two adjacent spreads split against a ground is undetermined.
    //
    let mut g = var_gen();
    let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(vec![Dim::Rank(a), Dim::Rank(b)], tprec(Prim::F32));
    let ground = Type::<TestEvidence>::Tensor(vec![name("x"), name("y")], tprec(Prim::F32));
    let err = unify(&row, &ground, &mut s).unwrap_err();
    assert!(
        err.message.contains("two adjacent rank spreads"),
        "{}",
        err.message
    );
}

#[test]
fn unify_identical_rows_ok() {
    // The body output `[..pre, ..post]` vs the declared return.
    //
    // `[..pre, ..post]` (same vars) is the identical-row self-check.
    //
    let mut g = var_gen();
    let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
        tprec(Prim::F32),
    );
    assert!(unify(&row, &row.clone(), &mut s).is_ok());
}

#[test]
fn unify_distinct_anchor_rows_rejected() {
    let mut g = var_gen();
    let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
    let mut s = Subst::<TestEvidence>::new();
    let r1 = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(a), name("seq"), Dim::Rank(b)],
        tprec(Prim::F32),
    );
    let r2 = Type::<TestEvidence>::Tensor(
        vec![Dim::Rank(a), name("head"), Dim::Rank(b)],
        tprec(Prim::F32),
    );
    assert!(unify(&r1, &r2, &mut s).is_err());
}

#[test]
fn unify_trailing_anchor_by_name() {
    // `[..pre, seq]` vs `[a, b, seq]`: pre:=[a,b], seq at the tail.
    //
    let mut g = var_gen();
    let pre = g.fresh_rvar();
    let mut s = Subst::<TestEvidence>::new();
    let row = Type::<TestEvidence>::Tensor(vec![Dim::Rank(pre), name("seq")], tprec(Prim::F32));
    let ground =
        Type::<TestEvidence>::Tensor(vec![name("a"), name("b"), name("seq")], tprec(Prim::F32));
    assert!(unify(&row, &ground, &mut s).is_ok());
    let out = Type::<TestEvidence>::Tensor(vec![Dim::Rank(pre)], tprec(Prim::F32));
    assert_eq!(
        s.apply(&out),
        Type::<TestEvidence>::Tensor(vec![name("a"), name("b")], tprec(Prim::F32)),
    );
}

#[test]
fn unify_name_with_lit_accepts() {
    // Issue Chelis-Lang/chelis#219 Option A: a Name slot on the.
    //
    // left and a concrete Lit on the right unify without binding.
    //
    // any substitution.
    //
    let mut s = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Name("batch".into()), &Dim::Lit(2), &mut s).is_ok());
    // Subst stays untouched — the Name carries diagnostics, the.
    //
    // Lit carries the value, nothing flows in.
    //
    assert_eq!(s.dims_len(), 0);
}

#[test]
fn unify_name_with_lit_accepts_symmetric() {
    // Issue #219 Option A: the symmetric direction (Lit on left,.
    //
    // Name on right) also unifies.
    //
    let mut s = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Lit(3), &Dim::Name("hidden".into()), &mut s).is_ok());
    assert_eq!(s.dims_len(), 0);
}

#[test]
fn unify_name_distinct_names_still_errors() {
    // Issue #219 regression-lock: the Name <-> Lit relaxation.
    //
    // must not bleed into Name <-> Name. Distinct symbolic names.
    //
    // continue to surface as DimensionMismatch (this is the.
    //
    // dim-polymorphism rigidity rule from §4.4).
    //
    let mut s = Subst::<TestEvidence>::new();
    let err = unify_dim(&Dim::Name("batch".into()), &Dim::Name("seq".into()), &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
}

#[test]
fn unify_lit_distinct_still_errors() {
    // Issue #219 regression-lock: distinct Lit <-> Lit still.
    //
    // errors. The permissive arm is narrowly Name <-> Lit.
    //
    let mut s = Subst::<TestEvidence>::new();
    let err = unify_dim(&Dim::Lit(2), &Dim::Lit(3), &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
}

#[test]
fn unify_dim_transitive_binding_resolves() {
    let mut g = var_gen();
    let d0 = g.fresh_dvar();
    let d1 = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Var(d0), &Dim::Var(d1), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(d1), &Dim::Name("batch".into()), &mut s).is_ok());
    assert_eq!(s.apply_dim(&Dim::Var(d0)), Dim::Name("batch".into()));
}

#[test]
fn unify_wildcard_matches_anything() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    assert!(unify(&t1, &t2, &mut s).is_ok());
}

#[test]
fn unify_wildcard_with_var_leaves_var_free() {
    // chelis#143 invariant: `unify_dim(Wildcard, Var(v))` must.
    //
    // succeed without binding `v`. Binding `v := Wildcard` would.
    //
    // freeze the dim var. a later concrete arg in the same sig.
    //
    // could not then constrain `v` because `apply_dim` would.
    //
    // resolve `Var(v) → Wildcard` and the permissive Wildcard arm.
    //
    // would silently accept any concrete value.
    //
    //
    // The current implementation satisfies this because both.
    //
    // wildcard arms match before the `(Var(_), _) => bind_dvar(...)`.
    //
    // arm (Rust `match` is first-match-wins) and neither binds the.
    //
    // dim var. chelis#1801 added the explicit.
    //
    // `(Wildcard, Var(v)) | (Var(v), Wildcard)` arm ahead of the.
    //
    // permissive one. it RECORDS the meeting in.
    //
    // `wildcard_touched_classes` and still leaves `v` free, which is.
    //
    // what lets the concrete binding below still happen. This test.
    //
    // pins the property in case a future refactor reorders the arms.
    //
    // or makes either wildcard arm bind.
    //
    let mut g = var_gen();
    let dv = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();

    // Wildcard ↔ Var: succeeds, var remains free.
    //
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(dv), &mut s).is_ok());
    assert_eq!(
        s.apply_dim(&Dim::Var(dv)),
        Dim::Var(dv),
        "Wildcard ↔ Var must not bind the var to Wildcard",
    );

    // Symmetric direction: Var ↔ Wildcard also leaves the var.
    //
    // free.
    //
    let dv2 = g.fresh_dvar();
    assert!(unify_dim(&Dim::Var(dv2), &Dim::Wildcard, &mut s).is_ok());
    assert_eq!(s.apply_dim(&Dim::Var(dv2)), Dim::Var(dv2));

    // A next concrete unification with the still-free var.
    //
    // binds it to the concrete dim.
    //
    assert!(unify_dim(&Dim::Var(dv), &Dim::Lit(2), &mut s).is_ok());
    assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Lit(2));

    // And once the var is concretely bound, a conflicting concrete.
    //
    // value trips DimensionMismatch — the cross-position contract.
    //
    // the sig promised is now enforced end to end.
    //
    let err = unify_dim(&Dim::Var(dv), &Dim::Lit(3), &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
}

#[test]
fn a_wildcard_meeting_a_class_is_recorded_and_survives_clone_and_compose() {
    // chelis#1801. `unify_dim`'s wildcard arm leaves the variable free.
    //
    // by design, so without this record nothing downstream can tell a.
    //
    // variable that met a runtime extent from one that met nothing. The.
    //
    // application rule reads the record to decide what an unconstrained.
    //
    // instantiation variable denotes (spec/04-type-system.md section.
    //
    // 3.2).
    //
    //
    // Evidentiary status: a mechanism test for a mechanism this change.
    //
    // introduces. There is no base-sha reading to compare against. what.
    //
    // it proves is that each of the three carriers keeps the evidence.
    //
    let mut g = var_gen();
    let touched = g.fresh_dvar();
    let untouched = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();

    assert!(!s.dvar_class_met_wildcard(touched));
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(touched), &mut s).is_ok());
    assert!(
        s.dvar_class_met_wildcard(touched),
        "the Wildcard-then-Var direction records the meeting",
    );
    assert!(
        !s.dvar_class_met_wildcard(untouched),
        "and records nothing about a variable that met nothing",
    );

    // The symmetric direction records too.
    //
    let mut symmetric = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Var(touched), &Dim::Wildcard, &mut symmetric).is_ok());
    assert!(symmetric.dvar_class_met_wildcard(touched));

    // A wildcard against a wildcard names no variable, so it records.
    //
    // nothing: the permissive arm is still what handles it.
    //
    let mut both_wild = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Wildcard, &Dim::Wildcard, &mut both_wild).is_ok());
    assert!(!both_wild.dvar_class_met_wildcard(touched));

    // `Subst::<TestEvidence>::clone` carries the record. A clone that dropped it would.
    //
    // silently un-record the meeting for every checking context that.
    //
    // clones the substitution into its own state.
    //
    let cloned = s.clone();
    assert!(cloned.dvar_class_met_wildcard(touched));
    assert!(!cloned.dvar_class_met_wildcard(untouched));

    // `Subst::<TestEvidence>::compose` carries it in BOTH directions: from the receiver.
    //
    // (through the trial clone) and from the argument (through.
    //
    // `absorb_wildcard_touches`). The argument direction is the one a.
    //
    // compose can lose, because `compose_bindings` merges only the.
    //
    // binding maps.
    //
    let mut receiver = Subst::<TestEvidence>::new();
    receiver.compose(&s).expect("composing a rename is valid");
    assert!(
        receiver.dvar_class_met_wildcard(touched),
        "compose adopts the argument's wildcard-meeting evidence",
    );

    let mut carrier = s.clone();
    carrier
        .compose(&Subst::<TestEvidence>::new())
        .expect("composing an empty substitution is valid");
    assert!(
        carrier.dvar_class_met_wildcard(touched),
        "and keeps its own",
    );
}

#[test]
fn a_recorded_meeting_does_not_by_itself_mean_the_class_is_still_free() {
    // chelis#1801 negative parity for the test above, and the reason the.
    //
    // absorbing site asks the substitution a second question. The record.
    //
    // is monotone: it says the meeting happened, never that the variable.
    //
    // is still unconstrained. A literal in a later argument of the same.
    //
    // call binds the variable, and `constraint_dim` is what reports that.
    //
    //
    // `apply_dim` cannot: it deliberately answers `Var(v)` for a.
    //
    // LABELLED variable that resolved to a concrete dim, so an absorber.
    //
    // written against it would overwrite the literal with `*`.
    //
    //
    // Evidentiary status: a disposition lock on which predicate answers.
    //
    // "is this variable still unbound", not a regression test. The state.
    //
    // is built here directly. Swapping the absorbing site to `apply_dim`.
    //
    // was measured and changed no source-level reading, so no source.
    //
    // fixture pins this. what it pins is the contract the absorbing site.
    //
    // relies on.
    //
    let mut g = var_gen();
    let dv = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    s.set_dimension_label(dv, "k".to_string());

    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(dv), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(dv), &Dim::Lit(3), &mut s).is_ok());

    assert!(
        s.dvar_class_met_wildcard(dv),
        "the meeting is still recorded after the literal binding",
    );
    assert_eq!(
        s.constraint_dim(&Dim::Var(dv)),
        Dim::Lit(3),
        "but the variable is bound, so it denotes the literal",
    );
    assert_eq!(
        s.apply_dim(&Dim::Var(dv)),
        Dim::Var(dv),
        "`apply_dim` keeps the label's identity and cannot answer this",
    );
}

#[test]
fn a_class_keeps_its_wildcard_evidence_when_a_later_binding_re_roots_it() {
    // chelis#1925 round 2, at the unit level. `unify_dim` resolves both.
    //
    // operands through `constraint_dim` before it matches, so the meeting.
    //
    // is recorded on whatever rooted the class at that moment, and a.
    //
    // later `bind_dvar` in the same call re-roots the class over it. Ask.
    //
    // the variable, or two ends of the class, and a meeting recorded on a.
    //
    // third member is invisible. the class that carried it is then.
    //
    // generalized instead of denoting the extent it met.
    //
    //
    // Evidentiary status: regression test for the merge this change.
    //
    // introduces. Making `merge_dvar_class_flags` a no-op fails the loop.
    //
    // below, which is how it was checked.
    //
    let mut g = var_gen();
    let (first, middle, last) = (g.fresh_dvar(), g.fresh_dvar(), g.fresh_dvar());
    let mut s = Subst::<TestEvidence>::new();

    // The meeting happens while `middle` still roots its own class.
    //
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(middle), &mut s).is_ok());
    assert!(s.dvar_class_met_wildcard(middle));

    // Two more members join and the class re-roots away from `middle`.
    //
    assert!(unify_dim(&Dim::Var(middle), &Dim::Var(last), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(first), &Dim::Var(middle), &mut s).is_ok());

    for member in [first, middle, last] {
        assert!(
            s.dvar_class_met_wildcard(member),
            "every member answers for the class, whichever member roots it",
        );
    }
}

#[test]
fn an_authored_binder_pins_its_class_even_after_it_meets_a_runtime_extent() {
    // Negative parity for the row above, and the exclusion.
    //
    // `spec/04-type-system.md` section 3.2 states: a class an argument.
    //
    // binds to a named dimension is a CLAIM on the runtime extent,.
    //
    // checked by a section 4.7 guard, so absorbing it to `*` would erase.
    //
    // the claim. The meeting is still recorded. the pin is what stops the.
    //
    // absorption, and both answers have to survive the same merge.
    //
    //
    // Evidentiary status: disposition lock on the two predicates the.
    //
    // absorbing site asks. The source-level twin is.
    //
    // `an_enclosing_binder_survives_a_wildcard_in_its_own_alias_class`.
    //
    let mut g = var_gen();
    let binder = g.fresh_dvar();
    let minted = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    s.protect_dimensions([binder]);
    s.set_dimension_label(binder, "seq".to_string());

    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(minted), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(minted), &Dim::Var(binder), &mut s).is_ok());

    assert!(
        s.dvar_class_met_wildcard(minted),
        "the class still carries the meeting after the binder joins it",
    );
    assert!(
        s.dvar_class_is_binder_pinned(minted),
        "and the authored binder pins the class against absorption",
    );
    assert!(
        s.dvar_class_is_binder_pinned(binder),
        "asked from either member",
    );
}

#[test]
fn a_name_arm_class_is_pinned_by_protection_not_by_a_second_ledger() {
    // chelis#1925 round 2, the first of the paths that pin or union.
    //
    // WITHOUT a plain `insert_dim`. `bind_dvar`'s name arm records the.
    //
    // name as a LABEL and returns, so `constraint_dim` still answers the.
    //
    // variable and the absorbing site has no binding to skip on.
    //
    //
    // The arm needs no ledger of its own. It fires only when the variable.
    //
    // is protected or already labelled, and a `Dim::Name` operand arises.
    //
    // only from the definition under check, whose binders are protected.
    //
    // The protected half is therefore the whole reachable half, and it.
    //
    // pins through the rigidity ledger.
    //
    //
    // Evidentiary status, per assertion. The protected rows are a.
    //
    // disposition lock on the pin the absorbing site relies on. The last.
    //
    // row records a state no source program reaches, measured rather than.
    //
    // assumed: `def named(w: tensor[batch, f32])` passed as a polymorphic.
    //
    // argument reads `() -> tensor[batch, f32]` and executes identically.
    //
    // with and without a pin on this arm, because the reachable spelling.
    //
    // BINDS the name rather than labelling it, and `constraint_dim`.
    //
    // reports that. If a change ever makes the labelled-but-unprotected.
    //
    // state reachable, this row is where the missing pin goes.
    //
    let mut g = var_gen();
    let protected = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    s.protect_dimensions([protected]);

    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(protected), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(protected), &Dim::Name("seq".to_string()), &mut s).is_ok());

    assert_eq!(
        s.constraint_dim(&Dim::Var(protected)),
        Dim::Var(protected),
        "the name arm binds nothing, so the class still reads as unbound",
    );
    assert!(s.dvar_class_met_wildcard(protected));
    assert!(
        s.dvar_class_is_binder_pinned(protected),
        "and the declared binder pins it against absorption",
    );

    let labelled = g.fresh_dvar();
    let mut unreached = Subst::<TestEvidence>::new();
    unreached.set_dimension_label(labelled, "seq".to_string());
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(labelled), &mut unreached).is_ok());
    assert!(
        unify_dim(
            &Dim::Var(labelled),
            &Dim::Name("seq".to_string()),
            &mut unreached
        )
        .is_ok()
    );
    assert!(
        !unreached.dvar_class_is_binder_pinned(labelled),
        "a labelled but UNPROTECTED variable is not pinned by this arm, and no \
             source program reaches that state today",
    );
}

#[test]
fn a_swapped_binding_direction_still_lands_the_evidence_on_the_merged_root() {
    // The second such path. When `v` is protected and the other operand.
    //
    // is not, `bind_dvar` recurses with the operands SWAPPED, so the.
    //
    // union direction flips. The merge therefore hooks the actual insert.
    //
    // rather than the entry. hooking the entry would key the evidence to.
    //
    // a variable that does not root the merged class.
    //
    //
    // Evidentiary status: disposition lock on where the hook sits.
    //
    let mut g = var_gen();
    let binder = g.fresh_dvar();
    let minted = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    s.protect_dimensions([binder]);

    // The meeting is recorded on the unprotected member, and the.
    //
    // protected one is unified FIRST, which is the direction that.
    //
    // recurses.
    //
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(minted), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(binder), &Dim::Var(minted), &mut s).is_ok());

    assert!(
        s.dvar_class_met_wildcard(binder),
        "the meeting survives the swap, asked from the protected member",
    );
    assert!(s.dvar_class_met_wildcard(minted), "and from the other one");
    assert!(
        s.dvar_class_is_binder_pinned(minted),
        "and the protected member pins the class it was merged into",
    );
}

#[test]
fn a_literal_claim_against_a_named_dimension_leaves_no_absorbable_class() {
    // The third such path. `unify_dim`'s `(Name, Lit)` arm returns `Ok`.
    //
    // binding nothing (issue #219 Option A), so a literal claim against a.
    //
    // named dimension produces no union event either. It needs no hook:.
    //
    // by the time that arm matches, `constraint_dim` resolved.
    //
    // both operands to non-variables, so no class is left for an.
    //
    // application to absorb, and the class the name came from was pinned.
    //
    // when the declared binder reached it.
    //
    //
    // Evidentiary status: a reachability receipt for that claim rather.
    //
    // than a lock on a repair. It is the row to re-run if the arm ever.
    //
    // starts binding.
    //
    let mut g = var_gen();
    let named = g.fresh_dvar();
    let bystander = g.fresh_dvar();
    let mut s = Subst::<TestEvidence>::new();
    s.protect_dimensions([named]);
    s.set_dimension_label(named, "batch".to_string());

    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(named), &mut s).is_ok());
    assert!(unify_dim(&Dim::Var(named), &Dim::Name("batch".to_string()), &mut s).is_ok());
    assert!(
        s.dvar_class_is_binder_pinned(named),
        "the declared binder pinned the class before any literal arrived",
    );

    // The literal claim now meets the NAME, not the variable.
    //
    assert!(unify_dim(&Dim::Name("batch".to_string()), &Dim::Lit(3), &mut s).is_ok());
    assert!(
        s.dvar_class_is_binder_pinned(named),
        "and the literal arm leaves the pin where it was",
    );
    assert!(
        !s.dvar_class_met_wildcard(bystander),
        "recording nothing about any other class",
    );
}

#[test]
fn compose_rekeys_class_evidence_onto_the_root_the_composed_graph_gives_it() {
    // The fourth such path, and the one a set union gets wrong.
    //
    // `compose_bindings` merges only the binding maps, so a compose can.
    //
    // JOIN two classes that were separate in either operand. Copying the.
    //
    // flags across leaves the meeting keyed to a variable that no longer.
    //
    // roots the joined class, and the query on the new root answers.
    //
    // false.
    //
    //
    // Evidentiary status: regression test for the re-canonicalization.
    //
    // Replacing `recanonicalize_class_flags` with the set union it.
    //
    // supersedes fails the second-to-last assertion, which is how it was.
    //
    // checked.
    //
    let mut g = var_gen();
    let met = g.fresh_dvar();
    let joiner = g.fresh_dvar();

    // The left operand knows the meeting and roots that class at `met`.
    //
    let mut left = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(met), &mut left).is_ok());
    assert!(left.dvar_class_met_wildcard(met));

    // The right operand knows an aliasing the left never saw, and it.
    //
    // re-roots that class onto `joiner`.
    //
    let mut right = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Var(met), &Dim::Var(joiner), &mut right).is_ok());
    assert!(!right.dvar_class_met_wildcard(joiner));

    left.compose(&right).expect("composing an alias is valid");

    assert_eq!(
        left.constraint_dim(&Dim::Var(met)),
        Dim::Var(joiner),
        "the compose joined the classes and `joiner` roots the result",
    );
    assert!(
        left.dvar_class_met_wildcard(joiner),
        "so the meeting has to be readable from the new root",
    );
    assert!(
        left.dvar_class_met_wildcard(met),
        "and from the member it was recorded on",
    );
}

#[test]
fn compose_keeps_the_binder_pin_when_it_demotes_the_protected_member() {
    // The pin's half of the path above, and the reason.
    //
    // `binder_pinned_classes` exists at all rather than the query reading.
    //
    // the rigidity ledger alone.
    //
    //
    // `bind_dvar` never demotes a protected member: when the protected.
    //
    // side would bind to an unprotected one it recurses with the operands.
    //
    // swapped, so protection stays at the root and the merge needs no.
    //
    // help. `compose_bindings` has no such rule, so a compose CAN put an.
    //
    // unprotected variable at the root of a class a declared binder pins,.
    //
    // and the set is what keeps that answer reachable.
    //
    //
    // Evidentiary status: regression test for the protection fold in.
    //
    // `recanonicalize_class_flags`. Removing that fold fails the last.
    //
    // assertion, which is how it was checked. Removing the same fold from.
    //
    // `merge_dvar_class_flags` fails nothing today, for the swap reason.
    //
    // above. it is kept because a class this compose pinned can still be.
    //
    // merged afterwards, and the set is then the only carrier.
    //
    let mut g = var_gen();
    let binder = g.fresh_dvar();
    let joiner = g.fresh_dvar();

    let mut left = Subst::<TestEvidence>::new();
    left.protect_dimensions([binder]);
    assert!(unify_dim(&Dim::Wildcard, &Dim::Var(binder), &mut left).is_ok());
    assert!(
        left.dvar_class_is_binder_pinned(binder),
        "the declared binder pins its own class before the compose",
    );

    // The other operand re-roots that class onto an unprotected variable,.
    //
    // which is the demotion `bind_dvar` would refused.
    //
    let mut right = Subst::<TestEvidence>::new();
    assert!(unify_dim(&Dim::Var(binder), &Dim::Var(joiner), &mut right).is_ok());

    left.compose(&right).expect("composing an alias is valid");

    assert_eq!(
        left.constraint_dim(&Dim::Var(binder)),
        Dim::Var(joiner),
        "the compose demoted the protected member out of the root",
    );
    assert!(
        left.dvar_class_is_binder_pinned(joiner),
        "and the pin has to survive it, or the absorption would erase the claim",
    );
}

#[test]
fn unify_tensor_rank_mismatch() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("a".into())], tprec(Prim::F32));
    let t2 = Type::<TestEvidence>::Tensor(
        vec![Dim::Name("a".into()), Dim::Name("b".into())],
        tprec(Prim::F32),
    );
    let err = unify(&t1, &t2, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
}

#[test]
fn unify_dim_cannot_recover_lit_from_two_var_tensors() {
    // chelis#158 cause partition: when two tensors both arrive at.
    //
    // a shared-dim sig with `Var(d)` (or `Wildcard`) dims and.
    //
    // never with a concrete `Lit(_)` source, `unify_dim` cannot.
    //
    // manufacture a mismatch. This is the unify-layer half of.
    //
    // chelis#158's root cause: `to_tensor` and `pad_sequences_to`.
    //
    // emit shape-erased tensors that reach the sig with no `Lit`.
    //
    // in any position. unify_dim correctly succeeds on every pair.
    //
    // and the sig's shared dim var stays free.
    //
    //
    // The fix for chelis#158 must inject a `Lit` somewhere.
    //
    // UPSTREAM of unify (most likely in the desugarer when.
    //
    // `to_tensor` is applied to a statically-known list literal).
    //
    // This test pins the unify-layer contract so any future.
    //
    // "unify_dim should be smarter" proposal can be measured.
    //
    // against the constraint that follows: from Var/Wildcard.
    //
    // inputs alone, no concrete dim can be inferred.
    //
    let mut g = var_gen();
    let d_sig = g.fresh_dvar();

    // Two sig-instantiated tensor params that share `d_sig`.
    //
    let sig_arg1 = Type::<TestEvidence>::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));
    let sig_arg2 = Type::<TestEvidence>::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));

    // Two caller tensors, both shape-erased to fresh `Var(d_caller_*)`.
    //
    // (modeling two independent to_tensor outputs).
    //
    let d_caller_a = g.fresh_dvar();
    let d_caller_b = g.fresh_dvar();
    let caller_a = Type::<TestEvidence>::Tensor(vec![Dim::Var(d_caller_a)], tprec(Prim::F32));
    let caller_b = Type::<TestEvidence>::Tensor(vec![Dim::Var(d_caller_b)], tprec(Prim::F32));

    let mut s = Subst::<TestEvidence>::new();
    // Simulate `pair_id(caller_a, caller_b)` where pair_id's sig.
    //
    // is `&tensor[d_sig, f32] -> &tensor[d_sig, f32] -> ...`.
    //
    assert!(unify(&sig_arg1, &caller_a, &mut s).is_ok());
    assert!(unify(&sig_arg2, &caller_b, &mut s).is_ok());

    // After both unifications, `d_sig` is bound to (some chain.
    //
    // of) dim vars but never to a concrete `Lit`.
    //
    let resolved = s.apply_dim(&Dim::Var(d_sig));
    assert!(
        matches!(resolved, Dim::Var(_) | Dim::Wildcard),
        "sig dim var resolved to a concrete value despite no \
             Lit source in either caller; this would mean unify_dim \
             manufactured a constraint from nothing, which is the \
             wrong place to fix chelis#158. Got: {resolved:?}"
    );

    // Symmetric: replacing one caller's Var with Wildcard (the.
    //
    // actual shape-erased symptom from to_tensor) also leaves.
    //
    // d_sig unable to bind to a Lit. No mismatch is producible.
    //
    let mut s2 = Subst::<TestEvidence>::new();
    let caller_a_wild = Type::<TestEvidence>::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
    let caller_b_wild = Type::<TestEvidence>::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
    assert!(unify(&sig_arg1, &caller_a_wild, &mut s2).is_ok());
    assert!(unify(&sig_arg2, &caller_b_wild, &mut s2).is_ok());
    let resolved2 = s2.apply_dim(&Dim::Var(d_sig));
    assert!(
        matches!(resolved2, Dim::Var(_) | Dim::Wildcard),
        "with Wildcard caller dims, d_sig also cannot resolve to \
             a concrete Lit. Got: {resolved2:?}. The fix for \
             chelis#158 must introduce a Lit at the desugar/builder \
             layer."
    );
}

// === WS-A5 precision polymorphism unification ===.
//

#[test]
fn static_dim_product_comparison_is_exact_beyond_i128() {
    let s = Subst::<TestEvidence>::new();
    let max = Dim::Lit(i64::MAX);
    let lhs = vec![max.clone(), max.clone(), max.clone(), Dim::Lit(2)];
    let rhs = vec![Dim::Lit(2), max.clone(), max.clone(), max.clone()];
    let unequal = vec![max.clone(), max.clone(), max, Dim::Lit(3)];

    assert_eq!(s.static_dim_product(&lhs), None);
    assert_eq!(s.static_dim_products_match(&lhs, &rhs), Some(true));
    assert_eq!(s.static_dim_products_match(&lhs, &unequal), Some(false));
}

#[test]
fn unify_tensor_prec_var_binds_to_concrete() {
    // tensor[batch, ?p] vs tensor[batch, f32].
    //
    // ?p must bind to f32 in the substitution.
    //
    let mut g = var_gen();
    let pv = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    unify(&t1, &t2, &mut s).expect("var precision should bind");
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(pv)),
        Type::<TestEvidence>::Prim(Prim::F32)
    );
    // After substitution, the var-precision tensor reads as f32 too.
    //
    match s.apply(&t1) {
        Type::<TestEvidence>::Tensor(_, TensorPrec::Concrete(Prim::F32)) => {}
        other => panic!("apply should resolve precision var to f32, got {other}"),
    }
}

#[test]
fn unify_tensor_prec_concrete_binds_var_other_side() {
    // Symmetric: concrete on left, var on right.
    //
    let mut g = var_gen();
    let pv = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
    unify(&t1, &t2, &mut s).expect("var precision should bind");
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(pv)),
        Type::<TestEvidence>::Prim(Prim::Bf16)
    );
}

#[test]
fn unify_two_tensor_prec_vars_link() {
    // Two precision vars unify by linking. the link survives later.
    //
    // binding through either var.
    //
    let mut g = var_gen();
    let p1 = g.fresh_tvar();
    let p2 = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p1));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p2));
    unify(&t1, &t2, &mut s).expect("two prec vars should link");
    // Now bind one of them to a concrete prim and confirm the other.
    //
    // resolves to the same prim through the link.
    //
    let t3 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F64));
    unify(&t1, &t3, &mut s).expect("link + bind should work");
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(p1)),
        Type::<TestEvidence>::Prim(Prim::F64)
    );
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(p2)),
        Type::<TestEvidence>::Prim(Prim::F64)
    );
}

#[test]
fn unify_tensor_prec_var_then_conflicting_concrete_errors() {
    // Once ?p is bound to f32, unifying tensor[..., ?p] with.
    //
    // tensor[..., bf16] must error with PrecisionMismatch.
    //
    let mut g = var_gen();
    let pv = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    let t_var = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
    let t_f32 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
    let t_bf16 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
    unify(&t_var, &t_f32, &mut s).expect("first call binds prec var to f32");
    let err = unify(&t_var, &t_bf16, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
}

#[test]
fn tensor_precision_cannot_overwrite_a_composite_type_binding() {
    let mut s = Subst::<TestEvidence>::new();
    let p = TypeVar(9001);
    let tensor = Type::<TestEvidence>::Tensor(vec![Dim::Lit(3)], tprec(Prim::F32));
    unify(&Type::<TestEvidence>::Var(p), &tensor, &mut s).unwrap();
    let err = unify_tensor_prec(
        &TensorPrec::Var(p),
        &TensorPrec::Concrete(Prim::F32),
        &mut s,
    )
    .expect_err("a tensor is not an f32 precision");
    assert!(matches!(err.kind, TypeErrorKind::TypeMismatch));
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(p)),
        tensor,
        "failed unification must retain the binding"
    );
}

#[test]
fn aliased_precision_cannot_form_a_composite_cycle() {
    let mut s = Subst::<TestEvidence>::new();
    let p = TypeVar(9002);
    let q = TypeVar(9003);
    unify(
        &Type::<TestEvidence>::Var(p),
        &Type::<TestEvidence>::Tensor(vec![Dim::Lit(3)], TensorPrec::Var(q)),
        &mut s,
    )
    .unwrap();
    let err = unify(
        &Type::<TestEvidence>::Var(q),
        &Type::<TestEvidence>::Tensor(vec![Dim::Lit(3)], TensorPrec::Var(p)),
        &mut s,
    )
    .expect_err("precision alias cycle must terminate with a diagnostic");
    assert!(matches!(err.kind, TypeErrorKind::OccursCheck));
}

#[test]
fn unify_same_prec_var_with_itself_succeeds() {
    let mut g = var_gen();
    let pv = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
    let t2 = Type::<TestEvidence>::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
    unify(&t1, &t2, &mut s).expect("self-unify must succeed");
}

#[test]
fn apply_tensor_prec_resolves_through_chain() {
    // Build a substitution that puts ?p1 -> ?p2 -> Prim::F32.
    //
    let mut g = var_gen();
    let p1 = g.fresh_tvar();
    let p2 = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    unify(
        &Type::<TestEvidence>::Var(p1),
        &Type::<TestEvidence>::Var(p2),
        &mut s,
    )
    .unwrap();
    unify(
        &Type::<TestEvidence>::Var(p2),
        &Type::<TestEvidence>::Prim(Prim::F32),
        &mut s,
    )
    .unwrap();
    // apply_tensor_prec on either var must collapse to Concrete(F32).
    //
    assert_eq!(
        s.apply_tensor_prec(&TensorPrec::Var(p1)),
        TensorPrec::Concrete(Prim::F32)
    );
    assert_eq!(
        s.apply_tensor_prec(&TensorPrec::Var(p2)),
        TensorPrec::Concrete(Prim::F32)
    );
}

#[test]
fn unify_adt_same() {
    let mut s = Subst::<TestEvidence>::new();
    let a1 =
        Type::<TestEvidence>::adt("Option".into(), vec![Type::<TestEvidence>::Prim(Prim::F32)]);
    let a2 =
        Type::<TestEvidence>::adt("Option".into(), vec![Type::<TestEvidence>::Prim(Prim::F32)]);
    assert!(unify(&a1, &a2, &mut s).is_ok());
}

#[test]
fn unify_adt_name_mismatch() {
    let mut s = Subst::<TestEvidence>::new();
    let a1 =
        Type::<TestEvidence>::adt("Option".into(), vec![Type::<TestEvidence>::Prim(Prim::F32)]);
    let a2 =
        Type::<TestEvidence>::adt("Result".into(), vec![Type::<TestEvidence>::Prim(Prim::F32)]);
    assert!(unify(&a1, &a2, &mut s).is_err());
}

#[test]
fn occurs_check_prevents_infinite_type() {
    let mut g = var_gen();
    let v = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    // Try to unify ?0 with List(?0) — should fail with occurs check.
    //
    let list_v = Type::<TestEvidence>::adt("List".into(), vec![Type::<TestEvidence>::Var(v)]);
    let err = unify(&Type::<TestEvidence>::Var(v), &list_v, &mut s).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::OccursCheck));
}

#[test]
fn error_type_unifies_with_anything() {
    // Per WS-A5 RT-3a F1 escalation: the permissive rule is retained.
    //
    // here so a single upstream error does not fan out a cascade of.
    //
    // secondary diagnostics from one root cause. the silent.
    //
    // passthrough at the def-body vs declared-sig boundary is.
    //
    // closed at the call site in `infer.rs`, not by tightening the.
    //
    // unification rule.
    //
    let mut s = Subst::<TestEvidence>::new();
    let err = Type::<TestEvidence>::Error(TestEvidence);
    assert!(unify(&err, &Type::<TestEvidence>::Prim(Prim::F32), &mut s).is_ok());
    assert!(unify(&Type::<TestEvidence>::Prim(Prim::F32), &err, &mut s).is_ok());
}

#[test]
fn unify_tuple_types() {
    let mut s = Subst::<TestEvidence>::new();
    let t1 = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Prim(Prim::F32),
        Type::<TestEvidence>::Prim(Prim::Int32),
    ]);
    let t2 = Type::<TestEvidence>::tuple(vec![
        Type::<TestEvidence>::Prim(Prim::F32),
        Type::<TestEvidence>::Prim(Prim::Int32),
    ]);
    assert!(unify(&t1, &t2, &mut s).is_ok());
}

#[test]
fn transitive_binding() {
    let mut g = var_gen();
    let a = g.fresh_tvar();
    let b = g.fresh_tvar();
    let mut s = Subst::<TestEvidence>::new();
    // a = b, b = f32 → a = f32.
    //
    unify(
        &Type::<TestEvidence>::Var(a),
        &Type::<TestEvidence>::Var(b),
        &mut s,
    )
    .unwrap();
    unify(
        &Type::<TestEvidence>::Var(b),
        &Type::<TestEvidence>::Prim(Prim::F32),
        &mut s,
    )
    .unwrap();
    assert_eq!(
        s.apply(&Type::<TestEvidence>::Var(a)),
        Type::<TestEvidence>::Prim(Prim::F32)
    );
}

#[path = "storage_tests.rs"]
mod storage;
