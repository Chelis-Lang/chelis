use super::*;

#[test]
fn long_alias_resolution_populates_only_transient_caches() {
    let subst = Subst::<TestEvidence>::new();
    subst.types.with(|map| {
        for id in 0..4096 {
            map.insert(TypeVar(id), Type::Var(TypeVar(id + 1)));
        }
        map.insert(TypeVar(4096), Type::Prim(Prim::F32));
    });
    subst.dims.with(|map| {
        for id in 0..4096 {
            map.insert(DimVar(id), Dim::Var(DimVar(id + 1)));
        }
        map.insert(DimVar(4096), Dim::Lit(7));
    });
    subst.ranks.with(|map| {
        for id in 0..4096 {
            map.insert(RankVar(id), vec![Dim::Rank(RankVar(id + 1))]);
        }
        map.insert(RankVar(4096), vec![Dim::Name("batch".into())]);
    });
    for _ in 0..2 {
        assert_eq!(subst.apply(&Type::Var(TypeVar(0))), Type::Prim(Prim::F32));
        assert_eq!(subst.apply_dim(&Dim::Var(DimVar(0))), Dim::Lit(7));
        assert_eq!(
            subst.resolve_rvar(RankVar(0)),
            vec![Dim::Name("batch".into())]
        );
    }
    assert!(subst.types.with(|map| map.resolved.len()) >= 4096);
    assert!(subst.dims.with(|map| map.resolved.len()) >= 4096);
    assert!(subst.ranks.with(|map| map.resolved.len()) >= 4096);
    assert_eq!(
        subst
            .types
            .with(|map| map.bindings.get(&TypeVar(0)).cloned()),
        Some(Type::Var(TypeVar(1)))
    );
    assert_eq!(
        subst.dims.with(|map| map.bindings.get(&DimVar(0)).cloned()),
        Some(Dim::Var(DimVar(1)))
    );
    let cloned = subst.clone();
    assert_eq!(cloned.types.with(|map| map.resolved.len()), 0);
    assert_eq!(cloned.dims.with(|map| map.resolved.len()), 0);
    assert_eq!(cloned.ranks.with(|map| map.resolved.len()), 0);
    cloned.dims.with(|map| {
        map.insert(DimVar(4096), Dim::Lit(9));
    });
    assert_eq!(cloned.apply_dim(&Dim::Var(DimVar(0))), Dim::Lit(9));
    assert_eq!(subst.apply_dim(&Dim::Var(DimVar(0))), Dim::Lit(7));
}

#[test]
fn interrupted_evidence_copy_rejects_later_store_access() {
    #[derive(Debug, PartialEq)]
    struct Evidence;
    impl Clone for Evidence {
        fn clone(&self) -> Self {
            panic!("interrupt evidence copy")
        }
    }
    let subst = Subst::<Evidence>::new();
    subst.types.with(|map| {
        map.insert(TypeVar(0), Type::Error(Evidence));
    });
    let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        subst.apply(&Type::Var(TypeVar(0)))
    }));
    assert!(interrupted.is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| subst.types_len())).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| subst.clone())).is_err());
    assert_eq!(subst.dims_len(), 0);
}

#[test]
fn current_deferred_ledgers_clone_without_shared_mutable_state() {
    let subst = Subst::<TestEvidence>::new();
    let variable = TypeVar(10);
    let gate = DeferredOperandGate::Copy {
        result: Box::new(Type::Var(TypeVar(11))),
        location: None,
    };
    subst.record_deferred_borrow_var(variable);
    subst.record_deferred_drop_operand(variable, None);
    subst.record_deferred_opaque_use(variable, DeferredOpaqueUse::Access);
    subst.record_deferred_opaque_use(variable, DeferredOpaqueUse::RecordUpdate);
    subst.record_deferred_tensor_operand(variable, gate.clone());
    subst.record_operand_gate_failure(OperandGateFailure::Rejected {
        gate,
        resolved: Type::Unit,
    });
    subst.record_result_constraint(ResultConstraint::Annotation {
        actual: Type::Unit,
        declared: Type::Unit,
    });
    subst.record_collection_contract(CollectionConstraint::Len {
        operand: Type::Var(variable),
        result: Type::Prim(Prim::Int64),
    });
    let cloned = subst.clone();
    for state in [&cloned, &subst] {
        assert_eq!(state.take_deferred_borrow_vars(), vec![variable]);
        assert_eq!(state.take_deferred_drop_operands().len(), 1);
        assert_eq!(
            state.take_deferred_opaque_uses(),
            vec![
                (variable, DeferredOpaqueUse::Access),
                (variable, DeferredOpaqueUse::RecordUpdate)
            ]
        );
        assert_eq!(state.take_deferred_tensor_operands().len(), 1);
        assert_eq!(state.take_operand_gate_failures().len(), 1);
        assert_eq!(state.take_result_constraints().len(), 1);
        assert_eq!(state.pending_collection_contracts().len(), 1);
    }
    let ids = cloned
        .pending_collection_contracts()
        .into_iter()
        .map(|(id, _, _)| id)
        .collect::<Vec<_>>();
    cloned.take_collection_contracts(&ids);
    assert!(cloned.pending_collection_contracts().is_empty());
    assert_eq!(subst.pending_collection_contracts().len(), 1);
}

#[test]
fn interrupted_composition_invalidates_the_target_store() {
    #[derive(Debug, PartialEq)]
    struct Evidence;
    impl Clone for Evidence {
        fn clone(&self) -> Self {
            panic!("interrupt composition evidence copy")
        }
    }
    let mut target = Subst::<Evidence>::new();
    target.types.with(|map| {
        map.insert(TypeVar(0), Type::Var(TypeVar(1)));
    });
    let other = Subst::<Evidence>::new();
    other.types.with(|map| {
        map.insert(TypeVar(1), Type::Error(Evidence));
    });
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.compose(&other))).is_err()
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.types_len())).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn evidence_serialization_can_access_its_substitution() {
    use std::sync::{Arc, OnceLock, Weak};
    #[derive(Clone)]
    struct Evidence(Arc<OnceLock<Weak<Subst<Evidence>>>>);
    impl PartialEq for Evidence {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.0, &other.0)
        }
    }
    impl serde::Serialize for Evidence {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let subst = self.0.get().unwrap().upgrade().unwrap();
            serde::Serialize::serialize(&subst.types_len(), serializer)
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let link = Arc::new(OnceLock::new());
        let subst = Subst::<Evidence>::new();
        subst.types.with(|map| {
            map.insert(TypeVar(0), Type::Error(Evidence(link.clone())));
        });
        let subst = Arc::new(subst);
        assert!(link.set(Arc::downgrade(&subst)).is_ok());
        let bytes = postcard::to_allocvec(subst.as_ref()).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(subst.types_len(), 1);
        tx.send(()).unwrap();
    });
    rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct CurrentLogicalSubst {
    types: PersistentMap<TypeVar, Type<TestEvidence>>,
    tvar_restrictions: PersistentMap<TypeVar, TypeVarRestriction>,
    key_free_tvars: PersistentMap<TypeVar, GenericParameter>,
    key_carrying_adts: PersistentSet<String>,
    dims: PersistentMap<DimVar, Dim>,
    dimension_labels: PersistentMap<DimVar, String>,
    ranks: PersistentMap<RankVar, Vec<Dim>>,
    current_level: u32,
    level_transitions: TransitionLog,
    lowered_tvar_levels: PersistentMap<TypeVar, u32>,
    lowered_dvar_levels: PersistentMap<DimVar, u32>,
    lowered_rvar_levels: PersistentMap<RankVar, u32>,
    resume_floors: VarWatermarks,
}

#[cfg(feature = "serde")]
#[test]
fn current_cache_layout_retains_logical_fields_and_omits_transient_state() {
    let mut subst = Subst::<TestEvidence>::new();
    subst.insert_type(TypeVar(1), Type::Unit).unwrap();
    subst.insert_dim(DimVar(2), Dim::Lit(3));
    subst.insert_rank(RankVar(4), vec![Dim::Name("batch".into())]);
    subst
        .narrow_tvar_restriction(TypeVar(5), TypeVarRestriction::ActiveFloat)
        .unwrap();
    subst.forbid_key_instantiation(
        TypeVar(5),
        GenericParameter {
            generic: Some("identity".into()),
            binder: Some("a".into()),
            value: false,
        },
    );
    subst.set_dimension_label(DimVar(6), "sequence".into());
    subst.record_deferred_borrow_var(TypeVar(7));
    subst.record_deferred_drop_operand(TypeVar(7), None);
    subst.record_deferred_opaque_use(TypeVar(7), DeferredOpaqueUse::RecordUpdate);
    subst.record_deferred_tensor_operand(
        TypeVar(7),
        DeferredOperandGate::Copy {
            result: Box::new(Type::Var(TypeVar(8))),
            location: None,
        },
    );
    subst.record_collection_contract(CollectionConstraint::Len {
        operand: Type::Var(TypeVar(7)),
        result: Type::Prim(Prim::Int64),
    });
    subst.record_result_constraint(ResultConstraint::Annotation {
        actual: Type::Unit,
        declared: Type::Unit,
    });
    let _ = subst.apply(&Type::Var(TypeVar(1)));
    let logical = CurrentLogicalSubst {
        types: subst.types.with(|value| value.bindings.clone()),
        tvar_restrictions: subst.tvar_restrictions.with(|value| value.clone()),
        key_free_tvars: subst.key_free_tvars.with(|value| value.clone()),
        key_carrying_adts: subst.key_carrying_adts.clone(),
        dims: subst.dims.with(|value| value.bindings.clone()),
        dimension_labels: subst.dimension_labels.with(|value| value.clone()),
        ranks: subst.ranks.with(|value| value.bindings.clone()),
        current_level: subst.current_level.clone(),
        level_transitions: subst.level_transitions.clone(),
        lowered_tvar_levels: subst.lowered_tvar_levels.clone(),
        lowered_dvar_levels: subst.lowered_dvar_levels.clone(),
        lowered_rvar_levels: subst.lowered_rvar_levels.clone(),
        resume_floors: subst.resume_floors.clone(),
    };
    let bytes = bincode::serialize(&subst).unwrap();
    assert_eq!(bytes, bincode::serialize(&logical).unwrap());
    let restored: Subst<TestEvidence> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(restored.types.with(|map| map.resolved.len()), 0);
    assert_eq!(restored.apply(&Type::Var(TypeVar(1))), Type::Unit);
    assert_eq!(
        restored.tvar_restriction(TypeVar(5)),
        Some(TypeVarRestriction::ActiveFloat)
    );
    assert_eq!(
        restored.dimension_label(DimVar(6)).as_deref(),
        Some("sequence")
    );
    assert!(restored.take_deferred_borrow_vars().is_empty());
    assert!(restored.take_deferred_drop_operands().is_empty());
    assert!(restored.take_deferred_opaque_uses().is_empty());
    assert!(restored.take_deferred_tensor_operands().is_empty());
    assert!(restored.pending_collection_contracts().is_empty());
    assert!(restored.take_result_constraints().is_empty());
    assert_eq!(restored.dims.with(|map| map.resolved.len()), 0);
    assert_eq!(restored.ranks.with(|map| map.resolved.len()), 0);
}
