use chelis_numeric_core::prim::Prim;
use chelis_type_core::*;
use core::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn concurrent_resolution_and_clone_state_remain_independent() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Type<Infallible>>();
    send_sync::<Scheme<Infallible>>();
    send_sync::<Subst<Infallible>>();
    let mut subst = Subst::<Infallible>::new();
    for id in 0..4096 {
        subst
            .insert_type(TypeVar(id), Type::Var(TypeVar(id + 1)))
            .unwrap();
        subst.insert_dim(DimVar(id), Dim::Var(DimVar(id + 1)));
        subst.insert_rank(RankVar(id), vec![Dim::Rank(RankVar(id + 1))]);
    }
    subst
        .insert_type(TypeVar(4096), Type::Prim(Prim::F32))
        .unwrap();
    subst.insert_dim(DimVar(4096), Dim::Lit(7));
    subst.insert_rank(RankVar(4096), vec![Dim::Name("batch".into())]);
    let shared = Arc::new(subst);
    let (tx, rx) = std::sync::mpsc::channel();
    for _ in 0..8 {
        let subst = shared.clone();
        let tx = tx.clone();
        std::thread::spawn(move || {
            for _ in 0..32 {
                assert_eq!(subst.apply(&Type::Var(TypeVar(0))), Type::Prim(Prim::F32));
                assert_eq!(subst.apply_dim(&Dim::Var(DimVar(0))), Dim::Lit(7));
                let tensor = Type::Tensor(vec![Dim::Rank(RankVar(0))], Prim::F32.into());
                assert_eq!(
                    subst.apply(&tensor),
                    Type::Tensor(vec![Dim::Name("batch".into())], Prim::F32.into())
                );
            }
            tx.send(()).unwrap();
        });
    }
    for _ in 0..8 {
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
    }
    let mut isolated = shared.as_ref().clone();
    isolated.insert_type(TypeVar(5000), Type::Unit).unwrap();
    isolated.insert_dim(DimVar(5000), Dim::Lit(5));
    isolated.insert_rank(RankVar(5000), vec![Dim::Lit(9)]);
    isolated.record_deferred_borrow_var(TypeVar(5001));
    assert_eq!(
        shared.apply(&Type::Var(TypeVar(5000))),
        Type::Var(TypeVar(5000))
    );
    assert_eq!(isolated.apply(&Type::Var(TypeVar(5000))), Type::Unit);
    assert_eq!(
        shared.apply_dim(&Dim::Var(DimVar(5000))),
        Dim::Var(DimVar(5000))
    );
    assert!(shared.take_deferred_borrow_vars().is_empty());
}

#[test]
fn shared_resolution_preserves_quantifier_alias_boundaries() {
    let scheme = Scheme::<Infallible> {
        result_origin: None,
        constraints: vec![],
        tvar_restrictions: vec![],
        tvars: vec![TypeVar(1)],
        dvars: vec![DimVar(1)],
        rvars: vec![RankVar(1)],
        body: Type::tuple(vec![
            Type::Var(TypeVar(0)),
            Type::Tensor(
                vec![Dim::Var(DimVar(0)), Dim::Rank(RankVar(0))],
                TensorPrec::Var(TypeVar(0)),
            ),
        ]),
    };
    let mut subst = Subst::<Infallible>::new();
    for index in 0..2 {
        subst
            .insert_type(TypeVar(index), Type::Var(TypeVar(index + 1)))
            .unwrap();
        subst.insert_dim(DimVar(index), Dim::Var(DimVar(index + 1)));
        subst.insert_rank(RankVar(index), vec![Dim::Rank(RankVar(index + 1))]);
    }
    let _ = subst.apply(&scheme.body);
    let expected = Type::tuple(vec![
        Type::Var(TypeVar(1)),
        Type::Tensor(
            vec![Dim::Var(DimVar(1)), Dim::Rank(RankVar(1))],
            TensorPrec::Var(TypeVar(1)),
        ),
    ]);
    assert_eq!(subst.apply_scheme(&scheme), expected);
}

#[test]
fn staged_instantiation_retains_all_generic_result_equations() {
    let mut vars = VarGen::default();
    let variable = vars.fresh_tvar();
    let body = Type::<Infallible>::function(
        vec![Type::Var(variable), Type::Var(variable)],
        Type::Var(variable),
    );
    let origin = ResultOrigin::aggregate(&body, &[variable], &[], &[], &mut vars).unwrap();
    let scheme = Scheme {
        body,
        tvars: vec![variable],
        dvars: vec![],
        rvars: vec![],
        constraints: vec![],
        tvar_restrictions: vec![],
        result_origin: Some(origin),
    };
    let staged_subst = Subst::new();
    let mut staged_vars = vars.clone();
    let (staged, equations) = instantiate_scheme_staged(&scheme, &mut staged_vars, &staged_subst);
    assert!(staged_subst.take_result_constraints().is_empty());
    assert!(
        equations
            .iter()
            .any(|equation| matches!(equation, ResultConstraint::Join { .. }))
    );
    assert!(
        equations
            .iter()
            .any(|equation| matches!(equation, ResultConstraint::Annotation { .. }))
    );
    let ordinary_subst = Subst::new();
    let ordinary = instantiate_scheme(&scheme, &mut vars, &ordinary_subst);
    assert_eq!(staged.ty, ordinary.ty);
    assert_eq!(equations, ordinary_subst.take_result_constraints());
    for equation in equations {
        staged_subst.record_result_constraint(equation);
    }
    assert!(!staged_subst.take_result_constraints().is_empty());
}

#[test]
fn shared_scheme_instances_are_independent() {
    let mut vars = VarGen::default();
    let precision = vars.fresh_tvar();
    let dimension = vars.fresh_dvar();
    let rank = vars.fresh_rvar();
    let scheme = Scheme::<Infallible> {
        result_origin: None,
        constraints: vec![],
        tvar_restrictions: vec![],
        tvars: vec![precision],
        dvars: vec![dimension],
        rvars: vec![rank],
        body: Type::Tensor(
            vec![Dim::Var(dimension), Dim::Rank(rank)],
            TensorPrec::Var(precision),
        ),
    };
    let mut subst = Subst::<Infallible>::new();
    let left = instantiate(&scheme, &mut vars, &subst);
    let right = instantiate(&scheme, &mut vars, &subst);
    let a = Type::Tensor(vec![Dim::Lit(2), Dim::Name("a".into())], Prim::F32.into());
    let b = Type::Tensor(vec![Dim::Lit(3), Dim::Name("b".into())], Prim::F64.into());
    unify(&left, &a, &mut subst).unwrap();
    unify(&right, &b, &mut subst).unwrap();
    assert_eq!(subst.apply(&left), a);
    assert_eq!(subst.apply(&right), b);
    assert!(unify(&left, &b, &mut subst).is_err());
    assert_eq!(
        scheme.body,
        Type::Tensor(
            vec![Dim::Var(dimension), Dim::Rank(rank)],
            TensorPrec::Var(precision)
        )
    );
}

#[test]
fn external_scheme_identifiers_do_not_capture_fresh_instances() {
    let scheme = Scheme::<Infallible> {
        result_origin: None,
        constraints: vec![CollectionConstraint::Len {
            operand: Type::Var(TypeVar(300)),
            result: Type::Prim(Prim::Int64),
        }],
        tvar_restrictions: vec![],
        tvars: vec![TypeVar(5)],
        dvars: vec![DimVar(6)],
        rvars: vec![RankVar(7)],
        body: Type::function(
            vec![Type::Tensor(
                vec![Dim::Var(DimVar(6)), Dim::Rank(RankVar(7))],
                TensorPrec::Var(TypeVar(5)),
            )],
            Type::tuple(vec![
                Type::Var(TypeVar(0)),
                Type::Tensor(
                    vec![Dim::Var(DimVar(200)), Dim::Rank(RankVar(99))],
                    TensorPrec::Concrete(Prim::F32),
                ),
            ]),
        ),
    };
    let mut vars = VarGen::default();
    let subst = Subst::<Infallible>::new();
    let first = instantiate(&scheme, &mut vars, &subst);
    let second = instantiate(&scheme, &mut vars, &subst);
    let (Type::Fn(first_args, first_result), Type::Fn(second_args, second_result)) =
        (first, second)
    else {
        panic!("function instances")
    };
    assert_ne!(first_args, second_args);
    assert_eq!(first_result, second_result);
    let Type::Tensor(dims, TensorPrec::Var(precision)) = &first_args[0] else {
        panic!("tensor parameter")
    };
    assert!(precision.0 > 300);
    assert!(matches!(dims[0], Dim::Var(DimVar(id)) if id > 200));
    assert!(matches!(dims[1], Dim::Rank(RankVar(id)) if id > 99));
    assert!(vars.fresh_tvar().0 > precision.0);
}
