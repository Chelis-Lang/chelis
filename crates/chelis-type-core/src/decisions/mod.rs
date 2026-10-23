//! Portable type decisions and failure data.

use crate::unify::{TypeError, TypeErrorKind};
use alloc::{string::String, vec::Vec};

pub mod borrow;
pub mod cast;
pub mod collection;
pub mod shape;

/// Opaque source data. Only the host interprets or attaches this label.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceLabel {
    pub span_offset: Option<usize>,
    pub span_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum DecisionFailureKind {
    TypeMismatch,
    PrecisionMismatch,
    CastNonTensor,
    UnsupportedTensorPrecision,
    Unification(TypeErrorKind),
}

/// A decision failure without diagnostic or evidence authority.
#[derive(Debug, Clone)]
pub struct DecisionFailure {
    pub kind: DecisionFailureKind,
    pub message: String,
    pub original_message: Option<String>,
    pub expected: Option<String>,
    pub got: Option<String>,
    pub suggestions: Vec<String>,
    pub location: Option<SourceLabel>,
}

impl DecisionFailure {
    pub fn new(kind: DecisionFailureKind, message: String, suggestions: Vec<String>) -> Self {
        Self {
            kind,
            message,
            suggestions,
            original_message: None,
            expected: None,
            got: None,
            location: None,
        }
    }

    pub fn with_types(
        kind: DecisionFailureKind,
        message: String,
        expected: String,
        got: String,
        suggestions: Vec<String>,
    ) -> Self {
        Self {
            kind,
            message,
            suggestions,
            original_message: None,
            expected: Some(expected),
            got: Some(got),
            location: None,
        }
    }
}

impl From<TypeError> for DecisionFailure {
    fn from(error: TypeError) -> Self {
        Self {
            message: error.message,
            original_message: None,
            kind: DecisionFailureKind::Unification(error.kind),
            expected: None,
            got: None,
            suggestions: Vec::new(),
            location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::borrow::{copy_result_from_source, settled_borrow_type};
    use super::cast::{
        binder_cast_result_from_settled_source, bounded_scalar_cast_result,
        cast_result_from_settled_source,
    };
    use super::collection::{
        CollectionCallEvidence, ConcatListInfo, TensorConcatCallEvidence,
        decide_collection_constraint, tensor_concat_result_type,
    };
    use super::shape::{SumResultSlot, settled_sum_result_precision, shape_route_result};
    use super::{DecisionFailureKind, SourceLabel};
    use crate::model::{CollectionConstraint, Dim, TensorPrec, Type, TypeVarRestriction, VarGen};
    use crate::unify::{ShapeRoute, Subst, unify};
    use alloc::{boxed::Box, vec};
    use chelis_deep::CastMode;
    use chelis_numeric_core::prim::Prim;
    use core::convert::Infallible;

    #[test]
    fn borrow_and_copy_preserve_owned_and_shared_types() {
        let tensor: Type<Infallible> =
            Type::Tensor(vec![Dim::Lit(3)], TensorPrec::Concrete(Prim::F32));
        let borrowed = settled_borrow_type(tensor.clone()).unwrap();
        assert_eq!(borrowed, Type::reference(tensor.clone()));
        assert_eq!(
            settled_borrow_type(borrowed.clone()),
            Some(borrowed.clone())
        );
        assert_eq!(copy_result_from_source(&borrowed), tensor);
        assert_eq!(
            settled_borrow_type::<Infallible>(Type::Prim(Prim::Int64)),
            None
        );
    }

    #[test]
    fn cast_failures_retain_structured_fields_without_evidence() {
        let subst = Subst::<Infallible>::new();
        let error = cast_result_from_settled_source(
            Type::Prim(Prim::String),
            Prim::F32,
            CastMode::Checked,
            &subst,
        )
        .unwrap_err();
        assert!(matches!(error.kind, DecisionFailureKind::CastNonTensor));
        assert_eq!(error.expected.as_deref(), Some("numeric or bool scalar"));
        assert_eq!(error.got.as_deref(), Some("string"));
        assert!(error.suggestions.is_empty());
        assert!(error.location.is_none());
        let error = cast_result_from_settled_source(
            Type::Prim(Prim::Key),
            Prim::Int64,
            CastMode::Checked,
            &subst,
        )
        .unwrap_err();
        assert!(matches!(
            error.kind,
            DecisionFailureKind::UnsupportedTensorPrecision
        ));
        assert_eq!(error.suggestions.len(), 1);
        assert!(
            bounded_scalar_cast_result::<Infallible>(
                TypeVarRestriction::ActiveFloat,
                Prim::F32,
                CastMode::Checked
            )
            .unwrap()
            .is_ok()
        );
    }

    #[test]
    fn cast_reads_one_tensor_borrow_and_preserves_nominal_refusal() {
        let subst = Subst::<Infallible>::new();
        let input = Type::reference(Type::Tensor(
            vec![Dim::Lit(4)],
            TensorPrec::Concrete(Prim::F32),
        ));
        assert_eq!(
            cast_result_from_settled_source(input, Prim::Int32, CastMode::Checked, &subst).unwrap(),
            Type::Tensor(vec![Dim::Lit(4)], TensorPrec::Concrete(Prim::Int32))
        );
        let mut vars = VarGen::default();
        let target = vars.fresh_tvar();
        assert_eq!(
            binder_cast_result_from_settled_source(Type::Prim(Prim::Bool), target, &subst).unwrap(),
            Type::Var(target)
        );
        assert!(
            binder_cast_result_from_settled_source(
                Type::adt("Box".into(), vec![Type::Prim(Prim::F32)]),
                target,
                &subst
            )
            .is_err()
        );
    }

    #[test]
    fn concat_evidence_preserves_ragged_literal_extent() {
        let subst = Subst::<Infallible>::new();
        let element = Type::Tensor(
            vec![Dim::Wildcard, Dim::Lit(2)],
            TensorPrec::Concrete(Prim::F32),
        );
        let list_info = ConcatListInfo::Direct(vec![
            vec![Dim::Lit(3), Dim::Lit(2)],
            vec![Dim::Lit(5), Dim::Lit(2)],
        ]);
        let expected = Type::Tensor(
            vec![Dim::Lit(8), Dim::Lit(2)],
            TensorPrec::Concrete(Prim::F32),
        );
        assert_eq!(
            tensor_concat_result_type(&element, Some(0), list_info.clone(), &subst).unwrap(),
            expected
        );
        let constraint = CollectionConstraint::Concat {
            lhs: Type::adt("List".into(), vec![element]),
            rhs: Type::Prim(Prim::Int32),
            result: expected.clone(),
        };
        let evidence = CollectionCallEvidence::TensorConcat(TensorConcatCallEvidence {
            raw_axis: Some(0),
            list_info,
        });
        let decision = decide_collection_constraint(&constraint, Some(&evidence), &subst)
            .unwrap()
            .unwrap();
        assert_eq!(decision.produced_result(), Some(expected));
    }

    #[test]
    fn shape_route_preserves_axis_and_mode_error_order() {
        let subst = Subst::<Infallible>::new();
        let tensor = Type::Tensor(vec![Dim::Lit(2)], TensorPrec::Concrete(Prim::F32));
        let route = ShapeRoute::Gather {
            op: "scatter".into(),
            indices: Box::new(Type::Tensor(
                vec![Dim::Lit(1)],
                TensorPrec::Concrete(Prim::Int64),
            )),
            raw_axis: Some(3),
            updates: Some(Box::new(tensor.clone())),
            mode: Some("bad".into()),
        };
        let mut vars = VarGen::default();
        assert_eq!(
            shape_route_result(
                &route,
                &tensor,
                SumResultSlot::Fresh(&mut vars),
                None,
                &subst
            )
            .unwrap_err(),
            "scatter axis 3 out of bounds for rank 1"
        );
        assert_eq!(
            settled_sum_result_precision::<Infallible>("trace", &Type::Prim(Prim::Int8)).unwrap(),
            Prim::Int32
        );
    }

    #[test]
    fn pending_trace_retains_opaque_source_label() {
        let mut subst = Subst::<Infallible>::new();
        let mut vars = VarGen::default();
        let precision = vars.fresh_tvar();
        let route = ShapeRoute::Trace {
            raw_axis1: None,
            raw_axis2: None,
        };
        let source = Type::Tensor(vec![Dim::Lit(2), Dim::Lit(2)], TensorPrec::Var(precision));
        let label = SourceLabel {
            span_offset: Some(19),
            span_id: Some("producer:trace".into()),
        };
        let (result, updates) = shape_route_result(
            &route,
            &source,
            SumResultSlot::Fresh(&mut vars),
            Some(label.clone()),
            &subst,
        )
        .unwrap();
        assert!(updates.is_none());
        unify(&Type::Var(precision), &Type::Prim(Prim::Bool), &mut subst).unwrap();
        let failures = subst.take_operand_gate_failures();
        assert_eq!(failures.len(), 1);
        let crate::unify::OperandGateFailure::Decision { error } = &failures[0] else {
            panic!("expected decision failure")
        };
        assert_eq!(error.location.as_ref(), Some(&label));
        assert!(matches!(
            subst.apply(&result),
            Type::Tensor(_, TensorPrec::Var(_))
        ));
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::shape::{
        bound_sum_result_precision, sum_result_bound_note, sum_result_widening_note,
    };
    use crate::model::{PrimSet, TensorPrec, Type, TypeVarRestriction, VarGen};
    use crate::unify::Subst;
    use chelis_numeric_core::prim::Prim;
    use core::convert::Infallible;

    #[test]
    fn bounded_sum_decisions_keep_family_and_widening_rules() {
        let subst = Subst::<Infallible>::new();
        let mut vars = VarGen::default();
        let floats = vars.fresh_tvar();
        subst
            .narrow_tvar_restriction(floats, TypeVarRestriction::ActiveFloat)
            .unwrap();
        assert_eq!(
            bound_sum_result_precision("sum", floats, &subst).unwrap(),
            TensorPrec::Var(floats)
        );
        let small = vars.fresh_tvar();
        subst
            .narrow_tvar_restriction(
                small,
                TypeVarRestriction::ActiveSet(PrimSet::from_members([Prim::Int8, Prim::Int16])),
            )
            .unwrap();
        assert_eq!(
            bound_sum_result_precision("sum", small, &subst).unwrap(),
            TensorPrec::Concrete(Prim::Int32)
        );
        let numeric = vars.fresh_tvar();
        subst
            .narrow_tvar_restriction(numeric, TypeVarRestriction::ActiveNumeric)
            .unwrap();
        assert!(bound_sum_result_precision("sum", numeric, &subst).is_err());
        assert!(
            sum_result_widening_note::<Infallible>(
                Some("sum"),
                &Type::Prim(Prim::Int8),
                &Type::Prim(Prim::Int32)
            )
            .unwrap()
            .starts_with("`sum` over i8 returns i32")
        );
        assert!(sum_result_bound_note(true, Prim::Int32).is_some());
        assert!(sum_result_bound_note(false, Prim::Int32).is_none());
    }
}
