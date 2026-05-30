//! Bridge from c-earchin's PropertySpec to chelis-prove's SmtProperty.
//!
//! Converts the structured property specification (serialized via JSON from
//! c-earchin) into the solver-agnostic SmtExpr representation.

use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::SmtProperty;
use serde::Deserialize;

/// Mirrors c-earchin's PropertySpec via serde (no crate dependency).
#[derive(Debug, Clone, Deserialize)]
pub struct PropertySpecInput {
    pub id: String,
    pub function_ref: Option<FunctionRefInput>,
    pub params: Vec<TypedParamInput>,
    pub preconditions: Vec<PredExprInput>,
    pub postcondition: PredExprInput,
    pub smt_amenability: SmtAmenabilityInput,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionRefInput {
    pub name: String,
    pub params: Vec<TypedParamInput>,
    pub return_type: ChelisTypeInput,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TypedParamInput {
    pub name: String,
    pub ty: ChelisTypeInput,
}

#[derive(Debug, Clone, Deserialize)]
pub enum ChelisTypeInput {
    Prim(String),
    Tensor(Vec<usize>, String),
}

#[derive(Debug, Clone, Deserialize)]
pub enum PredExprInput {
    Cmp(CmpOpInput, ArithExprInput, ArithExprInput),
    And(Vec<PredExprInput>),
    Or(Vec<PredExprInput>),
    Not(Box<PredExprInput>),
    VocabPred(String, Vec<ArithExprInput>),
    Call(String, Vec<ArithExprInput>),
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum CmpOpInput {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Deserialize)]
pub enum ArithExprInput {
    Var(String),
    Lit(f64),
    BinOp(ArithOpInput, Box<ArithExprInput>, Box<ArithExprInput>),
    Call(String, Vec<ArithExprInput>),
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ArithOpInput {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum SmtAmenabilityInput {
    Linear,
    Polynomial,
    Transcendental,
    Opaque,
}

/// Convert a PropertySpecInput into an SmtProperty for Tier B.
pub fn to_smt_property(spec: &PropertySpecInput) -> SmtProperty {
    SmtProperty {
        variables: spec
            .params
            .iter()
            .map(|p| (p.name.clone(), type_to_sort(&p.ty)))
            .collect(),
        preconditions: spec.preconditions.iter().map(pred_to_smt).collect(),
        postcondition: pred_to_smt(&spec.postcondition),
    }
}

/// Convert amenability classification for dispatch routing.
pub fn to_dispatch_amenability(a: SmtAmenabilityInput) -> crate::dispatch::SmtAmenability {
    match a {
        SmtAmenabilityInput::Linear => crate::dispatch::SmtAmenability::Linear,
        SmtAmenabilityInput::Polynomial => crate::dispatch::SmtAmenability::Polynomial,
        SmtAmenabilityInput::Transcendental => crate::dispatch::SmtAmenability::Transcendental,
        SmtAmenabilityInput::Opaque => crate::dispatch::SmtAmenability::Opaque,
    }
}

fn type_to_sort(ty: &ChelisTypeInput) -> SmtSort {
    match ty {
        ChelisTypeInput::Prim(n) => match n.as_str() {
            "f32" | "f64" => SmtSort::Real,
            "int32" | "int64" => SmtSort::Int,
            "bool" => SmtSort::Bool,
            _ => SmtSort::Real,
        },
        ChelisTypeInput::Tensor(_, _) => SmtSort::Real,
    }
}

fn pred_to_smt(pred: &PredExprInput) -> SmtExpr {
    match pred {
        PredExprInput::Cmp(op, l, r) => SmtExpr::Cmp(
            cmp_op(*op),
            Box::new(arith_to_smt(l)),
            Box::new(arith_to_smt(r)),
        ),
        PredExprInput::And(ps) => SmtExpr::Bool(BoolOp::And, ps.iter().map(pred_to_smt).collect()),
        PredExprInput::Or(ps) => SmtExpr::Bool(BoolOp::Or, ps.iter().map(pred_to_smt).collect()),
        PredExprInput::Not(p) => SmtExpr::Not(Box::new(pred_to_smt(p))),
        PredExprInput::VocabPred(name, args) => {
            SmtExpr::Apply(name.clone(), args.iter().map(arith_to_smt).collect())
        }
        PredExprInput::Call(name, args) => {
            SmtExpr::Apply(name.clone(), args.iter().map(arith_to_smt).collect())
        }
    }
}

fn arith_to_smt(expr: &ArithExprInput) -> SmtExpr {
    match expr {
        ArithExprInput::Var(n) => SmtExpr::Var(n.clone()),
        ArithExprInput::Lit(v) => SmtExpr::RealLit(*v),
        ArithExprInput::BinOp(op, l, r) => SmtExpr::Arith(
            arith_op(*op),
            Box::new(arith_to_smt(l)),
            Box::new(arith_to_smt(r)),
        ),
        ArithExprInput::Call(name, args) => {
            SmtExpr::Apply(name.clone(), args.iter().map(arith_to_smt).collect())
        }
    }
}

fn cmp_op(op: CmpOpInput) -> CmpOp {
    match op {
        CmpOpInput::Lt => CmpOp::Lt,
        CmpOpInput::Le => CmpOp::Le,
        CmpOpInput::Gt => CmpOp::Gt,
        CmpOpInput::Ge => CmpOp::Ge,
        CmpOpInput::Eq => CmpOp::Eq,
        CmpOpInput::Ne => CmpOp::Ne,
    }
}

fn arith_op(op: ArithOpInput) -> ArithOp {
    match op {
        ArithOpInput::Add => ArithOp::Add,
        ArithOpInput::Sub => ArithOp::Sub,
        ArithOpInput::Mul => ArithOp::Mul,
        ArithOpInput::Div => ArithOp::Div,
        ArithOpInput::Neg => ArithOp::Neg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_simple_property() {
        let spec = PropertySpecInput {
            id: "T-001".into(),
            function_ref: None,
            params: vec![TypedParamInput {
                name: "x".into(),
                ty: ChelisTypeInput::Prim("f32".into()),
            }],
            preconditions: vec![PredExprInput::Cmp(
                CmpOpInput::Gt,
                ArithExprInput::Var("x".into()),
                ArithExprInput::Lit(0.0),
            )],
            postcondition: PredExprInput::Cmp(
                CmpOpInput::Ge,
                ArithExprInput::BinOp(
                    ArithOpInput::Mul,
                    Box::new(ArithExprInput::Var("x".into())),
                    Box::new(ArithExprInput::Var("x".into())),
                ),
                ArithExprInput::Lit(0.0),
            ),
            smt_amenability: SmtAmenabilityInput::Polynomial,
        };
        let smt = to_smt_property(&spec);
        assert_eq!(smt.variables.len(), 1);
        assert_eq!(smt.variables[0], ("x".to_string(), SmtSort::Real));
        assert_eq!(smt.preconditions.len(), 1);
    }

    #[test]
    fn converts_multi_param_with_function_ref() {
        let spec = PropertySpecInput {
            id: "T-002".into(),
            function_ref: Some(FunctionRefInput {
                name: "f".into(),
                params: vec![
                    TypedParamInput {
                        name: "a".into(),
                        ty: ChelisTypeInput::Prim("f64".into()),
                    },
                    TypedParamInput {
                        name: "b".into(),
                        ty: ChelisTypeInput::Prim("int32".into()),
                    },
                ],
                return_type: ChelisTypeInput::Prim("f64".into()),
            }),
            params: vec![
                TypedParamInput {
                    name: "a".into(),
                    ty: ChelisTypeInput::Prim("f64".into()),
                },
                TypedParamInput {
                    name: "b".into(),
                    ty: ChelisTypeInput::Prim("int32".into()),
                },
            ],
            preconditions: vec![],
            postcondition: PredExprInput::Cmp(
                CmpOpInput::Ge,
                ArithExprInput::Var("a".into()),
                ArithExprInput::Lit(0.0),
            ),
            smt_amenability: SmtAmenabilityInput::Linear,
        };
        let smt = to_smt_property(&spec);
        assert_eq!(smt.variables.len(), 2);
        assert_eq!(smt.variables[0].1, SmtSort::Real);
        assert_eq!(smt.variables[1].1, SmtSort::Int);
        assert!(smt.preconditions.is_empty());
    }
}
