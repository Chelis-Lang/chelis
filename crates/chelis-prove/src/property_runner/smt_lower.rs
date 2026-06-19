//! Surf -> SMT lowering for the shared property runner's Tier B path.
//!
//! Extracted verbatim from `prove.rs` as a pure module split (W3+W4
//! Unit a). The `surf_expr_to_smt` family lowers a Surf property body
//! and its preconditions into `crate::solver::SmtExpr` form,
//! inlining in-module function calls (depth 3, cycle-guarded) and
//! substituting `Block` let-bindings. `if` => `ite` is already handled
//! here; record beta-reduction and case-of-known-constructor reduction
//! (RFC D-TIERB) are added in a later unit.
//!
//! It lives in chelis-prove so both the CLI prove path and the tide MCP
//! tool reach one Surf->SMT lowering through the shared property runner.

use std::cell::RefCell;

use chelis_surf::ast::{BinOp, Decl, Expr, LetPattern, Literal, Param, UnaryOp};

use crate::contracts::{NORMAL_CDF_IMPLEMENTATION, NORMAL_CDF_RANGE, NORMAL_CDF_REFLECTION};
use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};

pub(super) struct InlineCtx<'a> {
    pub(super) decls: &'a [Decl],
    pub(super) depth: usize,
    pub(super) max_depth: usize,
    pub(super) call_stack: Vec<String>,
    pub(super) contracts: Option<&'a RefCell<ContractAbstraction>>,
}

#[derive(Debug, Clone)]
struct ContractCall {
    symbol: String,
    arg: SmtExpr,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ContractAbstraction {
    normal_cdf_enabled: bool,
    normal_cdf_range: bool,
    normal_cdf_reflection: bool,
    normal_cdf_symbols: Vec<String>,
    normal_cdf_calls: Vec<ContractCall>,
}

impl ContractAbstraction {
    pub(super) fn for_contracts(contracts: &[String], trusted_contract_decls: &[Decl]) -> Self {
        let normal_cdf_reflection = contracts.iter().any(|id| id == NORMAL_CDF_REFLECTION);
        let normal_cdf_range =
            normal_cdf_reflection || contracts.iter().any(|id| id == NORMAL_CDF_RANGE);
        Self {
            normal_cdf_enabled: normal_cdf_range || normal_cdf_reflection,
            normal_cdf_range,
            normal_cdf_reflection,
            normal_cdf_symbols: trusted_normal_cdf_symbols(trusted_contract_decls),
            normal_cdf_calls: Vec::new(),
        }
    }

    pub(super) fn requires_normal_cdf(&self) -> bool {
        self.normal_cdf_enabled
    }

    pub(super) fn used_normal_cdf(&self) -> bool {
        !self.normal_cdf_calls.is_empty()
    }

    pub(super) fn requires_reflection_pair(&self) -> bool {
        self.normal_cdf_reflection
    }

    pub(super) fn has_reflection_pair(&self) -> bool {
        self.normal_cdf_calls.iter().enumerate().any(|(idx, left)| {
            self.normal_cdf_calls
                .iter()
                .skip(idx + 1)
                .any(|right| are_negated_args(&left.arg, &right.arg))
        })
    }

    pub(super) fn variables(&self) -> Vec<(String, SmtSort)> {
        self.normal_cdf_calls
            .iter()
            .map(|call| (call.symbol.clone(), SmtSort::Real))
            .collect()
    }

    pub(super) fn preconditions(&self) -> Vec<SmtExpr> {
        let mut out = Vec::new();
        if self.normal_cdf_range {
            for call in &self.normal_cdf_calls {
                let value = SmtExpr::Var(call.symbol.clone());
                out.push(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(value.clone()),
                    Box::new(SmtExpr::RealLit(0.0)),
                ));
                out.push(SmtExpr::Cmp(
                    CmpOp::Le,
                    Box::new(value),
                    Box::new(SmtExpr::RealLit(1.0)),
                ));
            }
        }
        if self.normal_cdf_reflection {
            for (idx, left) in self.normal_cdf_calls.iter().enumerate() {
                for right in self.normal_cdf_calls.iter().skip(idx + 1) {
                    if are_negated_args(&left.arg, &right.arg) {
                        out.push(SmtExpr::Cmp(
                            CmpOp::Eq,
                            Box::new(SmtExpr::Var(right.symbol.clone())),
                            Box::new(SmtExpr::Arith(
                                ArithOp::Sub,
                                Box::new(SmtExpr::RealLit(1.0)),
                                Box::new(SmtExpr::Var(left.symbol.clone())),
                            )),
                        ));
                    }
                }
            }
        }
        out
    }

    fn abstract_call(&mut self, name: &str, args: &[SmtExpr]) -> Option<SmtExpr> {
        if !self.normal_cdf_enabled
            || args.len() != 1
            || !self.normal_cdf_symbols.iter().any(|symbol| symbol == name)
        {
            return None;
        }
        let symbol = format!("__contract_std_normal_cdf_{}", self.normal_cdf_calls.len());
        self.normal_cdf_calls.push(ContractCall {
            symbol: symbol.clone(),
            arg: args[0].clone(),
        });
        Some(SmtExpr::Var(symbol))
    }
}

fn trusted_normal_cdf_symbols(decls: &[Decl]) -> Vec<String> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::FunDef { name, params, .. }
                if (name == NORMAL_CDF_IMPLEMENTATION
                    || name == "pkg__chelis__std__Std__Contracts__normal_cdf")
                    && params.len() == 1 =>
            {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

fn are_negated_args(left: &SmtExpr, right: &SmtExpr) -> bool {
    matches!(left, SmtExpr::Arith(ArithOp::Neg, inner, _) if inner.as_ref() == right)
        || matches!(right, SmtExpr::Arith(ArithOp::Neg, inner, _) if inner.as_ref() == left)
}

fn lookup_fun_body<'a>(decls: &'a [Decl], name: &str) -> Option<(&'a [Param], &'a Expr)> {
    decls.iter().find_map(|d| match d {
        Decl::FunDef {
            name: n,
            params,
            body,
            ..
        } if n == name => Some((params.as_slice(), body)),
        _ => None,
    })
}

pub(super) fn surf_expr_to_smt(expr: &Expr, ctx: &InlineCtx) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{BoolOp as SB, CmpOp as SC, SmtExpr};
    match expr {
        Expr::Binary(BinOp::Ge, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ge,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Le, l, r, _) => Some(SmtExpr::Cmp(
            SC::Le,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Gt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Gt,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Lt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Lt,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Eq, l, r, _) => Some(SmtExpr::Cmp(
            SC::Eq,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Ne, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ne,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::And, l, r, _) => Some(SmtExpr::Bool(
            SB::And,
            vec![surf_expr_to_smt(l, ctx)?, surf_expr_to_smt(r, ctx)?],
        )),
        Expr::Binary(BinOp::Or, l, r, _) => Some(SmtExpr::Bool(
            SB::Or,
            vec![surf_expr_to_smt(l, ctx)?, surf_expr_to_smt(r, ctx)?],
        )),
        Expr::Lit(Literal::Bool(v), _) => Some(SmtExpr::BoolLit(*v)),
        _ => None,
    }
}

fn surf_arith(expr: &Expr, ctx: &InlineCtx) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, SmtExpr};
    match expr {
        Expr::Var(name, _) => Some(SmtExpr::Var(name.clone())),
        Expr::Lit(Literal::Float(v), _) => Some(SmtExpr::RealLit(*v)),
        Expr::Lit(Literal::Int(v), _) => Some(SmtExpr::IntLit(*v)),
        Expr::Unary(UnaryOp::Neg, inner, _) => Some(SmtExpr::Arith(
            SA::Neg,
            Box::new(surf_arith(inner, ctx)?),
            Box::new(SmtExpr::IntLit(0)),
        )),
        Expr::Binary(BinOp::Add, l, r, _) => Some(SmtExpr::Arith(
            SA::Add,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Sub, l, r, _) => Some(SmtExpr::Arith(
            SA::Sub,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Mul, l, r, _) => Some(SmtExpr::Arith(
            SA::Mul,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Binary(BinOp::Div, l, r, _) => Some(SmtExpr::Arith(
            SA::Div,
            Box::new(surf_arith(l, ctx)?),
            Box::new(surf_arith(r, ctx)?),
        )),
        Expr::Apply(func, args, _) => {
            let name = match func.as_ref() {
                Expr::Var(n, _) => n.clone(),
                _ => return None,
            };
            let smt_args: Option<Vec<_>> = args.iter().map(|a| surf_arith(a, ctx)).collect();
            let smt_args = smt_args?;
            if name == "neg" && smt_args.len() == 1 {
                return Some(SmtExpr::Arith(
                    SA::Neg,
                    Box::new(smt_args[0].clone()),
                    Box::new(SmtExpr::IntLit(0)),
                ));
            }
            if let Some(contracts) = ctx.contracts
                && let Some(abs) = contracts.borrow_mut().abstract_call(&name, &smt_args)
            {
                return Some(abs);
            }
            if let Some((params, body)) = lookup_fun_body(ctx.decls, &name) {
                if ctx.call_stack.contains(&name) {
                    return None;
                }
                if ctx.depth >= ctx.max_depth {
                    return None;
                }
                if params.len() != args.len() {
                    return None;
                }
                let subst: std::collections::HashMap<String, &Expr> = params
                    .iter()
                    .zip(args.iter())
                    .map(|(p, a)| (p.name.clone(), a))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
                    contracts: ctx.contracts,
                    call_stack: {
                        let mut s = ctx.call_stack.clone();
                        s.push(name.clone());
                        s
                    },
                };
                return surf_arith_subst(body, &subst, &deeper);
            }
            Some(SmtExpr::Apply(name, smt_args))
        }
        Expr::If(cond, then_e, else_e, _) => Some(SmtExpr::Ite(
            Box::new(surf_expr_to_smt(cond, ctx)?),
            Box::new(surf_arith(then_e, ctx)?),
            Box::new(surf_arith(else_e, ctx)?),
        )),
        _ => None,
    }
}

fn surf_arith_subst(
    expr: &Expr,
    subst: &std::collections::HashMap<String, &Expr>,
    ctx: &InlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, BoolOp as SB, CmpOp as SC, SmtExpr};
    match expr {
        Expr::Var(name, _) => {
            if let Some(replacement) = subst.get(name.as_str()) {
                surf_arith(replacement, ctx)
            } else {
                Some(SmtExpr::Var(name.clone()))
            }
        }
        Expr::Lit(Literal::Float(v), _) => Some(SmtExpr::RealLit(*v)),
        Expr::Lit(Literal::Int(v), _) => Some(SmtExpr::IntLit(*v)),
        Expr::Unary(UnaryOp::Neg, inner, _) => Some(SmtExpr::Arith(
            SA::Neg,
            Box::new(surf_arith_subst(inner, subst, ctx)?),
            Box::new(SmtExpr::IntLit(0)),
        )),
        Expr::Binary(BinOp::Add, l, r, _) => Some(SmtExpr::Arith(
            SA::Add,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Sub, l, r, _) => Some(SmtExpr::Arith(
            SA::Sub,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Mul, l, r, _) => Some(SmtExpr::Arith(
            SA::Mul,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Div, l, r, _) => Some(SmtExpr::Arith(
            SA::Div,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Ge, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ge,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Le, l, r, _) => Some(SmtExpr::Cmp(
            SC::Le,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Gt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Gt,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Lt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Lt,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Eq, l, r, _) => Some(SmtExpr::Cmp(
            SC::Eq,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Ne, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ne,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::And, l, r, _) => Some(SmtExpr::Bool(
            SB::And,
            vec![
                surf_expr_to_smt_subst(l, subst, ctx)?,
                surf_expr_to_smt_subst(r, subst, ctx)?,
            ],
        )),
        Expr::Binary(BinOp::Or, l, r, _) => Some(SmtExpr::Bool(
            SB::Or,
            vec![
                surf_expr_to_smt_subst(l, subst, ctx)?,
                surf_expr_to_smt_subst(r, subst, ctx)?,
            ],
        )),
        Expr::If(cond, then_e, else_e, _) => {
            let c = surf_expr_to_smt_subst(cond, subst, ctx)?;
            let t = surf_arith_subst(then_e, subst, ctx)?;
            let e = surf_arith_subst(else_e, subst, ctx)?;
            Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)))
        }
        Expr::Apply(func, args, _) => {
            let name = match func.as_ref() {
                Expr::Var(n, _) => n.clone(),
                _ => return None,
            };
            let smt_args: Option<Vec<_>> = args
                .iter()
                .map(|a| surf_arith_subst(a, subst, ctx))
                .collect();
            let smt_args = smt_args?;
            if name == "neg" && smt_args.len() == 1 {
                return Some(SmtExpr::Arith(
                    SA::Neg,
                    Box::new(smt_args[0].clone()),
                    Box::new(SmtExpr::IntLit(0)),
                ));
            }
            if let Some(contracts) = ctx.contracts
                && let Some(abs) = contracts.borrow_mut().abstract_call(&name, &smt_args)
            {
                return Some(abs);
            }
            if let Some((params, body)) = lookup_fun_body(ctx.decls, &name) {
                if ctx.call_stack.contains(&name) {
                    return None;
                }
                if ctx.depth >= ctx.max_depth {
                    return None;
                }
                if params.len() != args.len() {
                    return None;
                }
                let inner_subst: std::collections::HashMap<String, &Expr> = params
                    .iter()
                    .zip(args.iter())
                    .map(|(p, a)| (p.name.clone(), a))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
                    contracts: ctx.contracts,
                    call_stack: {
                        let mut s = ctx.call_stack.clone();
                        s.push(name);
                        s
                    },
                };
                return surf_arith_subst(body, &inner_subst, &deeper);
            }
            Some(SmtExpr::Apply(name, smt_args))
        }
        Expr::Block(bindings, body, _) => {
            let mut extended_subst = subst.clone();
            for binding in bindings {
                if let LetPattern::Var(name, _) = &binding.pattern {
                    extended_subst.insert(name.clone(), &binding.value);
                } else {
                    return None;
                }
            }
            surf_arith_subst(body, &extended_subst, ctx)
        }
        _ => None,
    }
}

fn surf_expr_to_smt_subst(
    expr: &Expr,
    subst: &std::collections::HashMap<String, &Expr>,
    ctx: &InlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{BoolOp as SB, CmpOp as SC, SmtExpr};
    match expr {
        Expr::Binary(BinOp::Ge, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ge,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Le, l, r, _) => Some(SmtExpr::Cmp(
            SC::Le,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Gt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Gt,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Lt, l, r, _) => Some(SmtExpr::Cmp(
            SC::Lt,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Eq, l, r, _) => Some(SmtExpr::Cmp(
            SC::Eq,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::Ne, l, r, _) => Some(SmtExpr::Cmp(
            SC::Ne,
            Box::new(surf_arith_subst(l, subst, ctx)?),
            Box::new(surf_arith_subst(r, subst, ctx)?),
        )),
        Expr::Binary(BinOp::And, l, r, _) => Some(SmtExpr::Bool(
            SB::And,
            vec![
                surf_expr_to_smt_subst(l, subst, ctx)?,
                surf_expr_to_smt_subst(r, subst, ctx)?,
            ],
        )),
        Expr::Binary(BinOp::Or, l, r, _) => Some(SmtExpr::Bool(
            SB::Or,
            vec![
                surf_expr_to_smt_subst(l, subst, ctx)?,
                surf_expr_to_smt_subst(r, subst, ctx)?,
            ],
        )),
        Expr::Lit(Literal::Bool(v), _) => Some(SmtExpr::BoolLit(*v)),
        _ => None,
    }
}
