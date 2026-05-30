//! Surf AST → SmtExpr conversion.
//!
//! Extracted from `chelis-cli/src/prove.rs` to allow reuse across the
//! verification pipeline without depending on the CLI crate.

use std::collections::HashMap;

use chelis_surf::ast::{BinOp, Decl, Expr, LetPattern, Literal, Param};

use crate::solver::{ArithOp as SA, BoolOp as SB, CmpOp as SC, SmtExpr};

/// Context for inlining function calls during Surf→SMT conversion.
pub struct InlineCtx<'a> {
    pub decls: &'a [Decl],
    pub depth: usize,
    pub max_depth: usize,
    pub call_stack: Vec<String>,
}

/// Look up a function body by name in a slice of declarations.
pub fn lookup_fun_body<'a>(decls: &'a [Decl], name: &str) -> Option<(&'a [Param], &'a Expr)> {
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

/// Convert a boolean Surf expression to an SMT expression.
pub fn surf_expr_to_smt(expr: &Expr, ctx: &InlineCtx) -> Option<SmtExpr> {
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

/// Convert an arithmetic Surf expression to an SMT expression.
pub fn surf_arith(expr: &Expr, ctx: &InlineCtx) -> Option<SmtExpr> {
    match expr {
        Expr::Var(name, _) => Some(SmtExpr::Var(name.clone())),
        Expr::Lit(Literal::Float(v), _) => Some(SmtExpr::RealLit(*v)),
        Expr::Lit(Literal::Int(v), _) => Some(SmtExpr::IntLit(*v)),
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
                let subst: HashMap<String, &Expr> = params
                    .iter()
                    .zip(args.iter())
                    .map(|(p, a)| (p.name.clone(), a))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
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

/// Convert an arithmetic Surf expression to SMT with variable substitution.
pub fn surf_arith_subst<'a>(
    expr: &Expr,
    subst: &HashMap<String, &'a Expr>,
    ctx: &InlineCtx,
) -> Option<SmtExpr> {
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
                let inner_subst: HashMap<String, &Expr> = params
                    .iter()
                    .zip(args.iter())
                    .map(|(p, a)| (p.name.clone(), a))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
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

/// Convert a boolean Surf expression to SMT with variable substitution.
pub fn surf_expr_to_smt_subst<'a>(
    expr: &Expr,
    subst: &HashMap<String, &'a Expr>,
    ctx: &InlineCtx,
) -> Option<SmtExpr> {
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
