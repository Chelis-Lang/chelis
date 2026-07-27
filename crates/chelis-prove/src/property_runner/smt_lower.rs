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

use chelis_deep::DeepTag;
use std::cell::RefCell;

use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList};
use chelis_surf::ast::{BinOp, Decl, Expr, LetPattern, Literal, Param, UnaryOp};

use crate::contracts::{
    NORMAL_CDF_IMPLEMENTATION, NORMAL_CDF_MONOTONICITY, NORMAL_CDF_RANGE, NORMAL_CDF_REFLECTION,
};
use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};

pub(super) struct InlineCtx<'a> {
    pub(super) decls: &'a [Decl],
    pub(super) depth: usize,
    pub(super) max_depth: usize,
    pub(super) call_stack: Vec<String>,
    pub(super) contracts: Option<&'a RefCell<ContractAbstraction>>,
}

pub(super) struct DeepInlineCtx<'a> {
    pub(super) exprs: &'a [DeepExpr],
    pub(super) depth: usize,
    pub(super) max_depth: usize,
    pub(super) call_stack: Vec<String>,
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
    normal_cdf_monotonicity: bool,
    normal_cdf_symbols: Vec<String>,
    normal_cdf_calls: Vec<ContractCall>,
}

impl ContractAbstraction {
    pub(super) fn for_contracts(contracts: &[String], trusted_contract_decls: &[Decl]) -> Self {
        let normal_cdf_reflection = contracts.iter().any(|id| id == NORMAL_CDF_REFLECTION);
        let normal_cdf_range =
            normal_cdf_reflection || contracts.iter().any(|id| id == NORMAL_CDF_RANGE);
        let normal_cdf_monotonicity = contracts.iter().any(|id| id == NORMAL_CDF_MONOTONICITY);
        Self {
            normal_cdf_enabled: normal_cdf_range
                || normal_cdf_reflection
                || normal_cdf_monotonicity,
            normal_cdf_range,
            normal_cdf_reflection,
            normal_cdf_monotonicity,
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
        // chelis#674: monotonicity relational injection. For every ordered pair
        // of CDF calls, inject (=> (<= arg_i arg_j) (<= N(arg_i) N(arg_j))).
        // This enables the SMT solver to use argument ordering to derive output
        // ordering, which is needed for price-positivity proofs.
        if self.normal_cdf_monotonicity {
            for (idx, left) in self.normal_cdf_calls.iter().enumerate() {
                for right in self.normal_cdf_calls.iter().skip(idx + 1) {
                    // Inject: (arg_left <= arg_right) => (N_left <= N_right)
                    let arg_le = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(left.arg.clone()),
                        Box::new(right.arg.clone()),
                    );
                    let output_le = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::Var(left.symbol.clone())),
                        Box::new(SmtExpr::Var(right.symbol.clone())),
                    );
                    out.push(SmtExpr::Bool(
                        crate::solver::BoolOp::Implies,
                        vec![arg_le, output_le],
                    ));

                    // Inject the reverse: (arg_right <= arg_left) => (N_right <= N_left)
                    let arg_ge = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(right.arg.clone()),
                        Box::new(left.arg.clone()),
                    );
                    let output_ge = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::Var(right.symbol.clone())),
                        Box::new(SmtExpr::Var(left.symbol.clone())),
                    );
                    out.push(SmtExpr::Bool(
                        crate::solver::BoolOp::Implies,
                        vec![arg_ge, output_ge],
                    ));
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

/// The interpreted arithmetic op a call-form arithmetic primitive lowers to,
/// so call-form `mul(x, x)` lowers to the same interpreted `SmtExpr::Arith`
/// node as operator-form `x * x` (chelis#422). A call-form arithmetic op must
/// NOT fall through to an uninterpreted `SmtExpr::Apply`, which the
/// inlineability classifier rejects (only the intrinsic whitelist is
/// inlineable), silently dropping the property to Tier C fuzz.
fn call_form_arith_op(name: &str) -> Option<crate::solver::ArithOp> {
    use crate::solver::ArithOp as SA;
    match name {
        "add" => Some(SA::Add),
        "sub" => Some(SA::Sub),
        "mul" => Some(SA::Mul),
        "div" => Some(SA::Div),
        _ => None,
    }
}

/// The comparison op a call-form comparison primitive lowers to, so a
/// top-level call-form predicate (`gte(mul(x, x), 0.0)`) lowers to the same
/// `SmtExpr::Cmp` node as the operator-form (`(x * x) >= 0.0`) instead of
/// silently dropping to a fuzz pass (chelis#422). Both the named comparison
/// primitives and the desugared operator name `cmplt` (`<`) are mapped.
fn call_form_cmp_op(name: &str) -> Option<crate::solver::CmpOp> {
    use crate::solver::CmpOp as SC;
    match name {
        "gt" => Some(SC::Gt),
        "gte" => Some(SC::Ge),
        "lt" | "cmplt" => Some(SC::Lt),
        "lte" => Some(SC::Le),
        "eq" => Some(SC::Eq),
        "neq" => Some(SC::Ne),
        _ => None,
    }
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
        // A top-level call-form comparison predicate (`gte(mul(x, x), 0.0)`)
        // lowers to the same `Cmp` node as its operator-form (chelis#422).
        // Without this arm such predicates returned `None` and silently
        // dropped to a Tier C fuzz pass -- a false green on a measure-zero
        // false call-form. `and`/`or`/`not` connectives also lower call-form.
        Expr::Apply(func, args, _) => {
            let name = match func.as_ref() {
                Expr::Var(n, _) => n.as_str(),
                _ => return None,
            };
            if let Some(op) = call_form_cmp_op(name) {
                if args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Cmp(
                    op,
                    Box::new(surf_arith(&args[0], ctx)?),
                    Box::new(surf_arith(&args[1], ctx)?),
                ));
            }
            match (name, args.as_slice()) {
                ("and", [l, r]) => Some(SmtExpr::Bool(
                    SB::And,
                    vec![surf_expr_to_smt(l, ctx)?, surf_expr_to_smt(r, ctx)?],
                )),
                ("or", [l, r]) => Some(SmtExpr::Bool(
                    SB::Or,
                    vec![surf_expr_to_smt(l, ctx)?, surf_expr_to_smt(r, ctx)?],
                )),
                ("not", [inner]) => Some(SmtExpr::Not(Box::new(surf_expr_to_smt(inner, ctx)?))),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn deep_expr_to_smt(
    expr: &DeepExpr,
    ctx: &DeepInlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{BoolOp as SB, CmpOp as SC, SmtExpr};
    match deep_app_name_and_args(expr) {
        Some((name, args)) => {
            if let Some(op) = call_form_cmp_op(name) {
                if args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Cmp(
                    op,
                    Box::new(deep_arith(&args[0], ctx)?),
                    Box::new(deep_arith(&args[1], ctx)?),
                ));
            }
            match (name, args) {
                ("and", [left, right]) => Some(SmtExpr::Bool(
                    SB::And,
                    vec![deep_expr_to_smt(left, ctx)?, deep_expr_to_smt(right, ctx)?],
                )),
                ("or", [left, right]) => Some(SmtExpr::Bool(
                    SB::Or,
                    vec![deep_expr_to_smt(left, ctx)?, deep_expr_to_smt(right, ctx)?],
                )),
                ("not", [inner]) => Some(SmtExpr::Not(Box::new(deep_expr_to_smt(inner, ctx)?))),
                _ => None,
            }
        }
        _ => match deep_bool_lit(expr) {
            Some(value) => Some(SmtExpr::BoolLit(value)),
            None => {
                if let Some((left, right)) = deep_builtin_cmp(expr, "cmplt") {
                    return Some(SmtExpr::Cmp(
                        SC::Lt,
                        Box::new(deep_arith(left, ctx)?),
                        Box::new(deep_arith(right, ctx)?),
                    ));
                }
                None
            }
        },
    }
}

fn deep_arith(expr: &DeepExpr, ctx: &DeepInlineCtx) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, SmtExpr};
    if let Some(name) = deep_var_name(expr) {
        return Some(SmtExpr::Var(name.to_string()));
    }
    if let Some(value) = deep_float_lit(expr) {
        return Some(SmtExpr::RealLit(value));
    }
    if let Some(value) = deep_int_lit(expr) {
        return Some(SmtExpr::IntLit(value));
    }
    if let Some((name, args)) = deep_app_name_and_args(expr) {
        let smt_args: Option<Vec<_>> = args.iter().map(|arg| deep_arith(arg, ctx)).collect();
        let smt_args = smt_args?;
        if name == "neg" && smt_args.len() == 1 {
            return Some(SmtExpr::Arith(
                SA::Neg,
                Box::new(smt_args[0].clone()),
                Box::new(SmtExpr::IntLit(0)),
            ));
        }
        if let Some(op) = call_form_arith_op(name) {
            if smt_args.len() != 2 {
                return None;
            }
            return Some(SmtExpr::Arith(
                op,
                Box::new(smt_args[0].clone()),
                Box::new(smt_args[1].clone()),
            ));
        }
        if let Some((params, body)) = lookup_deep_fun_body(ctx.exprs, name) {
            if ctx.call_stack.iter().any(|existing| existing == name) {
                return None;
            }
            if ctx.depth >= ctx.max_depth || params.len() != args.len() {
                return None;
            }
            let subst = params
                .iter()
                .zip(smt_args.iter())
                .map(|(param, arg)| (param.clone(), arg.clone()))
                .collect();
            let deeper = DeepInlineCtx {
                exprs: ctx.exprs,
                depth: ctx.depth + 1,
                max_depth: ctx.max_depth,
                call_stack: {
                    let mut stack = ctx.call_stack.clone();
                    stack.push(name.to_string());
                    stack
                },
            };
            return deep_arith_subst(body, &subst, &deeper);
        }
        return Some(SmtExpr::Apply(name.to_string(), smt_args));
    }
    if deep_tag(expr) == Some(DeepTag::If) {
        let list = deep_list(expr)?;
        let cond = list.elements.get(2)?;
        let then_expr = list.elements.get(3)?;
        let else_expr = list.elements.get(4)?;
        return Some(SmtExpr::Ite(
            Box::new(deep_expr_to_smt(cond, ctx)?),
            Box::new(deep_arith(then_expr, ctx)?),
            Box::new(deep_arith(else_expr, ctx)?),
        ));
    }
    None
}

fn deep_arith_subst(
    expr: &DeepExpr,
    subst: &std::collections::HashMap<String, crate::solver::SmtExpr>,
    ctx: &DeepInlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, SmtExpr};
    if let Some(name) = deep_var_name(expr) {
        return subst
            .get(name)
            .cloned()
            .or_else(|| Some(SmtExpr::Var(name.to_string())));
    }
    if let Some(value) = deep_float_lit(expr) {
        return Some(SmtExpr::RealLit(value));
    }
    if let Some(value) = deep_int_lit(expr) {
        return Some(SmtExpr::IntLit(value));
    }
    if let Some((name, args)) = deep_app_name_and_args(expr) {
        let smt_args: Option<Vec<_>> = args
            .iter()
            .map(|arg| deep_arith_subst(arg, subst, ctx))
            .collect();
        let smt_args = smt_args?;
        if name == "neg" && smt_args.len() == 1 {
            return Some(SmtExpr::Arith(
                SA::Neg,
                Box::new(smt_args[0].clone()),
                Box::new(SmtExpr::IntLit(0)),
            ));
        }
        if let Some(op) = call_form_arith_op(name) {
            if smt_args.len() != 2 {
                return None;
            }
            return Some(SmtExpr::Arith(
                op,
                Box::new(smt_args[0].clone()),
                Box::new(smt_args[1].clone()),
            ));
        }
        if let Some((params, body)) = lookup_deep_fun_body(ctx.exprs, name) {
            if ctx.call_stack.iter().any(|existing| existing == name) {
                return None;
            }
            if ctx.depth >= ctx.max_depth || params.len() != args.len() {
                return None;
            }
            let inner_subst = params
                .iter()
                .zip(smt_args.iter())
                .map(|(param, arg)| (param.clone(), arg.clone()))
                .collect();
            let deeper = DeepInlineCtx {
                exprs: ctx.exprs,
                depth: ctx.depth + 1,
                max_depth: ctx.max_depth,
                call_stack: {
                    let mut stack = ctx.call_stack.clone();
                    stack.push(name.to_string());
                    stack
                },
            };
            return deep_arith_subst(body, &inner_subst, &deeper);
        }
        return Some(SmtExpr::Apply(name.to_string(), smt_args));
    }
    if deep_tag(expr) == Some(DeepTag::If) {
        let list = deep_list(expr)?;
        let cond = list.elements.get(2)?;
        let then_expr = list.elements.get(3)?;
        let else_expr = list.elements.get(4)?;
        return Some(SmtExpr::Ite(
            Box::new(deep_expr_to_smt_subst(cond, subst, ctx)?),
            Box::new(deep_arith_subst(then_expr, subst, ctx)?),
            Box::new(deep_arith_subst(else_expr, subst, ctx)?),
        ));
    }
    if deep_tag(expr) == Some(DeepTag::Let) {
        let list = deep_list(expr)?;
        let bind = list.elements.get(2)?;
        let body = list.elements.get(3)?;
        let mut extended = subst.clone();
        for (name, value) in deep_bind_pairs(bind)? {
            let lowered = deep_arith_subst(value, &extended, ctx)?;
            extended.insert(name.to_string(), lowered);
        }
        return deep_arith_subst(body, &extended, ctx);
    }
    None
}

fn deep_expr_to_smt_subst(
    expr: &DeepExpr,
    subst: &std::collections::HashMap<String, crate::solver::SmtExpr>,
    ctx: &DeepInlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{BoolOp as SB, CmpOp as SC, SmtExpr};
    match deep_app_name_and_args(expr) {
        Some((name, args)) => {
            if let Some(op) = call_form_cmp_op(name) {
                if args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Cmp(
                    op,
                    Box::new(deep_arith_subst(&args[0], subst, ctx)?),
                    Box::new(deep_arith_subst(&args[1], subst, ctx)?),
                ));
            }
            match (name, args) {
                ("and", [left, right]) => Some(SmtExpr::Bool(
                    SB::And,
                    vec![
                        deep_expr_to_smt_subst(left, subst, ctx)?,
                        deep_expr_to_smt_subst(right, subst, ctx)?,
                    ],
                )),
                ("or", [left, right]) => Some(SmtExpr::Bool(
                    SB::Or,
                    vec![
                        deep_expr_to_smt_subst(left, subst, ctx)?,
                        deep_expr_to_smt_subst(right, subst, ctx)?,
                    ],
                )),
                ("not", [inner]) => Some(SmtExpr::Not(Box::new(deep_expr_to_smt_subst(
                    inner, subst, ctx,
                )?))),
                _ => None,
            }
        }
        _ => match deep_bool_lit(expr) {
            Some(value) => Some(SmtExpr::BoolLit(value)),
            None => {
                if let Some((left, right)) = deep_builtin_cmp(expr, "cmplt") {
                    return Some(SmtExpr::Cmp(
                        SC::Lt,
                        Box::new(deep_arith_subst(left, subst, ctx)?),
                        Box::new(deep_arith_subst(right, subst, ctx)?),
                    ));
                }
                None
            }
        },
    }
}

fn lookup_deep_fun_body<'a>(
    exprs: &'a [DeepExpr],
    name: &str,
) -> Option<(Vec<String>, &'a DeepExpr)> {
    let mut stack: Vec<&DeepExpr> = exprs.iter().collect();
    while let Some(expr) = stack.pop() {
        let Some(list) = deep_list(expr) else {
            continue;
        };
        match list_tag_from_list(list) {
            Some(DeepTag::Module) => {
                stack.extend(list.elements.iter().skip(3));
            }
            Some(DeepTag::Def) if list.elements.get(2).and_then(deep_symbol_text) == Some(name) => {
                let fn_expr = list.elements.get(3)?;
                let params = deep_fn_param_names(fn_expr)?;
                let body = deep_fn_body(fn_expr)?;
                return Some((params, body));
            }
            _ => {}
        }
    }
    None
}

fn deep_fn_param_names(expr: &DeepExpr) -> Option<Vec<String>> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Fn) {
        return None;
    }
    let params = list.elements.get(2).and_then(deep_list)?;
    if list_tag_from_list(params) != Some(DeepTag::Params) {
        return None;
    }
    params
        .elements
        .iter()
        .skip(2)
        .map(|param| {
            deep_symbol_text(param)
                .or_else(|| {
                    let list = deep_list(param)?;
                    list.elements.first().and_then(deep_symbol_text)
                })
                .map(str::to_string)
        })
        .collect()
}

fn deep_fn_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Fn) {
        return None;
    }
    list.elements.get(3)
}

fn deep_bind_pairs(expr: &DeepExpr) -> Option<Vec<(&str, &DeepExpr)>> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Bind) {
        return None;
    }
    let mut pairs = Vec::new();
    let mut children = list.elements.iter().skip(2);
    while let Some(name) = children.next() {
        let value = children.next()?;
        pairs.push((deep_symbol_text(name)?, value));
    }
    Some(pairs)
}

fn deep_builtin_cmp<'a>(expr: &'a DeepExpr, name: &str) -> Option<(&'a DeepExpr, &'a DeepExpr)> {
    let (found, args) = deep_app_name_and_args(expr)?;
    (found == name && args.len() == 2).then_some((&args[0], &args[1]))
}

fn deep_app_name_and_args(expr: &DeepExpr) -> Option<(&str, &[DeepExpr])> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::App) {
        return None;
    }
    let name = list.elements.get(2).and_then(deep_var_name)?;
    Some((name, &list.elements[3..]))
}

fn deep_var_name(expr: &DeepExpr) -> Option<&str> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Var) {
        return None;
    }
    list.elements.get(2).and_then(deep_symbol_text)
}

fn deep_float_lit(expr: &DeepExpr) -> Option<f64> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Lit) {
        return None;
    }
    match list.elements.get(2)? {
        DeepExpr::Atom(DeepAtom::Float(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_int_lit(expr: &DeepExpr) -> Option<i64> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Lit) {
        return None;
    }
    match list.elements.get(2)? {
        DeepExpr::Atom(DeepAtom::Int(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_bool_lit(expr: &DeepExpr) -> Option<bool> {
    let list = deep_list(expr)?;
    if list_tag_from_list(list) != Some(DeepTag::Lit) {
        return None;
    }
    match list.elements.get(2)? {
        DeepExpr::Atom(DeepAtom::Bool(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_tag(expr: &DeepExpr) -> Option<DeepTag> {
    match expr {
        DeepExpr::List(list, _) => list.tag(),
        _ => None,
    }
}

fn list_tag_from_list(list: &DeepList) -> Option<DeepTag> {
    list.tag()
}

fn deep_list(expr: &DeepExpr) -> Option<&DeepList> {
    match expr {
        DeepExpr::List(list, _) => Some(list),
        _ => None,
    }
}

fn deep_symbol_text(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Symbol(value), _) => Some(value.as_str()),
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
            // Call-form arithmetic (`mul(x, x)`) lowers to the same
            // interpreted `Arith` node as operator-form (`x * x`), not an
            // uninterpreted `Apply` the inlineability classifier would reject
            // (chelis#422).
            if let Some(op) = call_form_arith_op(&name) {
                if smt_args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Arith(
                    op,
                    Box::new(smt_args[0].clone()),
                    Box::new(smt_args[1].clone()),
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
                // Bind each parameter to its argument ALREADY LOWERED in the
                // caller's scope (`smt_args`), so a nested call inside `body`
                // sees fully-ground operands. Re-lowering the raw argument
                // Expr inside the callee body would resolve its free vars in
                // the WRONG (callee) scope -- the chelis#426 collapse, where
                // two call-sites with different args produced identical SMT.
                let subst: std::collections::HashMap<String, SmtExpr> = params
                    .iter()
                    .zip(smt_args.iter())
                    .map(|(p, a)| (p.name.clone(), a.clone()))
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
    subst: &std::collections::HashMap<String, SmtExpr>,
    ctx: &InlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, BoolOp as SB, CmpOp as SC, SmtExpr};
    match expr {
        Expr::Var(name, _) => {
            if let Some(replacement) = subst.get(name.as_str()) {
                // The replacement is already lowered in the binding scope;
                // return it verbatim (do NOT re-lower in this scope).
                Some(replacement.clone())
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
            // Call-form arithmetic lowers to an interpreted `Arith` node here
            // too, so an inlined call body carrying `mul(...)` lowers the same
            // way the operator-form does (chelis#422).
            if let Some(op) = call_form_arith_op(&name) {
                if smt_args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Arith(
                    op,
                    Box::new(smt_args[0].clone()),
                    Box::new(smt_args[1].clone()),
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
                // Bind each parameter to its argument ALREADY LOWERED under the
                // CURRENT substitution (`smt_args`), so the parent scope's
                // bindings flow into the nested callee body. Binding the raw
                // argument Expr and re-lowering it inside the callee would lose
                // the parent bindings -- the chelis#426 nested-call collapse.
                let inner_subst: std::collections::HashMap<String, SmtExpr> = params
                    .iter()
                    .zip(smt_args.iter())
                    .map(|(p, a)| (p.name.clone(), a.clone()))
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
            // A let-binding's value lowers under the CURRENT substitution (the
            // same scope-correctness the call path needs): a binding RHS that
            // references an outer-scope param must resolve through `subst`, not
            // be re-lowered later in a scope that has lost it.
            let mut extended_subst = subst.clone();
            for binding in bindings {
                if let LetPattern::Var(name, _) = &binding.pattern {
                    let value = surf_arith_subst(&binding.value, &extended_subst, ctx)?;
                    extended_subst.insert(name.clone(), value);
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
    subst: &std::collections::HashMap<String, SmtExpr>,
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
        // Call-form comparison predicates lower under substitution too, so an
        // inlined predicate body written in call-form lowers to a `Cmp` node
        // rather than silently dropping to fuzz (chelis#422).
        Expr::Apply(func, args, _) => {
            let name = match func.as_ref() {
                Expr::Var(n, _) => n.as_str(),
                _ => return None,
            };
            if let Some(op) = call_form_cmp_op(name) {
                if args.len() != 2 {
                    return None;
                }
                return Some(SmtExpr::Cmp(
                    op,
                    Box::new(surf_arith_subst(&args[0], subst, ctx)?),
                    Box::new(surf_arith_subst(&args[1], subst, ctx)?),
                ));
            }
            match (name, args.as_slice()) {
                ("and", [l, r]) => Some(SmtExpr::Bool(
                    SB::And,
                    vec![
                        surf_expr_to_smt_subst(l, subst, ctx)?,
                        surf_expr_to_smt_subst(r, subst, ctx)?,
                    ],
                )),
                ("or", [l, r]) => Some(SmtExpr::Bool(
                    SB::Or,
                    vec![
                        surf_expr_to_smt_subst(l, subst, ctx)?,
                        surf_expr_to_smt_subst(r, subst, ctx)?,
                    ],
                )),
                ("not", [inner]) => Some(SmtExpr::Not(Box::new(surf_expr_to_smt_subst(
                    inner, subst, ctx,
                )?))),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, CmpOp, SmtExpr};
    use chelis_deep::Span;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string(), sp())
    }

    fn apply(name: &str, args: Vec<Expr>) -> Expr {
        Expr::Apply(Box::new(var(name)), args, sp())
    }

    fn float(v: f64) -> Expr {
        Expr::Lit(Literal::Float(v), sp())
    }

    fn ctx<'a>(decls: &'a [Decl]) -> InlineCtx<'a> {
        InlineCtx {
            decls,
            depth: 0,
            max_depth: 3,
            call_stack: vec![],
            contracts: None,
        }
    }

    // chelis#422: call-form arithmetic must lower to an interpreted `Arith`
    // node, identical to operator-form, not an uninterpreted `Apply` that the
    // inlineability classifier rejects (which silently drops to fuzz).
    #[test]
    fn call_form_mul_lowers_to_interpreted_arith() {
        let decls: Vec<Decl> = Vec::new();
        let call = apply("mul", vec![var("x"), var("x")]);
        let lowered = surf_arith(&call, &ctx(&decls)).expect("mul lowers");
        assert_eq!(
            lowered,
            SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Var("x".into())),
            )
        );
    }

    // chelis#422: a top-level call-form comparison predicate lowers to the
    // same `Cmp` node as operator-form (`gte(mul(x, x), 0.0)` == `x*x >= 0.0`).
    #[test]
    fn call_form_comparison_predicate_lowers_to_cmp() {
        let decls: Vec<Decl> = Vec::new();
        let pred = apply(
            "gte",
            vec![apply("mul", vec![var("x"), var("x")]), float(0.0)],
        );
        let lowered = surf_expr_to_smt(&pred, &ctx(&decls)).expect("gte predicate lowers");
        assert_eq!(
            lowered,
            SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::Var("x".into())),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            )
        );
    }

    // Every call-form comparison primitive maps to the matching `CmpOp`, with
    // `cmplt` (the desugared `<`) mapping to `Lt`.
    #[test]
    fn every_call_form_comparison_maps_to_its_op() {
        for (name, op) in [
            ("gt", CmpOp::Gt),
            ("gte", CmpOp::Ge),
            ("lt", CmpOp::Lt),
            ("cmplt", CmpOp::Lt),
            ("lte", CmpOp::Le),
            ("eq", CmpOp::Eq),
            ("neq", CmpOp::Ne),
        ] {
            assert_eq!(call_form_cmp_op(name), Some(op), "{name}");
        }
    }

    // Negative parity: a call-form predicate that is NOT a comparison or a
    // boolean connective must return `None` (so the caller surfaces it rather
    // than minting a bogus interpreted node). This is the boundary the false
    // green crossed.
    #[test]
    fn non_predicate_call_form_does_not_lower_to_a_predicate() {
        let decls: Vec<Decl> = Vec::new();
        let pred = apply("mystery", vec![var("x")]);
        assert!(surf_expr_to_smt(&pred, &ctx(&decls)).is_none());
    }

    fn param(name: &str) -> Param {
        Param {
            name: name.to_string(),
            ty: None,
            span: sp(),
        }
    }

    fn fun_def(name: &str, params: &[&str], body: Expr) -> Decl {
        Decl::FunDef {
            name: name.to_string(),
            dim_params: Vec::new(),
            params: params.iter().map(|p| param(p)).collect(),
            ret_ty: None,
            effects: None,
            body,
            span: sp(),
        }
    }

    fn binop(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary(op, Box::new(l), Box::new(r), sp())
    }

    // chelis#426: two calls of a def whose body CALLS an ITE-bodied helper,
    // with DIFFERENT arguments, must lower to DISTINCT SMT terms. The bug:
    // inlining the nested helper call lost the outer call-site's argument
    // bindings, so both calls collapsed to the same term (`<=` then trivially
    // true => false-prove). The two `Cmp` operands must NOT be structurally
    // equal, and the RHS must mention the swapped `w*` vars.
    #[test]
    fn two_calls_of_ite_bodied_def_lower_to_distinct_terms() {
        // fmax(a, b) = if a >= b then a else b
        let fmax = fun_def(
            "fmax",
            &["a", "b"],
            Expr::If(
                Box::new(binop(BinOp::Ge, var("a"), var("b"))),
                Box::new(var("a")),
                Box::new(var("b")),
                sp(),
            ),
        );
        // bs(x, y) = fmax(x, y)  -- a def that CALLS the ITE-bodied helper
        let bs = fun_def("bs", &["x", "y"], apply("fmax", vec![var("x"), var("y")]));
        let decls = vec![fmax, bs];

        // Goal body: bs(v0, v1) <= bs(w0, w1)
        let body = binop(
            BinOp::Le,
            apply("bs", vec![var("v0"), var("v1")]),
            apply("bs", vec![var("w0"), var("w1")]),
        );
        let lowered = surf_expr_to_smt(&body, &ctx(&decls)).expect("goal lowers");
        let (lhs, rhs) = match &lowered {
            SmtExpr::Cmp(CmpOp::Le, l, r) => (l.as_ref(), r.as_ref()),
            other => panic!("expected a <= comparison, got {other:?}"),
        };
        // The collapse made lhs == rhs; faithful lowering keeps them distinct.
        assert_ne!(
            lhs, rhs,
            "two call-sites with different args collapsed to identical terms (chelis#426)"
        );
        // The first call lowers over v0/v1, the second over w0/w1: confirm the
        // swapped vars actually reach the RHS rather than being dropped.
        assert!(
            term_mentions(lhs, "v0") && term_mentions(lhs, "v1"),
            "lhs lost its own call-site bindings: {lhs:?}"
        );
        assert!(
            term_mentions(rhs, "w0") && term_mentions(rhs, "w1"),
            "rhs collapsed onto the first call-site's vars instead of w0/w1: {rhs:?}"
        );
        assert!(
            !term_mentions(rhs, "v0") && !term_mentions(rhs, "v1"),
            "rhs leaked the first call-site's vars (collapse): {rhs:?}"
        );
    }

    fn term_mentions(expr: &SmtExpr, name: &str) -> bool {
        match expr {
            SmtExpr::Var(n) => n == name,
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                term_mentions(l, name) || term_mentions(r, name)
            }
            SmtExpr::Ite(c, t, e) => {
                term_mentions(c, name) || term_mentions(t, name) || term_mentions(e, name)
            }
            SmtExpr::Not(inner) => term_mentions(inner, name),
            SmtExpr::Bool(_, parts) | SmtExpr::Apply(_, parts) => {
                parts.iter().any(|p| term_mentions(p, name))
            }
            _ => false,
        }
    }
}
