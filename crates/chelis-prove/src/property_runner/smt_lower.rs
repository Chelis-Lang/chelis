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

use chelis_deep::{DeepTag, ExprCarrier};
use chelis_unord::{UnordMap, UnordSet};
use std::cell::RefCell;

use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr};
use chelis_surf::ast::{BinOp, Decl, Expr, LetPattern, Literal, Param, TypeExpr, UnaryOp};

use crate::contracts::{
    NORMAL_CDF_IMPLEMENTATION, NORMAL_CDF_MONOTONICITY, NORMAL_CDF_RANGE, NORMAL_CDF_REFLECTION,
    QUANTILE_BOUNDARY, QUANTILE_MONOTONICITY, QUANTILE_RANGE,
};
use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};

const NESTED_GRAD_SMT_BOUNDARY: &str = "scalar grad SMT lowering does not support nested gradients";

pub(super) struct InlineCtx<'a> {
    pub(super) decls: &'a [Decl],
    pub(super) depth: usize,
    pub(super) max_depth: usize,
    pub(super) call_stack: Vec<String>,
    pub(super) contracts: Option<&'a RefCell<ContractAbstraction>>,
    /// First structured capability boundary encountered while lowering a
    /// scalar `grad` application.
    pub(super) grad_diagnostic: Option<&'a RefCell<Option<String>>>,
}

pub(super) struct DeepInlineCtx<'a> {
    pub(super) exprs: &'a [DeepExpr],
    pub(super) depth: usize,
    pub(super) max_depth: usize,
    pub(super) call_stack: Vec<String>,
    /// First structured capability boundary encountered while lowering a
    /// canonical Deep scalar `grad` application.
    pub(super) grad_diagnostic: Option<&'a RefCell<Option<String>>>,
}

#[derive(Debug, Clone)]
struct ContractCall {
    symbol: String,
    arg: SmtExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DatasetIdentity(String);

#[derive(Debug, Clone)]
struct QuantileContractCall {
    symbol: String,
    dataset: DatasetIdentity,
    q: SmtExpr,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ContractAbstraction {
    normal_cdf_enabled: bool,
    normal_cdf_range: bool,
    normal_cdf_reflection: bool,
    normal_cdf_monotonicity: bool,
    normal_cdf_symbols: Vec<String>,
    normal_cdf_calls: Vec<ContractCall>,
    quantile_enabled: bool,
    quantile_monotonicity: bool,
    quantile_has_unsupported_contract: bool,
    quantile_symbols: Vec<String>,
    quantile_calls: Vec<QuantileContractCall>,
}

impl ContractAbstraction {
    pub(super) fn for_contracts(contracts: &[String], trusted_contract_decls: &[Decl]) -> Self {
        let normal_cdf_reflection = contracts.iter().any(|id| id == NORMAL_CDF_REFLECTION);
        let normal_cdf_range =
            normal_cdf_reflection || contracts.iter().any(|id| id == NORMAL_CDF_RANGE);
        let normal_cdf_monotonicity = contracts.iter().any(|id| id == NORMAL_CDF_MONOTONICITY);
        let quantile_monotonicity = contracts.iter().any(|id| id == QUANTILE_MONOTONICITY);
        let quantile_has_unsupported_contract = contracts
            .iter()
            .any(|id| matches!(id.as_str(), QUANTILE_RANGE | QUANTILE_BOUNDARY));
        Self {
            normal_cdf_enabled: normal_cdf_range
                || normal_cdf_reflection
                || normal_cdf_monotonicity,
            normal_cdf_range,
            normal_cdf_reflection,
            normal_cdf_monotonicity,
            normal_cdf_symbols: trusted_normal_cdf_symbols(trusted_contract_decls),
            normal_cdf_calls: Vec::new(),
            quantile_enabled: quantile_monotonicity || quantile_has_unsupported_contract,
            quantile_monotonicity,
            quantile_has_unsupported_contract,
            quantile_symbols: trusted_quantile_symbols(trusted_contract_decls),
            quantile_calls: Vec::new(),
        }
    }

    pub(super) fn requires_normal_cdf(&self) -> bool {
        self.normal_cdf_enabled
    }

    pub(super) fn used_normal_cdf(&self) -> bool {
        !self.normal_cdf_calls.is_empty()
    }

    pub(super) fn requires_quantile(&self) -> bool {
        self.quantile_enabled
    }

    pub(super) fn used_quantile(&self) -> bool {
        !self.quantile_calls.is_empty()
    }

    pub(super) fn has_unsupported_quantile_contract(&self) -> bool {
        self.quantile_has_unsupported_contract
    }

    pub(super) fn has_quantile_monotonicity_pair(&self) -> bool {
        self.quantile_calls.iter().enumerate().any(|(idx, left)| {
            self.quantile_calls
                .iter()
                .skip(idx + 1)
                .any(|right| left.dataset == right.dataset)
        })
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
            .chain(
                self.quantile_calls
                    .iter()
                    .map(|call| (call.symbol.clone(), SmtSort::Real)),
            )
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
        // chelis#979: quantile monotonicity is relational and is sound only
        // for calls over the same compiler-bound dataset. The dataset itself
        // never enters scalar SMT lowering; its linker-rewritten AST identity
        // selects the pairs whose scalar q arguments may be coupled.
        if self.quantile_monotonicity {
            for (idx, left) in self.quantile_calls.iter().enumerate() {
                for right in self.quantile_calls.iter().skip(idx + 1) {
                    if left.dataset != right.dataset {
                        continue;
                    }
                    let q_le = quantile_order_domain(&left.q, &right.q);
                    let value_le = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::Var(left.symbol.clone())),
                        Box::new(SmtExpr::Var(right.symbol.clone())),
                    );
                    out.push(SmtExpr::Bool(
                        crate::solver::BoolOp::Implies,
                        vec![q_le, value_le],
                    ));
                    let q_ge = quantile_order_domain(&right.q, &left.q);
                    let value_ge = SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::Var(right.symbol.clone())),
                        Box::new(SmtExpr::Var(left.symbol.clone())),
                    );
                    out.push(SmtExpr::Bool(
                        crate::solver::BoolOp::Implies,
                        vec![q_ge, value_ge],
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

    fn abstract_quantile_call(
        &mut self,
        name: &str,
        dataset_expr: &Expr,
        q: SmtExpr,
    ) -> Option<SmtExpr> {
        if !self.quantile_enabled || !self.quantile_symbols.iter().any(|symbol| symbol == name) {
            return None;
        }
        let dataset = dataset_identity(dataset_expr)?;
        let symbol = format!("__contract_std_quantile_{}", self.quantile_calls.len());
        self.quantile_calls.push(QuantileContractCall {
            symbol: symbol.clone(),
            dataset,
            q,
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

const LINKED_NAUTILUS_QUANTILE: &str = "pkg__nautilus__Nautilus__Stats__quantile_vec";

fn trusted_quantile_symbols(decls: &[Decl]) -> Vec<String> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::FunDef {
                name,
                type_binders,
                params,
                ret_ty,
                ..
            } if name == LINKED_NAUTILUS_QUANTILE
                && type_binders
                    .iter()
                    .map(|binder| (binder.name.as_str(), binder.bound))
                    .eq([("n", None)])
                && params.len() == 2
                && matches!(
                    params[0].ty.as_ref(),
                    Some(TypeExpr::Ref(inner, _))
                        if matches!(
                            inner.as_ref(),
                            TypeExpr::Tensor(dimensions, precision, _)
                                if precision == "f32"
                                    && matches!(
                                        dimensions.as_slice(),
                                        [TypeExpr::Named(dimension, _)] if dimension == "n"
                                    )
                        )
                )
                && matches!(params[1].ty.as_ref(), Some(TypeExpr::Named(name, _)) if name == "f32")
                && matches!(ret_ty.as_ref(), Some(TypeExpr::Named(name, _)) if name == "f32") =>
            {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

/// Stable identity for the tensor operand of a quantile call. Linker rewriting
/// has already replaced author names with compiler-owned symbols at this
/// point. Spans are deliberately ignored; two source occurrences of the same
/// bound dataset must compare equal.
fn dataset_identity(expr: &Expr) -> Option<DatasetIdentity> {
    match expr {
        Expr::Borrow(inner, _) | Expr::Annotate(inner, _, _) => dataset_identity(inner),
        Expr::Var(name, _) => Some(DatasetIdentity(format!("var:{name}"))),
        Expr::Apply(func, args, _) if args.is_empty() => match func.as_ref() {
            Expr::Var(name, _) => Some(DatasetIdentity(format!("call:{name}"))),
            _ => None,
        },
        _ => None,
    }
}

fn quantile_order_domain(low: &SmtExpr, high: &SmtExpr) -> SmtExpr {
    SmtExpr::Bool(
        crate::solver::BoolOp::And,
        vec![
            SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::RealLit(0.0)),
                Box::new(low.clone()),
            ),
            SmtExpr::Cmp(CmpOp::Le, Box::new(low.clone()), Box::new(high.clone())),
            SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(high.clone()),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        ],
    )
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
    if let Some(reason) = deep_grad_capability_reason(expr, ctx) {
        record_deep_grad_diagnostic(ctx, reason);
        return None;
    }
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
                grad_diagnostic: ctx.grad_diagnostic,
            };
            return deep_arith_subst(body, &subst, &deeper);
        }
        return Some(SmtExpr::Apply(name.to_string(), smt_args));
    }
    if deep_tag(expr) == Some(DeepTag::If) {
        let (_, children) = deep_node_parts(expr)?;
        let cond = children.first()?;
        let then_expr = children.get(1)?;
        let else_expr = children.get(2)?;
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
    subst: &chelis_unord::UnordMap<String, crate::solver::SmtExpr>,
    ctx: &DeepInlineCtx,
) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, SmtExpr};
    if let Some(reason) = deep_grad_capability_reason(expr, ctx) {
        record_deep_grad_diagnostic(ctx, reason);
        return None;
    }
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
                grad_diagnostic: ctx.grad_diagnostic,
            };
            return deep_arith_subst(body, &inner_subst, &deeper);
        }
        return Some(SmtExpr::Apply(name.to_string(), smt_args));
    }
    if deep_tag(expr) == Some(DeepTag::If) {
        let (_, children) = deep_node_parts(expr)?;
        let cond = children.first()?;
        let then_expr = children.get(1)?;
        let else_expr = children.get(2)?;
        return Some(SmtExpr::Ite(
            Box::new(deep_expr_to_smt_subst(cond, subst, ctx)?),
            Box::new(deep_arith_subst(then_expr, subst, ctx)?),
            Box::new(deep_arith_subst(else_expr, subst, ctx)?),
        ));
    }
    if deep_tag(expr) == Some(DeepTag::Let) {
        let (_, children) = deep_node_parts(expr)?;
        let bind = children.first()?;
        let body = children.get(1)?;
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
    subst: &chelis_unord::UnordMap<String, crate::solver::SmtExpr>,
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
        let Some((tag, children)) = deep_node_parts(expr) else {
            continue;
        };
        match tag {
            DeepTag::Module => {
                stack.extend(children.iter().skip(1));
            }
            DeepTag::Def if children.first().and_then(deep_symbol_text) == Some(name) => {
                let fn_expr = children.get(1)?;
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
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Fn {
        return None;
    }
    let (params_tag, params) = deep_node_parts(children.first()?)?;
    if params_tag != DeepTag::Params {
        return None;
    }
    params
        .iter()
        .map(|param| deep_param_name(param).map(str::to_string))
        .collect()
}

fn deep_fn_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Fn {
        return None;
    }
    children.get(1)
}

fn deep_bind_pairs(expr: &DeepExpr) -> Option<Vec<(&str, &DeepExpr)>> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Bind {
        return None;
    }
    let mut pairs = Vec::new();
    let mut children = children.iter();
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
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::App {
        return None;
    }
    let name = children.first().and_then(deep_var_name)?;
    Some((name, children.get(1..)?))
}

fn deep_var_name(expr: &DeepExpr) -> Option<&str> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Var {
        return None;
    }
    children.first().and_then(deep_symbol_text)
}

fn deep_float_lit(expr: &DeepExpr) -> Option<f64> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Lit {
        return None;
    }
    match children.first()? {
        DeepExpr::Atom(DeepAtom::Float(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_int_lit(expr: &DeepExpr) -> Option<i64> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Lit {
        return None;
    }
    match children.first()? {
        DeepExpr::Atom(DeepAtom::Int(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_bool_lit(expr: &DeepExpr) -> Option<bool> {
    let (tag, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Lit {
        return None;
    }
    match children.first()? {
        DeepExpr::Atom(DeepAtom::Bool(value), _) => Some(*value),
        _ => None,
    }
}

fn deep_tag(expr: &DeepExpr) -> Option<DeepTag> {
    deep_node_parts(expr).map(|(tag, _)| tag)
}

fn deep_node_parts(expr: &DeepExpr) -> Option<(DeepTag, &[DeepExpr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => Some((tag, children)),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn deep_param_name(expr: &DeepExpr) -> Option<&str> {
    if let Some(name) = deep_symbol_text(expr) {
        return Some(name);
    }
    match expr.carrier() {
        ExprCarrier::StructuralList(elements) => elements.first().and_then(deep_symbol_text),
        ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn deep_symbol_text(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Name(value), _) => Some(value.as_str()),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ScalarDual {
    value: crate::solver::SmtExpr,
    tangent: crate::solver::SmtExpr,
}

fn real(value: f64) -> crate::solver::SmtExpr {
    crate::solver::SmtExpr::RealLit(value)
}

fn arith(
    op: crate::solver::ArithOp,
    left: crate::solver::SmtExpr,
    right: crate::solver::SmtExpr,
) -> crate::solver::SmtExpr {
    use crate::solver::{ArithOp, SmtExpr};
    match (&op, &left, &right) {
        (ArithOp::Add, SmtExpr::RealLit(0.0), _) => return right,
        (ArithOp::Add, _, SmtExpr::RealLit(0.0)) | (ArithOp::Sub, _, SmtExpr::RealLit(0.0)) => {
            return left;
        }
        (ArithOp::Mul, SmtExpr::RealLit(0.0), _) | (ArithOp::Mul, _, SmtExpr::RealLit(0.0)) => {
            return real(0.0);
        }
        (ArithOp::Mul, SmtExpr::RealLit(1.0), _) => return right,
        (ArithOp::Mul, _, SmtExpr::RealLit(1.0)) | (ArithOp::Div, _, SmtExpr::RealLit(1.0)) => {
            return left;
        }
        _ => {}
    }
    SmtExpr::Arith(op, Box::new(left), Box::new(right))
}

fn record_grad_diagnostic(ctx: &InlineCtx, reason: String) {
    if let Some(diagnostic) = ctx.grad_diagnostic {
        let mut diagnostic = diagnostic.borrow_mut();
        if diagnostic.is_none() {
            *diagnostic = Some(reason);
        }
    }
}

fn record_deep_grad_diagnostic(ctx: &DeepInlineCtx, reason: String) {
    if let Some(diagnostic) = ctx.grad_diagnostic {
        let mut diagnostic = diagnostic.borrow_mut();
        if diagnostic.is_none() {
            *diagnostic = Some(reason);
        }
    }
}

fn deep_grad_capability_reason(expr: &DeepExpr, ctx: &DeepInlineCtx) -> Option<String> {
    let (DeepTag::App, app_children) = deep_node_parts(expr)? else {
        return None;
    };
    let (DeepTag::Grad, grad_children) = deep_node_parts(app_children.first()?)? else {
        return None;
    };
    let target = grad_children.first()?;
    match deep_tag(target) {
        Some(DeepTag::Grad) => Some(NESTED_GRAD_SMT_BOUNDARY.to_string()),
        _ => {
            let name = deep_var_name(target)?;
            lookup_deep_fun_body(ctx.exprs, name)
                .is_none()
                .then(|| format!("scalar grad SMT lowering cannot resolve function `{name}`"))
        }
    }
}

fn scalar_param(param: &Param) -> bool {
    matches!(
        param.ty.as_ref(),
        Some(TypeExpr::Named(name, _)) if matches!(name.as_str(), "f32" | "f64")
    )
}

fn scalar_float_type(ty: Option<&TypeExpr>) -> bool {
    matches!(
        ty,
        Some(TypeExpr::Named(name, _)) if matches!(name.as_str(), "f32" | "f64")
    )
}

/// Fail-closed result-kind check for an inline gradient target.
///
/// Surf lambdas do not carry an explicit result annotation in the AST, so the
/// prover cannot read their checker-inferred result directly. This deliberately
/// recognizes only the same small floating arithmetic subset the symbolic-dual
/// lowering accepts. In particular, a bare/typed integer result is never
/// promoted to a Real merely because its derivative is zero.
fn inline_scalar_float_result(expr: &Expr, float_names: &UnordSet<String>, decls: &[Decl]) -> bool {
    match expr {
        Expr::Var(name, _) => float_names.contains(name),
        Expr::Lit(Literal::Float(_) | Literal::TypedFloat(_, _), _) => true,
        Expr::Unary(UnaryOp::Neg, inner, _) => {
            inline_scalar_float_result(inner, float_names, decls)
        }
        Expr::Binary(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, left, right, _) => {
            inline_scalar_float_result(left, float_names, decls)
                && inline_scalar_float_result(right, float_names, decls)
        }
        Expr::Apply(func, args, _) => {
            let Expr::Var(name, _) = func.as_ref() else {
                return false;
            };
            let declared_result = decls.iter().find_map(|decl| match decl {
                Decl::FunDef {
                    name: candidate,
                    ret_ty,
                    ..
                } if candidate == name => Some(ret_ty.as_ref()),
                _ => None,
            });
            // Unknown calls remain eligible here so `scalar_dual` can report
            // its more specific unsupported-intrinsic reason. They cannot
            // become green: that lowering rejects every unknown call.
            let known_float_callable = call_form_arith_op(name).is_some()
                || name == "neg"
                || declared_result.is_none()
                || declared_result.is_some_and(scalar_float_type);
            known_float_callable
                && args
                    .iter()
                    .all(|arg| inline_scalar_float_result(arg, float_names, decls))
        }
        Expr::Block(bindings, body, _) => {
            let mut names = float_names.clone();
            for binding in bindings {
                let LetPattern::Var(name, _) = &binding.pattern else {
                    return false;
                };
                if !inline_scalar_float_result(&binding.value, &names, decls)
                    || binding
                        .ty
                        .as_ref()
                        .is_some_and(|ty| !scalar_float_type(Some(ty)))
                {
                    return false;
                }
                names.insert(name.clone());
            }
            inline_scalar_float_result(body, &names, decls)
        }
        Expr::Annotate(inner, ty, _) => {
            scalar_float_type(Some(ty)) && inline_scalar_float_result(inner, float_names, decls)
        }
        _ => false,
    }
}

fn scalar_grad_application(
    target: &Expr,
    wrt: Option<&[String]>,
    args: &[Expr],
    ctx: &InlineCtx,
    lower_arg: impl Fn(&Expr) -> Option<crate::solver::SmtExpr>,
) -> Result<crate::solver::SmtExpr, String> {
    let (params, body, target_name, declared_result) = match target {
        Expr::Lambda(params, body, _) => (params.as_slice(), body.as_ref(), None, None),
        Expr::Var(name, _) => {
            let Some((params, body, effects, ret_ty)) =
                ctx.decls.iter().find_map(|decl| match decl {
                    Decl::FunDef {
                        name: candidate,
                        params,
                        body,
                        effects,
                        ret_ty,
                        ..
                    } if candidate == name => {
                        Some((params.as_slice(), body, effects.as_ref(), ret_ty.as_ref()))
                    }
                    _ => None,
                })
            else {
                return Err(format!(
                    "scalar grad SMT lowering cannot resolve function `{name}`"
                ));
            };
            if effects.is_some_and(|effects| !effects.is_empty()) {
                return Err(format!(
                    "scalar grad SMT lowering does not support effectful function `{name}`"
                ));
            }
            (params, body, Some(name.as_str()), ret_ty)
        }
        Expr::Grad(_, _, _) => {
            return Err(NESTED_GRAD_SMT_BOUNDARY.to_string());
        }
        Expr::Vmap(_, _, _) => {
            return Err("scalar grad SMT lowering does not support nested `vmap`".to_string());
        }
        Expr::Jit(_, _) => {
            return Err("scalar grad SMT lowering does not support nested `jit`".to_string());
        }
        _ => {
            return Err(
                "scalar grad SMT lowering requires an inline lambda or top-level function"
                    .to_string(),
            );
        }
    };
    if params.len() != args.len() {
        return Err(format!(
            "scalar grad SMT lowering expected {} arguments, found {}",
            params.len(),
            args.len()
        ));
    }
    if !params.iter().all(scalar_param) {
        return Err("scalar grad SMT lowering supports only f32/f64 parameters".to_string());
    }
    let has_float_result = match declared_result {
        Some(ty) => scalar_float_type(Some(ty)),
        None => {
            let float_names = params
                .iter()
                .map(|param| param.name.clone())
                .collect::<UnordSet<_>>();
            inline_scalar_float_result(body, &float_names, ctx.decls)
        }
    };
    if !has_float_result {
        return Err("scalar grad SMT lowering requires an f32/f64 result".to_string());
    }
    let [wrt_name] = wrt.unwrap_or_default() else {
        return Err(
            "scalar grad SMT lowering requires exactly one explicit `wrt` parameter".to_string(),
        );
    };
    if !params.iter().any(|param| param.name == *wrt_name) {
        return Err(format!(
            "scalar grad SMT lowering cannot find `wrt` parameter `{wrt_name}`"
        ));
    }
    let lowered_args = args
        .iter()
        .map(lower_arg)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            "scalar grad SMT lowering requires scalar arithmetic arguments".to_string()
        })?;
    if let Some(name) = target_name
        && ctx.call_stack.iter().any(|active| active == name)
    {
        return Err(format!(
            "scalar grad SMT lowering does not support recursion through `{name}`"
        ));
    }
    let env = params
        .iter()
        .zip(lowered_args)
        .map(|(param, value)| {
            (
                param.name.clone(),
                ScalarDual {
                    value,
                    tangent: real(if param.name == *wrt_name { 1.0 } else { 0.0 }),
                },
            )
        })
        .collect();
    let mut call_stack = ctx.call_stack.clone();
    if let Some(name) = target_name {
        call_stack.push(name.to_string());
    }
    scalar_dual(
        body,
        &env,
        &InlineCtx {
            decls: ctx.decls,
            depth: ctx.depth + usize::from(target_name.is_some()),
            max_depth: ctx.max_depth,
            call_stack,
            contracts: ctx.contracts,
            grad_diagnostic: ctx.grad_diagnostic,
        },
    )
    .map(|dual| dual.tangent)
}

fn scalar_dual(
    expr: &Expr,
    env: &UnordMap<String, ScalarDual>,
    ctx: &InlineCtx,
) -> Result<ScalarDual, String> {
    use crate::solver::{ArithOp, SmtExpr};
    let binary = |op, left: &Expr, right: &Expr| -> Result<ScalarDual, String> {
        let left = scalar_dual(left, env, ctx)?;
        let right = scalar_dual(right, env, ctx)?;
        let value = arith(op, left.value.clone(), right.value.clone());
        let tangent = match op {
            ArithOp::Add | ArithOp::Sub => arith(op, left.tangent, right.tangent),
            ArithOp::Mul => arith(
                ArithOp::Add,
                arith(ArithOp::Mul, left.tangent, right.value.clone()),
                arith(ArithOp::Mul, left.value, right.tangent),
            ),
            ArithOp::Div => arith(
                ArithOp::Div,
                arith(
                    ArithOp::Sub,
                    arith(ArithOp::Mul, left.tangent, right.value.clone()),
                    arith(ArithOp::Mul, left.value, right.tangent),
                ),
                arith(ArithOp::Mul, right.value.clone(), right.value),
            ),
            ArithOp::Neg => unreachable!(),
        };
        Ok(ScalarDual { value, tangent })
    };

    match expr {
        Expr::Var(name, _) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("scalar grad SMT lowering cannot resolve scalar `{name}`")),
        Expr::Lit(Literal::Float(value) | Literal::TypedFloat(value, _), _) => Ok(ScalarDual {
            value: SmtExpr::RealLit(*value),
            tangent: real(0.0),
        }),
        Expr::Lit(Literal::Int(value) | Literal::TypedInt(value, _), _) => Ok(ScalarDual {
            value: SmtExpr::IntLit(*value),
            tangent: real(0.0),
        }),
        Expr::Unary(UnaryOp::Neg, inner, _) => {
            let inner = scalar_dual(inner, env, ctx)?;
            Ok(ScalarDual {
                value: arith(ArithOp::Neg, inner.value, SmtExpr::IntLit(0)),
                tangent: arith(ArithOp::Neg, inner.tangent, SmtExpr::IntLit(0)),
            })
        }
        Expr::Binary(BinOp::Add, left, right, _) => binary(ArithOp::Add, left, right),
        Expr::Binary(BinOp::Sub, left, right, _) => binary(ArithOp::Sub, left, right),
        Expr::Binary(BinOp::Mul, left, right, _) => binary(ArithOp::Mul, left, right),
        Expr::Binary(BinOp::Div, left, right, _) => binary(ArithOp::Div, left, right),
        Expr::Apply(func, args, _) => {
            let Expr::Var(name, _) = func.as_ref() else {
                return Err(
                    "scalar grad SMT lowering supports only named scalar helper calls".to_string(),
                );
            };
            let dual_args = args
                .iter()
                .map(|arg| scalar_dual(arg, env, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(op) = call_form_arith_op(name) {
                let [left, right] = dual_args.as_slice() else {
                    return Err(format!(
                        "scalar grad SMT lowering found wrong arity for `{name}`"
                    ));
                };
                let helper_env = UnordMap::from([
                    ("left".to_string(), left.clone()),
                    ("right".to_string(), right.clone()),
                ]);
                let span = chelis_deep::Span::new(0, 0);
                return scalar_dual(
                    &Expr::Binary(
                        match op {
                            ArithOp::Add => BinOp::Add,
                            ArithOp::Sub => BinOp::Sub,
                            ArithOp::Mul => BinOp::Mul,
                            ArithOp::Div => BinOp::Div,
                            ArithOp::Neg => unreachable!(),
                        },
                        Box::new(Expr::Var("left".to_string(), span)),
                        Box::new(Expr::Var("right".to_string(), span)),
                        span,
                    ),
                    &helper_env,
                    ctx,
                );
            }
            if name == "neg" {
                let [inner] = dual_args.as_slice() else {
                    return Err("scalar grad SMT lowering found wrong arity for `neg`".to_string());
                };
                return Ok(ScalarDual {
                    value: arith(ArithOp::Neg, inner.value.clone(), SmtExpr::IntLit(0)),
                    tangent: arith(ArithOp::Neg, inner.tangent.clone(), SmtExpr::IntLit(0)),
                });
            }
            let Some((params, body, effects, ret_ty)) =
                ctx.decls.iter().find_map(|decl| match decl {
                    Decl::FunDef {
                        name: candidate,
                        params,
                        body,
                        effects,
                        ret_ty,
                        ..
                    } if candidate == name => {
                        Some((params, body, effects.as_ref(), ret_ty.as_ref()))
                    }
                    _ => None,
                })
            else {
                return Err(format!(
                    "scalar grad SMT lowering does not support call `{name}`"
                ));
            };
            if effects.is_some_and(|effects| !effects.is_empty()) {
                return Err(format!(
                    "scalar grad SMT lowering does not support effectful helper `{name}`"
                ));
            }
            if ctx.depth >= ctx.max_depth {
                return Err(
                    "scalar grad SMT lowering exceeded the helper inlining depth".to_string(),
                );
            }
            if ctx.call_stack.iter().any(|active| active == name) {
                return Err(format!(
                    "scalar grad SMT lowering does not support recursion through `{name}`"
                ));
            }
            let helper_has_float_result = scalar_float_type(ret_ty) || {
                let float_names = params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<UnordSet<_>>();
                ret_ty.is_none() && inline_scalar_float_result(body, &float_names, ctx.decls)
            };
            if params.len() != dual_args.len()
                || !params.iter().all(scalar_param)
                || !helper_has_float_result
            {
                return Err(format!(
                    "scalar grad SMT lowering requires scalar f32/f64 helper `{name}`"
                ));
            }
            let helper_env = params
                .iter()
                .zip(dual_args)
                .map(|(param, dual)| (param.name.clone(), dual))
                .collect();
            let mut call_stack = ctx.call_stack.clone();
            call_stack.push(name.clone());
            scalar_dual(
                body,
                &helper_env,
                &InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
                    call_stack,
                    contracts: ctx.contracts,
                    grad_diagnostic: ctx.grad_diagnostic,
                },
            )
        }
        // Keep the prover-owned transform inside the compiler's executable
        // scalar-AD intersection. The compiler currently rejects conditional
        // scalar gradients, so proving an independently reconstructed ITE
        // derivative here would certify a program that cannot be built.
        Expr::If(_, _, _, _) => {
            Err("scalar grad SMT lowering does not support conditionals".to_string())
        }
        Expr::Block(bindings, body, _) => {
            let mut env = env.clone();
            for binding in bindings {
                let LetPattern::Var(name, _) = &binding.pattern else {
                    return Err(
                        "scalar grad SMT lowering supports only named scalar block bindings"
                            .to_string(),
                    );
                };
                let value = scalar_dual(&binding.value, &env, ctx)?;
                env.insert(name.clone(), value);
            }
            scalar_dual(body, &env, ctx)
        }
        Expr::Cast(_, precision, _, _) if matches!(precision.as_str(), "f32" | "f64") => Err(
            "scalar grad SMT lowering does not support casts in differentiated bodies".to_string(),
        ),
        Expr::Annotate(inner, _, _) => scalar_dual(inner, env, ctx),
        Expr::Grad(_, _, _) => Err(NESTED_GRAD_SMT_BOUNDARY.to_string()),
        Expr::Vmap(_, _, _) => Err("scalar grad SMT lowering does not support `vmap`".to_string()),
        Expr::Jit(_, _) => Err("scalar grad SMT lowering does not support `jit`".to_string()),
        _ => Err("scalar grad SMT lowering encountered a non-scalar operation".to_string()),
    }
}

pub(super) fn surf_arith(expr: &Expr, ctx: &InlineCtx) -> Option<crate::solver::SmtExpr> {
    use crate::solver::{ArithOp as SA, SmtExpr};
    match expr {
        Expr::Var(name, _) => Some(SmtExpr::Var(name.clone())),
        Expr::Lit(Literal::Float(v) | Literal::TypedFloat(v, _), _) => Some(SmtExpr::RealLit(*v)),
        Expr::Lit(Literal::Int(v) | Literal::TypedInt(v, _), _) => Some(SmtExpr::IntLit(*v)),
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
        Expr::Apply(func, args, _) if matches!(func.as_ref(), Expr::Grad(_, _, _)) => {
            let Expr::Grad(target, wrt, _) = func.as_ref() else {
                unreachable!()
            };
            match scalar_grad_application(target, wrt.as_deref(), args, ctx, |arg| {
                surf_arith(arg, ctx)
            }) {
                Ok(gradient) => Some(gradient),
                Err(reason) => {
                    record_grad_diagnostic(ctx, reason);
                    None
                }
            }
        }
        Expr::Apply(func, args, _) => {
            let name = match func.as_ref() {
                Expr::Var(n, _) => n.clone(),
                _ => return None,
            };
            // chelis#979: intercept the real linked quantile call before the
            // generic scalar argument pass. Its tensor operand is identified
            // from the compiler AST and must not be reconstructed or lowered
            // as a scalar; only q enters SMT.
            if args.len() == 2
                && let Some(contracts) = ctx.contracts
                && contracts.borrow().requires_quantile()
            {
                let q = surf_arith(&args[1], ctx)?;
                if let Some(abs) = contracts
                    .borrow_mut()
                    .abstract_quantile_call(&name, &args[0], q)
                {
                    return Some(abs);
                }
            }
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
                let subst: chelis_unord::UnordMap<String, SmtExpr> = params
                    .iter()
                    .zip(smt_args.iter())
                    .map(|(p, a)| (p.name.clone(), a.clone()))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
                    contracts: ctx.contracts,
                    grad_diagnostic: ctx.grad_diagnostic,
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
        Expr::Cast(inner, precision, _, _) if matches!(precision.as_str(), "f32" | "f64") => {
            surf_arith(inner, ctx)
        }
        Expr::Annotate(inner, _, _) => surf_arith(inner, ctx),
        _ => None,
    }
}

fn surf_arith_subst(
    expr: &Expr,
    subst: &chelis_unord::UnordMap<String, SmtExpr>,
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
        Expr::Lit(Literal::Float(v) | Literal::TypedFloat(v, _), _) => Some(SmtExpr::RealLit(*v)),
        Expr::Lit(Literal::Int(v) | Literal::TypedInt(v, _), _) => Some(SmtExpr::IntLit(*v)),
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
        Expr::Apply(func, args, _) if matches!(func.as_ref(), Expr::Grad(_, _, _)) => {
            let Expr::Grad(target, wrt, _) = func.as_ref() else {
                unreachable!()
            };
            match scalar_grad_application(target, wrt.as_deref(), args, ctx, |arg| {
                surf_arith_subst(arg, subst, ctx)
            }) {
                Ok(gradient) => Some(gradient),
                Err(reason) => {
                    record_grad_diagnostic(ctx, reason);
                    None
                }
            }
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
                let inner_subst: chelis_unord::UnordMap<String, SmtExpr> = params
                    .iter()
                    .zip(smt_args.iter())
                    .map(|(p, a)| (p.name.clone(), a.clone()))
                    .collect();
                let deeper = InlineCtx {
                    decls: ctx.decls,
                    depth: ctx.depth + 1,
                    max_depth: ctx.max_depth,
                    contracts: ctx.contracts,
                    grad_diagnostic: ctx.grad_diagnostic,
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
        Expr::Cast(inner, precision, _, _) if matches!(precision.as_str(), "f32" | "f64") => {
            surf_arith_subst(inner, subst, ctx)
        }
        Expr::Annotate(inner, _, _) => surf_arith_subst(inner, subst, ctx),
        _ => None,
    }
}

fn surf_expr_to_smt_subst(
    expr: &Expr,
    subst: &chelis_unord::UnordMap<String, SmtExpr>,
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
            grad_diagnostic: None,
        }
    }

    #[test]
    fn deep_param_name_rejects_unknown_form_but_reads_a_structural_param() {
        let span = sp();
        let structural = DeepExpr::BareList(
            vec![
                DeepExpr::Atom(DeepAtom::Name("x".into()), span),
                DeepExpr::Map(chelis_deep::Metadata::default(), span),
            ],
            span,
        );
        let unknown = DeepExpr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "x".into(),
            meta: chelis_deep::Metadata::default(),
            children: Vec::new(),
            span,
        }));
        let malformed = DeepExpr::BareList(
            vec![
                DeepExpr::Atom(DeepAtom::Name("x".into()), span),
                DeepExpr::Atom(DeepAtom::Int(0), span),
            ],
            span,
        );

        assert_eq!(deep_param_name(&structural), Some("x"));
        assert_eq!(deep_param_name(&unknown), None);
        assert_eq!(deep_param_name(&malformed), Some("x"));
    }

    #[test]
    fn quantile_monotonicity_implication_carries_validated_unit_interval_domain() {
        let low = SmtExpr::Var("p".into());
        let high = SmtExpr::Var("q".into());
        assert_eq!(
            quantile_order_domain(&low, &high),
            SmtExpr::Bool(
                crate::solver::BoolOp::And,
                vec![
                    SmtExpr::Cmp(
                        CmpOp::Le,
                        Box::new(SmtExpr::RealLit(0.0)),
                        Box::new(low.clone()),
                    ),
                    SmtExpr::Cmp(CmpOp::Le, Box::new(low), Box::new(high.clone())),
                    SmtExpr::Cmp(CmpOp::Le, Box::new(high), Box::new(SmtExpr::RealLit(1.0)),),
                ],
            )
        );
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
            type_binders: Vec::new(),
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

    fn parsed_grad_error(source: &str, target: &str, wrt: &[&str]) -> String {
        let decls = chelis_surf::parser::parse_str(source).expect("parse boundary fixture");
        let wrt = wrt
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        scalar_grad_application(
            &var(target),
            Some(&wrt),
            &[var("argument")],
            &ctx(&decls),
            |_| Some(SmtExpr::Var("argument".into())),
        )
        .expect_err("boundary must fail closed")
    }

    #[test]
    fn scalar_grad_multi_wrt_reports_specific_boundary() {
        let error = parsed_grad_error("def f(x: f32) -> f32 = x * x\n", "f", &["x", "also_x"]);
        assert_eq!(
            error,
            "scalar grad SMT lowering requires exactly one explicit `wrt` parameter"
        );
    }

    #[test]
    fn scalar_grad_tensor_and_adt_params_report_specific_boundary() {
        for (source, target) in [
            (
                "def tensor_loss(x: tensor[2, f32]) -> f32 = sum(x, 0)\n",
                "tensor_loss",
            ),
            (
                "type Box =\n\
                 \x20 | Box { value: f32 }\n\
                 def record_loss(x: Box) -> f32 = 0.0\n",
                "record_loss",
            ),
        ] {
            assert_eq!(
                parsed_grad_error(source, target, &["x"]),
                "scalar grad SMT lowering supports only f32/f64 parameters",
                "{target}"
            );
        }
    }

    #[test]
    fn scalar_grad_effectful_target_reports_specific_boundary() {
        // chelis#2413 retired the `Random` effect; `Resource` is the effect
        // that remains, and the boundary is about any declared effect.
        assert_eq!(
            parsed_grad_error(
                "def device_loss(x: f32) -> f32 ! { Resource(\"gpu:0\") } = x * x\n",
                "device_loss",
                &["x"],
            ),
            "scalar grad SMT lowering does not support effectful function `device_loss`"
        );
    }

    #[test]
    fn scalar_grad_recursive_target_reports_specific_boundary() {
        assert_eq!(
            parsed_grad_error(
                "def recursive_loss(x: f32) -> f32 = x * recursive_loss(x)\n",
                "recursive_loss",
                &["x"],
            ),
            "scalar grad SMT lowering does not support recursion through `recursive_loss`"
        );
    }

    #[test]
    fn scalar_grad_named_non_float_result_fails_closed() {
        assert_eq!(
            parsed_grad_error(
                "def integer_value(x: f32) -> i32 = 1\n",
                "integer_value",
                &["x"],
            ),
            "scalar grad SMT lowering requires an f32/f64 result"
        );
    }

    #[test]
    fn scalar_grad_inline_non_float_result_fails_closed() {
        let parameter = Param {
            name: "x".to_string(),
            ty: Some(TypeExpr::Named("f32".to_string(), sp())),
            span: sp(),
        };
        let target = Expr::Lambda(
            vec![parameter],
            Box::new(Expr::Lit(Literal::Int(1), sp())),
            sp(),
        );
        let error = scalar_grad_application(
            &target,
            Some(&["x".to_string()]),
            &[var("argument")],
            &ctx(&[]),
            |_| Some(SmtExpr::Var("argument".into())),
        )
        .expect_err("integer-result lambda must fail closed");
        assert_eq!(error, "scalar grad SMT lowering requires an f32/f64 result");
    }
}
