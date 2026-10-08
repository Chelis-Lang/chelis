//! Opt-in ideal-real model of a source-bound Clarabel QP call.

use std::cell::RefCell;
use std::collections::BTreeMap;

use chelis_reef::PreparedReefGraph;
use chelis_surf::ast::{
    BinOp, Decl, Expr, LetPattern, Literal, LiteralSuffix, Pattern, TypeExpr, UnaryOp,
};
use num_rational::BigRational;
use sha2::{Digest, Sha256};

use crate::composition::{
    AssumptionDischarge, AssumptionRecord, DischargeMethod, DischargeTier, NonVacuityStatus,
};
use crate::smt_names::NameSupply;
use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::{SmtProperty, TierBResult};
use crate::{discharge::Goal, engine_registry::DischargeRegistry};

use super::{Property, PropertyOutcome, PropertyRunOptions, PropertyStatus, PropertyTier};

const CONTRACT: &str = "clarabel.qp.ideal_optimality";
const PSD_CONTRACT: &str = "clarabel.qp.assume_psd";
const SOLVED_CTOR: &str = "Pkg__chelis__clarabel__Clarabel__Qp__Solved";
const ZERO_CTOR: &str = "Pkg__chelis__clarabel__Clarabel__Qp__ZeroCone";
const NONNEGATIVE_CTOR: &str = "Pkg__chelis__clarabel__Clarabel__Qp__NonnegativeCone";
const SETTINGS_CTOR: &str = "Pkg__chelis__clarabel__Clarabel__Qp__Settings";

struct Qp {
    p: Vec<Vec<SmtExpr>>,
    q: Vec<SmtExpr>,
    a: Vec<Vec<SmtExpr>>,
    b: Vec<SmtExpr>,
    cones: Vec<Cone>,
    psd: PsdEvidence,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PsdEvidence {
    Literal,
    Gram,
    Assumed,
}

enum Cone {
    Zero(usize),
    Nonnegative(usize),
}

fn unsupported(reason: impl Into<String>) -> String {
    format!("{CONTRACT}: {}", reason.into())
}

fn app<'a>(expr: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    match expr {
        Expr::Apply(callee, args, _) if matches!(callee.as_ref(), Expr::Var(found, _) if found == name) => {
            Some(args)
        }
        _ => None,
    }
}

fn float_literal(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Lit(Literal::TypedFloat(value, LiteralSuffix::F64), _) if value.is_finite() => {
            Some(*value)
        }
        Expr::Unary(UnaryOp::Neg, value, _) => float_literal(value).map(|value| -value),
        _ => None,
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(Literal::Int(value) | Literal::TypedInt(value, _), _) => Some(*value),
        _ => None,
    }
}

fn fixed_f64_vector_extent(ty: &TypeExpr) -> Option<usize> {
    let TypeExpr::Tensor(dims, precision, _) = ty else {
        return None;
    };
    if precision.as_str() != "f64" || dims.len() != 1 {
        return None;
    }
    match dims.first()? {
        TypeExpr::Named(extent, _) => extent.parse::<usize>().ok(),
        TypeExpr::DimensionLiteral(extent, _) => extent.to_string().parse::<usize>().ok(),
        _ => None,
    }
}

fn fixed_f64_matrix_shape(ty: &TypeExpr) -> Option<(usize, usize)> {
    let TypeExpr::Tensor(dims, precision, _) = ty else {
        return None;
    };
    if precision.as_str() != "f64" || dims.len() != 2 {
        return None;
    }
    let dimension = |dim: &TypeExpr| match dim {
        TypeExpr::Named(extent, _) => extent.parse::<usize>().ok(),
        TypeExpr::DimensionLiteral(extent, _) => extent.to_string().parse::<usize>().ok(),
        _ => None,
    };
    Some((dimension(&dims[0])?, dimension(&dims[1])?))
}

fn empty_list_alias(expr: &Expr, decls: &[Decl]) -> bool {
    match expr {
        Expr::List(items, _) => items.is_empty(),
        Expr::Var(name, _) => decls.iter().any(|decl| {
            matches!(decl, Decl::LetDef { name: found, value: Expr::List(items, _), .. } if found == name && items.is_empty())
        }),
        _ => false,
    }
}

fn dense_vector(expr: &Expr, decls: &[Decl]) -> Option<Vec<f64>> {
    let [arg] = app(expr, "to_tensor")? else {
        return None;
    };
    if empty_list_alias(arg, decls) {
        return Some(Vec::new());
    }
    let Expr::List(items, _) = arg else {
        return None;
    };
    items.iter().map(float_literal).collect()
}

fn dense_matrix(expr: &Expr, decls: &[Decl], n: usize) -> Option<Vec<Vec<f64>>> {
    if let Some([tensor, shape]) = app(expr, "reshape")
        && dense_vector(tensor, decls)?.is_empty()
        && let Expr::List(dims, _) = shape
        && dims.len() == 2
        && int_literal(&dims[0]) == Some(0)
        && int_literal(&dims[1]) == i64::try_from(n).ok()
    {
        return Some(Vec::new());
    }
    let [arg] = app(expr, "to_tensor")? else {
        return None;
    };
    let Expr::List(rows, _) = arg else {
        return None;
    };
    rows.iter()
        .map(|row| {
            let Expr::List(values, _) = row else {
                return None;
            };
            (values.len() == n).then(|| values.iter().map(float_literal).collect())?
        })
        .collect()
}

fn parse_cones(expr: &Expr, rows: usize) -> Result<Vec<Cone>, String> {
    let Expr::List(items, _) = expr else {
        return Err(unsupported("cone descriptions must have fixed list shape"));
    };
    let mut cones = Vec::with_capacity(items.len());
    let mut total = 0usize;
    for item in items {
        let Expr::Apply(callee, args, _) = item else {
            return Err(unsupported("cone descriptions must be fixed constructors"));
        };
        let (Expr::Constructor(name, _) | Expr::Var(name, _)) = callee.as_ref() else {
            return Err(unsupported("cone descriptions must be fixed constructors"));
        };
        let [dimension] = args.as_slice() else {
            return Err(unsupported(
                "only fixed zero and nonnegative cones lower to SMT",
            ));
        };
        let count = int_literal(dimension)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| unsupported("cone dimension must be a nonnegative literal"))?;
        if count == 0 {
            return Err(unsupported(
                "zero and nonnegative cone dimensions must be positive",
            ));
        }
        total = total
            .checked_add(count)
            .ok_or_else(|| unsupported("cone dimensions overflow"))?;
        cones.push(match name.as_str() {
            ZERO_CTOR => Cone::Zero(count),
            NONNEGATIVE_CTOR => Cone::Nonnegative(count),
            _ => return Err(unsupported("only zero and nonnegative cones lower to SMT")),
        });
    }
    if total != rows {
        return Err(unsupported("cone dimensions do not match A and b"));
    }
    Ok(cones)
}

fn exact_psd(p: &[Vec<f64>]) -> bool {
    let n = p.len();
    let Some(mut a) = p
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(BigRational::from_float)
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if (0..n).any(|i| (0..n).any(|j| a[i][j] != a[j][i])) {
        return false;
    }
    let zero = BigRational::from_integer(0.into());
    for pivot in 0..n {
        let diagonal = a[pivot][pivot].clone();
        if diagonal < zero {
            return false;
        }
        if diagonal == zero {
            if (pivot + 1..n).any(|column| a[pivot][column] != zero) {
                return false;
            }
            continue;
        }
        for row in pivot + 1..n {
            for column in pivot + 1..n {
                let adjustment =
                    a[row][pivot].clone() * a[pivot][column].clone() / diagonal.clone();
                a[row][column] -= adjustment;
            }
        }
    }
    true
}

fn parse_qp(
    args: &[Expr],
    decls: &[Decl],
    lowering: &Scalarization<'_>,
    assume_psd: bool,
) -> Result<Qp, String> {
    let [p, q, a, b, cones, settings] = args else {
        return Err(unsupported(
            "solve call must carry P, q, A, b, cones, settings",
        ));
    };
    let Expr::Record(name, fields, _) = settings else {
        return Err(unsupported("settings must be a fixed Settings record"));
    };
    if name != SETTINGS_CTOR
        || fields.len() != 4
        || !fields.iter().any(|(name, value)| {
            name == "max_iterations" && int_literal(value).is_some_and(|value| value > 0)
        })
        || [
            "absolute_gap_tolerance",
            "relative_gap_tolerance",
            "feasibility_tolerance",
        ]
        .iter()
        .any(|field| {
            !fields.iter().any(|(name, value)| {
                name == field && float_literal(value).is_some_and(|value| value > 0.0)
            })
        })
    {
        return Err(unsupported(
            "settings must have a positive iteration limit and finite positive tolerances",
        ));
    }
    let q = dense_vector(q, decls)
        .map(|values| values.into_iter().map(real).collect())
        .or_else(|| lowering.vector(q))
        .ok_or_else(|| unsupported("q must be a fixed-shape f64 vector"))?;
    let n = q.len();
    if n == 0 {
        return Err(unsupported("QP dimension n must be positive"));
    }
    let (p, psd) = if let Some(literal) = dense_matrix(p, decls, n) {
        let psd = if assume_psd {
            PsdEvidence::Assumed
        } else if literal.len() == n && exact_psd(&literal) {
            PsdEvidence::Literal
        } else {
            return Err(unsupported(
                "P must have an exact symmetric positive semidefinite certificate",
            ));
        };
        (
            literal
                .into_iter()
                .map(|row| row.into_iter().map(real).collect())
                .collect(),
            psd,
        )
    } else if is_gram(p, lowering) {
        (
            lowering
                .matrix(p)
                .ok_or_else(|| unsupported("Gram P does not lower to a fixed matrix"))?,
            if assume_psd {
                PsdEvidence::Assumed
            } else {
                PsdEvidence::Gram
            },
        )
    } else if assume_psd {
        (
            lowering
                .matrix(p)
                .ok_or_else(|| unsupported("P must lower to a fixed matrix"))?,
            PsdEvidence::Assumed,
        )
    } else {
        return Err(unsupported(
            "P requires a verified literal or Gram positive semidefinite certificate",
        ));
    };
    if p.len() != n || p.iter().any(|row: &Vec<SmtExpr>| row.len() != n) {
        return Err(unsupported("P dimensions do not match q"));
    }
    let b: Vec<SmtExpr> = dense_vector(b, decls)
        .map(|values| values.into_iter().map(real).collect())
        .or_else(|| lowering.vector(b))
        .ok_or_else(|| unsupported("b must be a fixed-shape f64 vector"))?;
    let a: Vec<Vec<SmtExpr>> = dense_matrix(a, decls, n)
        .map(|matrix| {
            matrix
                .into_iter()
                .map(|row| row.into_iter().map(real).collect())
                .collect()
        })
        .or_else(|| lowering.matrix(a))
        .ok_or_else(|| unsupported("A must be a fixed-shape f64 matrix"))?;
    if a.len() != b.len() || a.iter().any(|row| row.len() != n) {
        return Err(unsupported("A and b extents must match q and each other"));
    }
    let cones = parse_cones(cones, b.len())?;
    Ok(Qp {
        p,
        q,
        a,
        b,
        cones,
        psd,
    })
}

fn is_gram(expr: &Expr, lowering: &Scalarization<'_>) -> bool {
    let Some([left, right]) = app(expr, "matmul") else {
        return false;
    };
    let Some([source, first_axis, second_axis]) = app(left, "permute") else {
        return false;
    };
    int_literal(first_axis) == Some(1)
        && int_literal(second_axis) == Some(0)
        && matches!(
            (lowering.matrix(source), lowering.matrix(right)),
            (Some(source), Some(right)) if !source.is_empty() && source == right
        )
}

fn determinant(matrix: &[Vec<SmtExpr>]) -> SmtExpr {
    if matrix.len() == 1 {
        return matrix[0][0].clone();
    }
    sum(matrix[0].iter().enumerate().map(|(column, coefficient)| {
        let minor = matrix[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(index, _)| *index != column)
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let term = arith(ArithOp::Mul, coefficient.clone(), determinant(&minor));
        if column % 2 == 0 {
            term
        } else {
            arith(ArithOp::Neg, term, real(0.0))
        }
    }))
}

fn psd_conditions(p: &[Vec<SmtExpr>]) -> Result<Vec<SmtExpr>, String> {
    let n = p.len();
    if n == 0 || n > 4 || p.iter().any(|row| row.len() != n) {
        return Err(unsupported(
            "PSD principal-minor expansion requires a square matrix of dimension 1..4",
        ));
    }
    let mut conditions = Vec::new();
    for row in 0..n {
        for column in row + 1..n {
            conditions.push(cmp(
                CmpOp::Eq,
                p[row][column].clone(),
                p[column][row].clone(),
            ));
        }
    }
    for mask in 1..(1usize << n) {
        let indices = (0..n)
            .filter(|index| mask & (1 << index) != 0)
            .collect::<Vec<_>>();
        let minor = indices
            .iter()
            .map(|row| {
                indices
                    .iter()
                    .map(|column| p[*row][*column].clone())
                    .collect()
            })
            .collect::<Vec<Vec<SmtExpr>>>();
        conditions.push(cmp(CmpOp::Ge, determinant(&minor), real(0.0)));
    }
    Ok(conditions)
}

fn substitute(expr: &Expr, values: &BTreeMap<String, Expr>) -> Option<Expr> {
    match expr {
        Expr::Var(name, _) => Some(values.get(name).cloned().unwrap_or_else(|| expr.clone())),
        Expr::Lit(..) | Expr::Constructor(..) => Some(expr.clone()),
        Expr::Apply(callee, args, span) => Some(Expr::Apply(
            Box::new(substitute(callee, values)?),
            args.iter()
                .map(|arg| substitute(arg, values))
                .collect::<Option<Vec<_>>>()?,
            *span,
        )),
        Expr::List(items, span) => Some(Expr::List(
            items
                .iter()
                .map(|item| substitute(item, values))
                .collect::<Option<Vec<_>>>()?,
            *span,
        )),
        Expr::Record(name, fields, span) => Some(Expr::Record(
            name.clone(),
            fields
                .iter()
                .map(|(field, value)| Some((field.clone(), substitute(value, values)?)))
                .collect::<Option<Vec<_>>>()?,
            *span,
        )),
        Expr::Unary(op, value, span) => Some(Expr::Unary(
            *op,
            Box::new(substitute(value, values)?),
            *span,
        )),
        Expr::Binary(op, left, right, span) => Some(Expr::Binary(
            *op,
            Box::new(substitute(left, values)?),
            Box::new(substitute(right, values)?),
            *span,
        )),
        _ => None,
    }
}

fn resolved_solve_args(call: &Expr, binding: &str, decls: &[Decl]) -> Result<Vec<Expr>, String> {
    let Expr::Apply(callee, args, _) = call else {
        return Err(unsupported(
            "match scrutinee must call solve or a direct wrapper",
        ));
    };
    let Expr::Var(name, _) = callee.as_ref() else {
        return Err(unsupported("solve call must resolve to a declaration"));
    };
    if name == binding {
        return Ok(args.clone());
    }
    let Some(Decl::FunDef { params, body, .. }) = decls
        .iter()
        .find(|decl| matches!(decl, Decl::FunDef { name: found, .. } if found == name))
    else {
        return Err(unsupported(
            "solve call is not the registered dependency declaration",
        ));
    };
    if params.len() != args.len() {
        return Err(unsupported(
            "wrapper call arity differs from its declaration",
        ));
    }
    let substitutions = params
        .iter()
        .zip(args)
        .map(|(param, actual)| (param.name.clone(), actual.clone()))
        .collect::<BTreeMap<_, _>>();
    let body = substitute(body, &substitutions)
        .ok_or_else(|| unsupported("wrapper body cannot be resolved to one solve call"))?;
    let Expr::Apply(inner_callee, inner_args, _) = body else {
        return Err(unsupported("wrapper must return one direct solve call"));
    };
    if !matches!(inner_callee.as_ref(), Expr::Var(found, _) if found == binding) {
        return Err(unsupported(
            "wrapper does not call the registered dependency declaration",
        ));
    }
    Ok(inner_args)
}

fn solved_property_body<'a>(
    property: &'a Property,
    binding: &str,
    decls: &[Decl],
) -> Result<(Vec<Expr>, &'a str, &'a Expr), String> {
    let Expr::Match(call, arms, _) = &property.body else {
        return Err(unsupported("property must match the solve result"));
    };
    let args = resolved_solve_args(call, binding, decls)?;
    let [solved, stopped] = arms.as_slice() else {
        return Err(unsupported(
            "property needs a Solved arm and a true fallback",
        ));
    };
    let Pattern::Record(ctor, fields, _) = &solved.pattern else {
        return Err(unsupported("first match arm must be Solved"));
    };
    if ctor != SOLVED_CTOR || solved.guard.is_some() {
        return Err(unsupported(
            "first match arm must be an unguarded Solved arm",
        ));
    }
    if fields.iter().any(|(_, pattern)| {
        matches!(pattern, Pattern::Var(name, _) if property.params.iter().any(|param| param.name == *name))
    }) {
        return Err(unsupported("Solved field binder shadows a property parameter"));
    }
    let primal = fields
        .iter()
        .find_map(|(field, pattern)| {
            if field == "primal" {
                if let Pattern::Var(name, _) = pattern {
                    Some(name.as_str())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .ok_or_else(|| unsupported("Solved arm must bind its primal field"))?;
    if !matches!(stopped.pattern, Pattern::Wildcard(_))
        || stopped.guard.is_some()
        || !matches!(stopped.body, Expr::Lit(Literal::Bool(true), _))
    {
        return Err(unsupported("the non-Solved arm must return true"));
    }
    Ok((args, primal, &solved.body))
}

fn real(value: f64) -> SmtExpr {
    SmtExpr::RealLit(value)
}

fn arith(op: ArithOp, left: SmtExpr, right: SmtExpr) -> SmtExpr {
    SmtExpr::Arith(op, Box::new(left), Box::new(right))
}

fn cmp(op: CmpOp, left: SmtExpr, right: SmtExpr) -> SmtExpr {
    SmtExpr::Cmp(op, Box::new(left), Box::new(right))
}

fn conjunction(values: Vec<SmtExpr>) -> SmtExpr {
    match values.len() {
        0 => SmtExpr::BoolLit(true),
        1 => values.into_iter().next().expect("one conjunct"),
        _ => SmtExpr::Bool(BoolOp::And, values),
    }
}

fn sum(values: impl IntoIterator<Item = SmtExpr>) -> SmtExpr {
    values
        .into_iter()
        .fold(real(0.0), |left, right| arith(ArithOp::Add, left, right))
}

fn dot(left: &[SmtExpr], right: &[SmtExpr]) -> SmtExpr {
    sum(left
        .iter()
        .cloned()
        .zip(right.iter().cloned())
        .map(|(a, b)| arith(ArithOp::Mul, a, b)))
}

fn matrix_row(row: &[SmtExpr], vector: &[SmtExpr]) -> SmtExpr {
    dot(row, vector)
}

fn objective(qp: &Qp, vector: &[SmtExpr]) -> SmtExpr {
    let quadratic = sum(qp.p.iter().enumerate().map(|(row, coefficients)| {
        arith(
            ArithOp::Mul,
            vector[row].clone(),
            matrix_row(coefficients, vector),
        )
    }));
    let linear = dot(&qp.q, vector);
    arith(
        ArithOp::Add,
        arith(ArithOp::Mul, real(0.5), quadratic),
        linear,
    )
}

fn feasible(qp: &Qp, vector: &[SmtExpr], strict_nonnegative: bool) -> Vec<SmtExpr> {
    let mut out = Vec::with_capacity(qp.b.len());
    let mut row = 0usize;
    for cone in &qp.cones {
        let (extent, comparison) = match cone {
            Cone::Zero(extent) => (*extent, CmpOp::Eq),
            Cone::Nonnegative(extent) if strict_nonnegative => (*extent, CmpOp::Lt),
            Cone::Nonnegative(extent) => (*extent, CmpOp::Le),
        };
        for _ in 0..extent {
            out.push(cmp(
                comparison,
                matrix_row(&qp.a[row], vector),
                qp.b[row].clone(),
            ));
            row += 1;
        }
    }
    out
}

fn optimizer_axioms(
    qp: &Qp,
    x: &[SmtExpr],
    comparisons: &[Vec<SmtExpr>],
    fresh: &mut NameSupply,
    full: bool,
) -> Vec<SmtExpr> {
    let mut out = feasible(qp, x, false);
    let stationarity = || {
        (0..qp.q.len())
            .map(|row| {
                cmp(
                    CmpOp::Eq,
                    arith(ArithOp::Add, matrix_row(&qp.p[row], x), qp.q[row].clone()),
                    real(0.0),
                )
            })
            .collect::<Vec<_>>()
    };
    if qp.b.is_empty() {
        // For a PSD unconstrained quadratic, stationarity is equivalent to
        // global minimality. This avoids a quantified nonlinear SMT goal.
        out.extend(stationarity());
        for candidate in comparisons {
            out.push(cmp(CmpOp::Le, objective(qp, x), objective(qp, candidate)));
        }
        return out;
    }
    if full {
        let y_names = (0..qp.q.len())
            .map(|index| fresh.fresh(&format!("__clarabel_y_{index}")))
            .collect::<Vec<_>>();
        let y = y_names
            .iter()
            .cloned()
            .map(SmtExpr::Var)
            .collect::<Vec<_>>();
        out.push(SmtExpr::Forall(
            y_names
                .into_iter()
                .map(|name| (name, SmtSort::Real))
                .collect(),
            Box::new(SmtExpr::Bool(
                BoolOp::Implies,
                vec![
                    conjunction(feasible(qp, &y, false)),
                    cmp(CmpOp::Le, objective(qp, x), objective(qp, &y)),
                ],
            )),
        ));
    } else {
        for candidate in comparisons {
            out.push(cmp(CmpOp::Le, objective(qp, x), objective(qp, candidate)));
        }
    }
    if qp
        .cones
        .iter()
        .all(|cone| matches!(cone, Cone::Nonnegative(_)))
    {
        out.push(SmtExpr::Bool(
            BoolOp::Implies,
            vec![
                conjunction(feasible(qp, x, true)),
                conjunction(stationarity()),
            ],
        ));
    }
    out
}

fn symbolic_data(qp: &Qp) -> bool {
    qp.p.iter()
        .flat_map(|row| row.iter())
        .chain(qp.q.iter())
        .chain(qp.a.iter().flat_map(|row| row.iter()))
        .chain(qp.b.iter())
        .any(|value| !matches!(value, SmtExpr::RealLit(_)))
}

struct Scalarization<'a> {
    primal_name: &'a str,
    primal: &'a [SmtExpr],
    scalars: &'a BTreeMap<String, SmtExpr>,
    vectors: &'a BTreeMap<String, Vec<SmtExpr>>,
    matrices: &'a BTreeMap<String, Vec<Vec<SmtExpr>>>,
    decls: &'a [Decl],
    depth: usize,
}

impl Scalarization<'_> {
    fn call_helper<T>(
        &self,
        expr: &Expr,
        lower: impl FnOnce(&Scalarization<'_>, &Expr) -> Option<T>,
    ) -> Option<T> {
        let Expr::Apply(callee, args, _) = expr else {
            return None;
        };
        let Expr::Var(name, _) = callee.as_ref() else {
            return None;
        };
        if self.depth >= 8 {
            return None;
        }
        let Decl::FunDef {
            params,
            effects,
            body,
            ..
        } = self
            .decls
            .iter()
            .find(|decl| matches!(decl, Decl::FunDef { name: found, .. } if found == name))?
        else {
            return None;
        };
        if params.len() != args.len() || !effects.as_ref().is_none_or(Vec::is_empty) {
            return None;
        }
        let mut scalars = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        let mut matrices = BTreeMap::new();
        for (param, arg) in params.iter().zip(args) {
            match &param.ty {
                Some(ty) if fixed_f64_vector_extent(ty).is_some() => {
                    let vector = self.vector(arg)?;
                    if vector.len() != fixed_f64_vector_extent(ty)? {
                        return None;
                    }
                    vectors.insert(param.name.clone(), vector);
                }
                Some(ty) if fixed_f64_matrix_shape(ty).is_some() => {
                    let matrix = self.matrix(arg)?;
                    let shape = fixed_f64_matrix_shape(ty)?;
                    if matrix.len() != shape.0 || matrix.iter().any(|row| row.len() != shape.1) {
                        return None;
                    }
                    matrices.insert(param.name.clone(), matrix);
                }
                Some(TypeExpr::Named(..)) => {
                    scalars.insert(param.name.clone(), self.scalar(arg)?);
                }
                _ => return None,
            }
        }
        lower(
            &Scalarization {
                primal_name: "",
                primal: &[],
                scalars: &scalars,
                vectors: &vectors,
                matrices: &matrices,
                decls: self.decls,
                depth: self.depth + 1,
            },
            body,
        )
    }

    fn vector(&self, expr: &Expr) -> Option<Vec<SmtExpr>> {
        if matches!(expr, Expr::Var(name, _) if name == self.primal_name) {
            return Some(self.primal.to_vec());
        }
        if let Expr::Var(name, _) = expr
            && let Some(vector) = self.vectors.get(name)
        {
            return Some(vector.clone());
        }
        if let Some([arg]) = app(expr, "to_tensor") {
            let Expr::List(items, _) = arg else {
                return None;
            };
            return items.iter().map(|item| self.scalar(item)).collect();
        }
        if let Some([matrix, shape]) = app(expr, "reshape") {
            let Expr::List(dims, _) = shape else {
                return None;
            };
            if let [extent] = dims.as_slice() {
                let rows = self.matrix(matrix)?;
                let flat = rows.into_iter().flatten().collect::<Vec<_>>();
                return (int_literal(extent) == i64::try_from(flat.len()).ok()).then_some(flat);
            }
        }
        for (name, op) in [
            ("add", ArithOp::Add),
            ("sub", ArithOp::Sub),
            ("mul", ArithOp::Mul),
        ] {
            if let Some([left, right]) = app(expr, name) {
                let left = self.vector(left)?;
                let right = self.vector(right)?;
                if left.len() != right.len() {
                    return None;
                }
                return Some(
                    left.into_iter()
                        .zip(right)
                        .map(|(a, b)| arith(op, a, b))
                        .collect(),
                );
            }
        }
        self.call_helper(expr, |nested, body| nested.vector(body))
    }

    fn matrix(&self, expr: &Expr) -> Option<Vec<Vec<SmtExpr>>> {
        if let Expr::Var(name, _) = expr
            && let Some(matrix) = self.matrices.get(name)
        {
            return Some(matrix.clone());
        }
        if let Some([arg]) = app(expr, "to_tensor") {
            let Expr::List(rows, _) = arg else {
                return None;
            };
            let matrix = rows
                .iter()
                .map(|row| {
                    let Expr::List(items, _) = row else {
                        return None;
                    };
                    items.iter().map(|item| self.scalar(item)).collect()
                })
                .collect::<Option<Vec<Vec<_>>>>()?;
            if matrix.windows(2).any(|rows| rows[0].len() != rows[1].len()) {
                return None;
            }
            return Some(matrix);
        }
        if let Some([value, first_axis, second_axis]) = app(expr, "permute") {
            if int_literal(first_axis) != Some(1) || int_literal(second_axis) != Some(0) {
                return None;
            }
            let matrix = self.matrix(value)?;
            let width = matrix.first()?.len();
            if matrix.iter().any(|row| row.len() != width) {
                return None;
            }
            return Some(
                (0..width)
                    .map(|column| matrix.iter().map(|row| row[column].clone()).collect())
                    .collect(),
            );
        }
        if let Some([value, shape]) = app(expr, "reshape") {
            let Expr::List(dims, _) = shape else {
                return None;
            };
            let [rows, columns] = dims.as_slice() else {
                return None;
            };
            let rows = usize::try_from(int_literal(rows)?).ok()?;
            let columns = usize::try_from(int_literal(columns)?).ok()?;
            let values = self.vector(value)?;
            if rows.checked_mul(columns)? != values.len() || columns == 0 {
                return None;
            }
            return Some(values.chunks(columns).map(<[_]>::to_vec).collect());
        }
        if let Some([left, right]) = app(expr, "matmul") {
            let left = self.matrix(left)?;
            let right = self.matrix(right)?;
            let inner = left.first()?.len();
            let columns = right.first()?.len();
            if left.iter().any(|row| row.len() != inner)
                || right.len() != inner
                || right.iter().any(|row| row.len() != columns)
            {
                return None;
            }
            return Some(
                left.iter()
                    .map(|row| {
                        (0..columns)
                            .map(|column| {
                                sum((0..inner).map(|index| {
                                    arith(
                                        ArithOp::Mul,
                                        row[index].clone(),
                                        right[index][column].clone(),
                                    )
                                }))
                            })
                            .collect()
                    })
                    .collect(),
            );
        }
        self.call_helper(expr, |nested, body| nested.matrix(body))
    }

    fn scalar(&self, expr: &Expr) -> Option<SmtExpr> {
        if let Some(value) = float_literal(expr) {
            return Some(real(value));
        }
        match expr {
            Expr::Lit(Literal::Int(value) | Literal::TypedInt(value, _), _) => {
                Some(SmtExpr::IntLit(*value))
            }
            Expr::Var(name, _) => self.scalars.get(name).cloned(),
            Expr::Block(bindings, body, _) => {
                let mut scalars = self.scalars.clone();
                let mut vectors = self.vectors.clone();
                let mut matrices = self.matrices.clone();
                for binding in bindings {
                    let LetPattern::Var(name, _) = &binding.pattern else {
                        return None;
                    };
                    let nested = Scalarization {
                        primal_name: self.primal_name,
                        primal: self.primal,
                        scalars: &scalars,
                        vectors: &vectors,
                        matrices: &matrices,
                        decls: self.decls,
                        depth: self.depth,
                    };
                    if let Some(value) = nested.vector(&binding.value) {
                        vectors.insert(name.clone(), value);
                    } else if let Some(value) = nested.matrix(&binding.value) {
                        matrices.insert(name.clone(), value);
                    } else {
                        scalars.insert(name.clone(), nested.scalar(&binding.value)?);
                    }
                }
                Scalarization {
                    primal_name: self.primal_name,
                    primal: self.primal,
                    scalars: &scalars,
                    vectors: &vectors,
                    matrices: &matrices,
                    decls: self.decls,
                    depth: self.depth,
                }
                .scalar(body)
            }
            Expr::Unary(UnaryOp::Neg, value, _) => {
                Some(arith(ArithOp::Neg, self.scalar(value)?, SmtExpr::IntLit(0)))
            }
            Expr::Binary(op, left, right, _) => {
                let op = match op {
                    BinOp::Add => ArithOp::Add,
                    BinOp::Sub => ArithOp::Sub,
                    BinOp::Mul => ArithOp::Mul,
                    _ => return None,
                };
                Some(arith(op, self.scalar(left)?, self.scalar(right)?))
            }
            _ => {
                for (name, op) in [
                    ("add", ArithOp::Add),
                    ("sub", ArithOp::Sub),
                    ("mul", ArithOp::Mul),
                ] {
                    if let Some([left, right]) = app(expr, name) {
                        return Some(arith(op, self.scalar(left)?, self.scalar(right)?));
                    }
                }
                if let Some([value]) = app(expr, "neg") {
                    return Some(arith(ArithOp::Neg, self.scalar(value)?, SmtExpr::IntLit(0)));
                }
                if let Some([reduction]) = app(expr, "tensor_to_scalar")
                    && let Some([vector, axis]) = app(reduction, "sum")
                    && int_literal(axis) == Some(0)
                {
                    return Some(sum(self.vector(vector)?));
                }
                self.call_helper(expr, |nested, body| nested.scalar(body))
            }
        }
    }

    fn boolean(&self, expr: &Expr) -> Option<SmtExpr> {
        match expr {
            Expr::Lit(Literal::Bool(value), _) => Some(SmtExpr::BoolLit(*value)),
            Expr::Unary(UnaryOp::Not, inner, _) => {
                Some(SmtExpr::Not(Box::new(self.boolean(inner)?)))
            }
            Expr::Binary(BinOp::And | BinOp::Or, left, right, _) => Some(SmtExpr::Bool(
                if matches!(expr, Expr::Binary(BinOp::And, ..)) {
                    BoolOp::And
                } else {
                    BoolOp::Or
                },
                vec![self.boolean(left)?, self.boolean(right)?],
            )),
            Expr::Binary(op, left, right, _) => {
                let op = match op {
                    BinOp::Eq => CmpOp::Eq,
                    BinOp::Ne => CmpOp::Ne,
                    BinOp::Lt => CmpOp::Lt,
                    BinOp::Le => CmpOp::Le,
                    BinOp::Gt => CmpOp::Gt,
                    BinOp::Ge => CmpOp::Ge,
                    _ => return None,
                };
                Some(cmp(op, self.scalar(left)?, self.scalar(right)?))
            }
            Expr::If(condition, then_branch, else_branch, _) => Some(SmtExpr::Ite(
                Box::new(self.boolean(condition)?),
                Box::new(self.boolean(then_branch)?),
                Box::new(self.boolean(else_branch)?),
            )),
            _ => {
                for (name, op) in [
                    ("eq", CmpOp::Eq),
                    ("neq", CmpOp::Ne),
                    ("lt", CmpOp::Lt),
                    ("lte", CmpOp::Le),
                    ("gt", CmpOp::Gt),
                    ("gte", CmpOp::Ge),
                ] {
                    if let Some([left, right]) = app(expr, name) {
                        return Some(cmp(op, self.scalar(left)?, self.scalar(right)?));
                    }
                }
                if let Some([left, right]) = app(expr, "and") {
                    return Some(SmtExpr::Bool(
                        BoolOp::And,
                        vec![self.boolean(left)?, self.boolean(right)?],
                    ));
                }
                if let Some([left, right]) = app(expr, "or") {
                    return Some(SmtExpr::Bool(
                        BoolOp::Or,
                        vec![self.boolean(left)?, self.boolean(right)?],
                    ));
                }
                if let Some([inner]) = app(expr, "not") {
                    return Some(SmtExpr::Not(Box::new(self.boolean(inner)?)));
                }
                None
            }
        }
    }
}

thread_local! {
    static SOLVE_BINDING: RefCell<Option<chelis_compiler_api::RegisteredClarabelProvider>> = const { RefCell::new(None) };
}

/// Run a linked property batch with the exact provider declaration admitted
/// by the Reef package graph. The source hash, package owner and version are
/// checked before any property receives the binding.
pub fn with_registered_clarabel_contract<R>(
    graph: &PreparedReefGraph,
    run: impl FnOnce() -> R,
) -> Result<R, String> {
    struct Restore(Option<chelis_compiler_api::RegisteredClarabelProvider>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SOLVE_BINDING.with(|binding| *binding.borrow_mut() = self.0.take());
        }
    }
    let provider = chelis_compiler_api::registered_clarabel_provider(graph)?;
    let old = SOLVE_BINDING.with(|binding| binding.replace(provider));
    let _restore = Restore(old);
    Ok(run())
}

pub(super) fn prove(
    decls: &[Decl],
    trusted_contract_decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);
    let unsupported_outcome = |reason: String| {
        PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some(reason),
            false,
            Vec::new(),
        )
    };
    let contracts = super::expanded_contracts(property);
    let assume_psd = contracts == [PSD_CONTRACT, CONTRACT];
    if contracts != [CONTRACT] && !assume_psd {
        return unsupported_outcome(unsupported(
            "ideal optimality requires only its own contract, optionally paired with assume_psd",
        ));
    }
    if options.tier != "auto" && options.tier != "smt-only" {
        return unsupported_outcome(unsupported("ideal optimality requires the SMT tier"));
    }
    let Some(binding) = SOLVE_BINDING.with(|binding| binding.borrow().clone()) else {
        return unsupported_outcome(unsupported("registered Clarabel package source is absent"));
    };
    if !trusted_contract_decls
        .iter()
        .any(|decl| matches!(decl, Decl::FunDef { name, .. } if name == &binding.solve_symbol))
    {
        return unsupported_outcome(unsupported(
            "linked solve declaration is not a trusted dependency",
        ));
    }
    let (args, primal_name, solved_body) =
        match solved_property_body(property, &binding.solve_symbol, decls) {
            Ok(value) => value,
            Err(reason) => return unsupported_outcome(reason),
        };
    let deep = match chelis_surf::desugar::desugar_program(decls) {
        Ok(value) => value,
        Err(error) => {
            return unsupported_outcome(unsupported(format!(
                "cannot reserve source names: {error}"
            )));
        }
    };
    let mut names = NameSupply::for_deep_program(&deep);
    let mut scalars = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    let mut matrices = BTreeMap::new();
    let mut variables = Vec::new();
    for param in &property.params {
        match &param.ty {
            Some(TypeExpr::Named(name, _)) if name == "f64" || name == "f32" => {
                let variable = names.fresh(&format!("__clarabel_{}", param.name));
                scalars.insert(param.name.clone(), SmtExpr::Var(variable.clone()));
                variables.push((variable, SmtSort::Real));
            }
            Some(TypeExpr::Named(name, _)) if name == "i64" || name == "i32" => {
                let variable = names.fresh(&format!("__clarabel_{}", param.name));
                scalars.insert(param.name.clone(), SmtExpr::Var(variable.clone()));
                variables.push((variable, SmtSort::Int));
            }
            Some(TypeExpr::Named(name, _)) if name == "bool" => {
                let variable = names.fresh(&format!("__clarabel_{}", param.name));
                scalars.insert(param.name.clone(), SmtExpr::Var(variable.clone()));
                variables.push((variable, SmtSort::Bool));
            }
            Some(ty) if fixed_f64_vector_extent(ty).is_some() => {
                let extent = fixed_f64_vector_extent(ty).expect("guarded fixed extent");
                if extent > 64 || variables.len() + extent > 64 {
                    return unsupported_outcome(unsupported(
                        "symbolic QP scalarization exceeds the 64-scalar resource limit",
                    ));
                }
                let mut components = Vec::with_capacity(extent);
                for index in 0..extent {
                    let variable = names.fresh(&format!("__clarabel_{}_{}", param.name, index));
                    components.push(SmtExpr::Var(variable.clone()));
                    variables.push((variable, SmtSort::Real));
                }
                vectors.insert(param.name.clone(), components);
            }
            Some(ty) if fixed_f64_matrix_shape(ty).is_some() => {
                let (rows, columns) = fixed_f64_matrix_shape(ty).expect("guarded fixed shape");
                let Some(count) = rows.checked_mul(columns) else {
                    return unsupported_outcome(unsupported("symbolic matrix extent overflow"));
                };
                if count > 64 || variables.len() + count > 64 {
                    return unsupported_outcome(unsupported(
                        "symbolic QP scalarization exceeds the 64-scalar resource limit",
                    ));
                }
                let mut matrix = Vec::with_capacity(rows);
                for row in 0..rows {
                    let mut values = Vec::with_capacity(columns);
                    for column in 0..columns {
                        let variable =
                            names.fresh(&format!("__clarabel_{}_{}_{}", param.name, row, column));
                        values.push(SmtExpr::Var(variable.clone()));
                        variables.push((variable, SmtSort::Real));
                    }
                    matrix.push(values);
                }
                matrices.insert(param.name.clone(), matrix);
            }
            _ => {
                return unsupported_outcome(unsupported(
                    "property parameters must be fixed f64 vectors/matrices or scalar numeric/bool values",
                ));
            }
        }
    }
    let argument_lowering = Scalarization {
        primal_name: "",
        primal: &[],
        scalars: &scalars,
        vectors: &vectors,
        matrices: &matrices,
        decls,
        depth: 0,
    };
    let qp = match parse_qp(&args, decls, &argument_lowering, assume_psd) {
        Ok(value) => value,
        Err(reason) => return unsupported_outcome(reason),
    };
    let psd_premise = if assume_psd {
        match psd_conditions(&qp.p) {
            Ok(conditions) => conditions,
            Err(reason) => return unsupported_outcome(reason),
        }
    } else {
        Vec::new()
    };
    let call_fingerprint = {
        let bytes = serde_json::to_vec(&(&binding.solve_symbol, &property.body, &args))
            .expect("checked Surf expression serializes");
        format!("{:x}", Sha256::digest(bytes))
    };
    let x_names = (0..qp.q.len())
        .map(|index| names.fresh(&format!("__clarabel_x_{index}")))
        .collect::<Vec<_>>();
    let x = x_names
        .iter()
        .cloned()
        .map(SmtExpr::Var)
        .collect::<Vec<_>>();
    let lowering = Scalarization {
        primal_name,
        primal: &x,
        scalars: &scalars,
        vectors: &vectors,
        matrices: &matrices,
        decls,
        depth: 0,
    };
    let Some(postcondition) = lowering.boolean(solved_body) else {
        return unsupported_outcome(unsupported(
            "Solved arm does not lower to scalar SMT arithmetic",
        ));
    };
    let mut preconditions = Vec::new();
    for condition in &property.preconditions {
        let Some(lowered) = lowering.boolean(condition) else {
            return unsupported_outcome(unsupported(
                "property precondition does not lower to scalar SMT arithmetic",
            ));
        };
        preconditions.push(lowered);
    }
    let user_preconditions = preconditions.clone();
    preconditions.extend(psd_premise.clone());
    variables.extend(x_names.into_iter().map(|name| (name, SmtSort::Real)));
    let symbolic = symbolic_data(&qp);
    let mut comparisons = Vec::new();
    for candidate in vectors.values().filter(|vector| vector.len() == qp.q.len()) {
        if qp.b.is_empty() {
            comparisons.push(candidate.clone());
            continue;
        }
        if symbolic {
            let feasibility_goal = SmtProperty {
                variables: variables.clone(),
                preconditions: preconditions.clone(),
                postcondition: conjunction(feasible(&qp, candidate, false)),
            };
            if matches!(
                DischargeRegistry::with_builtin_engines()
                    .dispatch(&Goal::smt(feasibility_goal), options.smt_timeout_ms)
                    .into_result(),
                TierBResult::Proved
            ) {
                comparisons.push(candidate.clone());
            }
        }
    }
    preconditions.extend(optimizer_axioms(
        &qp,
        &x,
        &comparisons,
        &mut names,
        !symbolic,
    ));
    let smt_prop = SmtProperty {
        variables: variables.clone(),
        preconditions,
        postcondition,
    };
    let full_axiom_prop = SmtProperty {
        variables,
        preconditions: user_preconditions
            .iter()
            .cloned()
            .chain(psd_premise.clone())
            .chain(optimizer_axioms(&qp, &x, &[], &mut names, true))
            .collect(),
        postcondition: SmtExpr::BoolLit(true),
    };
    let discharge = DischargeRegistry::with_builtin_engines()
        .dispatch(&Goal::smt(smt_prop.clone()), options.smt_timeout_ms);
    let base = Some((discharge.soundness(), discharge.qualifier_set().clone()));
    match discharge.into_result() {
        TierBResult::Proved => {
            let mut non_vacuity =
                super::smt_non_vacuity_record(&full_axiom_prop, options.smt_timeout_ms);
            if symbolic && matches!(non_vacuity.status, NonVacuityStatus::Unsupported) {
                // The unrestricted quantified SAT query can be unknown even
                // when a simple concrete valuation satisfies the full axiom.
                // This is a stronger SAT query, never a replacement for the
                // global-optimality formula.
                let mut witness = full_axiom_prop.clone();
                for (name, sort) in &witness.variables {
                    let variable = SmtExpr::Var(name.clone());
                    witness.preconditions.push(match sort {
                        SmtSort::Real => cmp(CmpOp::Eq, variable, real(0.0)),
                        SmtSort::Int => cmp(CmpOp::Eq, variable, SmtExpr::IntLit(0)),
                        SmtSort::Bool => SmtExpr::Not(Box::new(variable)),
                    });
                }
                let zero_witness = super::smt_non_vacuity_record(&witness, options.smt_timeout_ms);
                if matches!(zero_witness.status, NonVacuityStatus::Established) {
                    non_vacuity = zero_witness;
                    if let Some(evidence) = non_vacuity.evidence.as_object_mut() {
                        evidence.insert("witness".to_string(), serde_json::json!("all_zero"));
                    }
                }
            }
            if !matches!(non_vacuity.status, NonVacuityStatus::Established) {
                return unsupported_outcome(non_vacuity.reason.unwrap_or_else(|| {
                    unsupported("solver assumption non-vacuity could not be established")
                }));
            }
            let axiom = AssumptionRecord::new(
                CONTRACT,
                Some(AssumptionDischarge::new(
                    DischargeMethod::Axiom,
                    serde_json::json!({
                        "status": "asserted",
                        "arith_model": "real",
                        "provider": "chelis-clarabel/0.1.0",
                        "call": binding.solve_symbol,
                        "abi_version": binding.abi_version,
                        "provider_archive_sha256": binding.archive_sha256,
                        "call_fingerprint_sha256": call_fingerprint,
                        "psd_evidence": match qp.psd {
                            PsdEvidence::Literal => "exact_literal",
                            PsdEvidence::Gram => "verified_gram",
                            PsdEvidence::Assumed => "assume_psd",
                        },
                        "justification": "author opted into ideal exact-real optimizer contract",
                    }),
                )),
                Some(non_vacuity.clone()),
            )
            .with_source("native_provider", binding.solve_symbol.clone())
            .with_discharge_tier(DischargeTier::new(
                DischargeMethod::Axiom.engine(),
                DischargeMethod::Axiom,
                Some(binding.solve_symbol.clone()),
            ));
            let mut assumptions = if user_preconditions.is_empty() {
                Vec::new()
            } else {
                let user_goal = SmtProperty {
                    variables: smt_prop.variables.clone(),
                    preconditions: user_preconditions,
                    postcondition: SmtExpr::BoolLit(true),
                };
                let user_non_vacuity =
                    super::smt_non_vacuity_record(&user_goal, options.smt_timeout_ms);
                if !matches!(user_non_vacuity.status, NonVacuityStatus::Established) {
                    return unsupported_outcome(user_non_vacuity.reason.unwrap_or_else(|| {
                        unsupported("property precondition non-vacuity could not be established")
                    }));
                }
                super::property_assumption_records(
                    &property.name,
                    &user_goal,
                    AssumptionDischarge::new(
                        DischargeMethod::Smt,
                        serde_json::json!({
                            "status": "proved", "property": property.name, "arith_model": "real",
                        }),
                    ),
                    user_non_vacuity,
                )
            };
            if assume_psd {
                assumptions.push(
                    AssumptionRecord::new(
                        PSD_CONTRACT,
                        Some(AssumptionDischarge::new(
                            DischargeMethod::Axiom,
                            serde_json::json!({
                                "status": "asserted",
                                "arith_model": "real",
                                "provider": "chelis-clarabel/0.1.0",
                                "call": binding.solve_symbol,
                                "abi_version": binding.abi_version,
                                "provider_archive_sha256": binding.archive_sha256,
                                "call_fingerprint_sha256": call_fingerprint,
                                "condition": "symmetric and all principal minors nonnegative",
                                "justification": "author opted into the call-bound ideal-real PSD premise",
                            }),
                        )),
                        Some(non_vacuity.clone()),
                    )
                    .with_source("native_provider", binding.solve_symbol.clone())
                    .with_discharge_tier(DischargeTier::new(
                        DischargeMethod::Axiom.engine(),
                        DischargeMethod::Axiom,
                        Some(binding.solve_symbol.clone()),
                    )),
                );
            }
            assumptions.push(axiom);
            PropertyOutcome::with_base_discharge(
                property.name.clone(),
                PropertyStatus::Passed,
                PropertyTier::Smt,
                0,
                seed,
                None,
                None,
                false,
                assumptions,
                base,
            )
        }
        TierBResult::Disproved(model) => PropertyOutcome::with_base_discharge(
            property.name.clone(),
            PropertyStatus::Failed,
            PropertyTier::Smt,
            0,
            seed,
            Some(model),
            None,
            false,
            Vec::new(),
            base,
        ),
        TierBResult::Timeout => unsupported_outcome(unsupported("SMT timeout")),
        TierBResult::Unknown => unsupported_outcome(unsupported("SMT unknown")),
        TierBResult::Error(reason) => {
            unsupported_outcome(unsupported(format!("SMT lowering error: {reason}")))
        }
    }
}
