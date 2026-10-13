//! Checked scalar `grad(f, wrt=x)(...)` through the compiler's IR AD pass.
//! The forward graph comes from the same linked Surf declarations as an
//! ordinary scalar Beacon call. Both translations are deliberately closed.

use super::*;
use chelis_ir::dag::{ComparisonKind, Dag, NodeId, RiscOp, TensorType};
use chelis_ir::grad::grad_dag_checked;
use chelis_types::dtype_semantics::scalar_from_f64;

pub(super) fn lower(
    outer: &mut ScalarGraph<'_>,
    target: &Expr,
    wrt: &Option<Vec<String>>,
    args: &[Expr],
    bindings: &BTreeMap<String, u64>,
    declaration: u64,
) -> Result<u64, String> {
    let Expr::Var(name, _) = target else {
        return Err("Beacon scalar AD requires a resolved direct function".into());
    };
    if bindings.contains_key(name) {
        return Err("Beacon scalar AD does not differentiate a shadowed value".into());
    }
    let [wrt_name] = wrt.as_deref().unwrap_or(&[]) else {
        return Err("Beacon scalar AD requires one explicit wrt parameter".into());
    };
    let candidates = outer
        .decls
        .iter()
        .filter(|decl| matches!(decl, Decl::FunDef { name: defined, .. } if defined == name))
        .collect::<Vec<_>>();
    let [
        Decl::FunDef {
            type_binders,
            params,
            ret_ty,
            effects,
            body,
            ..
        },
    ] = candidates.as_slice()
    else {
        return Err(format!(
            "Beacon scalar AD cannot resolve unique function `{name}`"
        ));
    };
    if !type_binders.is_empty()
        || effects.as_ref().is_some_and(|effects| !effects.is_empty())
        || !matches!(ret_ty, Some(TypeExpr::Named(dtype, _)) if dtype == outer.source_dtype.name())
        || params.len() != args.len()
        || params.iter().any(|param| !matches!(&param.ty, Some(TypeExpr::Named(dtype, _)) if dtype == outer.source_dtype.name()))
    {
        return Err(format!("Beacon scalar AD requires a pure same-dtype f32 or f64 signature for `{name}`"));
    }
    let wrt_index = params
        .iter()
        .position(|param| &param.name == wrt_name)
        .ok_or_else(|| format!("Beacon scalar AD has no parameter `{wrt_name}` in `{name}`"))?;
    if params
        .iter()
        .filter(|param| &param.name == wrt_name)
        .count()
        != 1
    {
        return Err(format!(
            "Beacon scalar AD has duplicate parameter `{wrt_name}` in `{name}`"
        ));
    }

    // Differentiation happens before actual arguments are substituted. In
    // particular, `vega(s, ..., 0.2f32, ...)` differentiates the formal
    // sigma slot even though its caller supplies a constant.
    let mut forward = ScalarGraph {
        decls: outer.decls,
        source_dtype: outer.source_dtype,
        dag: WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            declarations: Vec::new(),
            nodes: Vec::new(),
            roots: Vec::new(),
        },
        active_calls: vec![name.clone()],
    };
    let row = forward.declaration(name);
    let mut formals = BTreeMap::new();
    let mut formal_ids = Vec::new();
    for param in params.iter() {
        let id = forward.node(
            row,
            WireRiscOp::Load {
                name: param.name.clone(),
            },
            vec![],
        );
        if formals.insert(param.name.clone(), id).is_some() {
            return Err(format!(
                "Beacon scalar AD has duplicate parameter in `{name}`"
            ));
        }
        formal_ids.push(id);
    }
    let output = forward.expression(body, &formals, row)?;
    forward.dag.roots.push(output);
    forward
        .dag
        .validate_wire_contract()
        .map_err(|error| format!("Beacon scalar AD forward graph rejected: {error}"))?;
    let ir = checked_ir_forward(&forward.dag, outer.source_dtype)?;
    let result = grad_dag_checked(
        &ir,
        NodeId(output as usize),
        &[NodeId(formal_ids[wrt_index] as usize)],
    )
    .map_err(|error| format!("Beacon scalar AD rejected `{name}`: {error}"))?;
    let gradient = result
        .grad_nodes
        .get(&NodeId(formal_ids[wrt_index] as usize))
        .copied()
        .ok_or_else(|| format!("Beacon scalar AD returned no gradient for `{wrt_name}`"))?;

    let actuals = args
        .iter()
        .map(|arg| outer.expression(arg, bindings, declaration))
        .collect::<Result<Vec<_>, _>>()?;
    splice_ad_graph(outer, &result.dag, &formal_ids, &actuals, gradient)
}

fn checked_ir_forward(wire: &WireDag, source_dtype: Prim) -> Result<Dag, String> {
    let mut dag = Dag::new();
    let declarations = wire
        .declarations
        .iter()
        .map(|name| dag.declare(name.clone()))
        .collect::<Vec<_>>();
    for node in &wire.nodes {
        if node.id != dag.nodes().len() as u64
            || !node.shape_deps.is_empty()
            || node.activation.is_some()
        {
            return Err("Beacon scalar AD forward graph has non-scalar dependencies".into());
        }
        let op = match &node.op {
            WireRiscOp::Load { name } => RiscOp::Load {
                name: name.clone().into(),
            },
            WireRiscOp::Const { value } if value.prim() == Prim::F64 => RiscOp::Const {
                value: scalar_from_f64(
                    "Beacon scalar AD source constant",
                    source_dtype,
                    value.as_f64_lossy(),
                )
                .map_err(|error| error.to_string())?,
            },
            WireRiscOp::Copy => RiscOp::Copy,
            WireRiscOp::Add => RiscOp::Add,
            WireRiscOp::Sub => RiscOp::Sub,
            WireRiscOp::Mul => RiscOp::Mul,
            WireRiscOp::Div => RiscOp::Div,
            WireRiscOp::Neg => RiscOp::Neg,
            WireRiscOp::Exp => RiscOp::Exp,
            WireRiscOp::Log => RiscOp::Log,
            WireRiscOp::Sqrt => RiscOp::Sqrt,
            WireRiscOp::Erfc => RiscOp::Erfc,
            WireRiscOp::Abs => RiscOp::Abs,
            WireRiscOp::Where {} => RiscOp::Where,
            WireRiscOp::Compare { comparison } => RiscOp::Compare(ir_comparison(*comparison)),
            other => {
                return Err(format!(
                    "Beacon scalar AD cannot translate forward op `{other:?}`"
                ));
            }
        };
        let precision = match node.output_type.precision.as_str() {
            "f64" => source_dtype,
            "bool" => Prim::Bool,
            other => {
                return Err(format!(
                    "Beacon scalar AD forward graph has unsupported dtype `{other}`"
                ));
            }
        };
        if !node.output_type.dims.is_empty() {
            return Err("Beacon scalar AD requires rank-zero forward nodes".into());
        }
        let inputs = node.inputs.iter().map(|id| NodeId(*id as usize)).collect();
        let declared = *declarations
            .get(node.declaration as usize)
            .ok_or("Beacon scalar AD forward graph has an unknown declaration")?;
        dag.add_node(
            declared,
            op,
            inputs,
            TensorType {
                dims: vec![],
                precision,
            },
            None,
        );
    }
    for root in &wire.roots {
        dag.add_root(NodeId(*root as usize));
    }
    Ok(dag)
}

fn ir_comparison(kind: WireComparisonKind) -> ComparisonKind {
    match kind {
        WireComparisonKind::CmpLt => ComparisonKind::CmpLt,
        WireComparisonKind::Lt => ComparisonKind::Lt,
        WireComparisonKind::Eq => ComparisonKind::Eq,
        WireComparisonKind::Neq => ComparisonKind::Neq,
        WireComparisonKind::Gt => ComparisonKind::Gt,
        WireComparisonKind::Gte => ComparisonKind::Gte,
        WireComparisonKind::Lte => ComparisonKind::Lte,
    }
}

fn wire_comparison(kind: ComparisonKind) -> WireComparisonKind {
    match kind {
        ComparisonKind::CmpLt => WireComparisonKind::CmpLt,
        ComparisonKind::Lt => WireComparisonKind::Lt,
        ComparisonKind::Eq => WireComparisonKind::Eq,
        ComparisonKind::Neq => WireComparisonKind::Neq,
        ComparisonKind::Gt => WireComparisonKind::Gt,
        ComparisonKind::Gte => WireComparisonKind::Gte,
        ComparisonKind::Lte => WireComparisonKind::Lte,
    }
}

fn splice_ad_graph(
    outer: &mut ScalarGraph<'_>,
    dag: &Dag,
    formal_ids: &[u64],
    actuals: &[u64],
    gradient: NodeId,
) -> Result<u64, String> {
    let formal_map = formal_ids
        .iter()
        .copied()
        .zip(actuals.iter().copied())
        .collect::<BTreeMap<_, _>>();
    let mut mapped = Vec::with_capacity(dag.nodes().len());
    for node in dag.nodes() {
        if let Some(actual) = formal_map.get(&(node.id.0 as u64)) {
            if !matches!(node.op, RiscOp::Load { .. }) {
                return Err("Beacon scalar AD formal no longer names a load".into());
            }
            mapped.push(*actual);
            continue;
        }
        if !node.output_type.dims.is_empty()
            || !node.shape_deps.is_empty()
            || !node.result_claim_deps.is_empty()
            || node.owner.activation.is_some()
        {
            return Err("Beacon scalar AD emitted a non-scalar or activated node".into());
        }
        let declaration = outer.declaration(&dag.declaration(node.owner.decl).name);
        let inputs = node
            .inputs
            .iter()
            .map(|id| {
                mapped
                    .get(id.0)
                    .copied()
                    .ok_or("Beacon scalar AD node has an unbound input")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let id = match &node.op {
            RiscOp::Const { value } if value.prim() == outer.source_dtype => {
                outer.constant(declaration, value.as_f64_lossy(), outer.source_dtype)?
            }
            RiscOp::Copy => outer.node(declaration, WireRiscOp::Copy, inputs),
            RiscOp::Add => outer.node(declaration, WireRiscOp::Add, inputs),
            RiscOp::Sub => outer.node(declaration, WireRiscOp::Sub, inputs),
            RiscOp::Mul => outer.node(declaration, WireRiscOp::Mul, inputs),
            RiscOp::Div => outer.node(declaration, WireRiscOp::Div, inputs),
            RiscOp::Neg => outer.node(declaration, WireRiscOp::Neg, inputs),
            RiscOp::Exp => outer.node(declaration, WireRiscOp::Exp, inputs),
            RiscOp::Log => outer.node(declaration, WireRiscOp::Log, inputs),
            RiscOp::Sqrt => outer.node(declaration, WireRiscOp::Sqrt, inputs),
            RiscOp::Erfc => outer.node(declaration, WireRiscOp::Erfc, inputs),
            RiscOp::Abs => outer.node(declaration, WireRiscOp::Abs, inputs),
            RiscOp::Where => outer.node(declaration, WireRiscOp::Where {}, inputs),
            RiscOp::Compare(kind) => outer.typed_node(
                declaration,
                WireRiscOp::Compare {
                    comparison: wire_comparison(*kind),
                },
                inputs,
                "bool",
            ),
            RiscOp::Cast { new_precision }
                if *new_precision == outer.source_dtype
                    && node.inputs.len() == 1
                    && dag
                        .get(node.inputs[0])
                        .is_some_and(|operand| operand.output_type.precision == Prim::Bool) =>
            {
                // The IR abs adjoint casts ordered comparison results to the
                // float dtype. Its exact-real image is the selected 0 or 1.
                let one = outer.constant(declaration, 1.0, outer.source_dtype)?;
                let zero = outer.constant(declaration, 0.0, outer.source_dtype)?;
                outer.node(
                    declaration,
                    WireRiscOp::Where {},
                    vec![inputs[0], one, zero],
                )
            }
            other => {
                return Err(format!(
                    "Beacon scalar AD cannot translate adjoint op `{other:?}`"
                ));
            }
        };
        mapped.push(id);
    }
    mapped
        .get(gradient.0)
        .copied()
        .ok_or("Beacon scalar AD gradient root is missing".into())
}
