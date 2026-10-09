//! Proof-only scalar graph extraction. This never changes host execution or
//! asks the ordinary tensor-entry lowerer to reinterpret a scalar function.

use super::*;
use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireDag, WireDagNode, WireRiscOp, WireTensorType,
};
use chelis_surf::ast::{LiteralSuffix, UnaryOp};
use chelis_types::{dtype_semantics::scalar_from_f64, types::Prim};

pub(super) fn lower(
    decls: &[Decl],
    property: &Property,
    expression: &Expr,
) -> Result<(WireDag, u64), String> {
    if property.params.is_empty() {
        return Err("Beacon scalar goal requires a named input".into());
    }
    let mut graph = ScalarGraph {
        decls,
        dag: WireDag {
            schema_version: WIRE_DAG_SCHEMA_VERSION,
            declarations: vec!["beacon_goal_output".into()],
            nodes: Vec::new(),
            roots: Vec::new(),
        },
        active_calls: Vec::new(),
    };
    let mut bindings = BTreeMap::new();
    for param in &property.params {
        if !matches!(&param.ty, Some(TypeExpr::Named(name, _)) if name == "f64")
            || bindings.contains_key(&param.name)
        {
            return Err("Beacon scalar goal requires distinct named f64 inputs".into());
        }
        let id = graph.node(
            0,
            WireRiscOp::Load {
                name: param.name.clone(),
            },
            vec![],
        );
        bindings.insert(param.name.clone(), id);
    }
    let root = graph.expression(expression, &bindings, 0)?;
    graph.dag.roots.push(root);
    graph
        .dag
        .validate_wire_contract()
        .map_err(|error| format!("Beacon scalar graph contract rejected: {error}"))?;
    Ok((graph.dag, root))
}

struct ScalarGraph<'a> {
    decls: &'a [Decl],
    dag: WireDag,
    active_calls: Vec<String>,
}

impl ScalarGraph<'_> {
    fn node(&mut self, declaration: u64, op: WireRiscOp, inputs: Vec<u64>) -> u64 {
        let id = self.dag.nodes.len() as u64;
        self.dag.nodes.push(WireDagNode {
            shape_deps: Vec::new(),
            span_id: None,
            merged_spans: Vec::new(),
            declaration,
            activation: None,
            id,
            op,
            inputs,
            output_type: WireTensorType {
                dims: Vec::new(),
                precision: "f64".into(),
            },
        });
        id
    }

    fn declaration(&mut self, name: &str) -> u64 {
        if let Some(index) = self
            .dag
            .declarations
            .iter()
            .position(|existing| existing == name)
        {
            index as u64
        } else {
            let index = self.dag.declarations.len() as u64;
            self.dag.declarations.push(name.into());
            index
        }
    }

    fn expression(
        &mut self,
        expr: &Expr,
        bindings: &BTreeMap<String, u64>,
        declaration: u64,
    ) -> Result<u64, String> {
        match expr {
            Expr::Var(name, _) => bindings
                .get(name)
                .copied()
                .ok_or_else(|| format!("Beacon scalar graph has unsupported free value `{name}`")),
            Expr::Lit(Literal::TypedFloat(value, LiteralSuffix::F64), _) if value.is_finite() => {
                let value = scalar_from_f64("Beacon scalar constant", Prim::F64, *value)
                    .map_err(|error| error.to_string())?;
                Ok(self.node(declaration, WireRiscOp::Const { value }, vec![]))
            }
            Expr::Unary(UnaryOp::Neg, value, _) => {
                let value = self.expression(value, bindings, declaration)?;
                Ok(self.node(declaration, WireRiscOp::Neg, vec![value]))
            }
            Expr::Binary(
                op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div),
                left,
                right,
                _,
            ) => {
                let left = self.expression(left, bindings, declaration)?;
                let right = self.expression(right, bindings, declaration)?;
                match op {
                    BinOp::Add => Ok(self.node(declaration, WireRiscOp::Add, vec![left, right])),
                    // Beacon's scalar schema 27 cone admits Add and Neg but
                    // rejects Sub. These are equivalent in its real model;
                    // the proof remains explicitly qualified as real arithmetic.
                    BinOp::Sub => {
                        let negated = self.node(declaration, WireRiscOp::Neg, vec![right]);
                        Ok(self.node(declaration, WireRiscOp::Add, vec![left, negated]))
                    }
                    BinOp::Mul => Ok(self.node(declaration, WireRiscOp::Mul, vec![left, right])),
                    BinOp::Div => Ok(self.node(declaration, WireRiscOp::Div, vec![left, right])),
                    _ => unreachable!(),
                }
            }
            Expr::Apply(function, args, _) => {
                let Expr::Var(name, _) = function.as_ref() else {
                    return Err("Beacon scalar graph requires a resolved direct call".into());
                };
                if bindings.contains_key(name) {
                    return Err("Beacon scalar graph does not call a shadowed value".into());
                }
                let candidates = self
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
                        "Beacon scalar graph cannot resolve unique function `{name}`"
                    ));
                };
                if !type_binders.is_empty()
                    || effects.as_ref().is_some_and(|effects| !effects.is_empty())
                    || !matches!(ret_ty, Some(TypeExpr::Named(dtype, _)) if dtype == "f64")
                    || params.len() != args.len()
                    || params.iter().any(|param| !matches!(&param.ty, Some(TypeExpr::Named(dtype, _)) if dtype == "f64"))
                {
                    return Err(format!("Beacon scalar graph requires a pure f64 signature for `{name}`"));
                }
                if self.active_calls.contains(name) {
                    return Err(format!(
                        "Beacon scalar graph does not unfold recursion in `{name}`"
                    ));
                }
                let mut local = BTreeMap::new();
                for (param, arg) in params.iter().zip(args) {
                    let value = self.expression(arg, bindings, declaration)?;
                    if local.insert(param.name.clone(), value).is_some() {
                        return Err(format!(
                            "Beacon scalar graph has duplicate parameter in `{name}`"
                        ));
                    }
                }
                let row = self.declaration(name);
                self.active_calls.push(name.clone());
                let result = self.expression(body, &local, row);
                self.active_calls.pop();
                result.map(|value| self.node(row, WireRiscOp::Copy, vec![value]))
            }
            _ => Err("Beacon scalar graph has an unsupported expression".into()),
        }
    }
}
