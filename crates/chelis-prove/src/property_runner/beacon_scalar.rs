//! Proof-only scalar graph extraction. This never changes host execution or
//! asks the ordinary tensor-entry lowerer to reinterpret a scalar function.

use super::*;
use chelis_compiler_api::schema::{
    WIRE_DAG_SCHEMA_VERSION, WireComparisonKind, WireDag, WireDagNode, WireRiscOp, WireTensorType,
};
use chelis_numeric_core::activation::{ActivationGraph, DerivedActivation, lower_activation};
use chelis_numeric_core::dtype_semantics::{FloatBinOp, FloatUnOp, scalar_from_f64};
use chelis_numeric_core::prim::Prim;
use chelis_surf::ast::{LetPattern, LiteralSuffix, UnaryOp};

mod ad;

pub(super) fn lower(
    decls: &[Decl],
    property: &Property,
    expression: &Expr,
) -> Result<(WireDag, u64), String> {
    if property.params.is_empty() {
        return Err("Beacon scalar goal requires a named input".into());
    }
    let source_dtype = match &property.params[0].ty {
        Some(TypeExpr::Named(name, _)) if name == "f32" => Prim::F32,
        Some(TypeExpr::Named(name, _)) if name == "f64" => Prim::F64,
        _ => return Err("Beacon scalar goal requires f32 or f64 inputs".into()),
    };
    let mut graph = ScalarGraph {
        decls,
        source_dtype,
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
        if !matches!(&param.ty, Some(TypeExpr::Named(name, _)) if name == source_dtype.name())
            || bindings.contains_key(&param.name)
        {
            return Err("Beacon scalar goal requires distinct same-dtype f32 or f64 inputs".into());
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
    source_dtype: Prim,
    dag: WireDag,
    active_calls: Vec<String>,
}

impl ScalarGraph<'_> {
    fn node(&mut self, declaration: u64, op: WireRiscOp, inputs: Vec<u64>) -> u64 {
        self.typed_node(declaration, op, inputs, "f64")
    }

    fn typed_node(
        &mut self,
        declaration: u64,
        op: WireRiscOp,
        inputs: Vec<u64>,
        precision: &str,
    ) -> u64 {
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
                precision: precision.into(),
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
            Expr::Lit(Literal::TypedFloat(value, suffix), _)
                if value.is_finite()
                    && matches!(
                        (self.source_dtype, suffix),
                        (Prim::F32, LiteralSuffix::F32) | (Prim::F64, LiteralSuffix::F64)
                    ) =>
            {
                // The proof graph is an f64 carrier for exact real arithmetic.
                // Every finite f32 stored value widens to f64 without loss.
                self.constant(declaration, *value, self.source_dtype)
            }
            Expr::Cast(value, dtype, _, _)
                if dtype == self.source_dtype.name()
                    && matches!(value.as_ref(), Expr::Lit(Literal::Float(_), _)) =>
            {
                let Expr::Lit(Literal::Float(value), _) = value.as_ref() else {
                    unreachable!("cast guard requires a bare float literal")
                };
                self.constant(declaration, *value, self.source_dtype)
            }
            Expr::Block(bindings_in_block, body, _) => {
                let mut locals = bindings.clone();
                for binding in bindings_in_block {
                    let LetPattern::Var(name, _) = &binding.pattern else {
                        return Err("Beacon scalar graph requires scalar local bindings".into());
                    };
                    let value = self.expression(&binding.value, &locals, declaration)?;
                    if self.dag.nodes[value as usize].output_type.precision != "f64" {
                        return Err("Beacon scalar local binding must be f64".into());
                    }
                    locals.insert(name.clone(), value);
                }
                self.expression(body, &locals, declaration)
            }
            Expr::If(condition, yes, no, _) => {
                let condition = self.expression(condition, bindings, declaration)?;
                let yes = self.expression(yes, bindings, declaration)?;
                let no = self.expression(no, bindings, declaration)?;
                if self.dag.nodes[condition as usize].output_type.precision != "bool"
                    || self.dag.nodes[yes as usize].output_type.precision != "f64"
                    || self.dag.nodes[no as usize].output_type.precision != "f64"
                {
                    return Err("Beacon scalar conditional requires a bool and f64 arms".into());
                }
                Ok(self.node(declaration, WireRiscOp::Where {}, vec![condition, yes, no]))
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
                if let Expr::Grad(target, wrt, _) = function.as_ref() {
                    return ad::lower(self, target, wrt, args, bindings, declaration);
                }
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
                if candidates.is_empty()
                    && let Some(result) = self.builtin(name, args, bindings, declaration)?
                {
                    return Ok(result);
                }
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
                    || !matches!(ret_ty, Some(TypeExpr::Named(dtype, _)) if dtype == self.source_dtype.name())
                    || params.len() != args.len()
                    || params.iter().any(|param| !matches!(&param.ty, Some(TypeExpr::Named(dtype, _)) if dtype == self.source_dtype.name()))
                {
                    return Err(format!("Beacon scalar graph requires a pure same-dtype f32 or f64 signature for `{name}`"));
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

    fn constant(&mut self, declaration: u64, raw: f64, prim: Prim) -> Result<u64, String> {
        let stored = scalar_from_f64("Beacon source scalar constant", prim, raw)
            .map_err(|error| error.to_string())?;
        let value = scalar_from_f64(
            "Beacon scalar graph constant",
            Prim::F64,
            stored.as_f64_lossy(),
        )
        .map_err(|error| error.to_string())?;
        Ok(self.node(declaration, WireRiscOp::Const { value }, vec![]))
    }

    fn builtin(
        &mut self,
        name: &str,
        args: &[Expr],
        bindings: &BTreeMap<String, u64>,
        declaration: u64,
    ) -> Result<Option<u64>, String> {
        let arity = match name {
            "neg" | "abs" | "log" | "sqrt" | "exp" | "standard_normal_cdf" => 1,
            "add" | "sub" | "mul" | "div" | "lt" => 2,
            _ => return Ok(None),
        };
        if args.len() != arity {
            return Err(format!("Beacon scalar builtin `{name}` has wrong arity"));
        }
        let operands = args
            .iter()
            .map(|arg| self.expression(arg, bindings, declaration))
            .collect::<Result<Vec<_>, _>>()?;
        if operands
            .iter()
            .any(|id| self.dag.nodes[*id as usize].output_type.precision != "f64")
        {
            return Err(format!(
                "Beacon scalar builtin `{name}` requires f64 operands"
            ));
        }
        let op = match name {
            "neg" => WireRiscOp::Neg,
            "abs" => WireRiscOp::Abs,
            "log" => WireRiscOp::Log,
            "sqrt" => WireRiscOp::Sqrt,
            "exp" => WireRiscOp::Exp,
            "add" => WireRiscOp::Add,
            "sub" => WireRiscOp::Sub,
            "mul" => WireRiscOp::Mul,
            "div" => WireRiscOp::Div,
            "lt" => WireRiscOp::Compare {
                comparison: WireComparisonKind::Lt,
            },
            "standard_normal_cdf" => {
                let mut graph = ScalarActivationGraph {
                    graph: self,
                    declaration,
                };
                return lower_activation(
                    &mut graph,
                    DerivedActivation::StandardNormalCdf,
                    operands[0],
                )
                .map(Some);
            }
            _ => unreachable!("builtin arity admitted only known names"),
        };
        Ok(Some(if name == "lt" {
            self.typed_node(declaration, op, operands, "bool")
        } else {
            self.node(declaration, op, operands)
        }))
    }
}

struct ScalarActivationGraph<'a, 'b> {
    graph: &'a mut ScalarGraph<'b>,
    declaration: u64,
}

impl ActivationGraph for ScalarActivationGraph<'_, '_> {
    type Value = u64;
    type Predicate = u64;
    type Error = String;

    fn operand_prim(&self) -> Prim {
        self.graph.source_dtype
    }

    fn constant(&mut self, value: f64, prim: Prim) -> Result<u64, String> {
        self.graph.constant(self.declaration, value, prim)
    }

    fn unary(&mut self, op: FloatUnOp, x: u64) -> Result<u64, String> {
        let wire = match op {
            FloatUnOp::Neg => WireRiscOp::Neg,
            FloatUnOp::Abs => WireRiscOp::Abs,
            FloatUnOp::Erfc => WireRiscOp::Erfc,
            FloatUnOp::Exp => WireRiscOp::Exp,
            _ => return Err(format!("Beacon scalar activation cannot lower `{op:?}`")),
        };
        Ok(self.graph.node(self.declaration, wire, vec![x]))
    }

    fn binary(&mut self, op: FloatBinOp, lhs: u64, rhs: u64) -> Result<u64, String> {
        let wire = match op {
            FloatBinOp::Add => WireRiscOp::Add,
            FloatBinOp::Sub => WireRiscOp::Sub,
            FloatBinOp::Mul => WireRiscOp::Mul,
            _ => return Err(format!("Beacon scalar activation cannot lower `{op:?}`")),
        };
        Ok(self.graph.node(self.declaration, wire, vec![lhs, rhs]))
    }

    fn less_than(&mut self, lhs: u64, rhs: u64) -> Result<u64, String> {
        Ok(self.graph.typed_node(
            self.declaration,
            WireRiscOp::Compare {
                comparison: WireComparisonKind::CmpLt,
            },
            vec![lhs, rhs],
            "bool",
        ))
    }

    fn select(&mut self, condition: u64, yes: u64, no: u64) -> Result<u64, String> {
        Ok(self.graph.node(
            self.declaration,
            WireRiscOp::Where {},
            vec![condition, yes, no],
        ))
    }

    fn convert(&mut self, x: u64, prim: Prim) -> Result<u64, String> {
        if prim == self.graph.source_dtype {
            Ok(x)
        } else {
            Err("Beacon scalar activation cannot change dtype".into())
        }
    }
}
