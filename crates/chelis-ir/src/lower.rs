//! Deep AST to RISC DAG lowering.
//!
//! Walks the Deep AST and produces a flat DAG of RISC primitive nodes.

use std::collections::HashMap;

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::types::Prim;

use crate::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use crate::tier2;

/// Lower a sequence of top-level Deep expressions into a RISC DAG.
pub fn lower_program(exprs: &[Expr]) -> Dag {
    let mut ctx = LowerCtx::new();
    for expr in exprs {
        ctx.lower_top_level(expr);
    }
    ctx.dag
}

struct LowerCtx {
    dag: Dag,
    bindings: HashMap<String, NodeId>,
}

impl LowerCtx {
    fn new() -> Self {
        Self {
            dag: Dag::new(),
            bindings: HashMap::new(),
        }
    }

    /// Default tensor type when we don't have richer type info.
    fn default_type() -> TensorType {
        TensorType::scalar_f32()
    }

    /// Extract a type from a metadata map if one is present, otherwise return a default.
    fn type_from_meta(meta: &[(String, Expr)]) -> TensorType {
        for (key, val) in meta {
            if key == "type" {
                if let Some(prim) = Self::try_extract_prim(val) {
                    return TensorType {
                        dims: vec![],
                        precision: prim,
                    };
                }
                if let Some(tt) = Self::try_extract_tensor_type(val) {
                    return tt;
                }
            }
        }
        Self::default_type()
    }

    fn try_extract_prim(expr: &Expr) -> Option<Prim> {
        // (t-prim {} f32)
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-prim"
            && let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2]
        {
            return Prim::parse_name(name);
        }
        None
    }

    fn try_extract_tensor_type(expr: &Expr) -> Option<TensorType> {
        // (t-tensor {} (t-dims {} ...) (t-prim {} p))
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 4
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-tensor"
        {
            let dims = Self::try_extract_dims(&list.elements[2])?;
            let prim = Self::try_extract_prim(&list.elements[3])?;
            return Some(TensorType {
                dims,
                precision: prim,
            });
        }
        None
    }

    fn try_extract_dims(expr: &Expr) -> Option<Vec<DimInfo>> {
        // (t-dims {} dim1 dim2 ...)
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 2
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-dims"
        {
            let mut dims = Vec::new();
            // Skip tag and meta (elements[0] and [1]).
            for elem in &list.elements[2..] {
                match elem {
                    Expr::Atom(Atom::Symbol(name), _) => {
                        dims.push(DimInfo::Named(name.clone(), None));
                    }
                    Expr::Atom(Atom::Int(n), _) => {
                        dims.push(DimInfo::Lit(*n as usize));
                    }
                    _ => {}
                }
            }
            return Some(dims);
        }
        None
    }

    fn lower_top_level(&mut self, expr: &Expr) {
        if let Expr::List(list, _) = expr
            && let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
        {
            match tag.as_str() {
                // Skip type-level declarations.
                "defsig" | "deftype" | "typealias" => return,
                _ => {}
            }
        }
        self.lower_expr(expr);
    }

    fn lower_expr(&mut self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Atom(atom, _) => self.lower_atom(atom),
            Expr::List(list, _) => self.lower_list(list),
            Expr::Map(_, _) => {
                // Bare metadata map -- shouldn't appear as an expression to lower.
                self.dag
                    .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
            }
            Expr::MetaExpr(meta_expr, _) => self.lower_expr(&meta_expr.expr),
        }
    }

    fn lower_atom(&mut self, atom: &Atom) -> NodeId {
        match atom {
            Atom::Symbol(name) => {
                if let Some(&id) = self.bindings.get(name) {
                    id
                } else {
                    self.dag.add_node(
                        RiscOp::Load { name: name.clone() },
                        vec![],
                        Self::default_type(),
                    )
                }
            }
            Atom::Int(n) => self.dag.add_node(
                RiscOp::Const { value: *n as f64 },
                vec![],
                Self::default_type(),
            ),
            Atom::Float(f) => {
                self.dag
                    .add_node(RiscOp::Const { value: *f }, vec![], Self::default_type())
            }
            Atom::Bool(b) => self.dag.add_node(
                RiscOp::Const {
                    value: if *b { 1.0 } else { 0.0 },
                },
                vec![],
                Self::default_type(),
            ),
            Atom::Str(_) | Atom::Keyword(_) => {
                self.dag
                    .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
            }
        }
    }

    fn lower_list(&mut self, list: &List) -> NodeId {
        let elems = &list.elements;
        if elems.is_empty() {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }

        let tag = match &elems[0] {
            Expr::Atom(Atom::Symbol(s), _) => s.as_str(),
            _ => {
                return self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                );
            }
        };

        match tag {
            "def" => self.lower_def(elems),
            "let" => self.lower_let(elems),
            "lit" => self.lower_lit(elems),
            "var" => self.lower_var(elems),
            "app" => self.lower_app(elems),
            "fn" => self.lower_fn(elems),
            "pipe" => self.lower_pipe(elems),
            "cast" => self.lower_cast(elems),
            "grad" => self.lower_grad(elems),
            "defsig" | "deftype" | "typealias" => {
                self.dag
                    .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
            }
            _ => {
                let mut last =
                    self.dag
                        .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
                for elem in &elems[2..] {
                    last = self.lower_expr(elem);
                }
                last
            }
        }
    }

    /// `(def {meta...} name body)`
    fn lower_def(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let name = match &elems[2] {
            Expr::Atom(Atom::Symbol(s), _) => s.clone(),
            _ => String::new(),
        };
        let body_id = self.lower_expr(&elems[3]);
        if !name.is_empty() {
            self.bindings.insert(name, body_id);
        }
        body_id
    }

    /// `(let {} (bind {} name expr) body)`
    fn lower_let(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        if let Expr::List(bind_list, _) = &elems[2]
            && bind_list.elements.len() >= 4
            && let Expr::Atom(Atom::Symbol(name), _) = &bind_list.elements[2]
        {
            let val_id = self.lower_expr(&bind_list.elements[3]);
            self.bindings.insert(name.clone(), val_id);
        }
        self.lower_expr(&elems[3])
    }

    /// `(lit {type: T} value)`
    fn lower_lit(&mut self, elems: &[Expr]) -> NodeId {
        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        let value = if let Some(val_expr) = elems.get(2) {
            match val_expr {
                Expr::Atom(Atom::Int(n), _) => *n as f64,
                Expr::Atom(Atom::Float(f), _) => *f,
                Expr::Atom(Atom::Bool(b), _) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        self.dag.add_node(RiscOp::Const { value }, vec![], ty)
    }

    /// `(var {meta...} name)`
    fn lower_var(&mut self, elems: &[Expr]) -> NodeId {
        // C6: Extract type from metadata if available.
        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        if let Some(Expr::Atom(Atom::Symbol(name), _)) = elems.get(2) {
            if let Some(&id) = self.bindings.get(name) {
                return id;
            }
            return self
                .dag
                .add_node(RiscOp::Load { name: name.clone() }, vec![], ty);
        }
        self.dag
            .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
    }

    /// `(app {meta...} func arg1 arg2 ...)`
    fn lower_app(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }

        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        // Check if func is a known built-in: (var {} name).
        if let Expr::List(func_list, _) = &elems[2]
            && let Some(Expr::Atom(Atom::Symbol(func_tag), _)) = func_list.elements.first()
            && func_tag == "var"
            && let Some(Expr::Atom(Atom::Symbol(func_name), _)) = func_list.elements.get(2)
        {
            return self.lower_builtin_app(func_name, &elems[3..], &ty);
        }

        // Not a recognized built-in -- lower func and args, return last.
        let mut last = self.lower_expr(&elems[2]);
        for arg in &elems[3..] {
            last = self.lower_expr(arg);
        }
        last
    }

    fn lower_builtin_app(&mut self, func_name: &str, args: &[Expr], ty: &TensorType) -> NodeId {
        match func_name {
            // Tier 1: binary elementwise
            "add" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                self.dag.add_node(RiscOp::Add, vec![a, b], ty.clone())
            }
            "mul" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                self.dag.add_node(RiscOp::Mul, vec![a, b], ty.clone())
            }
            "cmplt" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                // C5: CmpLt always produces Bool output regardless of input precision.
                let bool_ty = TensorType {
                    dims: ty.dims.clone(),
                    precision: Prim::Bool,
                };
                self.dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty)
            }
            "max_elem" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                self.dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone())
            }

            // Tier 1: unary elementwise
            "neg" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                self.dag.add_node(RiscOp::Neg, vec![x], ty.clone())
            }
            "exp" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                self.lower_transcendental(RiscOp::Exp, x, ty)
            }
            "log" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                self.lower_transcendental(RiscOp::Log, x, ty)
            }
            "sin" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                self.lower_transcendental(RiscOp::Sin, x, ty)
            }
            "sqrt" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                self.lower_transcendental(RiscOp::Sqrt, x, ty)
            }

            // Tier 2 decompositions
            "sub" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_sub(&mut self.dag, a, b, ty)
            }
            "relu" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                tier2::lower_relu(&mut self.dag, x, ty)
            }
            "sigmoid" if args.len() == 1 => {
                let x = self.lower_expr(&args[0]);
                tier2::lower_sigmoid(&mut self.dag, x, ty)
            }
            "div" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_div(&mut self.dag, a, b, ty)
            }

            // H1: Tier 2 comparison ops
            "gt" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_gt(&mut self.dag, a, b, ty)
            }
            "gte" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_gte(&mut self.dag, a, b, ty)
            }
            "lte" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_lte(&mut self.dag, a, b, ty)
            }
            "eq" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_eq(&mut self.dag, a, b, ty)
            }
            "neq" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_neq(&mut self.dag, a, b, ty)
            }
            "min_elem" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_min_elem(&mut self.dag, a, b, ty)
            }

            // H2: Boolean operators
            "and" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_and(&mut self.dag, a, b, ty)
            }
            "or" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                tier2::lower_or(&mut self.dag, a, b, ty)
            }
            "not" if args.len() == 1 => {
                let a = self.lower_expr(&args[0]);
                tier2::lower_not(&mut self.dag, a, ty)
            }

            // Tier 1: reductions
            "sum" if args.len() == 2 => {
                let x = self.lower_expr(&args[0]);
                let axis = self.extract_axis(&args[1]);
                self.dag.add_node(RiscOp::Sum { axis }, vec![x], ty.clone())
            }
            "max_reduce" if args.len() == 2 => {
                let x = self.lower_expr(&args[0]);
                let axis = self.extract_axis(&args[1]);
                self.dag
                    .add_node(RiscOp::MaxReduce { axis }, vec![x], ty.clone())
            }

            // H3: Movement op stubs -- recognized but passthrough for Phase 0.
            "reshape" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                self.dag.add_node(
                    RiscOp::Reshape {
                        new_shape: ty.dims.clone(),
                    },
                    vec![x],
                    ty.clone(),
                )
            }
            "permute" if args.len() >= 2 => {
                let x = self.lower_expr(&args[0]);
                // Extract axes from remaining args (stub: empty for now).
                let axes = vec![];
                self.dag
                    .add_node(RiscOp::Permute { axes }, vec![x], ty.clone())
            }
            "expand" if args.len() >= 2 => {
                let x = self.lower_expr(&args[0]);
                self.dag
                    .add_node(RiscOp::Expand { axis: 0, size: 1 }, vec![x], ty.clone())
            }
            "pad" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                self.dag.add_node(
                    RiscOp::Pad {
                        padding: vec![],
                        fill: 0.0,
                    },
                    vec![x],
                    ty.clone(),
                )
            }
            "shrink" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                self.dag
                    .add_node(RiscOp::Shrink { bounds: vec![] }, vec![x], ty.clone())
            }
            "stride" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                self.dag
                    .add_node(RiscOp::Stride { strides: vec![] }, vec![x], ty.clone())
            }

            // Fallback: unknown function.
            _ => {
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.to_string(),
                    },
                    vec![],
                    Self::default_type(),
                )
            }
        }
    }

    /// Extract an axis value from an expression (for sum/max_reduce).
    fn extract_axis(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Atom(Atom::Int(n), _) => *n as usize,
            // Handle (lit {} n) form.
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    *n as usize
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// C4: Enforce float-only for transcendental ops (exp, log, sin, sqrt).
    /// If the input is not float, produce a Const(0) error placeholder.
    fn lower_transcendental(&mut self, op: RiscOp, x: NodeId, ty: &TensorType) -> NodeId {
        let input_prec = self
            .dag
            .get(x)
            .map(|n| n.output_type.precision)
            .unwrap_or(Prim::F32);
        if input_prec.is_float() {
            self.dag.add_node(op, vec![x], ty.clone())
        } else {
            // Non-float input: produce a zero constant as error placeholder.
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone())
        }
    }

    /// `(fn {} (params {} p1 p2 ...) body)`
    fn lower_fn(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        // Register params as Load nodes.
        if let Expr::List(params_list, _) = &elems[2] {
            for param in &params_list.elements[2..] {
                if let Expr::Atom(Atom::Symbol(name), _) = param {
                    let load_id = self.dag.add_node(
                        RiscOp::Load { name: name.clone() },
                        vec![],
                        Self::default_type(),
                    );
                    self.bindings.insert(name.clone(), load_id);
                }
                // Handle (param {} name) form.
                if let Expr::List(param_list, _) = param
                    && param_list.elements.len() >= 3
                    && let Expr::Atom(Atom::Symbol(name), _) = &param_list.elements[2]
                {
                    let load_id = self.dag.add_node(
                        RiscOp::Load { name: name.clone() },
                        vec![],
                        Self::default_type(),
                    );
                    self.bindings.insert(name.clone(), load_id);
                }
            }
        }
        self.lower_expr(&elems[3])
    }

    /// `(pipe {} x f g ...)` -- chain: lower x, then apply f, then g, etc.
    fn lower_pipe(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 3 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let mut current = self.lower_expr(&elems[2]);
        for func_expr in &elems[3..] {
            if let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                let ty = Self::default_type();
                current = match fname.as_str() {
                    "neg" => self.dag.add_node(RiscOp::Neg, vec![current], ty),
                    "exp" => self.dag.add_node(RiscOp::Exp, vec![current], ty),
                    "log" => self.dag.add_node(RiscOp::Log, vec![current], ty),
                    "sin" => self.dag.add_node(RiscOp::Sin, vec![current], ty),
                    "sqrt" => self.dag.add_node(RiscOp::Sqrt, vec![current], ty),
                    "relu" => tier2::lower_relu(&mut self.dag, current, &ty),
                    "sigmoid" => tier2::lower_sigmoid(&mut self.dag, current, &ty),
                    _ => current,
                };
                continue;
            }
            // Fallback: just lower the expression.
            current = self.lower_expr(func_expr);
        }
        current
    }

    /// `(cast {} expr prec)` -- precision cast.
    fn lower_cast(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let x = self.lower_expr(&elems[2]);
        let new_precision = if let Expr::Atom(Atom::Symbol(pname), _) = &elems[3] {
            Prim::parse_name(pname).unwrap_or(Prim::F32)
        } else {
            Prim::F32
        };
        let ty = TensorType {
            dims: vec![],
            precision: new_precision,
        };
        self.dag
            .add_node(RiscOp::Cast { new_precision }, vec![x], ty)
    }

    /// `(grad {} f)` -- placeholder: just lower f for now.
    fn lower_grad(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        lower_program(&exprs)
    }

    #[test]
    fn lower_single_const() {
        let dag = parse_and_lower("(def {} x (lit {type: (t-prim {} f32)} 1.0))");
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 1.0 });
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_add_two_consts() {
        let src = r#"
            (def {} a (lit {type: (t-prim {} f32)} 1.0))
            (def {} b (lit {type: (t-prim {} f32)} 2.0))
            (def {} c (app {} (var {} add) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_neg() {
        let src = r#"
            (def {} x (lit {type: (t-prim {} f32)} 5.0))
            (def {} y (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        let neg_node = dag.get(NodeId(1)).unwrap();
        assert_eq!(neg_node.op, RiscOp::Neg);
        assert_eq!(neg_node.inputs, vec![NodeId(0)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_sub_decomposes() {
        let src = r#"
            (def {} a (lit {} 3.0))
            (def {} b (lit {} 1.0))
            (def {} c (app {} (var {} sub) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a=Const(3), b=Const(1), Neg(b), Add(a, Neg(b))
        assert_eq!(dag.len(), 4);
        assert!(verify::verify(&dag).is_empty());
        let last = dag.get(NodeId(3)).unwrap();
        assert_eq!(last.op, RiscOp::Add);
    }

    #[test]
    fn lower_relu_decomposes() {
        let src = r#"
            (def {} x (lit {} -2.0))
            (def {} y (app {} (var {} relu) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(-2), Const(0), MaxElem(x, 0)
        assert_eq!(dag.len(), 3);
        assert!(verify::verify(&dag).is_empty());
        let last = dag.get(NodeId(2)).unwrap();
        assert_eq!(last.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_let_binding() {
        let src = r#"
            (let {} (bind {} x (lit {} 10.0)) (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(10), Neg(x)
        assert_eq!(dag.len(), 2);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_unknown_var_becomes_load() {
        let src = "(def {} y (var {} weights))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Load {
                name: "weights".to_string()
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H1: Tier 2 comparison ops lowering ---

    #[test]
    fn lower_gt_decomposes() {
        let src = r#"
            (def {} a (lit {} 5.0))
            (def {} b (lit {} 3.0))
            (def {} c (app {} (var {} gt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(b, a)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // Args are swapped: b, a
        assert_eq!(node.inputs, vec![NodeId(1), NodeId(0)]);
    }

    #[test]
    fn lower_gte_decomposes() {
        let src = r#"
            (def {} a (lit {} 5.0))
            (def {} b (lit {} 3.0))
            (def {} c (app {} (var {} gte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(dag.len(), 5);
        let node = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
    }

    #[test]
    fn lower_lte_decomposes() {
        let src = r#"
            (def {} a (lit {} 3.0))
            (def {} b (lit {} 5.0))
            (def {} c (app {} (var {} lte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 5);
    }

    #[test]
    fn lower_eq_decomposes() {
        let src = r#"
            (def {} a (lit {} 3.0))
            (def {} b (lit {} 3.0))
            (def {} c (app {} (var {} eq) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(dag.len(), 7);
    }

    #[test]
    fn lower_min_elem_decomposes() {
        let src = r#"
            (def {} a (lit {} 5.0))
            (def {} b (lit {} 3.0))
            (def {} c (app {} (var {} min_elem) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(dag.len(), 6);
        let node = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert_eq!(node.op, RiscOp::Neg);
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H2: Boolean operators ---

    #[test]
    fn lower_and_decomposes() {
        let src = r#"
            (def {} a (lit {} 1.0))
            (def {} b (lit {} 0.0))
            (def {} c (app {} (var {} and) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, Mul(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
    }

    #[test]
    fn lower_or_decomposes() {
        let src = r#"
            (def {} a (lit {} 0.0))
            (def {} b (lit {} 1.0))
            (def {} c (app {} (var {} or) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, MaxElem(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_not_decomposes() {
        let src = r#"
            (def {} a (lit {} 1.0))
            (def {} b (app {} (var {} not) (var {} a)))
        "#;
        let dag = parse_and_lower(src);
        // a, Const(1), CmpLt(a, 1)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
    }

    // --- H3: Movement op stubs ---

    #[test]
    fn lower_reshape_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} reshape) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Reshape { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_pad_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} pad) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Pad { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_shrink_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} shrink) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Shrink { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_stride_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} stride) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Stride { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H4: sum/max_reduce lowering ---

    #[test]
    fn lower_sum_reduction() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} sum) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(1), axis_const=Const(0) is lowered inline, Sum{axis:0}
        let found_sum = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Sum { axis: 0 }));
        assert!(found_sum, "expected a Sum{{axis:0}} node");
    }

    #[test]
    fn lower_max_reduce_reduction() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {} (var {} max_reduce) (var {} x) (lit {} 1)))
        "#;
        let dag = parse_and_lower(src);
        let found = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::MaxReduce { axis: 1 }));
        assert!(found, "expected a MaxReduce{{axis:1}} node");
    }

    // --- C5: CmpLt lowering produces Bool ---

    #[test]
    fn lower_cmplt_produces_bool_output() {
        let src = r#"
            (def {} a (lit {type: (t-prim {} f32)} 1.0))
            (def {} b (lit {type: (t-prim {} f32)} 2.0))
            (def {} c (app {} (var {} cmplt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        let cmplt_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::CmpLt))
            .expect("expected CmpLt node");
        assert_eq!(
            cmplt_node.output_type.precision,
            Prim::Bool,
            "CmpLt output must be Bool"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- C6: type propagation from metadata ---

    #[test]
    fn lower_lit_with_type_metadata() {
        let src = "(def {} x (lit {type: (t-prim {} f64)} 3.14))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    #[test]
    fn lower_var_with_type_metadata() {
        let src = "(def {} y (var {type: (t-prim {} f64)} weights))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }
}
