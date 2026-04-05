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
        // Flat format: (t-tensor {} dim1 dim2 ... (t-prim {} p))
        // Children after tag+meta: dimension nodes followed by a t-prim node as the last child.
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-tensor"
        {
            // elements[0] = tag, elements[1] = meta, elements[2..] = children
            let children = &list.elements[2..];
            if children.is_empty() {
                return None;
            }
            // Last child is the precision (t-prim {} name).
            let prim = Self::try_extract_prim(children.last()?)?;
            // All children before the last are dimension nodes.
            let mut dims = Vec::new();
            for child in &children[..children.len() - 1] {
                if let Some(dim) = Self::try_extract_dim(child) {
                    dims.push(dim);
                }
            }
            return Some(TensorType {
                dims,
                precision: prim,
            });
        }
        None
    }

    /// Extract a single dimension from a dimension node.
    fn try_extract_dim(expr: &Expr) -> Option<DimInfo> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        {
            match tag.as_str() {
                "d-name" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-var" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-lit" => {
                    if let Expr::Atom(Atom::Int(n), _) = &list.elements[2] {
                        return Some(DimInfo::Lit(*n as usize));
                    }
                }
                _ => {}
            }
        }
        // Also handle bare symbols/ints for backward compat.
        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(DimInfo::Named(name.clone(), None)),
            Expr::Atom(Atom::Int(n), _) => Some(DimInfo::Lit(*n as usize)),
            _ => None,
        }
    }

    /// Extract a list of dimensions from a `(t-dims {} dim1 dim2 ...)` expression.
    fn try_extract_dims(expr: &Expr) -> Option<Vec<DimInfo>> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 2
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-dims"
        {
            // Skip element [0] (tag) and [1] (empty map / metadata), parse remaining as dims.
            let mut dims = Vec::new();
            for elem in list.elements.iter().skip(2) {
                if let Some(d) = Self::try_extract_dim(elem) {
                    dims.push(d);
                }
            }
            if !dims.is_empty() {
                return Some(dims);
            }
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
            "if" => self.lower_if(elems),
            "tuple" => self.lower_tuple(elems),
            "par" => self.lower_par(elems),
            "realize" | "copy" => self.lower_identity(elems),
            "tuple-get" => self.lower_tuple_get(elems),
            "match" => self.lower_match(elems),
            "grad" => self.lower_grad(elems),
            "vmap" | "jit" => self.lower_unsupported(tag, elems),
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

    /// `(let {} (bind {} name1 expr1 name2 expr2 ...) body)`
    fn lower_let(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let saved = self.bindings.clone();

        // elems[2] = (bind {} name1 expr1 name2 expr2 ...)
        if let Expr::List(bind_list, _) = &elems[2] {
            // Skip tag and meta (elements[0] and [1]).
            let bind_kids = &bind_list.elements[2..];
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                if let Expr::Atom(Atom::Symbol(name), _) = &bind_kids[i] {
                    let val_id = self.lower_expr(&bind_kids[i + 1]);
                    self.bindings.insert(name.clone(), val_id);
                }
                i += 2;
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        result
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

            // Tier 2 higher-level ops (spec §3.4, §4.1–4.2)
            "matmul" if args.len() == 2 => {
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                let a_ty = self
                    .dag
                    .get(a)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let b_ty = self
                    .dag
                    .get(b)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                tier2::lower_matmul(&mut self.dag, a, b, &a_ty, &b_ty)
            }
            "softmax" if args.len() == 2 => {
                let x = self.lower_expr(&args[0]);
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                tier2::lower_softmax(&mut self.dag, x, axis, &x_ty)
            }
            "mean" if args.len() == 2 => {
                let x = self.lower_expr(&args[0]);
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                tier2::lower_mean(&mut self.dag, x, axis, &x_ty)
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

            // H3: Movement ops -- extract parameters from Deep AST args where possible.
            "reshape" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                // Try to extract new_shape from the second arg; fall back to output type dims.
                let new_shape = if args.len() >= 2 {
                    self.extract_dim_list(&args[1])
                        .unwrap_or_else(|| ty.dims.clone())
                } else {
                    ty.dims.clone()
                };
                let out_ty = TensorType {
                    dims: new_shape.clone(),
                    precision: ty.precision,
                };
                self.dag
                    .add_node(RiscOp::Reshape { new_shape }, vec![x], out_ty)
            }
            "permute" if args.len() >= 2 => {
                let x = self.lower_expr(&args[0]);
                // Extract axes ordering from remaining args.
                let axes = self.extract_usize_list(&args[1..]);
                self.dag
                    .add_node(RiscOp::Permute { axes }, vec![x], ty.clone())
            }
            "expand" if args.len() >= 2 => {
                let x = self.lower_expr(&args[0]);
                let axis = self.extract_usize_value(&args[1]).unwrap_or(0);
                let size = if args.len() >= 3 {
                    self.extract_usize_value(&args[2]).unwrap_or(1)
                } else {
                    1
                };
                self.dag
                    .add_node(RiscOp::Expand { axis, size }, vec![x], ty.clone())
            }
            "pad" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                let padding = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                let fill = if args.len() >= 3 {
                    self.extract_f64_value(&args[2]).unwrap_or(0.0)
                } else {
                    0.0
                };
                self.dag
                    .add_node(RiscOp::Pad { padding, fill }, vec![x], ty.clone())
            }
            "shrink" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                let bounds = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                self.dag
                    .add_node(RiscOp::Shrink { bounds }, vec![x], ty.clone())
            }
            "stride" if !args.is_empty() => {
                let x = self.lower_expr(&args[0]);
                let strides = if args.len() >= 2 {
                    self.extract_usize_list(&args[1..])
                } else {
                    vec![]
                };
                self.dag
                    .add_node(RiscOp::Stride { strides }, vec![x], ty.clone())
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

    /// Extract a single usize value from an expression.
    fn extract_usize_value(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as usize)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract an f64 value from an expression.
    fn extract_f64_value(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Atom(Atom::Float(f), _) => Some(*f),
            Expr::Atom(Atom::Int(n), _) => Some(*n as f64),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Float(f), _)) = list.elements.get(2) {
                    Some(*f)
                } else if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as f64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract a list of usize values from a slice of expressions.
    fn extract_usize_list(&self, exprs: &[Expr]) -> Vec<usize> {
        let mut result = Vec::new();
        for expr in exprs {
            if let Some(v) = self.extract_usize_value(expr) {
                result.push(v);
            }
        }
        result
    }

    /// Extract dimension info list from an expression (e.g., for reshape).
    fn extract_dim_list(&self, expr: &Expr) -> Option<Vec<DimInfo>> {
        // Handle (t-dims {} dim1 dim2 ...) form.
        if let Expr::List(list, _) = expr {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && tag == "t-dims"
            {
                return Self::try_extract_dims(expr);
            }
            // Try as a plain list of integers.
            let mut dims = Vec::new();
            for elem in &list.elements {
                match elem {
                    Expr::Atom(Atom::Int(n), _) => dims.push(DimInfo::Lit(*n as usize)),
                    Expr::Atom(Atom::Symbol(name), _) => {
                        dims.push(DimInfo::Named(name.clone(), None));
                    }
                    _ => {}
                }
            }
            if !dims.is_empty() {
                return Some(dims);
            }
        }
        None
    }

    /// Extract a list of (usize, usize) pairs from an expression (for pad/shrink bounds).
    fn extract_pair_list(&self, expr: &Expr) -> Option<Vec<(usize, usize)>> {
        if let Expr::List(list, _) = expr {
            let mut pairs = Vec::new();
            for elem in &list.elements {
                if let Expr::List(pair_list, _) = elem {
                    let vals: Vec<usize> = pair_list
                        .elements
                        .iter()
                        .filter_map(|e| {
                            if let Expr::Atom(Atom::Int(n), _) = e {
                                Some(*n as usize)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if vals.len() >= 2 {
                        pairs.push((vals[0], vals[1]));
                    }
                }
            }
            if !pairs.is_empty() {
                return Some(pairs);
            }
        }
        None
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
        let saved = self.bindings.clone();

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

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        result
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

    /// `(cast {} expr (t-prim {} name))` -- precision cast.
    fn lower_cast(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() < 4 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let x = self.lower_expr(&elems[2]);
        let new_precision = if let Some(prim) = Self::try_extract_prim(&elems[3]) {
            // Handle (t-prim {} name) form.
            prim
        } else if let Expr::Atom(Atom::Symbol(pname), _) = &elems[3] {
            // Fallback: bare symbol for backward compat.
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

    /// `(grad {} f)` -- Phase 2 feature, produce NaN warning.
    fn lower_grad(&mut self, elems: &[Expr]) -> NodeId {
        eprintln!(
            "WARNING: `grad` is a Phase 2 feature and is not yet supported in lowering. \
             Producing NaN placeholder."
        );
        // Lower the child so its side effects (bindings) still happen,
        // but discard the result and return NaN to signal the error.
        if elems.len() >= 3 {
            let _ = self.lower_expr(&elems[2]);
        }
        self.dag.add_node(
            RiscOp::Const { value: f64::NAN },
            vec![],
            Self::default_type(),
        )
    }

    /// `(if {} cond then else)` -- Phase 0: select via arithmetic on bools.
    fn lower_if(&mut self, elems: &[Expr]) -> NodeId {
        // elems: [tag, meta, cond, then_branch, else_branch]
        if elems.len() < 5 {
            return self
                .dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        }
        let cond = self.lower_expr(&elems[2]);
        let then_val = self.lower_expr(&elems[3]);
        let else_val = self.lower_expr(&elems[4]);
        let ty = Self::default_type();
        // not_cond = cmplt(cond, const(1))
        let one = self
            .dag
            .add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone());
        let bool_ty = TensorType {
            dims: ty.dims.clone(),
            precision: Prim::Bool,
        };
        let not_cond = self.dag.add_node(RiscOp::CmpLt, vec![cond, one], bool_ty);
        // result = add(mul(cond, then), mul(not_cond, else))
        let cond_then = self
            .dag
            .add_node(RiscOp::Mul, vec![cond, then_val], ty.clone());
        let not_cond_else = self
            .dag
            .add_node(RiscOp::Mul, vec![not_cond, else_val], ty.clone());
        self.dag
            .add_node(RiscOp::Add, vec![cond_then, not_cond_else], ty)
    }

    /// `(tuple {} elem1 elem2 ...)` -- Lower each element, return last.
    fn lower_tuple(&mut self, elems: &[Expr]) -> NodeId {
        let mut last =
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        for elem in &elems[2..] {
            last = self.lower_expr(elem);
        }
        last
    }

    /// `(par {} expr1 expr2 ...)` -- Lower each child sequentially, return last.
    fn lower_par(&mut self, elems: &[Expr]) -> NodeId {
        let mut last =
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type());
        for elem in &elems[2..] {
            last = self.lower_expr(elem);
        }
        last
    }

    /// `(realize {} expr)` or `(copy {} expr)` -- identity in Phase 0.
    fn lower_identity(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
        }
    }

    /// `(tuple-get {} tuple_expr index)` -- Phase 0: return the lowered tuple.
    fn lower_tuple_get(&mut self, elems: &[Expr]) -> NodeId {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
        }
    }

    /// `(match {} scrutinee (arm {} pattern body) ...)` -- Phase 0: lower first arm's body.
    fn lower_match(&mut self, elems: &[Expr]) -> NodeId {
        eprintln!(
            "WARNING: `match` lowering is incomplete in Phase 0. \
             Only the first arm's body is lowered."
        );
        // elems[2] = scrutinee, elems[3..] = arms
        if elems.len() >= 3 {
            let _ = self.lower_expr(&elems[2]); // lower scrutinee for side effects
        }
        // Lower first arm's body if available.
        if elems.len() >= 4
            && let Expr::List(arm, _) = &elems[3]
            && arm.elements.len() >= 4
        {
            return self.lower_expr(&arm.elements[arm.elements.len() - 1]);
        }
        self.dag
            .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
    }

    /// Unsupported Phase 2 constructs (vmap, jit).
    fn lower_unsupported(&mut self, tag: &str, _elems: &[Expr]) -> NodeId {
        eprintln!(
            "WARNING: `{tag}` is a Phase 2 feature and is not yet supported in lowering. \
             Producing NaN placeholder."
        );
        self.dag.add_node(
            RiscOp::Const { value: f64::NAN },
            vec![],
            Self::default_type(),
        )
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

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::dag::{DimInfo, NodeId, RiscOp};
    use crate::verify;
    use chelis_types::types::Prim;

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        lower_program(&exprs)
    }

    // Fix 1: Tensor type metadata with flat Deep shape format.
    #[test]
    fn fix1_tensor_type_flat_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![DimInfo::Named("batch".to_string(), None)]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_multiple_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("hidden".to_string(), None),
            ]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_lit_dim() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f64))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.dims, vec![DimInfo::Lit(512)]);
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    // Fix 3: Lexical scoping -- let restores bindings.
    #[test]
    fn fix3_let_multiple_bindings() {
        let src = r#"
            (let {} (bind {} x (lit {} 1.0) y (lit {} 2.0))
                (app {} (var {} add) (var {} x) (var {} y)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn fix3_let_scope_does_not_leak() {
        let src = r#"
            (let {} (bind {} x (lit {} 1.0)) (var {} x))
            (var {} x)
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "x"),
            "x should not be visible after let scope"
        );
    }

    #[test]
    fn fix3_fn_scope_does_not_leak() {
        let src = r#"
            (fn {} (params {} p) (var {} p))
            (var {} p)
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "p"),
            "fn param p should not be visible after fn scope"
        );
    }

    // Fix 4: Unsupported constructs.
    #[test]
    fn fix4_grad_produces_nan() {
        let src = "(grad {} (var {} f))";
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        if let RiscOp::Const { value } = &last.op {
            assert!(value.is_nan(), "grad should produce NaN placeholder");
        } else {
            panic!("grad should produce a Const(NaN) node, got {:?}", last.op);
        }
    }

    #[test]
    fn fix4_vmap_produces_nan() {
        let src = "(vmap {} (var {} f))";
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        if let RiscOp::Const { value } = &last.op {
            assert!(value.is_nan(), "vmap should produce NaN placeholder");
        } else {
            panic!("vmap should produce a Const(NaN) node");
        }
    }

    #[test]
    fn fix4_jit_produces_nan() {
        let src = "(jit {} (var {} f))";
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        if let RiscOp::Const { value } = &last.op {
            assert!(value.is_nan(), "jit should produce NaN placeholder");
        } else {
            panic!("jit should produce a Const(NaN) node");
        }
    }

    #[test]
    fn fix4_realize_is_identity() {
        let src = "(realize {} (lit {} 42.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Const { value: 42.0 }
        );
    }

    #[test]
    fn fix4_copy_is_identity() {
        let src = "(copy {} (lit {} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 7.0 });
    }

    // Fix 9: Cast with (t-prim {} bf16) node.
    #[test]
    fn fix9_cast_with_tprim_node() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) (t-prim {} bf16)))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::Bf16
            }
        );
        assert_eq!(cast_node.output_type.precision, Prim::Bf16);
    }

    #[test]
    fn fix9_cast_bare_symbol_still_works() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) f16))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::F16
            }
        );
    }
}
