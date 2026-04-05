//! Desugars Surface AST into Deep (s-expression) AST.
//!
//! Every Deep node is a 3-tuple: (tag {} children...)
//! where {} is an inline metadata map.

use std::collections::HashSet;

use chelis_deep::Span;
use chelis_deep::ast as deep;

use crate::ast::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn desugar_program(decls: &[Decl]) -> Vec<deep::Expr> {
    decls.iter().flat_map(desugar_decl).collect()
}

// ---------------------------------------------------------------------------
// Helpers for building Deep AST nodes (3-tuple format)
// ---------------------------------------------------------------------------

fn sp() -> Span {
    Span::new(0, 0)
}

fn sym(s: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(s.to_string()), sp())
}

fn meta_empty() -> deep::Expr {
    deep::Expr::Map(deep::MetaMap::default(), sp())
}

fn meta_with_type(ty: deep::Expr) -> deep::Expr {
    deep::Expr::Map(
        deep::MetaMap {
            entries: vec![("type".to_string(), ty)],
        },
        sp(),
    )
}

/// Build a 3-tuple Deep node: (tag {} children...)
fn node(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![sym(tag), meta_empty()];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, sp())
}

/// Build a 3-tuple Deep node with custom metadata: (tag {meta} children...)
fn node_meta(tag: &str, meta: deep::Expr, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![sym(tag), meta];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, sp())
}

/// Variable reference: (var {} name)
fn dvar(name: &str) -> deep::Expr {
    node("var", vec![sym(name)])
}

/// Build a bare list (no tag/meta) for structural helpers like params, bind
fn bare_list(elements: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::List(deep::List { elements }, sp())
}

fn lower_module_path(path: &str) -> String {
    path.to_ascii_lowercase()
}

fn desugar_param(param: &Param) -> deep::Expr {
    desugar_param_with_dims(param, &HashSet::new())
}

fn desugar_param_with_dims(param: &Param, dim_vars: &HashSet<String>) -> deep::Expr {
    match &param.ty {
        Some(ty) => deep::Expr::List(
            deep::List {
                elements: vec![
                    sym(&param.name),
                    meta_with_type(desugar_type_with_dims(ty, dim_vars)),
                ],
            },
            sp(),
        ),
        None => sym(&param.name),
    }
}

/// Inject a type annotation into the metadata of a desugared expression.
fn inject_type_metadata(expr: deep::Expr, ty: deep::Expr) -> deep::Expr {
    match expr {
        deep::Expr::List(list, span) => {
            let mut elements = list.elements;
            if elements.len() >= 2 {
                // Replace the metadata map (element[1]) with one containing the type
                elements[1] = meta_with_type(ty);
            }
            deep::Expr::List(deep::List { elements }, span)
        }
        // For atoms, wrap in an annotated var node
        other => node_meta("var", meta_with_type(ty), vec![other]),
    }
}

// ---------------------------------------------------------------------------
// Primitive type names
// ---------------------------------------------------------------------------

const PRIMITIVES: &[&str] = &[
    "f32", "f64", "f16", "bf16", "f8e4m3", "int8", "int32", "int64", "bool", "string", "unit",
];

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

fn desugar_decl(decl: &Decl) -> Vec<deep::Expr> {
    match decl {
        Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            body,
            ..
        } => desugar_fun_def(name, dim_params, params, ret_ty, body),

        Decl::LetDef {
            name,
            ty: Some(t),
            value,
            ..
        } => {
            vec![
                node("defsig", vec![sym(name), desugar_type(t)]),
                node("def", vec![sym(name), desugar_expr(value)]),
            ]
        }

        Decl::LetDef {
            name,
            ty: None,
            value,
            ..
        } => {
            vec![node("def", vec![sym(name), desugar_expr(value)])]
        }

        Decl::TypeDef {
            name,
            params,
            variants,
            ..
        } => vec![desugar_type_def(name, params, variants)],

        Decl::TypeAlias {
            name, params, ty, ..
        } => {
            let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
            vec![node(
                "typealias",
                vec![sym(name), param_list, desugar_type(ty)],
            )]
        }

        Decl::Sig { name, ty, .. } => vec![node("defsig", vec![sym(name), desugar_type(ty)])],

        Decl::Dim { names, .. } => names
            .iter()
            .map(|name| node("defdim", vec![sym(name)]))
            .collect(),

        Decl::Module { name, decls, .. } => {
            let mut children = vec![sym(&lower_module_path(name))];
            for d in decls {
                children.extend(desugar_decl(d));
            }
            vec![node("module", children)]
        }

        Decl::Import { module, kind, .. } => match kind {
            ImportKind::Names(ns) => {
                let name_list = bare_list(ns.iter().map(|n| sym(n)).collect());
                vec![node(
                    "import",
                    vec![sym(&lower_module_path(module)), name_list],
                )]
            }
            ImportKind::Qualified => vec![node(
                "import",
                vec![sym(&lower_module_path(module)), bare_list(vec![])],
            )],
            ImportKind::All => vec![node("import-all", vec![sym(&lower_module_path(module))])],
        },

        Decl::Export { names, .. } => {
            let mut children = Vec::new();
            for n in names {
                children.push(sym(n));
            }
            vec![node("export", children)]
        }
    }
}

fn desugar_fun_def(
    name: &str,
    dim_params: &[String],
    params: &[Param],
    ret_ty: &Option<TypeExpr>,
    body: &Expr,
) -> Vec<deep::Expr> {
    // Function-level dim params are polymorphic d-vars, NOT module-level defdim.
    // Build a set so desugar_type_with_dims treats them as d-var.
    let dim_set: HashSet<String> = dim_params.iter().cloned().collect();

    let param_names: Vec<deep::Expr> = params
        .iter()
        .map(|param| desugar_param_with_dims(param, &dim_set))
        .collect();
    let params_node = node("params", param_names);
    let fn_node = node("fn", vec![params_node, desugar_expr(body)]);
    let def_node = node("def", vec![sym(name), fn_node]);

    if params.iter().any(|p| p.ty.is_some()) || ret_ty.is_some() {
        let mut type_parts: Vec<deep::Expr> = params
            .iter()
            .map(|p| match &p.ty {
                Some(ty) => desugar_type_with_dims(ty, &dim_set),
                None => node("t-var", vec![sym("_")]),
            })
            .collect();
        type_parts.push(match ret_ty {
            Some(ty) => desugar_type_with_dims(ty, &dim_set),
            None => node("t-var", vec![sym("_")]),
        });
        let sig = node("defsig", vec![sym(name), node("t-fn", type_parts)]);
        vec![sig, def_node]
    } else {
        vec![def_node]
    }
}

fn desugar_type_def(name: &str, params: &[String], variants: &[Variant]) -> deep::Expr {
    let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
    let mut children = vec![sym(name), param_list];
    for v in variants {
        children.push(desugar_variant(v));
    }
    node("deftype", children)
}

fn desugar_variant(variant: &Variant) -> deep::Expr {
    match &variant.fields {
        VariantFields::Positional(fields) => {
            let mut children = vec![sym(&variant.name)];
            for f in fields {
                children.push(desugar_type(f));
            }
            node("variant", children)
        }
        VariantFields::Record(fields) => {
            let mut children = vec![sym(&variant.name)];
            for (field_name, field_ty) in fields {
                children.push(node("field", vec![sym(field_name), desugar_type(field_ty)]));
            }
            node("variant", children)
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn desugar_expr(expr: &Expr) -> deep::Expr {
    match expr {
        Expr::Lit(lit, _) => desugar_literal(lit),
        Expr::Var(name, _) => dvar(name),
        Expr::Constructor(name, _) => dvar(name),
        Expr::Record(name, fields, _) => {
            let mut fields = fields.clone();
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            let mut children = vec![sym(name)];
            for (field, value) in fields {
                children.push(node("kv", vec![sym(&field), desugar_expr(&value)]));
            }
            node("record", children)
        }
        Expr::Access(target, field, _) => node("access", vec![desugar_expr(target), sym(field)]),
        Expr::TupleGet(target, index, _) => node(
            "tuple-get",
            vec![
                desugar_expr(target),
                node_meta(
                    "lit",
                    meta_with_type(node("t-prim", vec![sym("int32")])),
                    vec![deep::Expr::Atom(deep::Atom::Int(*index), sp())],
                ),
            ],
        ),

        Expr::Apply(func, args, _) => desugar_apply(func, args),

        Expr::Binary(op, lhs, rhs, _) => {
            let op_name = binop_name(*op);
            // a > b -> (app {} (var {} cmplt) b' a') -- swap operands
            match op {
                BinOp::Gt => node(
                    "app",
                    vec![dvar(op_name), desugar_expr(rhs), desugar_expr(lhs)],
                ),
                _ => node(
                    "app",
                    vec![dvar(op_name), desugar_expr(lhs), desugar_expr(rhs)],
                ),
            }
        }

        Expr::Unary(op, operand, _) => {
            let op_name = match op {
                UnaryOp::Neg => "neg",
                UnaryOp::Not => "not",
            };
            node("app", vec![dvar(op_name), desugar_expr(operand)])
        }

        Expr::Pipe(head, stages, _) => {
            let mut children = vec![desugar_expr(head)];
            children.extend(stages.iter().map(desugar_expr));
            node("pipe", children)
        }

        Expr::If(cond, then_e, else_e, _) => node(
            "if",
            vec![
                desugar_expr(cond),
                desugar_expr(then_e),
                desugar_expr(else_e),
            ],
        ),

        Expr::Match(scrutinee, arms, _) => {
            let mut children = vec![desugar_expr(scrutinee)];
            for arm in arms {
                let guard = arm
                    .guard
                    .as_ref()
                    .map(desugar_expr)
                    .unwrap_or_else(|| bare_list(vec![]));
                children.push(node(
                    "arm",
                    vec![
                        desugar_pattern(&arm.pattern),
                        guard,
                        desugar_expr(&arm.body),
                    ],
                ));
            }
            node("match", children)
        }

        Expr::Let(bindings, body, _) => desugar_let_bindings(bindings, desugar_expr(body)),

        Expr::Lambda(params, body, _) => {
            let param_names: Vec<deep::Expr> = params.iter().map(desugar_param).collect();
            let params_node = node("params", param_names);
            node("fn", vec![params_node, desugar_expr(body)])
        }

        Expr::Tuple(elems, _) if elems.is_empty() => node_meta(
            "lit",
            meta_with_type(node("t-unit", vec![])),
            vec![bare_list(vec![])],
        ),
        Expr::Tuple(elems, _) => node("tuple", elems.iter().map(desugar_expr).collect()),

        Expr::Cast(e, prec, _) => node(
            "cast",
            vec![desugar_expr(e), node("t-prim", vec![sym(prec)])],
        ),

        Expr::Grad(f, _) => node("grad", vec![desugar_expr(f)]),

        Expr::Vmap(f, axis, _) => {
            let axis_node = node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym("int32")])),
                vec![deep::Expr::Atom(deep::Atom::Int(axis.unwrap_or(0)), sp())],
            );
            node("vmap", vec![desugar_expr(f), axis_node])
        }

        Expr::Jit(f, _) => node("jit", vec![desugar_expr(f)]),
        Expr::Realize(f, _) => node("realize", vec![desugar_expr(f)]),
        Expr::Copy(f, _) => node("copy", vec![desugar_expr(f)]),
        Expr::Par(exprs, _) => node("par", exprs.iter().map(desugar_expr).collect()),

        Expr::Annotate(e, ty, _) => {
            // Type annotation pushed into metadata of the desugared expression
            let desugared = desugar_expr(e);
            inject_type_metadata(desugared, desugar_type(ty))
        }

        Expr::Block(bindings, final_expr, _) => {
            if bindings.is_empty() {
                desugar_expr(final_expr)
            } else {
                desugar_let_bindings(bindings, desugar_expr(final_expr))
            }
        }
    }
}

fn tuple_index_expr(target: deep::Expr, index: i64) -> deep::Expr {
    node(
        "tuple-get",
        vec![
            target,
            node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym("int32")])),
                vec![deep::Expr::Atom(deep::Atom::Int(index), sp())],
            ),
        ],
    )
}

fn bind_name_value(name: &str, value: deep::Expr, body: deep::Expr) -> deep::Expr {
    let bind_node = node("bind", vec![sym(name), value]);
    node("let", vec![bind_node, body])
}

fn destructure_pattern(
    pattern: &LetPattern,
    source_name: &str,
    body: deep::Expr,
    next_tmp: &mut usize,
) -> deep::Expr {
    match pattern {
        LetPattern::Var(name, _) => bind_name_value(name, dvar(source_name), body),
        LetPattern::Wildcard(_) => body,
        LetPattern::Tuple(parts, _) => {
            let mut out = body;
            for (index, part) in parts.iter().enumerate().rev() {
                let tuple_value = tuple_index_expr(dvar(source_name), index as i64);
                let tmp_name = format!("__chelis_tmp{}", *next_tmp);
                *next_tmp += 1;
                out = destructure_pattern(part, &tmp_name, out, next_tmp);
                out = bind_name_value(&tmp_name, tuple_value, out);
            }
            out
        }
    }
}

fn desugar_let_bindings(bindings: &[LetBinding], body: deep::Expr) -> deep::Expr {
    let mut out = body;
    let mut next_tmp = 0usize;
    for binding in bindings.iter().rev() {
        match &binding.pattern {
            LetPattern::Var(name, _) => {
                let value = desugar_expr(&binding.value);
                if let Some(ty) = &binding.ty {
                    out = bind_name_value(name, inject_type_metadata(value, desugar_type(ty)), out);
                } else {
                    out = bind_name_value(name, value, out);
                }
            }
            pattern => {
                let temp_name = format!("__chelis_tmp{}", next_tmp);
                next_tmp += 1;
                out = destructure_pattern(pattern, &temp_name, out, &mut next_tmp);
                out = bind_name_value(&temp_name, desugar_expr(&binding.value), out);
            }
        }
    }
    out
}

fn desugar_literal(lit: &Literal) -> deep::Expr {
    match lit {
        Literal::Int(n) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("int32")])),
            vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
        ),
        Literal::Float(f) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("f32")])),
            vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
        ),
        Literal::Bool(b) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("bool")])),
            vec![deep::Expr::Atom(deep::Atom::Bool(*b), sp())],
        ),
        Literal::Str(s) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("string")])),
            vec![deep::Expr::Atom(deep::Atom::Str(s.clone()), sp())],
        ),
    }
}

fn desugar_apply(func: &Expr, args: &[Expr]) -> deep::Expr {
    // Flatten nested Apply chains
    let mut all_args = Vec::new();
    let base_func = collect_apply_chain(func, &mut all_args);
    for arg in args {
        all_args.push(desugar_expr(arg));
    }
    let mut children = vec![desugar_expr(base_func)];
    children.extend(all_args);
    node("app", children)
}

fn collect_apply_chain<'a>(expr: &'a Expr, args: &mut Vec<deep::Expr>) -> &'a Expr {
    match expr {
        Expr::Apply(inner_func, inner_args, _) => {
            let base = collect_apply_chain(inner_func, args);
            for arg in inner_args {
                args.push(desugar_expr(arg));
            }
            base
        }
        other => other,
    }
}

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Mod => "mod",
        BinOp::Eq => "eq",
        BinOp::Ne => "neq",
        BinOp::Lt => "cmplt",
        BinOp::Gt => "cmplt", // handled specially with swap
        BinOp::Le => "lte",
        BinOp::Ge => "gte",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

// ---------------------------------------------------------------------------
// Type Expressions
// ---------------------------------------------------------------------------

/// Desugar a type with no declared dim params (module-level context).
fn desugar_type(ty: &TypeExpr) -> deep::Expr {
    desugar_type_with_dims(ty, &HashSet::new())
}

/// Desugar a type with declared dimension parameters.
/// Names in `dim_vars` become d-var regardless of length.
fn desugar_type_with_dims(ty: &TypeExpr, dim_vars: &HashSet<String>) -> deep::Expr {
    match ty {
        TypeExpr::Named(name, _) => {
            if PRIMITIVES.contains(&name.as_str()) {
                node("t-prim", vec![sym(name)])
            } else if name.starts_with(|c: char| c.is_uppercase()) {
                node("t-adt", vec![sym(name)])
            } else {
                node("t-var", vec![sym(name)])
            }
        }

        TypeExpr::Tensor(dims, precision, _) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|d| match d {
                    TypeExpr::Named(n, _) if n == "*" => node("d-name", vec![sym("*")]),
                    // Declared dim param → always d-var (polymorphic)
                    TypeExpr::Named(n, _) if dim_vars.contains(n.as_str()) => {
                        node("d-var", vec![sym(n)])
                    }
                    // Single lowercase letter → d-var (heuristic fallback)
                    TypeExpr::Named(n, _)
                        if n.len() == 1 && n.starts_with(|c: char| c.is_lowercase()) =>
                    {
                        node("d-var", vec![sym(n)])
                    }
                    // Everything else → d-name (concrete)
                    TypeExpr::Named(n, _) => node("d-name", vec![sym(n)]),
                    _ => node("d-var", vec![desugar_type_with_dims(d, dim_vars)]),
                })
                .collect();
            children.push(node("t-prim", vec![sym(precision)]));
            node("t-tensor", children)
        }

        TypeExpr::Arrow(params, ret, _) => {
            let mut children: Vec<deep::Expr> = params
                .iter()
                .map(|p| desugar_type_with_dims(p, dim_vars))
                .collect();
            children.push(desugar_type_with_dims(ret, dim_vars));
            node("t-fn", children)
        }

        TypeExpr::App(name, args, _) => {
            let mut children = vec![sym(name)];
            children.extend(args.iter().map(|a| desugar_type_with_dims(a, dim_vars)));
            node("t-adt", children)
        }

        TypeExpr::Tuple(elems, _) if elems.is_empty() => node("t-unit", vec![]),
        TypeExpr::Tuple(elems, _) => node(
            "t-tuple",
            elems
                .iter()
                .map(|e| desugar_type_with_dims(e, dim_vars))
                .collect(),
        ),

        TypeExpr::Infer(_) => node("t-var", vec![sym("_")]),
    }
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

fn desugar_pattern(pat: &Pattern) -> deep::Expr {
    match pat {
        Pattern::Wildcard(_) => node("pat-wild", vec![]),
        Pattern::Var(name, _) => node("pat-var", vec![sym(name)]),
        Pattern::Lit(lit, _) => {
            // pat-lit contains the raw literal value, NOT a typed (lit ...) node
            let val = match lit {
                Literal::Int(n) => deep::Expr::Atom(deep::Atom::Int(*n), sp()),
                Literal::Float(f) => deep::Expr::Atom(deep::Atom::Float(*f), sp()),
                Literal::Bool(b) => deep::Expr::Atom(deep::Atom::Bool(*b), sp()),
                Literal::Str(s) => deep::Expr::Atom(deep::Atom::Str(s.clone()), sp()),
            };
            node("pat-lit", vec![val])
        }
        Pattern::Constructor(name, sub_pats, _) => {
            let mut children = vec![sym(name)];
            children.extend(sub_pats.iter().map(desugar_pattern));
            node("pat-ctor", children)
        }
        Pattern::Tuple(pats, _) => node("pat-tuple", pats.iter().map(desugar_pattern).collect()),
        Pattern::Record(name, fields, _) => {
            let mut fields = fields.clone();
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            let mut children = vec![sym(name)];
            for (field_name, field_pat) in fields {
                children.push(node(
                    "kv",
                    vec![sym(&field_name), desugar_pattern(&field_pat)],
                ));
            }
            node("pat-record", children)
        }
        Pattern::As(name, inner, _) => node("pat-as", vec![sym(name), desugar_pattern(inner)]),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::printer::print_expr;

    fn s() -> Span {
        Span::new(0, 0)
    }

    fn tvar(name: &str) -> Expr {
        Expr::Var(name.to_string(), s())
    }

    fn int_lit(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), s())
    }

    fn float_lit(f: f64) -> Expr {
        Expr::Lit(Literal::Float(f), s())
    }

    fn param(name: &str, ty: Option<TypeExpr>) -> Param {
        Param {
            name: name.to_string(),
            ty,
            span: s(),
        }
    }

    fn named_ty(name: &str) -> TypeExpr {
        TypeExpr::Named(name.to_string(), s())
    }

    /// Helper: desugar a decl and print all resulting nodes.
    fn desugar_decl_strs(decl: &Decl) -> Vec<String> {
        desugar_decl(decl).iter().map(print_expr).collect()
    }

    // --- Literals ---

    #[test]
    fn test_int_literal() {
        let result = print_expr(&desugar_expr(&int_lit(42)));
        assert_eq!(result, "(lit {type: (t-prim {} int32)} 42)");
    }

    #[test]
    fn test_float_literal() {
        let result = print_expr(&desugar_expr(&float_lit(3.125)));
        assert_eq!(result, "(lit {type: (t-prim {} f32)} 3.125)");
    }

    #[test]
    fn test_bool_literal() {
        let result = print_expr(&desugar_expr(&Expr::Lit(Literal::Bool(true), s())));
        assert_eq!(result, "(lit {type: (t-prim {} bool)} true)");
    }

    #[test]
    fn test_string_literal() {
        let result = print_expr(&desugar_expr(&Expr::Lit(
            Literal::Str("hello".to_string()),
            s(),
        )));
        assert_eq!(result, "(lit {type: (t-prim {} string)} \"hello\")");
    }

    // --- Variables ---

    #[test]
    fn test_var() {
        assert_eq!(print_expr(&desugar_expr(&tvar("x"))), "(var {} x)");
    }

    // --- Binary operators (all go through app) ---

    #[test]
    fn test_add() {
        let expr = Expr::Binary(BinOp::Add, Box::new(tvar("a")), Box::new(tvar("b")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} add) (var {} a) (var {} b))"
        );
    }

    #[test]
    fn test_gt_swaps_operands() {
        let expr = Expr::Binary(BinOp::Gt, Box::new(tvar("a")), Box::new(tvar("b")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} cmplt) (var {} b) (var {} a))"
        );
    }

    #[test]
    fn test_nested_binops() {
        // a + b * c
        let expr = Expr::Binary(
            BinOp::Add,
            Box::new(tvar("a")),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(tvar("b")),
                Box::new(tvar("c")),
                s(),
            )),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} add) (var {} a) (app {} (var {} mul) (var {} b) (var {} c)))"
        );
    }

    // --- Unary ---

    #[test]
    fn test_unary_neg() {
        let expr = Expr::Unary(UnaryOp::Neg, Box::new(tvar("a")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} neg) (var {} a))"
        );
    }

    // --- Function application ---

    #[test]
    fn test_apply() {
        let expr = Expr::Apply(Box::new(tvar("f")), vec![tvar("x"), tvar("y")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} f) (var {} x) (var {} y))"
        );
    }

    #[test]
    fn test_apply_chain_flattening() {
        let inner = Expr::Apply(Box::new(tvar("f")), vec![tvar("x")], s());
        let outer = Expr::Apply(Box::new(inner), vec![tvar("y")], s());
        assert_eq!(
            print_expr(&desugar_expr(&outer)),
            "(app {} (var {} f) (var {} x) (var {} y))"
        );
    }

    // --- Pipe ---

    #[test]
    fn test_pipe() {
        let expr = Expr::Pipe(Box::new(tvar("x")), vec![tvar("f"), tvar("g")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(pipe {} (var {} x) (var {} f) (var {} g))"
        );
    }

    // --- If ---

    #[test]
    fn test_if() {
        let expr = Expr::If(
            Box::new(tvar("a")),
            Box::new(tvar("b")),
            Box::new(tvar("c")),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(if {} (var {} a) (var {} b) (var {} c))"
        );
    }

    // --- Match ---

    #[test]
    fn test_match() {
        let expr = Expr::Match(
            Box::new(tvar("x")),
            vec![
                MatchArm {
                    pattern: Pattern::Constructor(
                        "Some".to_string(),
                        vec![Pattern::Var("y".to_string(), s())],
                        s(),
                    ),
                    guard: None,
                    body: tvar("y"),
                    span: s(),
                },
                MatchArm {
                    pattern: Pattern::Constructor("None".to_string(), vec![], s()),
                    guard: None,
                    body: int_lit(0),
                    span: s(),
                },
            ],
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert_eq!(
            result,
            "(match {}\n  (var {} x)\n  (arm {} (pat-ctor {} Some (pat-var {} y)) () (var {} y))\n  (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int32)} 0)))"
        );
    }

    // --- Let ---

    #[test]
    fn test_let() {
        let expr = Expr::Let(
            vec![LetBinding {
                pattern: LetPattern::Var("x".to_string(), s()),
                ty: None,
                value: int_lit(1),
            }],
            Box::new(tvar("x")),
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert_eq!(
            result,
            "(let {} (bind {} x (lit {type: (t-prim {} int32)} 1)) (var {} x))"
        );
    }

    #[test]
    fn test_let_tuple_destructuring() {
        let expr = Expr::Let(
            vec![LetBinding {
                pattern: LetPattern::Tuple(
                    vec![
                        LetPattern::Var("a".to_string(), s()),
                        LetPattern::Wildcard(s()),
                        LetPattern::Var("c".to_string(), s()),
                    ],
                    s(),
                ),
                ty: None,
                value: tvar("triple"),
            }],
            Box::new(tvar("c")),
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert!(result.contains("(bind {} __chelis_tmp0 (var {} triple))"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 0)"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 1)"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 2)"));
        assert_eq!(result.matches("(tuple-get {}").count(), 3);
        assert!(result.contains("(bind {} a (var {} __chelis_tmp"));
        assert!(result.contains("(bind {} c (var {} __chelis_tmp"));
        assert!(!result.contains("(bind {} _ "));
    }

    // --- Lambda ---

    #[test]
    fn test_lambda() {
        let expr = Expr::Lambda(vec![param("x", None)], Box::new(tvar("x")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(fn {} (params {} x) (var {} x))"
        );
    }

    // --- Fun def (no types) ---

    #[test]
    fn test_fun_def() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            dim_params: vec![],
            params: vec![param("x", None)],
            ret_ty: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(def {} f (fn {} (params {} x) (var {} x)))");
    }

    // --- Fun def (with types) produces defsig + def ---

    #[test]
    fn test_fun_def_typed() {
        // def f(x: f32): f32 = x
        let decl = Decl::FunDef {
            name: "f".to_string(),
            dim_params: vec![],
            params: vec![param("x", Some(named_ty("f32")))],
            ret_ty: Some(named_ty("f32")),
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0],
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))"
        );
        assert_eq!(
            nodes[1],
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))"
        );
    }

    // --- Fun def with dim params ---

    #[test]
    fn test_fun_def_with_dim_params() {
        // def transpose[batch, hidden](x: tensor[batch, hidden, f32]): tensor[hidden, batch, f32] = x
        // batch and hidden should be d-var (polymorphic), NOT d-name
        let decl = Decl::FunDef {
            name: "transpose".to_string(),
            dim_params: vec!["batch".to_string(), "hidden".to_string()],
            params: vec![param(
                "x",
                Some(TypeExpr::Tensor(
                    vec![named_ty("batch"), named_ty("hidden")],
                    "f32".to_string(),
                    s(),
                )),
            )],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("hidden"), named_ty("batch")],
                "f32".to_string(),
                s(),
            )),
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2); // defsig + def, NO defdim
        // defsig must use d-var for batch and hidden (declared dim params)
        assert!(
            nodes[0].contains("(d-var {} batch)"),
            "expected d-var for 'batch' (declared dim param), got:\n{}",
            nodes[0]
        );
        assert!(
            nodes[0].contains("(d-var {} hidden)"),
            "expected d-var for 'hidden' (declared dim param), got:\n{}",
            nodes[0]
        );
        // Must NOT contain defdim (those are module-level)
        for n in &nodes {
            assert!(
                !n.contains("defdim"),
                "function dim params should NOT emit defdim, got:\n{n}"
            );
        }
    }

    // --- Let def (with type) produces defsig + def ---

    #[test]
    fn test_let_def_typed() {
        // let x: f32 = 1.0
        let decl = Decl::LetDef {
            name: "x".to_string(),
            ty: Some(named_ty("f32")),
            value: float_lit(1.0),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0], "(defsig {} x (t-prim {} f32))");
        assert_eq!(nodes[1], "(def {} x (lit {type: (t-prim {} f32)} 1.0))");
    }

    // --- Let def (no type) produces just def ---

    #[test]
    fn test_let_def_untyped() {
        let decl = Decl::LetDef {
            name: "x".to_string(),
            ty: None,
            value: int_lit(42),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(def {} x (lit {type: (t-prim {} int32)} 42))");
    }

    // --- Annotate preserves type in metadata ---

    #[test]
    fn test_annotate_var() {
        // x : f32
        let expr = Expr::Annotate(Box::new(tvar("x")), named_ty("f32"), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(var {type: (t-prim {} f32)} x)"
        );
    }

    // --- Transforms (tags, not app) ---

    #[test]
    fn test_grad() {
        let expr = Expr::Grad(Box::new(tvar("f")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(grad {} (var {} f))");
    }

    #[test]
    fn test_cast() {
        let expr = Expr::Cast(Box::new(tvar("x")), "bf16".to_string(), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {} (var {} x) (t-prim {} bf16))"
        );
    }

    #[test]
    fn test_jit() {
        let expr = Expr::Jit(Box::new(tvar("f")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(jit {} (var {} f))");
    }

    // --- Type expressions ---

    #[test]
    fn test_type_prim() {
        assert_eq!(
            print_expr(&desugar_type(&named_ty("f32"))),
            "(t-prim {} f32)"
        );
    }

    #[test]
    fn test_type_var() {
        // Lowercase non-primitive → t-var
        assert_eq!(print_expr(&desugar_type(&named_ty("a"))), "(t-var {} a)");
    }

    #[test]
    fn test_uppercase_named_is_adt() {
        // Uppercase non-primitive → t-adt (concrete ADT, zero args)
        assert_eq!(
            print_expr(&desugar_type(&named_ty("Activation"))),
            "(t-adt {} Activation)"
        );
        assert_eq!(
            print_expr(&desugar_type(&named_ty("MyType"))),
            "(t-adt {} MyType)"
        );
    }

    #[test]
    fn test_type_tensor() {
        let ty = TypeExpr::Tensor(
            vec![named_ty("batch"), named_ty("hidden")],
            "f32".to_string(),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_arrow() {
        let ty = TypeExpr::Arrow(
            vec![named_ty("f32"), named_ty("f32")],
            Box::new(named_ty("f32")),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_adt() {
        let ty = TypeExpr::App("Option".to_string(), vec![named_ty("f32")], s());
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-adt {} Option (t-prim {} f32))"
        );
    }

    // --- Type def with type variable ---

    #[test]
    fn test_type_def() {
        let decl = Decl::TypeDef {
            name: "Option".to_string(),
            params: vec!["a".to_string()],
            variants: vec![
                Variant {
                    name: "Some".to_string(),
                    fields: VariantFields::Positional(vec![named_ty("a")]),
                    span: s(),
                },
                Variant {
                    name: "None".to_string(),
                    fields: VariantFields::Positional(vec![]),
                    span: s(),
                },
            ],
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0],
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))"
        );
    }

    #[test]
    fn test_record_variant() {
        let decl = Decl::TypeDef {
            name: "T".to_string(),
            params: vec![],
            variants: vec![Variant {
                name: "V".to_string(),
                fields: VariantFields::Record(vec![("x".to_string(), named_ty("f32"))]),
                span: s(),
            }],
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0],
            "(deftype {} T () (variant {} V (field {} x (t-prim {} f32))))"
        );
    }

    // --- Tuple ---

    #[test]
    fn test_tuple() {
        let expr = Expr::Tuple(vec![tvar("a"), tvar("b")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(tuple {} (var {} a) (var {} b))"
        );
    }

    // --- Patterns ---

    #[test]
    fn test_pat_lit_int() {
        let pat = Pattern::Lit(Literal::Int(42), s());
        assert_eq!(print_expr(&desugar_pattern(&pat)), "(pat-lit {} 42)");
    }

    #[test]
    fn test_pat_tuple() {
        let pat = Pattern::Tuple(
            vec![
                Pattern::Var("a".to_string(), s()),
                Pattern::Var("b".to_string(), s()),
            ],
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-tuple {} (pat-var {} a) (pat-var {} b))"
        );
    }

    // --- Import ---

    #[test]
    fn test_import_all() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            kind: ImportKind::All,
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(import-all {} foo)");
    }

    #[test]
    fn test_import_selective() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            kind: ImportKind::Names(vec!["a".to_string(), "b".to_string()]),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(import {} foo (a b))");
    }

    // --- Record pattern ---

    #[test]
    fn test_pat_record() {
        let pat = Pattern::Record(
            "Adam".to_string(),
            vec![
                ("lr".to_string(), Pattern::Var("lr".to_string(), s())),
                ("eps".to_string(), Pattern::Var("eps".to_string(), s())),
            ],
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-record {} Adam (kv {} eps (pat-var {} eps)) (kv {} lr (pat-var {} lr)))"
        );
    }

    // --- As pattern ---

    #[test]
    fn test_pat_as() {
        let pat = Pattern::As(
            "y".to_string(),
            Box::new(Pattern::Constructor(
                "Some".to_string(),
                vec![Pattern::Var("z".to_string(), s())],
                s(),
            )),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-as {} y (pat-ctor {} Some (pat-var {} z)))"
        );
    }
}
