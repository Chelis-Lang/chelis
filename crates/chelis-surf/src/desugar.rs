//! Desugars Surface AST into Deep (s-expression) AST.
//!
//! Every Deep node is a 3-tuple: (tag {} children...)
//! where {} is an inline metadata map.

use chelis_deep::Span;
use chelis_deep::ast as deep;

use crate::ast::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn desugar_program(decls: &[Decl]) -> Vec<deep::Expr> {
    decls.iter().map(desugar_decl).collect()
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

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

fn desugar_decl(decl: &Decl) -> deep::Expr {
    match decl {
        Decl::FunDef {
            name,
            params,
            ret_ty,
            body,
            ..
        } => desugar_fun_def(name, params, ret_ty, body),

        Decl::LetDef { name, value, .. } => {
            // Top-level let: (def {} name value')
            node("def", vec![sym(name), desugar_expr(value)])
        }

        Decl::TypeDef {
            name,
            params,
            variants,
            ..
        } => desugar_type_def(name, params, variants),

        Decl::Module { name, decls, .. } => {
            let mut children = vec![sym(name)];
            for d in decls {
                children.push(desugar_decl(d));
            }
            node("module", children)
        }

        Decl::Import { module, names, .. } => match names {
            Some(ns) => {
                let name_list = bare_list(ns.iter().map(|n| sym(n)).collect());
                node("import", vec![sym(module), name_list])
            }
            None => node("import-all", vec![sym(module)]),
        },
    }
}

fn desugar_fun_def(
    name: &str,
    params: &[Param],
    ret_ty: &Option<TypeExpr>,
    body: &Expr,
) -> deep::Expr {
    // If type annotations exist, emit defsig first then def
    // For Phase 0, combine into single def with fn

    // Build params helper (no tag/meta — structural helper)
    let param_names: Vec<deep::Expr> = params.iter().map(|p| sym(&p.name)).collect();
    let mut params_elements = vec![sym("params")];
    params_elements.extend(param_names);
    let params_node = bare_list(params_elements);

    let fn_node = node("fn", vec![params_node, desugar_expr(body)]);

    // If we have type annotations, also emit defsig
    if params.iter().any(|p| p.ty.is_some()) || ret_ty.is_some() {
        let mut type_parts: Vec<deep::Expr> = params
            .iter()
            .map(|p| match &p.ty {
                Some(ty) => desugar_type(ty),
                None => node("t-var", vec![sym("_")]),
            })
            .collect();
        type_parts.push(match ret_ty {
            Some(ty) => desugar_type(ty),
            None => node("t-var", vec![sym("_")]),
        });
        let _sig = node("defsig", vec![sym(name), node("t-fn", type_parts)]);
        // For now, just emit the def (defsig handling is for later)
    }

    node("def", vec![sym(name), fn_node])
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

        Expr::Apply(func, args, _) => desugar_apply(func, args),

        Expr::Binary(op, lhs, rhs, _) => {
            let op_name = binop_name(*op);
            // a > b → (app {} (var {} cmplt) b' a') — swap operands
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
                let guard = bare_list(vec![]); // empty guard
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

        Expr::Let(bindings, body, _) => {
            // (let {} (bind name1 expr1 name2 expr2 ...) body)
            let mut bind_elements = vec![sym("bind")];
            for b in bindings {
                bind_elements.push(sym(&b.name));
                bind_elements.push(desugar_expr(&b.value));
            }
            let bind_node = bare_list(bind_elements);
            node("let", vec![bind_node, desugar_expr(body)])
        }

        Expr::Lambda(params, body, _) => {
            let mut params_elements = vec![sym("params")];
            params_elements.extend(params.iter().map(|p| sym(&p.name)));
            let params_node = bare_list(params_elements);
            node("fn", vec![params_node, desugar_expr(body)])
        }

        Expr::Tuple(elems, _) => node("tuple", elems.iter().map(desugar_expr).collect()),

        Expr::Cast(e, prec, _) => node(
            "cast",
            vec![desugar_expr(e), node("t-prim", vec![sym(prec)])],
        ),

        Expr::Grad(f, _) => node("grad", vec![desugar_expr(f)]),

        Expr::Vmap(f, axis, _) => {
            let axis_node = node(
                "d-var",
                vec![deep::Expr::Atom(deep::Atom::Int(axis.unwrap_or(0)), sp())],
            );
            node("vmap", vec![desugar_expr(f), axis_node])
        }

        Expr::Jit(f, _) => node("jit", vec![desugar_expr(f)]),

        Expr::Annotate(e, _ty, _) => {
            // Type annotation pushed into metadata of the desugared expression
            // For now, just desugar the inner expression
            desugar_expr(e)
        }

        Expr::Block(decls, final_expr, _) => {
            let mut children: Vec<deep::Expr> = decls.iter().map(desugar_decl).collect();
            children.push(desugar_expr(final_expr));
            node("block", children)
        }
    }
}

fn desugar_literal(lit: &Literal) -> deep::Expr {
    match lit {
        Literal::Int(n) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("int64")])),
            vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
        ),
        Literal::Float(f) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("f64")])),
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

fn desugar_type(ty: &TypeExpr) -> deep::Expr {
    match ty {
        TypeExpr::Named(name, _) => node("t-prim", vec![sym(name)]),

        TypeExpr::Tensor(dims, precision, _) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|d| match d {
                    TypeExpr::Named(n, _) => node("d-name", vec![sym(n)]),
                    _ => node("d-var", vec![desugar_type(d)]),
                })
                .collect();
            children.push(node("t-prim", vec![sym(precision)]));
            node("t-tensor", children)
        }

        TypeExpr::Arrow(params, ret, _) => {
            let mut children: Vec<deep::Expr> = params.iter().map(desugar_type).collect();
            children.push(desugar_type(ret));
            node("t-fn", children)
        }

        TypeExpr::App(name, args, _) => {
            let mut children = vec![sym(name)];
            children.extend(args.iter().map(desugar_type));
            node("t-adt", children)
        }

        TypeExpr::Tuple(elems, _) => node("t-tuple", elems.iter().map(desugar_type).collect()),

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
        Pattern::Lit(lit, _) => node("pat-lit", vec![desugar_literal(lit)]),
        Pattern::Constructor(name, sub_pats, _) => {
            let mut children = vec![sym(name)];
            children.extend(sub_pats.iter().map(desugar_pattern));
            node("pat-ctor", children)
        }
        Pattern::Tuple(pats, _) => node("t-tuple", pats.iter().map(desugar_pattern).collect()),
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

    // --- Literals ---

    #[test]
    fn test_int_literal() {
        let result = print_expr(&desugar_expr(&int_lit(42)));
        assert_eq!(result, "(lit {type: (t-prim {} int64)} 42)");
    }

    #[test]
    fn test_float_literal() {
        let result = print_expr(&desugar_expr(&float_lit(3.125)));
        assert_eq!(result, "(lit {type: (t-prim {} f64)} 3.125)");
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
                    body: tvar("y"),
                    span: s(),
                },
                MatchArm {
                    pattern: Pattern::Constructor("None".to_string(), vec![], s()),
                    body: int_lit(0),
                    span: s(),
                },
            ],
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert!(result.contains("(match {}"));
        assert!(result.contains("(arm {}"));
        assert!(result.contains("(pat-ctor {} Some"));
        assert!(result.contains("(pat-var {} y)"));
    }

    // --- Let ---

    #[test]
    fn test_let() {
        let expr = Expr::Let(
            vec![LetBinding {
                name: "x".to_string(),
                ty: None,
                value: int_lit(1),
            }],
            Box::new(tvar("x")),
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert!(result.contains("(let {}"));
        assert!(result.contains("(bind x"));
        assert!(result.contains("(var {} x)"));
    }

    // --- Lambda ---

    #[test]
    fn test_lambda() {
        let expr = Expr::Lambda(vec![param("x", None)], Box::new(tvar("x")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(fn {} (params x) (var {} x))"
        );
    }

    // --- Fun def ---

    #[test]
    fn test_fun_def() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            params: vec![param("x", None)],
            ret_ty: None,
            body: tvar("x"),
            span: s(),
        };
        let result = print_expr(&desugar_decl(&decl));
        assert_eq!(result, "(def {} f (fn {} (params x) (var {} x)))");
    }

    // --- Transforms (tags, not app) ---

    #[test]
    fn test_grad() {
        let expr = Expr::Grad(Box::new(tvar("loss")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(grad {} (var {} loss))");
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

    // --- Type def ---

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
        let result = print_expr(&desugar_decl(&decl));
        assert!(result.contains("(deftype {}"));
        assert!(result.contains("(variant {} Some"));
        assert!(result.contains("(variant {} None)"));
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
        let result = print_expr(&desugar_decl(&decl));
        assert!(result.contains("(field {} x (t-prim {} f32))"));
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

    // --- Import ---

    #[test]
    fn test_import_all() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            names: None,
            span: s(),
        };
        assert_eq!(print_expr(&desugar_decl(&decl)), "(import-all {} Foo)");
    }

    #[test]
    fn test_import_selective() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            names: Some(vec!["a".to_string(), "b".to_string()]),
            span: s(),
        };
        assert_eq!(print_expr(&desugar_decl(&decl)), "(import {} Foo (a b))");
    }
}
