//! Desugars Surface AST into Deep (s-expression) AST.

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
// Helpers for building Deep AST nodes
// ---------------------------------------------------------------------------

fn sp() -> Span {
    Span::new(0, 0)
}

fn sym(s: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(s.to_string()), sp())
}

fn int(n: i64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Int(n), sp())
}

fn float(f: f64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Float(f), sp())
}

fn str_lit(s: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Str(s.to_string()), sp())
}

fn bool_lit(b: bool) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Bool(b), sp())
}

fn keyword(k: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Keyword(k.to_string()), sp())
}

fn list(elements: Vec<deep::Expr>) -> deep::Expr {
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

        Decl::LetDef {
            name, ty, value, ..
        } => desugar_let_def(name, ty, value),

        Decl::TypeDef {
            name,
            params,
            variants,
            ..
        } => desugar_type_def(name, params, variants),

        Decl::Module { name, decls, .. } => {
            let mut elements = vec![sym("module"), sym(name)];
            for d in decls {
                elements.push(desugar_decl(d));
            }
            list(elements)
        }

        Decl::Import { module, names, .. } => {
            let mut elements = vec![sym("import"), sym(module)];
            if let Some(names) = names {
                let name_syms: Vec<deep::Expr> = names.iter().map(|n| sym(n)).collect();
                elements.push(list(name_syms));
            }
            list(elements)
        }
    }
}

fn desugar_fun_def(
    name: &str,
    params: &[Param],
    ret_ty: &Option<TypeExpr>,
    body: &Expr,
) -> deep::Expr {
    let name_sym = sym(name);

    // Build sig
    let sig = if params.iter().any(|p| p.ty.is_some()) || ret_ty.is_some() {
        let mut arrow_parts = vec![sym("->")];
        for p in params {
            arrow_parts.push(match &p.ty {
                Some(ty) => desugar_type(ty),
                None => sym("_"),
            });
        }
        arrow_parts.push(match ret_ty {
            Some(ty) => desugar_type(ty),
            None => sym("_"),
        });
        list(vec![sym("sig"), list(arrow_parts)])
    } else {
        list(vec![sym("sig"), sym("_")])
    };

    // Build fn body
    let param_names: Vec<deep::Expr> = params.iter().map(|p| sym(&p.name)).collect();
    let fn_form = list(vec![sym("fn"), list(param_names), desugar_expr(body)]);

    list(vec![sym("def"), name_sym, sig, fn_form])
}

fn desugar_let_def(name: &str, ty: &Option<TypeExpr>, value: &Expr) -> deep::Expr {
    let sig = match ty {
        Some(t) => list(vec![sym("sig"), desugar_type(t)]),
        None => list(vec![sym("sig"), sym("_")]),
    };
    list(vec![sym("def"), sym(name), sig, desugar_expr(value)])
}

fn desugar_type_def(name: &str, params: &[String], variants: &[Variant]) -> deep::Expr {
    let param_list = list(params.iter().map(|p| sym(p)).collect());
    let variant_exprs: Vec<deep::Expr> = variants.iter().map(desugar_variant).collect();
    list(vec![
        sym("type"),
        sym(name),
        param_list,
        list(variant_exprs),
    ])
}

fn desugar_variant(variant: &Variant) -> deep::Expr {
    match &variant.fields {
        VariantFields::Positional(fields) if fields.is_empty() => sym(&variant.name),
        VariantFields::Positional(fields) => {
            let mut elements = vec![sym(&variant.name)];
            for f in fields {
                elements.push(desugar_type(f));
            }
            list(elements)
        }
        VariantFields::Record(fields) => {
            let mut record_elements = vec![sym("record")];
            for (field_name, field_ty) in fields {
                record_elements.push(list(vec![sym(field_name), desugar_type(field_ty)]));
            }
            list(vec![sym(&variant.name), list(record_elements)])
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn desugar_expr(expr: &Expr) -> deep::Expr {
    match expr {
        Expr::Lit(lit, _) => desugar_literal(lit),
        Expr::Var(name, _) => sym(name),
        Expr::Constructor(name, _) => sym(name),

        Expr::Apply(func, args, _) => desugar_apply(func, args),

        Expr::Binary(op, lhs, rhs, _) => {
            let op_name = binop_name(*op);
            list(vec![sym(op_name), desugar_expr(lhs), desugar_expr(rhs)])
        }

        Expr::Unary(op, operand, _) => {
            let op_name = unaryop_name(*op);
            list(vec![sym(op_name), desugar_expr(operand)])
        }

        Expr::Pipe(head, stages, _) => {
            let mut elements = vec![sym("pipe"), desugar_expr(head)];
            for s in stages {
                elements.push(desugar_expr(s));
            }
            list(elements)
        }

        Expr::If(cond, then_branch, else_branch, _) => list(vec![
            sym("if"),
            desugar_expr(cond),
            desugar_expr(then_branch),
            desugar_expr(else_branch),
        ]),

        Expr::Match(scrutinee, arms, _) => {
            let mut elements = vec![sym("match"), desugar_expr(scrutinee)];
            for arm in arms {
                elements.push(list(vec![
                    sym("case"),
                    desugar_pattern(&arm.pattern),
                    desugar_expr(&arm.body),
                ]));
            }
            list(elements)
        }

        Expr::Let(bindings, body, _) => {
            let binding_exprs: Vec<deep::Expr> = bindings
                .iter()
                .map(|b| list(vec![sym(&b.name), desugar_expr(&b.value)]))
                .collect();
            list(vec![sym("let"), list(binding_exprs), desugar_expr(body)])
        }

        Expr::Lambda(params, body, _) => {
            let param_names: Vec<deep::Expr> = params.iter().map(|p| sym(&p.name)).collect();
            list(vec![sym("fn"), list(param_names), desugar_expr(body)])
        }

        Expr::Tuple(elements, _) => {
            let mut parts = vec![sym("tuple")];
            for e in elements {
                parts.push(desugar_expr(e));
            }
            list(parts)
        }

        Expr::Cast(expr, target_ty, _) => {
            list(vec![sym("cast"), desugar_expr(expr), sym(target_ty)])
        }

        Expr::Grad(expr, _) => list(vec![sym("grad"), desugar_expr(expr)]),

        Expr::Vmap(expr, axis, _) => {
            let axis_val = axis.unwrap_or(0);
            list(vec![
                sym("vmap"),
                desugar_expr(expr),
                keyword("axis"),
                int(axis_val),
            ])
        }

        Expr::Jit(expr, _) => list(vec![sym("jit"), desugar_expr(expr)]),

        Expr::Annotate(expr, ty, _) => list(vec![sym(":"), desugar_expr(expr), desugar_type(ty)]),

        Expr::Block(decls, final_expr, _) => {
            let mut elements = vec![sym("block")];
            for d in decls {
                elements.push(desugar_decl(d));
            }
            elements.push(desugar_expr(final_expr));
            list(elements)
        }
    }
}

fn desugar_literal(lit: &Literal) -> deep::Expr {
    match lit {
        Literal::Int(n) => int(*n),
        Literal::Float(f) => float(*f),
        Literal::Str(s) => str_lit(s),
        Literal::Bool(b) => bool_lit(*b),
    }
}

fn desugar_apply(func: &Expr, args: &[Expr]) -> deep::Expr {
    // Flatten nested Apply chains: Apply(Apply(f, [x]), [y]) -> (f x y)
    let mut all_args = Vec::new();
    let base_func = collect_apply_chain(func, &mut all_args);
    for arg in args {
        all_args.push(desugar_expr(arg));
    }
    let mut elements = vec![desugar_expr(base_func)];
    elements.extend(all_args);
    list(elements)
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
        BinOp::Ne => "ne",
        BinOp::Lt => "lt",
        BinOp::Gt => "gt",
        BinOp::Le => "le",
        BinOp::Ge => "ge",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

fn unaryop_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Not => "not",
    }
}

// ---------------------------------------------------------------------------
// Type Expressions
// ---------------------------------------------------------------------------

fn desugar_type(ty: &TypeExpr) -> deep::Expr {
    match ty {
        TypeExpr::Named(name, _) => sym(name),

        TypeExpr::Tensor(dims, precision, _) => {
            let mut elements = vec![sym("tensor")];
            for d in dims {
                elements.push(list(vec![sym("dim"), desugar_type(d)]));
            }
            elements.push(sym(precision));
            list(elements)
        }

        TypeExpr::Arrow(params, ret, _) => {
            let mut elements = vec![sym("->")];
            for p in params {
                elements.push(desugar_type(p));
            }
            elements.push(desugar_type(ret));
            list(elements)
        }

        TypeExpr::App(name, args, _) => {
            let mut elements = vec![sym("adt"), sym(name)];
            for a in args {
                elements.push(desugar_type(a));
            }
            list(elements)
        }

        TypeExpr::Tuple(elements, _) => {
            let mut parts = vec![sym("tuple_type")];
            for e in elements {
                parts.push(desugar_type(e));
            }
            list(parts)
        }

        TypeExpr::Infer(_) => sym("_"),
    }
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

fn desugar_pattern(pat: &Pattern) -> deep::Expr {
    match pat {
        Pattern::Wildcard(_) => sym("_"),
        Pattern::Var(name, _) => sym(name),
        Pattern::Lit(lit, _) => desugar_literal(lit),
        Pattern::Constructor(name, args, _) => {
            if args.is_empty() {
                sym(name)
            } else {
                let mut elements = vec![sym(name)];
                for a in args {
                    elements.push(desugar_pattern(a));
                }
                list(elements)
            }
        }
        Pattern::Tuple(elements, _) => {
            let mut parts = vec![sym("tuple")];
            for e in elements {
                parts.push(desugar_pattern(e));
            }
            list(parts)
        }
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

    fn var(name: &str) -> Expr {
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

    // --- Fun def ---

    #[test]
    fn test_fun_def_typed() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            params: vec![param("x", Some(named_ty("f32")))],
            ret_ty: Some(named_ty("f32")),
            body: var("x"),
            span: s(),
        };
        let result = print_expr(&desugar_decl(&decl));
        assert_eq!(result, "(def f (sig (-> f32 f32)) (fn (x) x))");
    }

    #[test]
    fn test_fun_def_untyped() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            params: vec![param("x", None)],
            ret_ty: None,
            body: var("x"),
            span: s(),
        };
        let result = print_expr(&desugar_decl(&decl));
        assert_eq!(result, "(def f (sig _) (fn (x) x))");
    }

    // --- Binary operators ---

    #[test]
    fn test_add_mul_precedence() {
        // a + b * c  (parser would give us Binary(Add, a, Binary(Mul, b, c)))
        let expr = Expr::Binary(
            BinOp::Add,
            Box::new(var("a")),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(var("b")),
                Box::new(var("c")),
                s(),
            )),
            s(),
        );
        assert_eq!(print_expr(&desugar_expr(&expr)), "(add a (mul b c))");
    }

    // --- Pipe ---

    #[test]
    fn test_pipe() {
        let expr = Expr::Pipe(Box::new(var("x")), vec![var("f"), var("g")], s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(pipe x f g)");
    }

    // --- Let expression ---

    #[test]
    fn test_let_expr() {
        let expr = Expr::Let(
            vec![LetBinding {
                name: "x".to_string(),
                ty: None,
                value: int_lit(1),
            }],
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(var("x")),
                Box::new(int_lit(1)),
                s(),
            )),
            s(),
        );
        assert_eq!(print_expr(&desugar_expr(&expr)), "(let ((x 1)) (add x 1))");
    }

    // --- Lambda ---

    #[test]
    fn test_lambda() {
        let expr = Expr::Lambda(vec![param("x", None)], Box::new(var("x")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(fn (x) x)");
    }

    // --- If ---

    #[test]
    fn test_if() {
        let expr = Expr::If(
            Box::new(var("a")),
            Box::new(var("b")),
            Box::new(var("c")),
            s(),
        );
        assert_eq!(print_expr(&desugar_expr(&expr)), "(if a b c)");
    }

    // --- Match ---

    #[test]
    fn test_match() {
        let expr = Expr::Match(
            Box::new(var("x")),
            vec![
                MatchArm {
                    pattern: Pattern::Constructor(
                        "Some".to_string(),
                        vec![Pattern::Var("y".to_string(), s())],
                        s(),
                    ),
                    body: var("y"),
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
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(match x (case (Some y) y) (case None 0))"
        );
    }

    // --- Type def ---

    #[test]
    fn test_type_def_adt() {
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
        assert_eq!(
            print_expr(&desugar_decl(&decl)),
            "(type Option (a) ((Some a) None))"
        );
    }

    #[test]
    fn test_type_def_record() {
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
        assert_eq!(
            print_expr(&desugar_decl(&decl)),
            "(type T () ((V (record (x f32)))))"
        );
    }

    // --- Function application ---

    #[test]
    fn test_apply() {
        let expr = Expr::Apply(Box::new(var("f")), vec![var("x"), var("y")], s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(f x y)");
    }

    #[test]
    fn test_apply_chain_flattening() {
        // Apply(Apply(f, [x]), [y]) -> (f x y)
        let inner = Expr::Apply(Box::new(var("f")), vec![var("x")], s());
        let outer = Expr::Apply(Box::new(inner), vec![var("y")], s());
        assert_eq!(print_expr(&desugar_expr(&outer)), "(f x y)");
    }

    // --- Cast ---

    #[test]
    fn test_cast() {
        let expr = Expr::Cast(Box::new(var("x")), "f64".to_string(), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(cast x f64)");
    }

    // --- Grad ---

    #[test]
    fn test_grad() {
        let expr = Expr::Grad(Box::new(var("loss")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(grad loss)");
    }

    // --- Vmap ---

    #[test]
    fn test_vmap_default_axis() {
        let expr = Expr::Vmap(Box::new(var("f")), None, s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(vmap f :axis 0)");
    }

    #[test]
    fn test_vmap_with_axis() {
        let expr = Expr::Vmap(Box::new(var("f")), Some(1), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(vmap f :axis 1)");
    }

    // --- Tensor type ---

    #[test]
    fn test_tensor_type() {
        let ty = TypeExpr::Tensor(
            vec![named_ty("batch"), named_ty("hidden")],
            "f32".to_string(),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(tensor (dim batch) (dim hidden) f32)"
        );
    }

    // --- Arrow type ---

    #[test]
    fn test_arrow_type() {
        let ty = TypeExpr::Arrow(
            vec![named_ty("f32"), named_ty("f32")],
            Box::new(named_ty("f32")),
            s(),
        );
        assert_eq!(print_expr(&desugar_type(&ty)), "(-> f32 f32 f32)");
    }

    // --- Import ---

    #[test]
    fn test_import_simple() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            names: None,
            span: s(),
        };
        assert_eq!(print_expr(&desugar_decl(&decl)), "(import Foo)");
    }

    #[test]
    fn test_import_with_names() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            names: Some(vec!["a".to_string(), "b".to_string()]),
            span: s(),
        };
        assert_eq!(print_expr(&desugar_decl(&decl)), "(import Foo (a b))");
    }

    // --- Tuple ---

    #[test]
    fn test_tuple() {
        let expr = Expr::Tuple(vec![var("a"), var("b"), var("c")], s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(tuple a b c)");
    }

    // --- Annotate ---

    #[test]
    fn test_annotate() {
        let expr = Expr::Annotate(Box::new(var("x")), named_ty("f32"), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(: x f32)");
    }

    // --- Literals ---

    #[test]
    fn test_literals() {
        assert_eq!(print_expr(&desugar_expr(&int_lit(42))), "42");
        assert_eq!(print_expr(&desugar_expr(&float_lit(3.125))), "3.125");
        assert_eq!(
            print_expr(&desugar_expr(&Expr::Lit(
                Literal::Str("hello".to_string()),
                s()
            ))),
            "\"hello\""
        );
        assert_eq!(
            print_expr(&desugar_expr(&Expr::Lit(Literal::Bool(true), s()))),
            "true"
        );
    }

    // --- Unary ---

    #[test]
    fn test_unary_neg() {
        let expr = Expr::Unary(UnaryOp::Neg, Box::new(var("a")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(neg a)");
    }

    #[test]
    fn test_unary_not() {
        let expr = Expr::Unary(UnaryOp::Not, Box::new(var("a")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(not a)");
    }

    // --- Let def (top-level) ---

    #[test]
    fn test_let_def() {
        let decl = Decl::LetDef {
            name: "x".to_string(),
            ty: None,
            value: int_lit(42),
            span: s(),
        };
        assert_eq!(print_expr(&desugar_decl(&decl)), "(def x (sig _) 42)");
    }

    // --- Module ---

    #[test]
    fn test_module() {
        let decl = Decl::Module {
            name: "M".to_string(),
            decls: vec![Decl::LetDef {
                name: "x".to_string(),
                ty: None,
                value: int_lit(1),
                span: s(),
            }],
            span: s(),
        };
        assert_eq!(
            print_expr(&desugar_decl(&decl)),
            "(module M (def x (sig _) 1))"
        );
    }

    // --- Jit ---

    #[test]
    fn test_jit() {
        let expr = Expr::Jit(Box::new(var("f")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(jit f)");
    }

    // --- Type App ---

    #[test]
    fn test_type_app() {
        let ty = TypeExpr::App("Option".to_string(), vec![named_ty("f32")], s());
        assert_eq!(print_expr(&desugar_type(&ty)), "(adt Option f32)");
    }

    // --- Tuple type ---

    #[test]
    fn test_tuple_type() {
        let ty = TypeExpr::Tuple(vec![named_ty("f32"), named_ty("f32")], s());
        assert_eq!(print_expr(&desugar_type(&ty)), "(tuple_type f32 f32)");
    }

    // --- Infer ---

    #[test]
    fn test_infer_type() {
        let ty = TypeExpr::Infer(s());
        assert_eq!(print_expr(&desugar_type(&ty)), "_");
    }

    // --- Wildcard pattern ---

    #[test]
    fn test_pattern_wildcard() {
        let pat = Pattern::Wildcard(s());
        assert_eq!(print_expr(&desugar_pattern(&pat)), "_");
    }

    // --- Tuple pattern ---

    #[test]
    fn test_pattern_tuple() {
        let pat = Pattern::Tuple(
            vec![
                Pattern::Var("a".to_string(), s()),
                Pattern::Var("b".to_string(), s()),
            ],
            s(),
        );
        assert_eq!(print_expr(&desugar_pattern(&pat)), "(tuple a b)");
    }

    // --- Block ---

    #[test]
    fn test_block() {
        let expr = Expr::Block(
            vec![Decl::LetDef {
                name: "x".to_string(),
                ty: None,
                value: int_lit(1),
                span: s(),
            }],
            Box::new(var("x")),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(block (def x (sig _) 1) x)"
        );
    }
}
