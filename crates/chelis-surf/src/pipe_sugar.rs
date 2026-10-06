//! Surf-to-Surf pipe normalization, before any dtype or callable inference.
use crate::ast::*;
use chelis_deep::Span;

/// Visit every expression, including macros, invariants and property options.
/// The post-order visitor never crosses a binder by substitution.
pub(crate) fn visit_expr_mut(expr: &mut Expr, visit: &mut impl FnMut(&mut Expr)) {
    match expr {
        Expr::Lit(..) | Expr::Var(..) | Expr::Constructor(..) => {}
        Expr::List(items, _) | Expr::Tuple(items, _) | Expr::Par(items, _) | Expr::Do(items, _) => {
            for item in items {
                visit_expr_mut(item, visit);
            }
        }
        Expr::Apply(head, args, _) => {
            visit_expr_mut(head, visit);
            for arg in args {
                visit_expr_mut(arg, visit);
            }
        }
        Expr::Record(_, fields, _) => {
            for (_, value) in fields {
                visit_expr_mut(value, visit);
            }
        }
        Expr::RecordUpdate(base, fields, _) => {
            visit_expr_mut(base, visit);
            for (_, value) in fields {
                visit_expr_mut(value, visit);
            }
        }
        Expr::Access(value, _, _)
        | Expr::Accumulate(value, _, _)
        | Expr::TupleGet(value, _, _)
        | Expr::Unary(_, value, _)
        | Expr::Cast(value, _, _, _)
        | Expr::Grad(value, _, _)
        | Expr::Vmap(value, _, _)
        | Expr::Jit(value, _)
        | Expr::Realize(value, _)
        | Expr::Copy(value, _)
        | Expr::Borrow(value, _)
        | Expr::Quote(value, _)
        | Expr::Unquote(value, _)
        | Expr::Splice(value, _)
        | Expr::Annotate(value, _, _) => visit_expr_mut(value, visit),
        Expr::Binary(_, a, b, _) | Expr::WithDevice(a, b, _) => {
            visit_expr_mut(a, visit);
            visit_expr_mut(b, visit);
        }
        Expr::Pipe(seed, stages, _) => {
            visit_expr_mut(seed, visit);
            for stage in stages {
                visit_expr_mut(&mut stage.expression, visit);
            }
        }
        Expr::If(c, a, b, _) => {
            visit_expr_mut(c, visit);
            visit_expr_mut(a, visit);
            visit_expr_mut(b, visit);
        }
        Expr::Match(value, arms, _) => {
            visit_expr_mut(value, visit);
            for arm in arms {
                if let Some(guard) = &mut arm.guard {
                    visit_expr_mut(guard, visit);
                }
                visit_expr_mut(&mut arm.body, visit);
            }
        }
        Expr::Lambda(_, body, _) => visit_expr_mut(body, visit),
        Expr::Block(bindings, tail, _) => {
            for binding in bindings {
                visit_expr_mut(&mut binding.value, visit);
            }
            visit_expr_mut(tail, visit);
        }
    }
    visit(expr);
}

/// Visit every expression read-only, in the same post-order as
/// [`visit_expr_mut`].
pub(crate) fn visit_expr(expr: &Expr, visit: &mut impl FnMut(&Expr)) {
    match expr {
        Expr::Lit(..) | Expr::Var(..) | Expr::Constructor(..) => {}
        Expr::List(items, _) | Expr::Tuple(items, _) | Expr::Par(items, _) | Expr::Do(items, _) => {
            for item in items {
                visit_expr(item, visit);
            }
        }
        Expr::Apply(head, args, _) => {
            visit_expr(head, visit);
            for arg in args {
                visit_expr(arg, visit);
            }
        }
        Expr::Record(_, fields, _) => {
            for (_, value) in fields {
                visit_expr(value, visit);
            }
        }
        Expr::RecordUpdate(base, fields, _) => {
            visit_expr(base, visit);
            for (_, value) in fields {
                visit_expr(value, visit);
            }
        }
        Expr::Access(value, _, _)
        | Expr::Accumulate(value, _, _)
        | Expr::TupleGet(value, _, _)
        | Expr::Unary(_, value, _)
        | Expr::Cast(value, _, _, _)
        | Expr::Grad(value, _, _)
        | Expr::Vmap(value, _, _)
        | Expr::Jit(value, _)
        | Expr::Realize(value, _)
        | Expr::Copy(value, _)
        | Expr::Borrow(value, _)
        | Expr::Quote(value, _)
        | Expr::Unquote(value, _)
        | Expr::Splice(value, _)
        | Expr::Annotate(value, _, _) => visit_expr(value, visit),
        Expr::Binary(_, a, b, _) | Expr::WithDevice(a, b, _) => {
            visit_expr(a, visit);
            visit_expr(b, visit);
        }
        Expr::Pipe(seed, stages, _) => {
            visit_expr(seed, visit);
            for stage in stages {
                visit_expr(&stage.expression, visit);
            }
        }
        Expr::If(c, a, b, _) => {
            visit_expr(c, visit);
            visit_expr(a, visit);
            visit_expr(b, visit);
        }
        Expr::Match(value, arms, _) => {
            visit_expr(value, visit);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    visit_expr(guard, visit);
                }
                visit_expr(&arm.body, visit);
            }
        }
        Expr::Lambda(_, body, _) => visit_expr(body, visit),
        Expr::Block(bindings, tail, _) => {
            for binding in bindings {
                visit_expr(&binding.value, visit);
            }
            visit_expr(tail, visit);
        }
    }
    visit(expr);
}

pub(crate) fn visit_program_mut(decls: &mut [Decl], visit: &mut impl FnMut(&mut Expr)) {
    for decl in decls {
        match decl {
            Decl::Module { decls, .. } => visit_program_mut(decls, visit),
            Decl::FunDef { body, .. } | Decl::MacroDef { body, .. } => visit_expr_mut(body, visit),
            Decl::LetDef { value, .. } => visit_expr_mut(value, visit),
            Decl::Property {
                preconditions,
                body,
                options,
                ..
            } => {
                for condition in preconditions {
                    visit_expr_mut(condition, visit);
                }
                visit_expr_mut(body, visit);
                for option in options {
                    match option {
                        PropertyOption::Tolerance(value, _)
                        | PropertyOption::Seed(value, _)
                        | PropertyOption::Samples(value, _) => visit_expr_mut(value, visit),
                        PropertyOption::Contract(..) => {}
                    }
                }
            }
            Decl::TypeDef {
                invariant: Some(invariant),
                ..
            } => visit_expr_mut(&mut invariant.body, visit),
            Decl::TypeDef {
                invariant: None, ..
            }
            | Decl::TypeAlias { .. }
            | Decl::Sig { .. }
            | Decl::Dim { .. }
            | Decl::Import { .. }
            | Decl::Export { .. } => {}
        }
    }
}

pub(crate) fn normalize(expr: &mut Expr) {
    let Expr::Pipe(seed, stages, _) = expr else {
        return;
    };
    // Move rather than repeatedly cloning a growing chain.
    let mut carried = *std::mem::replace(seed, Box::new(Expr::Tuple(vec![], Span::new(0, 0))));
    for stage in std::mem::take(stages) {
        let span = crate::parser::expression_span(&stage.expression);
        carried = match (stage.syntax, stage.expression) {
            (PipeStageSyntax::CallFirst, Expr::Apply(callee, mut arguments, _)) => {
                arguments.insert(0, carried);
                Expr::Apply(callee, arguments, span)
            }
            (PipeStageSyntax::CallFirst, Expr::Accumulate(call, precision, _)) => {
                let Expr::Apply(callee, mut arguments, _) = *call else {
                    unreachable!("parser-validated accumulator call stage")
                };
                arguments.insert(0, carried);
                Expr::Accumulate(
                    Box::new(Expr::Apply(callee, arguments, span)),
                    precision,
                    span,
                )
            }
            (PipeStageSyntax::Cast(mode), Expr::Apply(_, arguments, _)) => {
                let [Expr::Var(dtype, _)] = arguments.as_slice() else {
                    unreachable!("parser-validated cast stage");
                };
                Expr::Cast(Box::new(carried), dtype.clone(), mode, span)
            }
            (PipeStageSyntax::Copy, _) => Expr::Copy(Box::new(carried), span),
            (PipeStageSyntax::Realize, _) => Expr::Realize(Box::new(carried), span),
            (PipeStageSyntax::Callable, callable) => {
                Expr::Apply(Box::new(callable), vec![carried], span)
            }
            (PipeStageSyntax::CallFirst, _) => unreachable!("parser-validated call stage"),
            (PipeStageSyntax::Cast(_), _) => unreachable!("parser-validated cast stage"),
        };
    }
    *expr = carried;
}

fn validate_stage(expr: &Expr) -> Result<(), crate::desugar::DesugarError> {
    if let Expr::Pipe(_, stages, _) = expr {
        for stage in stages {
            let valid = match (&stage.syntax, &stage.expression) {
                (PipeStageSyntax::Callable, _) => true,
                (PipeStageSyntax::CallFirst, Expr::Apply(..)) => true,
                (PipeStageSyntax::CallFirst, Expr::Accumulate(call, _, _)) => {
                    matches!(call.as_ref(), Expr::Apply(..))
                }
                (PipeStageSyntax::Cast(mode), Expr::Apply(head, args, _)) => {
                    matches!(head.as_ref(), Expr::Var(name, _) if name == mode.keyword())
                        && matches!(args.as_slice(), [Expr::Var(..)])
                }
                (PipeStageSyntax::Copy, Expr::Var(name, _)) => name == "copy",
                (PipeStageSyntax::Realize, Expr::Var(name, _)) => name == "realize",
                _ => false,
            };
            if !valid {
                return Err(crate::desugar::DesugarError::InvalidPipeStage {
                    span: crate::parser::expression_span(&stage.expression),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn normalized_program(
    decls: &[Decl],
) -> Result<Vec<Decl>, crate::desugar::DesugarError> {
    let mut decls = decls.to_vec();
    let mut error = None;
    visit_program_mut(&mut decls, &mut |expr| {
        if error.is_none() {
            match validate_stage(expr) {
                Ok(()) => normalize(expr),
                Err(e) => error = Some(e),
            }
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(decls)
}

pub(crate) fn normalized_expression(expr: &Expr) -> Result<Expr, crate::desugar::DesugarError> {
    let mut expr = expr.clone();
    let mut error = None;
    visit_expr_mut(&mut expr, &mut |expr| {
        if error.is_none() {
            match validate_stage(expr) {
                Ok(()) => normalize(expr),
                Err(e) => error = Some(e),
            }
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(expr)
}
