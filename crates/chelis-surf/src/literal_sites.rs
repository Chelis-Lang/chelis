//! Where a numeric literal's dtype is stated (`spec/04-type-system.md` §5.6).
//!
//! A literal takes its dtype from its suffix, or from a dtype-stating
//! construct that directly contains it: a declaration, a checked `cast`, or
//! the dtype argument of `to_tensor`. The desugarer binds literals by the
//! sites this module finds, and the resugarer runs the same functions on the
//! Surf it prints to decide where a suffix is redundant, so the two cannot
//! disagree about which construct states a literal's dtype. Both read the
//! program after pipe normalization (`spec/02-surf-syntax.md` [02-PIPE-1]).

use chelis_unord::{UnordMap, UnordSet};

use crate::ast::{CastMode, Decl, Expr, Literal, LiteralSuffix, PropertyOption, TypeExpr, UnaryOp};
use crate::dtype_name::canonical_primitive_name;

/// The token kind of a numeric literal, which decides its default and the
/// dtypes it admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LiteralKind {
    Integer,
    Decimal,
}

/// A numeric literal or the unary negation of one, the only forms §5.6 lets
/// a construct contain "directly".
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericLiteral<'a> {
    /// The literal as written: a `Lit`, or the `Unary(Neg, Lit)` that folds
    /// into one signed literal.
    pub(crate) expr: &'a Expr,
    /// The `Lit` node itself.
    pub(crate) token: &'a Expr,
    pub(crate) kind: LiteralKind,
    pub(crate) suffix: Option<LiteralSuffix>,
}

impl NumericLiteral<'_> {
    /// The `spec/04-type-system.md` §5.3 default for the token.
    pub(crate) fn default_dtype(&self) -> &'static str {
        match self.kind {
            LiteralKind::Integer => "i32",
            LiteralKind::Decimal => "f32",
        }
    }

    /// Whether the literal's kind admits `primitive` (§5.6 Binding).
    pub(crate) fn admits(&self, primitive: &str) -> bool {
        kind_admits(self.kind, primitive)
    }

    pub(crate) fn is_unsuffixed(&self) -> bool {
        self.suffix.is_none()
    }
}

/// The literal at `expr`, when `expr` is a numeric literal or its negation.
pub(crate) fn numeric_literal(expr: &Expr) -> Option<NumericLiteral<'_>> {
    let token = match expr {
        Expr::Unary(UnaryOp::Neg, inner, _) => inner.as_ref(),
        other => other,
    };
    let Expr::Lit(literal, _) = token else {
        return None;
    };
    let (kind, suffix) = match literal {
        Literal::Int(_) => (LiteralKind::Integer, None),
        Literal::Float(_) => (LiteralKind::Decimal, None),
        Literal::TypedInt(_, suffix) => (LiteralKind::Integer, Some(*suffix)),
        Literal::TypedFloat(_, suffix) => (LiteralKind::Decimal, Some(*suffix)),
        Literal::Str(_) | Literal::Bool(_) => return None,
    };
    Some(NumericLiteral {
        expr,
        token,
        kind,
        suffix,
    })
}

/// An integer literal admits every numeric primitive and a decimal literal
/// every float primitive; no numeric literal admits `bool`, `string`, or
/// another non-numeric primitive.
pub(crate) fn kind_admits(kind: LiteralKind, primitive: &str) -> bool {
    let float = matches!(primitive, "f32" | "f64" | "bf16" | "f16");
    let integer = matches!(primitive, "i8" | "i16" | "i32" | "i64");
    match kind {
        LiteralKind::Integer => float || integer,
        LiteralKind::Decimal => float,
    }
}

/// The suffix that spells a numeric primitive, if it has one.
pub(crate) fn primitive_suffix(primitive: &str) -> Option<LiteralSuffix> {
    [
        LiteralSuffix::F32,
        LiteralSuffix::F64,
        LiteralSuffix::Bf16,
        LiteralSuffix::F16,
        LiteralSuffix::I8,
        LiteralSuffix::I16,
        LiteralSuffix::I32,
        LiteralSuffix::I64,
    ]
    .into_iter()
    .find(|suffix| suffix.t_prim_name() == primitive)
}

/// A dtype a construct states: a primitive, or a dtype binder of the
/// enclosing declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatedDtype<'a> {
    Primitive(&'static str),
    Binder(&'a str),
}

/// Resolves a dtype spelling. A primitive name always names the primitive; any
/// other name states a dtype only when it is a binder of the declaration.
pub(crate) fn resolve_dtype<'a>(
    name: &'a str,
    is_binder: &dyn Fn(&str) -> bool,
) -> Option<StatedDtype<'a>> {
    if let Some(primitive) = canonical_primitive_name(name) {
        return Some(StatedDtype::Primitive(primitive));
    }
    is_binder(name).then_some(StatedDtype::Binder(name))
}

/// What a declared type states for an initializer that is a literal.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DeclaredDtype<'a> {
    /// A scalar type spelled by name: a primitive or a binder.
    Scalar(&'a str),
    /// A tensor type and its element precision spelling.
    Tensor(&'a str),
}

/// The dtype a declared type spells. A type alias spells none: its name is
/// neither a primitive nor a binder, so [`resolve_dtype`] states nothing for
/// it.
pub(crate) fn declared_dtype(ty: &TypeExpr) -> Option<DeclaredDtype<'_>> {
    match ty {
        TypeExpr::Named(name, _) => Some(DeclaredDtype::Scalar(name)),
        TypeExpr::Tensor(_, precision, _) => Some(DeclaredDtype::Tensor(precision.as_str())),
        _ => None,
    }
}

/// A declaration's literal initializer, by the declared type.
pub(crate) enum DeclarationSite<'a> {
    /// The initializer is a literal or its negation, under a declared
    /// primitive or binder.
    Scalar {
        literal: NumericLiteral<'a>,
        stated: StatedDtype<'a>,
    },
    /// The initializer is a bare bracket literal under a declared tensor type:
    /// a tensor literal, whose literal elements take the element dtype.
    Tensor {
        items: &'a [Expr],
        stated: Option<StatedDtype<'a>>,
        precision: &'a str,
    },
}

/// The §5.6 Declaration site of `init` under `declared`, if any.
pub(crate) fn declaration_site<'a>(
    declared: &'a TypeExpr,
    init: &'a Expr,
    is_binder: &dyn Fn(&str) -> bool,
) -> Option<DeclarationSite<'a>> {
    match declared_dtype(declared)? {
        DeclaredDtype::Scalar(name) => {
            let literal = numeric_literal(init)?;
            let stated = resolve_dtype(name, is_binder)?;
            Some(DeclarationSite::Scalar { literal, stated })
        }
        DeclaredDtype::Tensor(precision) => {
            let Expr::List(items, _) = init else {
                return None;
            };
            Some(DeclarationSite::Tensor {
                items,
                stated: resolve_dtype(precision, is_binder),
                precision,
            })
        }
    }
}

/// The literal operand of a checked `cast` and the dtype its target states.
/// The named casts state no dtype.
pub(crate) fn cast_operand<'a>(
    operand: &'a Expr,
    target: &'a str,
    mode: CastMode,
    is_binder: &dyn Fn(&str) -> bool,
) -> Option<(NumericLiteral<'a>, Option<StatedDtype<'a>>)> {
    if !matches!(mode, CastMode::Checked) {
        return None;
    }
    Some((numeric_literal(operand)?, resolve_dtype(target, is_binder)))
}

/// The primitive a cast's literal operand binds at, when the cast denotes the
/// literal itself: an unsuffixed literal whose kind admits a primitive target.
pub(crate) fn cast_collapse(
    literal: &NumericLiteral<'_>,
    stated: Option<StatedDtype<'_>>,
) -> Option<&'static str> {
    match stated {
        Some(StatedDtype::Primitive(primitive))
            if literal.is_unsuffixed() && literal.admits(primitive) =>
        {
            Some(primitive)
        }
        _ => None,
    }
}

/// A call of the reserved `to_tensor`, with its optional dtype argument.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ToTensorCall<'a> {
    pub(crate) argument: &'a Expr,
    pub(crate) dtype: Option<&'a Expr>,
}

/// `to_tensor(xs)` or `to_tensor(xs, p)`. Any other arity is an ordinary call
/// that the checker rejects.
pub(crate) fn to_tensor_call<'a>(
    callee: &'a Expr,
    arguments: &'a [Expr],
) -> Option<ToTensorCall<'a>> {
    let Expr::Var(name, _) = callee else {
        return None;
    };
    if name != "to_tensor" {
        return None;
    }
    match arguments {
        [argument] => Some(ToTensorCall {
            argument,
            dtype: None,
        }),
        [argument, dtype] => Some(ToTensorCall {
            argument,
            dtype: Some(dtype),
        }),
        _ => None,
    }
}

/// The dtype a dtype argument names. The position holds a dtype, never a
/// value (`spec/02-surf-syntax.md` §P9), so only an identifier naming a
/// primitive or a binder in scope states one.
pub(crate) fn dtype_argument<'a>(
    dtype: &'a Expr,
    is_binder: &dyn Fn(&str) -> bool,
) -> Option<StatedDtype<'a>> {
    let Expr::Var(name, _) = dtype else {
        return None;
    };
    resolve_dtype(name, is_binder)
}

/// The declaration that a declaration-site literal initializes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DeclarationOwner<'a> {
    Binding(&'a str),
    Result(&'a str),
}

/// How the constructs around one numeric literal decide its dtype.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Site<'a> {
    /// No construct states a dtype: the literal keeps its suffix or default.
    Ordinary,
    /// A declaration states the dtype; `None` when its spelling names no
    /// primitive or binder.
    Declaration {
        stated: Option<StatedDtype<'a>>,
        owner: DeclarationOwner<'a>,
    },
    /// The operand of a checked cast, which never takes a default it would
    /// not take anyway: a cast collapses or converts (§5.6).
    Cast,
    /// An element of `to_tensor(xs, p)`; `None` when `p` names no dtype.
    DtypeArgument(Option<StatedDtype<'a>>),
    /// An element of `to_tensor(xs)`, which has no default.
    MissingDtype {
        argument: &'a Expr,
        cast_target: Option<&'a str>,
    },
}

/// Receives every numeric literal of a program with its site, and every
/// `to_tensor` dtype argument.
pub(crate) trait SiteVisitor<'a> {
    fn literal(&mut self, literal: NumericLiteral<'a>, site: Site<'a>);

    fn dtype_argument(&mut self, _dtype: &'a Expr, _stated: Option<StatedDtype<'a>>) {}
}

/// Signatures and binders by declaration name, which decide declared types
/// and binder scopes.
#[derive(Default)]
pub(crate) struct ProgramDeclarations<'a> {
    value_types: UnordMap<&'a str, &'a TypeExpr>,
    result_types: UnordMap<&'a str, &'a TypeExpr>,
    binders: UnordMap<&'a str, UnordSet<&'a str>>,
}

impl<'a> ProgramDeclarations<'a> {
    pub(crate) fn collect(decls: &'a [Decl]) -> Self {
        let mut declarations = Self::default();
        declarations.collect_into(decls);
        declarations
    }

    fn collect_into(&mut self, decls: &'a [Decl]) {
        for decl in decls {
            match decl {
                Decl::Module { decls, .. } => self.collect_into(decls),
                Decl::Sig {
                    name,
                    type_binders,
                    ty,
                    ..
                } => {
                    match ty {
                        TypeExpr::Arrow(_, result, _) => {
                            self.result_types.insert(name.as_str(), result.as_ref());
                        }
                        other => {
                            self.value_types.insert(name.as_str(), other);
                        }
                    }
                    self.add_binders(name, type_binders.iter().map(|b| b.name.as_str()));
                }
                Decl::FunDef {
                    name, type_binders, ..
                }
                | Decl::Property {
                    name, type_binders, ..
                } => self.add_binders(name, type_binders.iter().map(|b| b.name.as_str())),
                _ => {}
            }
        }
    }

    fn add_binders(&mut self, name: &'a str, binders: impl Iterator<Item = &'a str>) {
        let entry = self.binders.entry(name).or_default();
        entry.extend(binders);
    }

    /// A top-level binding's declared type: its inline annotation, or else
    /// its standalone value `sig`.
    pub(crate) fn binding_type<'t>(
        &self,
        name: &str,
        inline: Option<&'t TypeExpr>,
    ) -> Option<&'t TypeExpr>
    where
        'a: 't,
    {
        inline.or_else(|| self.value_types.get(name).copied())
    }

    /// A function's declared result type: its inline result type, or else
    /// the result of its standalone function `sig`.
    pub(crate) fn result_type<'t>(
        &self,
        name: &str,
        inline: Option<&'t TypeExpr>,
    ) -> Option<&'t TypeExpr>
    where
        'a: 't,
    {
        inline.or_else(|| self.result_types.get(name).copied())
    }

    fn binder_scope(&self, name: Option<&str>) -> Option<&UnordSet<&'a str>> {
        name.and_then(|name| self.binders.get(name))
    }
}

/// Visits every numeric literal of `decls` with its §5.6 site.
pub(crate) fn visit_program<'a>(decls: &'a [Decl], visitor: &mut impl SiteVisitor<'a>) {
    let declarations = ProgramDeclarations::collect(decls);
    let mut walker = Walker {
        declarations: &declarations,
        binders: None,
        visitor,
    };
    walker.decls(decls);
}

/// Visits every numeric literal of one expression with its §5.6 site, outside
/// any declaration.
pub(crate) fn visit_expression<'a>(expr: &'a Expr, visitor: &mut impl SiteVisitor<'a>) {
    let declarations = ProgramDeclarations::default();
    let mut walker = Walker {
        declarations: &declarations,
        binders: None,
        visitor,
    };
    walker.expr(expr, None);
}

struct Walker<'d, 'a, V> {
    declarations: &'d ProgramDeclarations<'a>,
    binders: Option<&'d UnordSet<&'a str>>,
    visitor: &'d mut V,
}

impl<'a, V: SiteVisitor<'a>> Walker<'_, 'a, V> {
    fn is_binder(&self) -> impl Fn(&str) -> bool + '_ {
        move |name| self.binders.is_some_and(|binders| binders.contains(name))
    }

    fn decls(&mut self, decls: &'a [Decl]) {
        for decl in decls {
            self.decl(decl);
        }
    }

    fn decl(&mut self, decl: &'a Decl) {
        match decl {
            Decl::Module { decls, .. } => self.decls(decls),
            Decl::FunDef {
                name, ret_ty, body, ..
            } => {
                self.binders = self.declarations.binder_scope(Some(name));
                let declared = self.declarations.result_type(name, ret_ty.as_ref());
                self.initializer(declared, body, DeclarationOwner::Result(name));
                self.binders = None;
            }
            Decl::LetDef {
                name, ty, value, ..
            } => {
                self.binders = self.declarations.binder_scope(Some(name));
                let declared = self.declarations.binding_type(name, ty.as_ref());
                self.initializer(declared, value, DeclarationOwner::Binding(name));
                self.binders = None;
            }
            Decl::Property {
                name,
                preconditions,
                body,
                options,
                ..
            } => {
                self.binders = self.declarations.binder_scope(Some(name));
                for condition in preconditions {
                    self.expr(condition, None);
                }
                self.expr(body, None);
                for option in options {
                    match option {
                        PropertyOption::Tolerance(value, _)
                        | PropertyOption::Seed(value, _)
                        | PropertyOption::Samples(value, _) => self.expr(value, None),
                        PropertyOption::Contract(..) => {}
                    }
                }
                self.binders = None;
            }
            Decl::MacroDef { body, .. } => self.expr(body, None),
            Decl::TypeDef {
                invariant: Some(invariant),
                ..
            } => self.expr(&invariant.body, None),
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

    /// A declaration's initializer under its declared type, if any.
    fn initializer(
        &mut self,
        declared: Option<&'a TypeExpr>,
        init: &'a Expr,
        owner: DeclarationOwner<'a>,
    ) {
        let site =
            declared.and_then(|declared| declaration_site(declared, init, &self.is_binder()));
        match site {
            Some(DeclarationSite::Scalar { literal, stated }) => self.visitor.literal(
                literal,
                Site::Declaration {
                    stated: Some(stated),
                    owner,
                },
            ),
            Some(DeclarationSite::Tensor { items, stated, .. }) => {
                self.elements(items, Site::Declaration { stated, owner })
            }
            None => self.expr(init, None),
        }
    }

    /// The items of a bracket literal whose literal elements take `site`.
    fn elements(&mut self, items: &'a [Expr], site: Site<'a>) {
        for item in items {
            if let Some(literal) = numeric_literal(item) {
                self.visitor.literal(literal, site);
            } else if let Expr::List(nested, _) = item {
                self.elements(nested, site);
            } else {
                self.expr(item, None);
            }
        }
    }

    /// `cast_target` is the primitive spelling of a checked cast whose operand
    /// is `expr`, for the missing-dtype diagnostic.
    fn expr(&mut self, expr: &'a Expr, cast_target: Option<&'a str>) {
        if let Some(literal) = numeric_literal(expr) {
            self.visitor.literal(literal, Site::Ordinary);
            return;
        }
        match expr {
            Expr::Lit(..) | Expr::Var(..) | Expr::Constructor(..) => {}
            Expr::Apply(callee, arguments, _) => {
                if let Some(call) = to_tensor_call(callee, arguments) {
                    self.tensor_conversion(call, cast_target);
                } else {
                    self.expr(callee, None);
                    for argument in arguments {
                        self.expr(argument, None);
                    }
                }
            }
            Expr::Accumulate(call, _, _) => self.expr(call, None),
            Expr::List(items, _)
            | Expr::Tuple(items, _)
            | Expr::Par(items, _)
            | Expr::Do(items, _) => {
                for item in items {
                    self.expr(item, None);
                }
            }
            Expr::Record(_, fields, _) => {
                for (_, value) in fields {
                    self.expr(value, None);
                }
            }
            Expr::RecordUpdate(base, fields, _) => {
                self.expr(base, None);
                for (_, value) in fields {
                    self.expr(value, None);
                }
            }
            Expr::Cast(operand, target, mode, _) => {
                let site = cast_operand(operand, target, *mode, &self.is_binder());
                if let Some((literal, _)) = site {
                    self.visitor.literal(literal, Site::Cast);
                } else {
                    let target = matches!(mode, CastMode::Checked).then_some(target.as_str());
                    self.expr(operand, target);
                }
            }
            Expr::Access(value, _, _)
            | Expr::TupleGet(value, _, _)
            | Expr::Unary(_, value, _)
            | Expr::Grad(value, _, _)
            | Expr::Vmap(value, _, _)
            | Expr::Jit(value, _)
            | Expr::Realize(value, _)
            | Expr::Copy(value, _)
            | Expr::Borrow(value, _)
            | Expr::Quote(value, _)
            | Expr::Unquote(value, _)
            | Expr::Splice(value, _)
            | Expr::Annotate(value, _, _) => self.expr(value, None),
            Expr::Binary(_, left, right, _) | Expr::WithDevice(left, right, _) => {
                self.expr(left, None);
                self.expr(right, None);
            }
            Expr::Pipe(seed, stages, _) => {
                // Pipes normalize before literal dtype selection; an
                // unnormalized pipe states nothing.
                self.expr(seed, None);
                for stage in stages {
                    self.expr(&stage.expression, None);
                }
            }
            Expr::If(condition, consequence, alternative, _) => {
                self.expr(condition, None);
                self.expr(consequence, None);
                self.expr(alternative, None);
            }
            Expr::Match(scrutinee, arms, _) => {
                self.expr(scrutinee, None);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.expr(guard, None);
                    }
                    self.expr(&arm.body, None);
                }
            }
            Expr::Lambda(_, body, _) => self.expr(body, None),
            Expr::Block(bindings, tail, _) => {
                for binding in bindings {
                    match (&binding.pattern, &binding.ty) {
                        (crate::ast::LetPattern::Var(name, _), Some(ty)) => self.initializer(
                            Some(ty),
                            &binding.value,
                            DeclarationOwner::Binding(name),
                        ),
                        _ => self.expr(&binding.value, None),
                    }
                }
                self.expr(tail, None);
            }
        }
    }

    fn tensor_conversion(&mut self, call: ToTensorCall<'a>, cast_target: Option<&'a str>) {
        let site = match call.dtype {
            Some(dtype) => {
                let stated = dtype_argument(dtype, &self.is_binder());
                self.visitor.dtype_argument(dtype, stated);
                Site::DtypeArgument(stated)
            }
            None => Site::MissingDtype {
                argument: call.argument,
                cast_target,
            },
        };
        match call.argument {
            Expr::List(items, _) => self.elements(items, site),
            other => self.expr(other, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    #[derive(Default)]
    struct Recorder(Vec<String>);

    impl<'a> SiteVisitor<'a> for Recorder {
        fn literal(&mut self, literal: NumericLiteral<'a>, site: Site<'a>) {
            let site = match site {
                Site::Ordinary => "ordinary".to_string(),
                Site::Declaration { stated, .. } => format!("declaration {stated:?}"),
                Site::Cast => "cast".to_string(),
                Site::DtypeArgument(stated) => format!("dtype {stated:?}"),
                Site::MissingDtype { cast_target, .. } => format!("missing {cast_target:?}"),
            };
            self.0.push(format!(
                "{} {site}",
                crate::format::format_expression(literal.expr)
            ));
        }
    }

    fn sites(source: &str) -> Vec<String> {
        let decls = parse_str(source).expect("parse");
        let mut recorder = Recorder::default();
        visit_program(&decls, &mut recorder);
        recorder.0
    }

    #[test]
    fn every_construct_states_the_dtype_of_a_literal_it_directly_contains() {
        assert_eq!(
            sites("x: f64 = 1.1\ny = cast(-2.5, f32)\nz = to_tensor([1, [2]], i64)\n"),
            [
                "1.1 declaration Some(Primitive(\"f64\"))",
                "-2.5 cast",
                "1 dtype Some(Primitive(\"i64\"))",
                "2 dtype Some(Primitive(\"i64\"))",
            ]
        );
        assert_eq!(
            sites("sig v: f64\nv = 1.1\nsig g: f64 -> f64\ndef g(a) = 2.5\n"),
            [
                "1.1 declaration Some(Primitive(\"f64\"))",
                "2.5 declaration Some(Primitive(\"f64\"))",
            ]
        );
        assert_eq!(
            sites("def k[p: Float](x: p) -> p = cast(1.5, p)\n"),
            ["1.5 cast"]
        );
    }

    #[test]
    fn anything_between_the_construct_and_the_literal_breaks_the_site() {
        assert_eq!(
            sites(
                "x: f64 = neg(1.1)\ntype P = f64\ny: P = 1.1\nz = cast_trunc(1.9, i32)\n\
                 w = to_tensor([id(1.5)], f64)\n"
            ),
            [
                "1.1 ordinary",
                "1.1 ordinary",
                "1.9 ordinary",
                "1.5 ordinary"
            ]
        );
        assert_eq!(
            sites("x = cast(to_tensor([1.1, v]), f64)\n"),
            ["1.1 missing Some(\"f64\")"]
        );
    }
}
