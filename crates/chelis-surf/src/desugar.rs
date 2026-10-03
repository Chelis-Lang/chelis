//! Desugars Surface AST into Deep (s-expression) AST.
//!
//! Every Deep node is a 3-tuple: (tag {} children...)
//! where {} is an inline metadata map.

use chelis_deep::{
    Atom as DeepAtom, DeepTag, LiteralFamilyFit, LiteralSource, classify_literal_source,
    encode_dtype_bounds,
};
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::Span;
use chelis_deep::annotations::{
    Amenability, BindingTypeOrigin, EffectMember, EffectSet as AstEffectSet, InvariantPredicate,
    LiteralOrigin, LiteralStyle, MetadataValue as M, PipeStageOrigin, PositiveInteger, Present,
    PropertyBinder, PropertyContracts, PropertyPreconditions, PropertyQuantifiers,
    PropertySourceKind, ResourceEffect, RuntimeExpression, SpanId, Spanned, TypeSyntax,
    VariableRef, VariableTuple, WrtTargets,
};
use chelis_deep::ast as deep;
use chelis_vocab::EffectKind;
use std::collections::BTreeMap;
use thiserror::Error;

use crate::ast::*;
pub(crate) use crate::dtype_name::{
    canonical_primitive_name, is_reserved_dtype_name, migrated_integer_dtype_name,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A Surf program that cannot be translated to semantically faithful Deep.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DesugarError {
    #[error("invalid Surf declaration ownership: {message}")]
    InvalidDeclarationOwnership { message: String },
    #[error(
        "unknown `grad` parameter `{parameter}` for callable `{callable}`; available parameters: {available:?}"
    )]
    UnknownGradParameter {
        callable: String,
        parameter: String,
        available: Vec<String>,
        span: Span,
    },
    #[error(
        "cannot resolve the parameter identity of `grad` target `{target}`; use a direct declaration, inline lambda, or immutable alias"
    )]
    UnresolvedGradTarget { target: String, span: Span },
    #[error("`grad` target `{target}` is not callable")]
    NonCallableGradTarget { target: String, span: Span },
}

impl DesugarError {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::InvalidDeclarationOwnership { .. } => None,
            Self::UnknownGradParameter { span, .. }
            | Self::UnresolvedGradTarget { span, .. }
            | Self::NonCallableGradTarget { span, .. } => Some(*span),
        }
    }
}

/// Desugar parser-validated declarations. Programmatic callers receive the
/// same typed failure as source callers; Deep is never constructed with a
/// guessed selector index.
pub fn desugar_program(decls: &[Decl]) -> Result<Vec<deep::Expr>, DesugarError> {
    desugar_program_with_context(decls, &[])
}

/// Desugar declarations with callable origins supplied by an already checked
/// linked library context.
pub fn desugar_program_with_context(
    decls: &[Decl],
    context: &[deep::Expr],
) -> Result<Vec<deep::Expr>, DesugarError> {
    crate::parser::validate_bound_ownership(decls).map_err(|error| {
        DesugarError::InvalidDeclarationOwnership {
            message: error.to_string(),
        }
    })?;
    let resolved_grad_indices = GradSelectorResolver::resolve_program_with_context(decls, context)?;
    let ctx = DesugarCtx::new(decls, resolved_grad_indices);
    Ok(decls
        .iter()
        .flat_map(|decl| ctx.desugar_decl(decl))
        .collect())
}

pub fn desugar_decl_only(decl: &Decl) -> Result<Vec<deep::Expr>, DesugarError> {
    desugar_program(std::slice::from_ref(decl))
}

pub fn desugar_expr_only(expr: &Expr) -> Result<deep::Expr, DesugarError> {
    let resolved_grad_indices = GradSelectorResolver::resolve_expression(expr)?;
    Ok(DesugarCtx::with_resolved_grad_indices(resolved_grad_indices).desugar_expr(expr))
}

/// Desugar an expression using the declarations that establish its callable
/// aliases and formal-parameter identities.
pub fn desugar_expr_in_program(decls: &[Decl], expr: &Expr) -> Result<deep::Expr, DesugarError> {
    desugar_expr_in_program_scope(decls, expr, &[])
}

/// Desugar an expression using its program and lexical binder context.
pub fn desugar_expr_in_program_scope(
    decls: &[Decl],
    expr: &Expr,
    bound_names: &[String],
) -> Result<deep::Expr, DesugarError> {
    crate::parser::validate_bound_ownership(decls).map_err(|error| {
        DesugarError::InvalidDeclarationOwnership {
            message: error.to_string(),
        }
    })?;
    let resolved_grad_indices =
        GradSelectorResolver::resolve_expression_in_program(decls, expr, bound_names)?;
    Ok(DesugarCtx::new(decls, resolved_grad_indices).desugar_expr_with_scope(expr, bound_names))
}

#[derive(Default)]
struct DesugarCtx {
    resolved_grad_indices: Vec<(usize, Vec<i64>)>,
    /// Per-function tensor element types declared in the function's
    /// signature, indexed by parameter position. `None` for non-tensor
    /// parameters or parameters with no declared type.
    ///
    /// Used by the contextual tensor-literal inference rule
    /// (`spec/02-surf-syntax.md` §P10b, `spec/04-type-system.md` §5.6),
    /// position 2: the corresponding argument position of a call whose
    /// callee has a declared signature with a tensor parameter at that
    /// position.
    top_level_fn_tensor_param_prec: UnordMap<String, Vec<Option<String>>>,
    /// Names that carry an explicit standalone `sig`/signature declaration
    /// (`Decl::Sig`). When a `def` of the same name also has inline
    /// annotations, `desugar_fun_def` would otherwise synthesize a second
    /// `defsig` filling every un-annotated position with a wildcard
    /// `(t-var {} _)`. That synthesized signature is last-write-wins in the
    /// type checker's defsig binding (`chelis-types` `collect_declarations`),
    /// so it silently overwrites the concrete explicit `sig`, dropping the
    /// body-vs-signature contract on the un-annotated positions
    /// (chelis#285). When an explicit sig exists, the synthesized one is
    /// strictly redundant and weaker, so we suppress it here.
    explicit_sig_names: UnordSet<String>,
    /// Standalone value signatures supply binding literal context just as
    /// inline annotations do. List signatures still preserve List values.
    top_level_binding_tensor_prec: UnordMap<String, String>,
    /// Explicit effect clauses (`! { ... }`) declared on each `def`, keyed by
    /// name. The effect upper-bound check reads the declared effect set only
    /// from a `defsig`'s `t-fn` `eff` metadata
    /// (`chelis-effects::declared_effects_from_defsig`). A `def`'s clause
    /// reaches that check solely via the synthesized `defsig` — but
    /// `explicit_sig_names` now suppresses that synthesized `defsig`. So when
    /// an explicit `sig` declares no effects of its own, `Decl::Sig`
    /// desugaring inherits the same-named `def`'s clause from this map;
    /// otherwise suppression would silently drop the def's effect contract
    /// (chelis#285). Stored even for an empty `! {}` (which declares "no
    /// effects" and is distinct from no annotation at all).
    def_effects: UnordMap<String, Vec<EffectExpr>>,
    /// Monotonic counter behind every `__chelis_tmpN` this context
    /// synthesizes for destructuring `let` patterns (chelis#1200).
    ///
    /// It lives on the context, not on `desugar_let_bindings`, because a
    /// per-call counter restarts at 0 for every nested block. Two
    /// destructures in nested blocks then both mint `__chelis_tmp0..2`,
    /// and the inner names SHADOW the outer ones in the linearity
    /// checker's scope. Since a component is an alias of its temp, an
    /// outer component's consume resolved to the inner block's temp
    /// entry: it either blamed the wrong binding or, when the inner temp
    /// was still `Live`, left the outer carrier unconsumed so a genuine
    /// double consume was silently accepted. The names are internal, so
    /// the fix is simply to never reuse one within a context.
    ///
    /// A `Cell` because the desugar walk takes `&self` throughout.
    next_destructure_temp: std::cell::Cell<usize>,
    /// Each declaration's inline or standalone-signature binders. `None`
    /// marks a declared but unbounded binder, which cannot adopt a literal.
    declared_type_binders: UnordMap<String, UnordMap<String, Option<chelis_deep::DtypeBound>>>,
    /// Binder scope installed while one declaration body is desugared.
    current_type_binders: std::cell::RefCell<UnordMap<String, Option<chelis_deep::DtypeBound>>>,
}

impl DesugarCtx {
    fn current_type_binder(&self, name: &str) -> Option<Option<chelis_deep::DtypeBound>> {
        // spec/02 §P4b: a quantifier list overrides the lexical
        // type-variable case split, not active primitive or rejected dtype
        // spellings. Keep that category decision at the body lookup as well
        // as type desugaring, so an explicit `sig f[f32]` or `sig f[u8]`
        // cannot turn a cast target into `(t-var ...)`.
        if canonical_primitive_name(name).is_some() || is_reserved_dtype_name(name) {
            return None;
        }
        self.current_type_binders.borrow().get(name).cloned()
    }

    fn desugar_body_annotation_type(&self, ty: &TypeExpr) -> deep::Expr {
        let tvar_set: UnordSet<String> = self
            .current_type_binders
            .borrow()
            .to_sorted()
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect();
        desugar_type_with_scope_mode(ty, &UnordSet::new(), &tvar_set, true)
    }

    fn new(decls: &[Decl], resolved_grad_indices: Vec<(usize, Vec<i64>)>) -> Self {
        let mut top_level_fn_tensor_param_prec = UnordMap::new();
        let mut explicit_sig_names = UnordSet::new();
        let mut top_level_binding_tensor_prec = UnordMap::new();
        let mut def_effects = UnordMap::new();
        let mut declared_type_binders = UnordMap::new();
        for decl in decls {
            for_each_decl(decl, &mut |d| {
                collect_top_level_fn_tensor_param_prec(d, &mut top_level_fn_tensor_param_prec);
                collect_explicit_sig_names(d, &mut explicit_sig_names);
                if let Decl::Sig { name, ty, .. } = d
                    && let Some(precision) = tensor_element_prim_name(ty)
                {
                    top_level_binding_tensor_prec.insert(name.clone(), precision);
                }
                collect_def_effects(d, &mut def_effects);
                collect_declared_type_binders(d, &mut declared_type_binders);
            });
        }
        Self {
            resolved_grad_indices,
            top_level_fn_tensor_param_prec,
            explicit_sig_names,
            top_level_binding_tensor_prec,
            def_effects,
            next_destructure_temp: std::cell::Cell::new(0),
            declared_type_binders,
            current_type_binders: std::cell::RefCell::new(UnordMap::new()),
        }
    }

    fn with_resolved_grad_indices(resolved_grad_indices: Vec<(usize, Vec<i64>)>) -> Self {
        Self {
            resolved_grad_indices,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CallableOrigin {
    Known {
        identity: LexicalCallableId,
        display_name: String,
        params: Vec<String>,
    },
    Unknown,
    NonCallable,
    Tuple(Vec<CallableOrigin>),
    Constructor {
        name: String,
        payloads: Vec<CallableOrigin>,
    },
    Record {
        name: String,
        fields: BTreeMap<String, CallableOrigin>,
    },
}

impl CallableOrigin {
    fn alternate(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Known {
                    identity: left_identity,
                    params: left_params,
                    ..
                },
                Self::Known {
                    identity: right_identity,
                    params: right_params,
                    ..
                },
            ) if left_identity == right_identity && left_params == right_params => self.clone(),
            (Self::Tuple(left), Self::Tuple(right)) if left.len() == right.len() => Self::Tuple(
                left.iter()
                    .zip(right)
                    .map(|(left, right)| left.alternate(right))
                    .collect(),
            ),
            (
                Self::Constructor {
                    name: left_name,
                    payloads: left,
                },
                Self::Constructor {
                    name: right_name,
                    payloads: right,
                },
            ) if left_name == right_name && left.len() == right.len() => Self::Constructor {
                name: left_name.clone(),
                payloads: left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| left.alternate(right))
                    .collect(),
            },
            (
                Self::Record {
                    name: left_name,
                    fields: left,
                },
                Self::Record {
                    name: right_name,
                    fields: right,
                },
            ) if left_name == right_name && left.keys().eq(right.keys()) => Self::Record {
                name: left_name.clone(),
                fields: left
                    .iter()
                    .map(|(field, left)| {
                        (
                            field.clone(),
                            left.alternate(
                                right
                                    .get(field)
                                    .expect("equal record key sets contain every left key"),
                            ),
                        )
                    })
                    .collect(),
            },
            (Self::NonCallable, Self::NonCallable) => Self::NonCallable,
            _ => Self::Unknown,
        }
    }
}

/// Resolver-local identity allocated in deterministic lexical traversal order.
///
/// Aliases copy this identity. Distinct declaration or lambda sites receive
/// distinct identities even when their display labels and formal names match.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LexicalCallableId(u64);

#[derive(Clone, Default)]
struct CallableScope {
    values: UnordMap<String, CallableOrigin>,
}

impl CallableScope {
    fn lookup(&self, name: &str) -> CallableOrigin {
        self.values
            .get(name)
            .cloned()
            .unwrap_or(CallableOrigin::Unknown)
    }

    fn bind(&mut self, name: String, value: CallableOrigin) {
        self.values.insert(name, value);
    }
}

#[derive(Default)]
struct GradSelectorResolver {
    globals: CallableScope,
    resolved: Vec<(usize, Vec<i64>)>,
    next_callable_identity: u64,
    deep_callable_identities: UnordMap<usize, LexicalCallableId>,
}

impl GradSelectorResolver {
    fn fresh_callable_identity(&mut self) -> LexicalCallableId {
        let identity = LexicalCallableId(self.next_callable_identity);
        self.next_callable_identity += 1;
        identity
    }

    fn resolve_program_with_context(
        decls: &[Decl],
        context: &[deep::Expr],
    ) -> Result<Vec<(usize, Vec<i64>)>, DesugarError> {
        let mut resolver = Self::default();
        resolver.preseed_deep_function_origins(context);
        resolver.seed_deep_origins(context);
        for decl in decls {
            resolver.seed_function_origins(decl);
        }
        for decl in decls {
            resolver.visit_decl(decl)?;
        }
        Ok(resolver.resolved)
    }

    fn preseed_deep_function_origins(&mut self, exprs: &[deep::Expr]) {
        for expr in exprs {
            let deep::ExprCarrier::DecodedNode(tag, _, children) = expr.carrier() else {
                continue;
            };
            match tag {
                DeepTag::Module => self.preseed_deep_function_origins(&children[1..]),
                DeepTag::Def if children.len() == 2 => {
                    let Some(name) = deep_symbol_name(&children[0]) else {
                        continue;
                    };
                    let Some(params) = deep_function_parameters(&children[1]) else {
                        continue;
                    };
                    let identity = self.deep_callable_identity(&children[1]);
                    self.globals.bind(
                        name.to_string(),
                        CallableOrigin::Known {
                            identity,
                            display_name: name.to_string(),
                            params,
                        },
                    );
                }
                _ => {}
            }
        }
    }

    fn deep_callable_identity(&mut self, expr: &deep::Expr) -> LexicalCallableId {
        let expr = strip_deep_metadata(expr);
        let key = expr as *const deep::Expr as usize;
        if let Some(identity) = self.deep_callable_identities.get(&key) {
            return *identity;
        }
        let identity = self.fresh_callable_identity();
        self.deep_callable_identities.insert(key, identity);
        identity
    }

    fn seed_deep_origins(&mut self, exprs: &[deep::Expr]) {
        for expr in exprs {
            let deep::ExprCarrier::DecodedNode(tag, _, children) = expr.carrier() else {
                continue;
            };
            match tag {
                DeepTag::Module => self.seed_deep_origins(&children[1..]),
                DeepTag::Def if children.len() == 2 => {
                    let Some(name) = deep_symbol_name(&children[0]) else {
                        continue;
                    };
                    let scope = self.globals.clone();
                    let value = self.deep_callable_origin(&children[1], &scope, name);
                    self.globals.bind(name.to_string(), value);
                }
                _ => {}
            }
        }
    }

    fn resolve_expression(expr: &Expr) -> Result<Vec<(usize, Vec<i64>)>, DesugarError> {
        let mut resolver = Self::default();
        resolver.visit_expr(expr, &CallableScope::default())?;
        Ok(resolver.resolved)
    }

    fn resolve_expression_in_program(
        decls: &[Decl],
        expr: &Expr,
        bound_names: &[String],
    ) -> Result<Vec<(usize, Vec<i64>)>, DesugarError> {
        let mut resolver = Self::default();
        for decl in decls {
            resolver.seed_function_origins(decl);
        }
        for decl in decls {
            resolver.visit_decl(decl)?;
        }
        let mut scope = resolver.globals.clone();
        for name in bound_names {
            scope.bind(name.clone(), CallableOrigin::Unknown);
        }
        resolver.visit_expr(expr, &scope)?;
        Ok(resolver.resolved)
    }

    fn seed_function_origins(&mut self, decl: &Decl) {
        match decl {
            Decl::Module { decls, .. } => {
                for decl in decls {
                    self.seed_function_origins(decl);
                }
            }
            Decl::FunDef { name, params, .. } | Decl::Property { name, params, .. } => {
                let identity = self.fresh_callable_identity();
                self.globals.bind(
                    name.clone(),
                    CallableOrigin::Known {
                        identity,
                        display_name: name.clone(),
                        params: params.iter().map(|param| param.name.clone()).collect(),
                    },
                );
            }
            _ => {}
        }
    }

    fn visit_decl(&mut self, decl: &Decl) -> Result<(), DesugarError> {
        match decl {
            Decl::Module { decls, .. } => {
                for decl in decls {
                    self.visit_decl(decl)?;
                }
            }
            Decl::FunDef {
                name, params, body, ..
            }
            | Decl::Property {
                name, params, body, ..
            } => {
                let callable = self.globals.lookup(name);
                let mut scope = self.globals.clone();
                for param in params {
                    scope.bind(param.name.clone(), CallableOrigin::Unknown);
                }
                if let Decl::Property {
                    preconditions,
                    options,
                    ..
                } = decl
                {
                    for precondition in preconditions {
                        self.visit_expr(precondition, &scope)?;
                    }
                    for option in options {
                        match option {
                            PropertyOption::Tolerance(value, _)
                            | PropertyOption::Seed(value, _)
                            | PropertyOption::Samples(value, _) => {
                                self.visit_expr(value, &scope)?;
                            }
                            PropertyOption::Contract(..) => {}
                        }
                    }
                }
                self.visit_expr(body, &scope)?;
                self.globals.bind(name.clone(), callable);
            }
            Decl::LetDef { name, value, .. } => {
                let scope = self.globals.clone();
                let value = self.visit_expr(value, &scope)?;
                self.globals.bind(name.clone(), value);
            }
            Decl::MacroDef {
                params, body, name, ..
            } => {
                let mut scope = self.globals.clone();
                for param in params {
                    scope.bind(param.clone(), CallableOrigin::Unknown);
                }
                let value = self.visit_expr(body, &scope)?;
                self.globals.bind(name.clone(), value);
            }
            Decl::TypeDef {
                invariant: Some(invariant),
                ..
            } => {
                let mut scope = self.globals.clone();
                scope.bind(invariant.binder.clone(), CallableOrigin::Unknown);
                self.visit_expr(&invariant.body, &scope)?;
            }
            Decl::Import { .. }
            | Decl::Sig { .. }
            | Decl::Dim { .. }
            | Decl::TypeDef {
                invariant: None, ..
            }
            | Decl::TypeAlias { .. }
            | Decl::Export { .. } => {}
        }
        Ok(())
    }

    fn visit_expr(
        &mut self,
        expr: &Expr,
        scope: &CallableScope,
    ) -> Result<CallableOrigin, DesugarError> {
        let ordinary = match expr {
            Expr::Lit(..) | Expr::Constructor(..) => CallableOrigin::NonCallable,
            Expr::Var(name, _) => scope.lookup(name),
            Expr::Apply(function, arguments, _) => {
                self.visit_expr(function, scope)?;
                let payloads = arguments
                    .iter()
                    .map(|argument| self.visit_expr(argument, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                match function.as_ref() {
                    Expr::Constructor(name, _) => CallableOrigin::Constructor {
                        name: name.clone(),
                        payloads,
                    },
                    _ => CallableOrigin::Unknown,
                }
            }
            Expr::List(items, _)
            | Expr::Par(items, _)
            | Expr::Do(items, _)
            | Expr::Tuple(items, _) => {
                let values = items
                    .iter()
                    .map(|item| self.visit_expr(item, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                if matches!(expr, Expr::Tuple(..)) {
                    CallableOrigin::Tuple(values)
                } else if matches!(expr, Expr::Do(..)) {
                    values
                        .last()
                        .cloned()
                        .unwrap_or(CallableOrigin::NonCallable)
                } else {
                    CallableOrigin::NonCallable
                }
            }
            Expr::Record(name, fields, _) => CallableOrigin::Record {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(field, value)| {
                        self.visit_expr(value, scope)
                            .map(|origin| (field.clone(), origin))
                    })
                    .collect::<Result<_, _>>()?,
            },
            Expr::RecordUpdate(base, fields, _) => {
                self.visit_expr(base, scope)?;
                for (_, value) in fields {
                    self.visit_expr(value, scope)?;
                }
                CallableOrigin::NonCallable
            }
            Expr::Access(target, _, _) => {
                self.visit_expr(target, scope)?;
                CallableOrigin::Unknown
            }
            Expr::TupleGet(target, index, _) => {
                let target = self.visit_expr(target, scope)?;
                match target {
                    CallableOrigin::Tuple(values) => usize::try_from(*index)
                        .ok()
                        .and_then(|index| values.get(index).cloned())
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                }
            }
            Expr::Binary(_, left, right, _) => {
                self.visit_expr(left, scope)?;
                self.visit_expr(right, scope)?;
                CallableOrigin::NonCallable
            }
            Expr::Unary(_, operand, _) => {
                self.visit_expr(operand, scope)?;
                CallableOrigin::NonCallable
            }
            Expr::Pipe(head, stages, _) => {
                self.visit_expr(head, scope)?;
                for stage in stages {
                    self.visit_expr(stage, scope)?;
                }
                CallableOrigin::Unknown
            }
            Expr::If(condition, consequence, alternative, _) => {
                self.visit_expr(condition, scope)?;
                let consequence = self.visit_expr(consequence, scope)?;
                let alternative = self.visit_expr(alternative, scope)?;
                consequence.alternate(&alternative)
            }
            Expr::Match(scrutinee, arms, _) => {
                let scrutinee = self.visit_expr(scrutinee, scope)?;
                let mut result: Option<CallableOrigin> = None;
                for arm in arms {
                    let mut arm_scope = scope.clone();
                    bind_match_pattern(&arm.pattern, &scrutinee, &mut arm_scope);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard, &arm_scope)?;
                    }
                    let arm_value = self.visit_expr(&arm.body, &arm_scope)?;
                    result = Some(
                        result
                            .map_or_else(|| arm_value.clone(), |prior| prior.alternate(&arm_value)),
                    );
                }
                result.unwrap_or(CallableOrigin::Unknown)
            }
            Expr::Lambda(params, body, _) => {
                let identity = self.fresh_callable_identity();
                let names = params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<Vec<_>>();
                let mut body_scope = scope.clone();
                for name in &names {
                    body_scope.bind(name.clone(), CallableOrigin::Unknown);
                }
                self.visit_expr(body, &body_scope)?;
                CallableOrigin::Known {
                    identity,
                    display_name: "<inline lambda>".to_string(),
                    params: names,
                }
            }
            Expr::Cast(value, _, _, _) => {
                self.visit_expr(value, scope)?;
                CallableOrigin::NonCallable
            }
            Expr::Grad(function, wrt, span) => {
                let target = self.visit_expr(function, scope)?;
                if let Some(wrt) = wrt {
                    let indices = resolve_selector_indices(function, &target, wrt, *span)?;
                    self.resolved
                        .push((function.as_ref() as *const Expr as usize, indices));
                }
                // `grad` returns a callable with the target's parameter
                // identity. Preserve that identity so a surrounding
                // `grad(..., wrt=...)` can validate its selector instead of
                // treating the nested transform as an unresolved dynamic
                // expression.
                target
            }
            Expr::Vmap(function, _, _)
            | Expr::Jit(function, _)
            | Expr::Realize(function, _)
            | Expr::Copy(function, _)
            | Expr::Borrow(function, _)
            | Expr::Quote(function, _)
            | Expr::Unquote(function, _)
            | Expr::Splice(function, _) => {
                self.visit_expr(function, scope)?;
                CallableOrigin::Unknown
            }
            Expr::WithDevice(argument, body, _) => {
                self.visit_expr(argument, scope)?;
                self.visit_expr(body, scope)?
            }
            Expr::Annotate(value, _, _) => self.visit_expr(value, scope)?,
            Expr::Block(bindings, body, _) => {
                let mut block_scope = scope.clone();
                for binding in bindings {
                    let value = self.visit_expr(&binding.value, &block_scope)?;
                    bind_let_pattern(&binding.pattern, &value, &mut block_scope);
                }
                self.visit_expr(body, &block_scope)?
            }
        };
        Ok(ordinary)
    }

    fn deep_callable_origin(
        &mut self,
        expr: &deep::Expr,
        scope: &CallableScope,
        lexical_name: &str,
    ) -> CallableOrigin {
        let expr = strip_deep_metadata(expr);
        let deep::ExprCarrier::DecodedNode(tag, _, children) = expr.carrier() else {
            return CallableOrigin::Unknown;
        };
        match tag {
            DeepTag::Var => children
                .first()
                .and_then(deep_symbol_name)
                .map_or(CallableOrigin::Unknown, |name| scope.lookup(name)),
            DeepTag::Fn => {
                let Some(deep::ExprCarrier::DecodedNode(DeepTag::Params, _, params)) =
                    children.first().map(deep::Expr::carrier)
                else {
                    return CallableOrigin::Unknown;
                };
                let Some(params) = params.iter().map(deep_parameter_name).collect() else {
                    return CallableOrigin::Unknown;
                };
                CallableOrigin::Known {
                    identity: self.deep_callable_identity(expr),
                    display_name: lexical_name.to_string(),
                    params,
                }
            }
            DeepTag::If if children.len() == 3 => {
                let consequence = self.deep_callable_origin(&children[1], scope, lexical_name);
                let alternative = self.deep_callable_origin(&children[2], scope, lexical_name);
                consequence.alternate(&alternative)
            }
            DeepTag::Match if children.len() >= 2 => {
                let scrutinee = self.deep_callable_origin(&children[0], scope, lexical_name);
                let mut result = None;
                for arm in &children[1..] {
                    let deep::ExprCarrier::DecodedNode(DeepTag::Arm, _, arm_children) =
                        arm.carrier()
                    else {
                        return CallableOrigin::Unknown;
                    };
                    if arm_children.len() != 3 {
                        return CallableOrigin::Unknown;
                    }
                    let mut arm_scope = scope.clone();
                    bind_deep_match_pattern(&arm_children[0], &scrutinee, &mut arm_scope);
                    let arm_origin =
                        self.deep_callable_origin(&arm_children[2], &arm_scope, lexical_name);
                    result = Some(result.map_or_else(
                        || arm_origin.clone(),
                        |prior: CallableOrigin| prior.alternate(&arm_origin),
                    ));
                }
                result.unwrap_or(CallableOrigin::Unknown)
            }
            DeepTag::Tuple => CallableOrigin::Tuple(
                children
                    .iter()
                    .map(|child| self.deep_callable_origin(child, scope, lexical_name))
                    .collect(),
            ),
            DeepTag::App
                if children
                    .first()
                    .and_then(deep_variable_name)
                    .is_some_and(is_constructor_name) =>
            {
                CallableOrigin::Constructor {
                    name: deep_variable_name(&children[0])
                        .expect("constructor guard established the name")
                        .to_string(),
                    payloads: children[1..]
                        .iter()
                        .map(|child| self.deep_callable_origin(child, scope, lexical_name))
                        .collect(),
                }
            }
            DeepTag::Record if !children.is_empty() => {
                let Some(name) = children.first().and_then(deep_symbol_name) else {
                    return CallableOrigin::Unknown;
                };
                let mut fields = BTreeMap::new();
                for field in &children[1..] {
                    let deep::ExprCarrier::DecodedNode(DeepTag::Kv, _, field_children) =
                        field.carrier()
                    else {
                        return CallableOrigin::Unknown;
                    };
                    let [field_name, value] = field_children else {
                        return CallableOrigin::Unknown;
                    };
                    let Some(field_name) = deep_symbol_name(field_name) else {
                        return CallableOrigin::Unknown;
                    };
                    fields.insert(
                        field_name.to_string(),
                        self.deep_callable_origin(value, scope, lexical_name),
                    );
                }
                CallableOrigin::Record {
                    name: name.to_string(),
                    fields,
                }
            }
            DeepTag::TupleGet if children.len() == 2 => {
                let tuple = self.deep_callable_origin(&children[0], scope, lexical_name);
                match tuple {
                    CallableOrigin::Tuple(values) => deep_integer_literal(&children[1])
                        .and_then(|index| usize::try_from(index).ok())
                        .and_then(|index| values.get(index).cloned())
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                }
            }
            DeepTag::Block => children
                .last()
                .map(|child| self.deep_callable_origin(child, scope, lexical_name))
                .unwrap_or(CallableOrigin::NonCallable),
            DeepTag::Let if children.len() == 2 => {
                let deep::ExprCarrier::DecodedNode(DeepTag::Bind, _, bindings) =
                    children[0].carrier()
                else {
                    return CallableOrigin::Unknown;
                };
                if !bindings.len().is_multiple_of(2) {
                    return CallableOrigin::Unknown;
                }
                let mut scoped = scope.clone();
                for pair in bindings.as_chunks::<2>().0 {
                    let Some(name) = deep_symbol_name(&pair[0]) else {
                        return CallableOrigin::Unknown;
                    };
                    let value = self.deep_callable_origin(&pair[1], &scoped, name);
                    scoped.bind(name.to_string(), value);
                }
                self.deep_callable_origin(&children[1], &scoped, lexical_name)
            }
            _ => CallableOrigin::Unknown,
        }
    }
}

fn resolve_selector_indices(
    target_expr: &Expr,
    target: &CallableOrigin,
    wrt: &[String],
    span: Span,
) -> Result<Vec<i64>, DesugarError> {
    let target_name = match target_expr {
        Expr::Var(name, _) => name.clone(),
        Expr::Lambda(..) => "<inline lambda>".to_string(),
        _ => "<dynamic expression>".to_string(),
    };
    let CallableOrigin::Known {
        display_name,
        params,
        ..
    } = target
    else {
        return Err(match target {
            CallableOrigin::NonCallable
            | CallableOrigin::Tuple(_)
            | CallableOrigin::Constructor { .. }
            | CallableOrigin::Record { .. } => DesugarError::NonCallableGradTarget {
                target: target_name,
                span,
            },
            CallableOrigin::Unknown => DesugarError::UnresolvedGradTarget {
                target: target_name,
                span,
            },
            CallableOrigin::Known { .. } => unreachable!(),
        });
    };
    wrt.iter()
        .map(|parameter| {
            params
                .iter()
                .position(|candidate| candidate == parameter)
                .map(|index| index as i64)
                .ok_or_else(|| DesugarError::UnknownGradParameter {
                    callable: display_name.clone(),
                    parameter: parameter.clone(),
                    available: params.clone(),
                    span,
                })
        })
        .collect()
}

fn bind_let_pattern(pattern: &LetPattern, value: &CallableOrigin, scope: &mut CallableScope) {
    match pattern {
        LetPattern::Var(name, _) => scope.bind(name.clone(), value.clone()),
        LetPattern::Wildcard(_) => {}
        LetPattern::Tuple(patterns, _) => {
            for (index, pattern) in patterns.iter().enumerate() {
                let value = match value {
                    CallableOrigin::Tuple(values) => values
                        .get(index)
                        .cloned()
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                };
                bind_let_pattern(pattern, &value, scope);
            }
        }
    }
}

fn bind_match_pattern(pattern: &Pattern, value: &CallableOrigin, scope: &mut CallableScope) {
    match pattern {
        Pattern::Wildcard(..) | Pattern::Lit(..) => {}
        Pattern::Var(name, _) => scope.bind(name.clone(), value.clone()),
        Pattern::Constructor(name, patterns, _) => {
            for (index, pattern) in patterns.iter().enumerate() {
                let value = match value {
                    CallableOrigin::Constructor {
                        name: value_name,
                        payloads,
                    } if value_name == name => payloads
                        .get(index)
                        .cloned()
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                };
                bind_match_pattern(pattern, &value, scope);
            }
        }
        Pattern::Tuple(patterns, _) => {
            for (index, pattern) in patterns.iter().enumerate() {
                let value = match value {
                    CallableOrigin::Tuple(values) => values
                        .get(index)
                        .cloned()
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                };
                bind_match_pattern(pattern, &value, scope);
            }
        }
        Pattern::Record(name, fields, _) => {
            for (field, pattern) in fields {
                let field_value = match value {
                    CallableOrigin::Record {
                        name: value_name,
                        fields,
                    } if value_name == name => fields
                        .get(field)
                        .cloned()
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                };
                bind_match_pattern(pattern, &field_value, scope);
            }
        }
        Pattern::As(name, pattern, _) => {
            scope.bind(name.clone(), value.clone());
            bind_match_pattern(pattern, value, scope);
        }
    }
}

fn deep_symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn deep_variable_name(expr: &deep::Expr) -> Option<&str> {
    let deep::ExprCarrier::DecodedNode(DeepTag::Var, _, children) = expr.carrier() else {
        return None;
    };
    children.first().and_then(deep_symbol_name)
}

fn is_constructor_name(name: &str) -> bool {
    name.rsplit(['.', '_'])
        .find(|component| !component.is_empty())
        .and_then(|component| component.chars().next())
        .is_some_and(char::is_uppercase)
}

fn deep_parameter_name(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name.clone()),
        deep::Expr::MetaExpr(meta, _) => deep_parameter_name(&meta.expr),
        deep::Expr::BareList(elements, _) => elements
            .first()
            .and_then(deep_symbol_name)
            .map(str::to_string),
        _ => None,
    }
}

fn deep_function_parameters(expr: &deep::Expr) -> Option<Vec<String>> {
    let expr = strip_deep_metadata(expr);
    let deep::ExprCarrier::DecodedNode(DeepTag::Fn, _, children) = expr.carrier() else {
        return None;
    };
    let deep::ExprCarrier::DecodedNode(DeepTag::Params, _, params) = children.first()?.carrier()
    else {
        return None;
    };
    params.iter().map(deep_parameter_name).collect()
}

fn strip_deep_metadata(mut expr: &deep::Expr) -> &deep::Expr {
    while let deep::Expr::MetaExpr(meta, _) = expr {
        expr = &meta.expr;
    }
    expr
}

fn deep_integer_literal(expr: &deep::Expr) -> Option<i64> {
    let deep::ExprCarrier::DecodedNode(DeepTag::Lit, _, children) = expr.carrier() else {
        return None;
    };
    let [deep::Expr::Atom(deep::Atom::Int(value), _)] = children else {
        return None;
    };
    Some(*value)
}

fn bind_deep_match_pattern(
    pattern: &deep::Expr,
    value: &CallableOrigin,
    scope: &mut CallableScope,
) {
    let deep::ExprCarrier::DecodedNode(tag, _, children) = pattern.carrier() else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(deep_symbol_name) {
                scope.bind(name.to_string(), value.clone());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(deep_symbol_name) {
                scope.bind(name.to_string(), value.clone());
            }
            if let Some(nested) = children.get(1) {
                bind_deep_match_pattern(nested, value, scope);
            }
        }
        DeepTag::PatTuple => {
            for (index, child) in children.iter().enumerate() {
                let child_value = match value {
                    CallableOrigin::Tuple(values) => values
                        .get(index)
                        .cloned()
                        .unwrap_or(CallableOrigin::Unknown),
                    _ => CallableOrigin::Unknown,
                };
                bind_deep_match_pattern(child, &child_value, scope);
            }
        }
        DeepTag::PatCtor => {
            let pattern_name = children.first().and_then(deep_symbol_name);
            for (index, child) in children.iter().skip(1).enumerate() {
                let child_value = match value {
                    CallableOrigin::Constructor { name, payloads }
                        if pattern_name.is_some_and(|pattern_name| pattern_name == name) =>
                    {
                        payloads
                            .get(index)
                            .cloned()
                            .unwrap_or(CallableOrigin::Unknown)
                    }
                    _ => CallableOrigin::Unknown,
                };
                bind_deep_match_pattern(child, &child_value, scope);
            }
        }
        DeepTag::PatRecord => {
            let pattern_name = children.first().and_then(deep_symbol_name);
            for field in children.iter().skip(1) {
                let deep::ExprCarrier::DecodedNode(DeepTag::Kv, _, field_children) =
                    field.carrier()
                else {
                    continue;
                };
                if let [field_name, value_pattern] = field_children {
                    let field_value = match value {
                        CallableOrigin::Record { name, fields }
                            if pattern_name.is_some_and(|pattern_name| pattern_name == name) =>
                        {
                            deep_symbol_name(field_name)
                                .and_then(|field_name| fields.get(field_name))
                                .cloned()
                                .unwrap_or(CallableOrigin::Unknown)
                        }
                        _ => CallableOrigin::Unknown,
                    };
                    bind_deep_match_pattern(value_pattern, &field_value, scope);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
fn desugar_decl(decl: &Decl) -> Vec<deep::Expr> {
    desugar_decl_only(decl).expect("internal Surf declaration fixture must desugar")
}

#[cfg(test)]
fn desugar_expr(expr: &Expr) -> deep::Expr {
    desugar_expr_only(expr).expect("internal Surf expression fixture must desugar")
}

// ---------------------------------------------------------------------------
// Helpers for building Deep AST nodes (3-tuple format)
// ---------------------------------------------------------------------------

fn sp() -> Span {
    Span::new(0, 0)
}

fn sym(s: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Name(s.to_string()), sp())
}

fn int(n: i64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Int(n), sp())
}

fn surf_span_id(span: Span) -> Option<String> {
    if span.len == 0 {
        None
    } else {
        Some(format!("surf:{}..{}", span.offset, span.end()))
    }
}

fn span_entry(span: Span) -> Option<M> {
    surf_span_id(span)
        .map(|id| M::Span(SpanId::try_new(id, sp()).expect("generated Surf span identifier")))
}
fn type_metadata(ty: deep::Expr) -> M {
    M::Type(TypeSyntax::try_new(ty).expect("desugared type syntax"))
}
fn meta_with_type(ty: deep::Expr) -> deep::Expr {
    meta_with_entries(vec![type_metadata(ty)])
}
fn meta_with_integer_float_type(ty: deep::Expr, style: Option<LiteralStyle>) -> deep::Expr {
    let mut entries = vec![
        type_metadata(ty),
        M::LiteralSource(Spanned::new(LiteralOrigin::Integer, sp())),
    ];
    if let Some(style) = style {
        entries.push(M::SurfLiteralStyle(Spanned::new(style, sp())));
    }
    meta_with_entries(entries)
}
fn meta_with_entries(entries: Vec<M>) -> deep::Expr {
    deep::Expr::Map(
        deep::Metadata::try_from_values(entries).expect("desugar emits distinct annotation keys"),
        sp(),
    )
}
fn numeric_literal_meta(ty: deep::Expr, style: LiteralStyle) -> deep::Expr {
    meta_with_entries(vec![
        type_metadata(ty),
        M::SurfLiteralStyle(Spanned::new(style, sp())),
    ])
}
fn with_metadata_value(expr: deep::Expr, value: M) -> deep::Expr {
    match expr {
        deep::Expr::Node(mut node, span) => {
            let mut meta = node.meta().clone();
            meta.replace(value);
            node.try_replace_meta(meta)
                .expect("desugared annotation has an admissible owner");
            deep::Expr::Node(node, span)
        }
        other => other,
    }
}
fn has_type_metadata(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::Node(node, _) => node.meta().ty().is_some(),
        _ => false,
    }
}

/// Build a stamped Deep node. Decode-once (chelis#731 Phase 3): the
/// desugarer is a typed producer, so the tag is validated at construction
/// time via `Node::new`.
fn node(tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::Node(
        Box::new(chelis_deep::node::Node::new(
            tag,
            deep::Metadata::default(),
            children,
        )),
        sp(),
    )
}

/// Build a compiler-internal pre-expansion form (`defmacro`), whose tag is
/// deliberately OUTSIDE the public 62-tag vocabulary (spec/03 macro
/// boundary rule) and therefore stays an undecoded head: an `UnknownForm`,
/// exactly as the stamper carries it. chelis-macros expands it away before
/// any public consumer dispatches on tags; it is a recorded raw-string entry
/// point per checker_totality.md §C1.2.
fn internal_node(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    debug_assert!(
        DeepTag::parse(tag).is_none(),
        "vocabulary tags must go through the typed `node` constructor"
    );
    deep::Expr::UnknownForm(Box::new(deep::UnknownFormData {
        head: tag.to_string(),
        meta: deep::Metadata::default(),
        children,
        span: sp(),
    }))
}

/// Attach `dtype_bounds` metadata for every bounded binder in `binders`
/// (`spec/03-deep-syntax.md` §1.1). A list with no bound leaves the node
/// untouched, so canonical Deep for an unbounded declaration is unchanged.
fn with_dtype_bounds(expr: deep::Expr, binders: &[TypeBinder]) -> deep::Expr {
    let bounds: Vec<(String, chelis_deep::DtypeBound)> = binders
        .iter()
        .filter_map(|binder| {
            binder
                .bound
                .clone()
                .map(|bound| (binder.name.clone(), bound))
        })
        .collect();
    if bounds.is_empty() {
        return expr;
    }
    let deep::Expr::Node(mut node, span) = expr else {
        panic!("dtype bounds attach to a stamped declaration node");
    };
    let mut meta = node.meta().clone();
    meta.insert(M::DtypeBounds(
        encode_dtype_bounds(&bounds, sp()).expect("distinct dtype binder names"),
    ))
    .expect("bounds attached once");
    node.try_replace_meta(meta)
        .expect("dtype-bound metadata preserves the stamped Node invariant");
    deep::Expr::Node(node, span)
}

/// Build a stamped Deep node with custom metadata. The `meta` argument
/// must be an `Expr::Map(Metadata { .. }, _)` — the Metadata is extracted
/// and passed to `Node::new`.
fn node_meta(tag: DeepTag, meta: deep::Expr, children: Vec<deep::Expr>) -> deep::Expr {
    let meta_map = match meta {
        deep::Expr::Map(m, _) => m,
        _ => panic!("node_meta: expected Expr::Map for metadata, got {meta:?}"),
    };
    deep::Expr::Node(
        Box::new(chelis_deep::node::Node::new(tag, meta_map, children)),
        sp(),
    )
}

/// Preserve a Surf type expression's byte range in the structural Deep span
/// without changing canonical Deep metadata or printer output. Type-resolution
/// diagnostics use this when no external `span` metadata is present.
fn with_structural_span(expr: deep::Expr, span: Span) -> deep::Expr {
    match expr {
        deep::Expr::Atom(atom, _) => deep::Expr::Atom(atom, span),
        deep::Expr::Map(map, _) => deep::Expr::Map(map, span),
        deep::Expr::MetaExpr(meta, _) => deep::Expr::MetaExpr(meta, span),
        deep::Expr::Node(node, _) => deep::Expr::Node(node, span),
        deep::Expr::BareList(elems, _) => deep::Expr::BareList(elems, span),
        deep::Expr::UnknownForm(data) => {
            let mut d = *data;
            d.span = span;
            deep::Expr::UnknownForm(Box::new(d))
        }
    }
}

fn type_expr_span(ty: &TypeExpr) -> Span {
    match ty {
        TypeExpr::Named(_, span)
        | TypeExpr::DimensionLiteral(_, span)
        | TypeExpr::Tensor(_, _, span)
        | TypeExpr::Arrow(_, _, span)
        | TypeExpr::Ref(_, span)
        | TypeExpr::App(_, _, span)
        | TypeExpr::Tuple(_, span)
        | TypeExpr::Infer(span)
        | TypeExpr::RankSpread(_, span) => *span,
    }
}

/// Variable reference: (var {} name)
fn dvar(name: &str) -> deep::Expr {
    node(DeepTag::Var, vec![sym(name)])
}

fn attach_span_metadata(expr: deep::Expr, span: Span) -> deep::Expr {
    match span_entry(span) {
        Some(entry) => with_metadata_value(expr, entry),
        None => expr,
    }
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::Lit(_, span)
        | Expr::Var(_, span)
        | Expr::Constructor(_, span)
        | Expr::Apply(_, _, span)
        | Expr::List(_, span)
        | Expr::Record(_, _, span)
        | Expr::RecordUpdate(_, _, span)
        | Expr::Access(_, _, span)
        | Expr::TupleGet(_, _, span)
        | Expr::Binary(_, _, _, span)
        | Expr::Unary(_, _, span)
        | Expr::Pipe(_, _, span)
        | Expr::If(_, _, _, span)
        | Expr::Match(_, _, span)
        | Expr::Lambda(_, _, span)
        | Expr::Tuple(_, span)
        | Expr::Cast(_, _, _, span)
        | Expr::Grad(_, _, span)
        | Expr::Vmap(_, _, span)
        | Expr::Jit(_, span)
        | Expr::Realize(_, span)
        | Expr::Copy(_, span)
        | Expr::Borrow(_, span)
        | Expr::WithDevice(_, _, span)
        | Expr::Par(_, span)
        | Expr::Do(_, span)
        | Expr::Quote(_, span)
        | Expr::Unquote(_, span)
        | Expr::Splice(_, span)
        | Expr::Annotate(_, _, span)
        | Expr::Block(_, _, span) => *span,
    }
}

/// Build a bare list (no tag/meta) for structural helpers like params, bind
fn bare_list(elements: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::BareList(elements, sp())
}

fn lower_module_path(path: &str) -> String {
    path.to_ascii_lowercase()
}

/// Desugar a parameter with both a declared dim-variable scope and a
/// declared type-variable scope. For `def f[..](...)` parameters, a name in
/// the precision slot of a `tensor[..., <name>]` annotation becomes
/// `(t-var {} <name>)` only when the declaration's complete `[..]` list
/// contains it (`spec/02-surf-syntax.md` §P4b).
fn desugar_param_with_scope(
    param: &Param,
    dim_vars: &UnordSet<String>,
    tvar_set: &UnordSet<String>,
) -> deep::Expr {
    let annotation = param
        .ty
        .as_ref()
        .map(|ty| desugar_type_with_scope(ty, dim_vars, tvar_set));
    desugar_param_with_annotation(param, annotation)
}

fn desugar_param_with_annotation(param: &Param, annotation: Option<deep::Expr>) -> deep::Expr {
    match annotation {
        Some(ty) if typed_param_needs_meta_wrapper(&param.name) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                metadata: deep::Metadata::from(type_metadata(ty)),
                expr: Box::new(sym(&param.name)),
            },
            sp(),
        ),
        Some(ty) => bare_list(vec![sym(&param.name), meta_with_type(ty)]),
        None => sym(&param.name),
    }
}

fn typed_param_needs_meta_wrapper(name: &str) -> bool {
    matches!(
        name,
        "module"
            | "import"
            | "import-all"
            | "export"
            | "def"
            | "defsig"
            | "deftype"
            | "typealias"
            | "variant"
            | "field"
            | "defdim"
            | "fn"
            | "app"
            | "let"
            | "match"
            | "arm"
            | "if"
            | "var"
            | "lit"
            | "record"
            | "access"
            | "pipe"
            | "block"
            | "tuple"
            | "tuple-get"
            | "record-update"
            | "par"
            | "pat-var"
            | "pat-lit"
            | "pat-ctor"
            | "pat-tuple"
            | "pat-record"
            | "pat-wild"
            | "pat-as"
            | "t-prim"
            | "t-fn"
            | "t-tensor"
            | "t-adt"
            | "t-var"
            | "t-unit"
            | "t-tuple"
            | "d-name"
            | "d-var"
            | "d-lit"
            | "grad"
            | "vmap"
            | "jit"
            | "realize"
            | "cast"
            | "copy"
            | "quote"
            | "unquote"
            | "splice"
            | "params"
            | "bind"
            | "kv"
            | "defmacro"
            | "expand"
            | "effects"
            | "resource"
            | "handle-effect"
    )
}

/// Inject a type annotation into the metadata of a desugared expression.
fn inject_type_metadata(expr: deep::Expr, ty: deep::Expr) -> deep::Expr {
    match expr {
        deep::Expr::Node(..) => with_metadata_value(expr, type_metadata(ty)),
        other => node_meta(DeepTag::Var, meta_with_type(ty), vec![other]),
    }
}

fn desugar_effect_set(effects: &[EffectExpr]) -> AstEffectSet {
    let values = effects
        .iter()
        .map(|effect| match effect {
            EffectExpr::Diff(_) => EffectMember::Name(Spanned::new("diff".into(), sp())),
            EffectExpr::Accum(_) => EffectMember::Name(Spanned::new("accum".into(), sp())),
            EffectExpr::Io(_) => EffectMember::Name(Spanned::new("io".into(), sp())),
            EffectExpr::Test(_) => EffectMember::Name(Spanned::new("test".into(), sp())),
            EffectExpr::Resource(device, _) => EffectMember::Resource(
                ResourceEffect::new(
                    Spanned::new(device.clone(), sp()),
                    deep::Metadata::default(),
                    sp(),
                )
                .expect("resource effect"),
            ),
        })
        .collect();
    AstEffectSet::new(deep::Metadata::default(), values, sp())
}

fn apply_effect_metadata(ty_expr: deep::Expr, effects: &Option<Vec<EffectExpr>>) -> deep::Expr {
    match (effects, ty_expr) {
        // An explicit `! { ... }` clause — even the empty `! {}` — must be preserved in
        // the Deep AST so the effect checker can distinguish "declared empty" from
        // "no annotation" when validating declared vs inferred effects.
        (Some(effects), deep::Expr::Node(mut node, span)) => {
            if node.tag() == DeepTag::TFn {
                let mut meta = node.meta().clone();
                meta.replace(M::Eff(desugar_effect_set(effects)));
                node.try_replace_meta(meta)
                    .expect("effect annotation must preserve the stamped Node invariant");
            }
            deep::Expr::Node(node, span)
        }
        (_, other) => other,
    }
}

fn fresh_pipe_param_name(stage: &Expr) -> String {
    let base = "__chelis_pipe";
    let mut index = 0;
    loop {
        let candidate = if index == 0 {
            base.to_string()
        } else {
            format!("{base}{index}")
        };
        if !expr_mentions_name(stage, &candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn is_first_argument_pipe_lambda(expr: &Expr) -> bool {
    let Expr::Lambda(params, body, _) = expr else {
        return false;
    };
    let [param] = params.as_slice() else {
        return false;
    };
    if param.ty.is_some() {
        return false;
    }
    let is_param = |expr: &Expr| matches!(expr, Expr::Var(name, _) if name == &param.name);
    match body.as_ref() {
        Expr::Apply(_, arguments, _) => arguments.first().is_some_and(is_param),
        Expr::Realize(argument, _) | Expr::Copy(argument, _) | Expr::Cast(argument, _, _, _) => {
            is_param(argument)
        }
        _ => false,
    }
}

fn mark_call_first_pipe_stage(expr: deep::Expr) -> deep::Expr {
    with_metadata_value(
        expr,
        M::SurfPipeStage(Spanned::new(PipeStageOrigin::CallFirst, sp())),
    )
}

fn expr_mentions_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Lit(_, _) => false,
        Expr::Var(found, _) | Expr::Constructor(found, _) => found == name,
        Expr::List(items, _) => items.iter().any(|item| expr_mentions_name(item, name)),
        Expr::Apply(func, args, _) => {
            expr_mentions_name(func, name) || args.iter().any(|arg| expr_mentions_name(arg, name))
        }
        Expr::Record(_, fields, _) => fields
            .iter()
            .any(|(_, value)| expr_mentions_name(value, name)),
        Expr::RecordUpdate(base, fields, _) => {
            expr_mentions_name(base, name)
                || fields
                    .iter()
                    .any(|(_, value)| expr_mentions_name(value, name))
        }
        Expr::Access(base, field, _) => expr_mentions_name(base, name) || field == name,
        Expr::TupleGet(base, _, _) => expr_mentions_name(base, name),
        Expr::Binary(_, lhs, rhs, _) => {
            expr_mentions_name(lhs, name) || expr_mentions_name(rhs, name)
        }
        Expr::Unary(_, operand, _) => expr_mentions_name(operand, name),
        Expr::Pipe(seed, stages, _) => {
            expr_mentions_name(seed, name)
                || stages.iter().any(|stage| expr_mentions_name(stage, name))
        }
        Expr::If(cond, then_e, else_e, _) => {
            expr_mentions_name(cond, name)
                || expr_mentions_name(then_e, name)
                || expr_mentions_name(else_e, name)
        }
        Expr::Match(scrutinee, arms, _) => {
            expr_mentions_name(scrutinee, name)
                || arms.iter().any(|arm| {
                    pattern_mentions_name(&arm.pattern, name)
                        || arm
                            .guard
                            .as_ref()
                            .is_some_and(|guard| expr_mentions_name(guard, name))
                        || expr_mentions_name(&arm.body, name)
                })
        }
        Expr::Block(bindings, body, _) => {
            bindings.iter().any(|binding| {
                let_pattern_mentions_name(&binding.pattern, name)
                    || binding
                        .ty
                        .as_ref()
                        .is_some_and(|ty| type_mentions_name(ty, name))
                    || expr_mentions_name(&binding.value, name)
            }) || expr_mentions_name(body, name)
        }
        Expr::Lambda(params, body, _) => {
            params.iter().any(|param| {
                param.name == name
                    || param
                        .ty
                        .as_ref()
                        .is_some_and(|ty| type_mentions_name(ty, name))
            }) || expr_mentions_name(body, name)
        }
        Expr::Tuple(items, _) | Expr::Par(items, _) | Expr::Do(items, _) => {
            items.iter().any(|item| expr_mentions_name(item, name))
        }
        Expr::Cast(expr, precision, _, _) => expr_mentions_name(expr, name) || precision == name,
        Expr::Grad(expr, wrt, _) => {
            expr_mentions_name(expr, name)
                || wrt
                    .as_ref()
                    .is_some_and(|names| names.iter().any(|wrt_name| wrt_name == name))
        }
        Expr::Vmap(expr, _, _)
        | Expr::Jit(expr, _)
        | Expr::Realize(expr, _)
        | Expr::Copy(expr, _)
        | Expr::Borrow(expr, _)
        | Expr::Quote(expr, _)
        | Expr::Unquote(expr, _)
        | Expr::Splice(expr, _)
        | Expr::Annotate(expr, _, _) => expr_mentions_name(expr, name),
        Expr::WithDevice(device, body, _) => {
            expr_mentions_name(device, name) || expr_mentions_name(body, name)
        }
    }
}

fn let_pattern_mentions_name(pattern: &LetPattern, name: &str) -> bool {
    match pattern {
        LetPattern::Var(found, _) => found == name,
        LetPattern::Wildcard(_) => false,
        LetPattern::Tuple(items, _) => items
            .iter()
            .any(|item| let_pattern_mentions_name(item, name)),
    }
}

fn pattern_mentions_name(pattern: &Pattern, name: &str) -> bool {
    match pattern {
        Pattern::Wildcard(_) | Pattern::Lit(_, _) => false,
        Pattern::Var(found, _) => found == name,
        Pattern::Constructor(found, items, _) => {
            found == name || items.iter().any(|item| pattern_mentions_name(item, name))
        }
        Pattern::Tuple(items, _) => items.iter().any(|item| pattern_mentions_name(item, name)),
        Pattern::Record(found, fields, _) => {
            found == name
                || fields
                    .iter()
                    .any(|(field, value)| field == name || pattern_mentions_name(value, name))
        }
        Pattern::As(found, inner, _) => found == name || pattern_mentions_name(inner, name),
    }
}

fn type_mentions_name(ty: &TypeExpr, name: &str) -> bool {
    match ty {
        TypeExpr::Named(found, _) => found == name,
        TypeExpr::DimensionLiteral(_, _) => false,
        TypeExpr::RankSpread(found, _) => found == name,
        TypeExpr::Tensor(items, precision, _) => {
            precision == name || items.iter().any(|item| type_mentions_name(item, name))
        }
        TypeExpr::Arrow(args, ret, _) => {
            args.iter().any(|arg| type_mentions_name(arg, name)) || type_mentions_name(ret, name)
        }
        TypeExpr::Ref(inner, _) => type_mentions_name(inner, name),
        TypeExpr::App(found, args, _) => {
            found == name || args.iter().any(|arg| type_mentions_name(arg, name))
        }
        TypeExpr::Tuple(items, _) => items.iter().any(|item| type_mentions_name(item, name)),
        TypeExpr::Infer(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

/// Visit `decl` and every declaration nested inside a `Decl::Module`,
/// calling `visit` on each. Centralizes the module descent shared by the
/// `DesugarCtx::new` pre-pass collectors so a future nesting variant only
/// needs handling in one place. Every idiomatic Surf source wraps its
/// declarations in a single `module`, so without this descent the collectors
/// would see only the wrapper and miss everything inside.
fn for_each_decl(decl: &Decl, visit: &mut impl FnMut(&Decl)) {
    visit(decl);
    if let Decl::Module { decls, .. } = decl {
        for d in decls {
            for_each_decl(d, visit);
        }
    }
}

/// Merge inline and standalone-signature binders by declaration name.
fn collect_declared_type_binders(
    decl: &Decl,
    out: &mut UnordMap<String, UnordMap<String, Option<chelis_deep::DtypeBound>>>,
) {
    let (name, type_binders) = match decl {
        Decl::FunDef {
            name, type_binders, ..
        } => (name, type_binders),
        Decl::Sig {
            name, type_binders, ..
        } => (name, type_binders),
        Decl::Property {
            name, type_binders, ..
        } => (name, type_binders),
        _ => return,
    };
    let entry = out.entry(name.clone()).or_default();
    for binder in type_binders {
        let slot = entry.entry(binder.name.clone()).or_insert(None);
        if slot.is_none() {
            *slot = binder.bound.clone();
        }
    }
    if entry.is_empty() {
        out.remove(name);
    }
}

/// Build canonical Deep `defsig` children. A declaration with no binders
/// keeps the compact `(defsig {} name type)` form; a polymorphic declaration
/// carries its complete explicit list as `(defsig {} name (binders...) type)`.
fn defsig_node(name: &str, type_binders: &[TypeBinder], ty: deep::Expr) -> deep::Expr {
    let mut children = vec![sym(name)];
    if !type_binders.is_empty() {
        children.push(bare_list(
            type_binders
                .iter()
                .map(|binder| sym(&binder.name))
                .collect(),
        ));
    }
    children.push(ty);
    node(DeepTag::Defsig, children)
}

/// Collect names with a standalone `sig`, suppressing a synthesized duplicate
/// `defsig` for the same annotated `def` (chelis#285).
fn collect_explicit_sig_names(decl: &Decl, out: &mut UnordSet<String>) {
    if let Decl::Sig { name, .. } = decl {
        out.insert(name.clone());
    }
}

/// Collect the explicit effect clause (`! { ... }`) declared on each `def`,
/// keyed by name, so `Decl::Sig` desugaring can inherit it when the explicit
/// sig declares no effects of its own (chelis#285 — see the `def_effects`
/// field doc). An empty `! {}` is stored too: it declares "no effects" and
/// must be distinguished from no annotation at all.
fn collect_def_effects(decl: &Decl, out: &mut UnordMap<String, Vec<EffectExpr>>) {
    if let Decl::FunDef {
        name,
        effects: Some(effects),
        ..
    } = decl
    {
        out.insert(name.clone(), effects.clone());
    }
}

/// Collect per-position tensor element-prim names from each top-level
/// function's declared signature. Used by the contextual tensor-literal
/// inference rule (spec §P10b / §5.6) to narrow numeric literals in
/// argument positions whose declared parameter type is a tensor.
///
/// `Decl::Sig` (a separate signature declaration) is also collected so
/// `sig f: tensor[3, f64] -> ...` followed by an untyped `def f` participates.
fn collect_top_level_fn_tensor_param_prec(
    decl: &Decl,
    out: &mut UnordMap<String, Vec<Option<String>>>,
) {
    match decl {
        Decl::FunDef { name, params, .. } => {
            let entry: Vec<Option<String>> = params
                .iter()
                .map(|p| p.ty.as_ref().and_then(tensor_element_prim_name))
                .collect();
            // Only insert if at least one parameter has a tensor element
            // type — otherwise an entry would still be returned but with
            // all-None which the lookup harmlessly ignores. Cheap
            // pre-filter to keep the map sparse.
            if entry.iter().any(Option::is_some) {
                out.insert(name.clone(), entry);
            }
        }
        Decl::Sig {
            name,
            ty: TypeExpr::Arrow(args, _ret, _),
            ..
        } => {
            // sig f: A -> B -> C is a flat Arrow; collect each non-return
            // arrow position.
            let entry: Vec<Option<String>> = args.iter().map(tensor_element_prim_name).collect();
            if entry.iter().any(Option::is_some) {
                out.insert(name.clone(), entry);
            }
        }
        _ => {}
    }
}

/// Return the precision name (e.g. `"f64"`, `"i32"`) for a tensor type
/// expression, or `None` for any other shape. Tensor type expressions in
/// Surf carry the spelling and exact token span in `TensorPrecision`.
fn tensor_element_prim_name(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Tensor(_, prec, _) => {
            Some(canonical_primitive_name(prec).unwrap_or(prec).to_owned())
        }
        _ => None,
    }
}

impl DesugarCtx {
    fn desugar_pipe_stage(&self, stage: &Expr, local_fn_params: &[String]) -> deep::Expr {
        match stage {
            Expr::Apply(func, args, span) if !args.is_empty() => {
                let pipe_param = fresh_pipe_param_name(stage);
                let mut applied_args = Vec::with_capacity(args.len() + 1);
                applied_args.push(Expr::Var(pipe_param.clone(), *span));
                applied_args.extend(args.iter().cloned());
                let lambda = Expr::Lambda(
                    vec![Param {
                        name: pipe_param,
                        ty: None,
                        span: *span,
                    }],
                    Box::new(Expr::Apply(func.clone(), applied_args, *span)),
                    *span,
                );
                mark_call_first_pipe_stage(self.desugar_expr_with_scope(&lambda, local_fn_params))
            }
            Expr::Lambda(..) if is_first_argument_pipe_lambda(stage) => {
                mark_call_first_pipe_stage(self.desugar_expr_with_scope(stage, local_fn_params))
            }
            _ => self.desugar_expr_with_scope(stage, local_fn_params),
        }
    }

    fn desugar_decl(&self, decl: &Decl) -> Vec<deep::Expr> {
        let mut lowered = match decl {
            Decl::FunDef {
                name,
                type_binders,
                params,
                ret_ty,
                effects,
                body,
                ..
            } => self.desugar_fun_def(name, type_binders, params, ret_ty, effects, body),

            Decl::Property {
                name,
                type_binders,
                params,
                preconditions,
                body,
                options,
                ..
            } => self.desugar_property(name, type_binders, params, preconditions, body, options),

            Decl::LetDef {
                name,
                ty: Some(t),
                value,
                ..
            } => {
                // Position 1 (spec §P10b / §5.6): RHS of a let-binding
                // whose declared type is a tensor type. Narrow numeric
                // literals in `value` to the tensor element type.
                let body = match (tensor_element_prim_name(t), value) {
                    (Some(prec), Expr::List(items, _)) => {
                        self.desugar_list_as_tensor_literal(items, &prec, &[])
                    }
                    _ => self.desugar_expr(value),
                };
                vec![
                    node(DeepTag::Defsig, vec![sym(name), desugar_type(t)]),
                    node(DeepTag::Def, vec![sym(name), body]),
                ]
            }

            Decl::LetDef {
                name,
                ty: None,
                value,
                ..
            } => {
                // chelis#1625: a lambda-valued top-level binding whose type
                // comes from a standalone `sig` must desugar `cast` targets
                // naming that sig's binders the same way `desugar_fun_def`
                // does for the `def f(...) = ...` spelling — as `(t-var {}
                // <name>)`, not `(t-prim {} <name>)`. Without installing this
                // scope, `current_type_binder` never sees the sig's binders
                // here, so `cast(v, p)` under `sig f[p]: p -> p` silently kept
                // the `t-prim` default and slipped past the [04-DTYPE-1]
                // classifier in `chelis_deep::literal_source`, which only
                // recognizes a `t-var` cast target.
                let restore_binders = self.current_type_binders.replace(
                    self.declared_type_binders
                        .get(name)
                        .cloned()
                        .unwrap_or_default(),
                );
                let body = match (self.top_level_binding_tensor_prec.get(name), value) {
                    (Some(precision), Expr::List(items, _)) => {
                        self.desugar_list_as_tensor_literal(items, precision, &[])
                    }
                    _ if !self.explicit_sig_names.contains(name)
                        && is_bare_numeric_tensor_literal(value) =>
                    {
                        self.desugar_default_tensor_binding(value)
                    }
                    _ => self.desugar_expr(value),
                };
                self.current_type_binders.replace(restore_binders);
                vec![node(DeepTag::Def, vec![sym(name), body])]
            }

            Decl::MacroDef {
                name, params, body, ..
            } => {
                let params = node(
                    DeepTag::Params,
                    params.iter().map(|param| sym(param)).collect(),
                );
                vec![internal_node(
                    "defmacro",
                    vec![sym(name), params, self.desugar_expr(body)],
                )]
            }

            Decl::TypeDef {
                name,
                params,
                variants,
                opaque,
                invariant,
                ..
            } => vec![self.desugar_type_def(name, params, variants, *opaque, invariant.as_ref())],

            Decl::TypeAlias {
                name, params, ty, ..
            } => {
                let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
                let explicit_params: UnordSet<String> = params.iter().cloned().collect();
                vec![node(
                    DeepTag::Typealias,
                    vec![
                        sym(name),
                        param_list,
                        desugar_declaration_type(ty, &explicit_params),
                    ],
                )]
            }

            Decl::Sig {
                name,
                type_binders,
                ty,
                effects,
                ..
            } => {
                // chelis#285: the synthesized `defsig` that used to carry a
                // same-named `def`'s `! { ... }` clause is suppressed when this
                // explicit sig exists (see `desugar_fun_def`). The effect
                // upper-bound check reads the declared effect set only from a
                // `defsig`'s `t-fn` `eff` metadata, so if this sig declares no
                // effects of its own, inherit the def's clause here. Otherwise
                // suppressing the synthesized `defsig` would silently drop the
                // def's effect contract and let its body leak effects unchecked.
                let effects = effects
                    .clone()
                    .or_else(|| self.def_effects.get(name).cloned());
                // §P4b: the sig's `[..]` list is the complete, authoritative,
                // unkinded binder set. No type, dimension, or rank variable
                // is introduced by occurrence alone.
                let declared: UnordSet<String> = type_binders
                    .iter()
                    .map(|binder| binder.name.clone())
                    .collect();
                vec![with_dtype_bounds(
                    defsig_node(
                        name,
                        type_binders,
                        apply_effect_metadata(desugar_sig_type(ty, &declared), &effects),
                    ),
                    type_binders,
                )]
            }

            Decl::Dim { names, .. } => names
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    if index == 0 {
                        node_meta(
                            DeepTag::Defdim,
                            meta_with_entries(vec![M::SurfDimGroupSize(
                                PositiveInteger::try_new(int(names.len() as i64))
                                    .expect("nonempty dimension group"),
                            )]),
                            vec![sym(name)],
                        )
                    } else {
                        node(DeepTag::Defdim, vec![sym(name)])
                    }
                })
                .collect(),

            Decl::Module { name, decls, .. } => {
                let mut children = vec![sym(&lower_module_path(name))];
                for d in decls {
                    children.extend(self.desugar_decl(d));
                }
                vec![node_meta(
                    DeepTag::Module,
                    meta_with_entries(vec![M::SurfPath(Spanned::new(name.to_string(), sp()))]),
                    children,
                )]
            }

            Decl::Import { module, kind, .. } => match kind {
                ImportKind::Names(ns) => {
                    let name_list = bare_list(ns.iter().map(|n| sym(n)).collect());
                    vec![node_meta(
                        DeepTag::Import,
                        meta_with_entries(vec![M::SurfPath(Spanned::new(
                            module.to_string(),
                            sp(),
                        ))]),
                        vec![sym(&lower_module_path(module)), name_list],
                    )]
                }
                ImportKind::Qualified => vec![node_meta(
                    DeepTag::Import,
                    meta_with_entries(vec![M::SurfPath(Spanned::new(module.to_string(), sp()))]),
                    vec![sym(&lower_module_path(module)), bare_list(vec![])],
                )],
                ImportKind::All => vec![node_meta(
                    DeepTag::ImportAll,
                    meta_with_entries(vec![M::SurfPath(Spanned::new(module.to_string(), sp()))]),
                    vec![sym(&lower_module_path(module))],
                )],
            },

            Decl::Export { names, .. } => {
                let mut children = Vec::new();
                for n in names {
                    children.push(sym(n));
                }
                vec![node(DeepTag::Export, children)]
            }
        };
        for expr in &mut lowered {
            if let deep::Expr::Node(node, _) = expr
                && matches!(
                    node.tag(),
                    DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias
                )
                && node.meta().span_id().is_none()
                && let Some(entry) = span_entry(decl.span())
            {
                let mut meta = node.meta().clone();
                meta.insert(entry)
                    .expect("declaration has no existing span ID");
                node.try_replace_meta(meta)
                    .expect("Surf declaration span is valid metadata");
            }
        }
        lowered
    }

    fn desugar_fun_def(
        &self,
        name: &str,
        type_binders: &[TypeBinder],
        params: &[Param],
        ret_ty: &Option<TypeExpr>,
        effects: &Option<Vec<EffectExpr>>,
        body: &Expr,
    ) -> Vec<deep::Expr> {
        // One declaration has one binder owner. For an inline-only def this
        // map contains the def's list; for a standalone sig plus matching def
        // it contains the sig's list. Matching-def annotations and the body
        // must resolve in that same scope even though the def cannot repeat it.
        let dim_set: UnordSet<String> = self
            .declared_type_binders
            .get(name)
            .map(|binders| {
                binders
                    .to_sorted()
                    .into_iter()
                    .map(|(binder, _)| binder.clone())
                    .collect()
            })
            .unwrap_or_default();
        let declares_bound = type_binders.iter().any(|binder| binder.bound.is_some());

        // P4b: names in the declaration's sole binder list become `t-var`,
        // `d-var`, or `d-rank` according to the annotation position. The
        // scope may come from this def or its standalone sig.
        let param_ann_tvar_set: UnordSet<String> = dim_set.clone();

        let synthesize_signature = (!type_binders.is_empty()
            || params.iter().any(|param| param.ty.is_some())
            || ret_ty.is_some()
            || effects.is_some()
            || declares_bound)
            && !self.explicit_sig_names.contains(name);
        let param_names: Vec<deep::Expr> = params
            .iter()
            .map(|param| {
                let annotation = param.ty.as_ref().map(|ty| {
                    if synthesize_signature {
                        // The signature owns the actual type. This hole keeps
                        // the source annotation's presence, but adds no constraint.
                        node(DeepTag::TVar, vec![sym("_")])
                    } else {
                        desugar_type_with_scope(ty, &dim_set, &param_ann_tvar_set)
                    }
                });
                desugar_param_with_annotation(param, annotation)
            })
            .collect();
        let params_node = node(DeepTag::Params, param_names);
        let body_scope: Vec<String> = params.iter().map(|param| param.name.clone()).collect();
        // Position 3 (spec §P10b / §5.6): body expression of a function
        // whose declared return type is a tensor type and whose body is
        // itself a tensor literal. Narrow numeric literals in `body` to
        // the tensor element type.
        // Binder scope controls `t-var` cast targets and literal adoption.
        let restore_binders = self.current_type_binders.replace(
            self.declared_type_binders
                .get(name)
                .cloned()
                .unwrap_or_default(),
        );
        let desugared_body = match (ret_ty.as_ref().and_then(tensor_element_prim_name), body) {
            (Some(prec), Expr::List(items, _)) => {
                self.desugar_list_as_tensor_literal(items, &prec, &body_scope)
            }
            _ => self.desugar_expr_with_scope(body, &body_scope),
        };
        self.current_type_binders.replace(restore_binders);
        let fn_node = node(DeepTag::Fn, vec![params_node, desugared_body]);
        let def_node = node(DeepTag::Def, vec![sym(name), fn_node]);
        // Bound ownership is validated before desugaring: a standalone sig
        // and its def cannot each author bounds. Only defsig carries them.

        // chelis#285: when an explicit standalone `sig` already declares this
        // name, the signature synthesized below from inline annotations is
        // redundant and weaker — it fills every un-annotated position with a
        // wildcard `(t-var {} _)` and (being last-write-wins in the checker's
        // defsig binding) would overwrite the concrete explicit sig, dropping
        // the body-vs-signature contract on those positions. Suppress it and
        // let the explicit sig drive body validation.
        // An explicit binder list forces the synthesized signature even when
        // every outer type slot is omitted: `defsig` is the declaration's
        // structural binder carrier, and P4b keeps those names in scope for
        // ordinary body annotations. A declared bound additionally requires
        // an occurrence in the declared type, so its existing validation
        // remains stricter than the unbounded body-only case.
        if synthesize_signature {
            // The explicit clause is the only binder source. Variable-shaped
            // uses not present in it remain undeclared at Deep resolution.
            let tvar_set: UnordSet<String> = dim_set.clone();
            let mut type_parts: Vec<deep::Expr> = params
                .iter()
                .map(|p| match &p.ty {
                    Some(ty) => desugar_type_with_scope(ty, &dim_set, &tvar_set),
                    None => node(DeepTag::TVar, vec![sym("_")]),
                })
                .collect();
            type_parts.push(match ret_ty {
                Some(ty) => desugar_type_with_scope(ty, &dim_set, &tvar_set),
                None => node(DeepTag::TVar, vec![sym("_")]),
            });
            let sig = with_dtype_bounds(
                defsig_node(
                    name,
                    type_binders,
                    apply_effect_metadata(node(DeepTag::TFn, type_parts), effects),
                ),
                type_binders,
            );
            vec![sig, def_node]
        } else {
            vec![def_node]
        }
    }

    fn desugar_property(
        &self,
        name: &str,
        type_binders: &[TypeBinder],
        params: &[Param],
        preconditions: &[Expr],
        body: &Expr,
        options: &[PropertyOption],
    ) -> Vec<deep::Expr> {
        let declared_binders = self
            .declared_type_binders
            .get(name)
            .cloned()
            .unwrap_or_default();
        let binder_names: UnordSet<String> = declared_binders
            .to_sorted()
            .into_iter()
            .map(|(binder, _)| binder.clone())
            .collect();
        let param_scope = params
            .iter()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>();
        let param_nodes = params
            .iter()
            .map(|param| desugar_param_with_scope(param, &binder_names, &binder_names))
            .collect::<Vec<_>>();
        let params_node = node(DeepTag::Params, param_nodes.clone());
        let restore_binders = self.current_type_binders.replace(declared_binders);
        let preconditions = PropertyPreconditions::new(
            deep::Metadata::default(),
            preconditions
                .iter()
                .map(|expr| {
                    RuntimeExpression::try_new(self.desugar_expr_with_scope(expr, &param_scope))
                        .expect("property expression")
                })
                .collect(),
            sp(),
        );
        let quantifiers = PropertyQuantifiers::new(
            deep::Metadata::default(),
            param_nodes
                .into_iter()
                .map(|p| PropertyBinder::try_from_expression(p).expect("property binder"))
                .collect(),
            sp(),
        );
        let mut meta_entries = vec![
            M::ChelisRole(Spanned::new("property".into(), sp())),
            M::PropertySourceKind(Spanned::new(PropertySourceKind::User, sp())),
            M::PropertyQuantifiers(quantifiers),
            M::PropertyPreconditions(preconditions),
        ];
        let mut contract_ids = Vec::new();
        for option in options {
            match option {
                PropertyOption::Tolerance(value, _) => meta_entries.push(M::PropertyTolerance(
                    RuntimeExpression::try_new(self.desugar_expr_with_scope(value, &param_scope))
                        .expect("property option expression"),
                )),
                PropertyOption::Seed(value, _) => meta_entries.push(M::PropertySeed(
                    RuntimeExpression::try_new(self.desugar_expr_with_scope(value, &param_scope))
                        .expect("property option expression"),
                )),
                PropertyOption::Samples(value, _) => meta_entries.push(M::PropertySamples(
                    RuntimeExpression::try_new(self.desugar_expr_with_scope(value, &param_scope))
                        .expect("property option expression"),
                )),
                PropertyOption::Contract(id, _) => {
                    contract_ids.push(Spanned::new(id.clone(), sp()))
                }
            }
        }
        if !contract_ids.is_empty() {
            meta_entries.push(M::PropertyContracts(PropertyContracts::new(
                deep::Metadata::default(),
                contract_ids,
                sp(),
            )));
        }

        let fn_node = node(
            DeepTag::Fn,
            vec![
                params_node,
                self.desugar_expr_with_scope(body, &param_scope),
            ],
        );
        self.current_type_binders.replace(restore_binders);
        let def_node = node_meta(
            DeepTag::Def,
            meta_with_entries(meta_entries),
            vec![sym(name), fn_node],
        );
        let mut type_parts = params
            .iter()
            .map(|param| {
                desugar_type_with_scope(
                    param.ty.as_ref().expect("property params are typed"),
                    &binder_names,
                    &binder_names,
                )
            })
            .collect::<Vec<_>>();
        type_parts.push(node(DeepTag::TPrim, vec![sym("bool")]));
        let sig_node = with_dtype_bounds(
            defsig_node(name, type_binders, node(DeepTag::TFn, type_parts)),
            type_binders,
        );
        vec![sig_node, def_node]
    }

    /// Desugar a type definition (RFC D-META). An opaque type emits
    /// `opaque: true`; an opaque type carrying a declared invariant also
    /// emits `invariant: (fn {} (params {} <binder>) <desugared body>)`
    /// and `invariant_amenability: "<class>"`. The predicate body is
    /// desugared with the binder in scope, and the amenability is
    /// computed by `chelis_pred::classify_predicate` over the just-built
    /// fn node.
    fn desugar_type_def(
        &self,
        name: &str,
        params: &[String],
        variants: &[Variant],
        opaque: bool,
        invariant: Option<&TypeInvariant>,
    ) -> deep::Expr {
        let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
        let explicit_params: UnordSet<String> = params.iter().cloned().collect();
        let mut children = vec![sym(name), param_list];
        for v in variants {
            children.push(desugar_variant(v, &explicit_params));
        }

        if !opaque {
            return node(DeepTag::Deftype, children);
        }

        let mut meta_entries = vec![M::Opaque(Present::new(sp()))];
        if let Some(inv) = invariant {
            // Predicate fn node: (fn {} (params {} <binder>) <body>).
            // The body is desugared with the binder in scope.
            let params_node = node(DeepTag::Params, vec![sym(&inv.binder)]);
            let body = self.desugar_expr_with_scope(&inv.body, std::slice::from_ref(&inv.binder));
            let fn_node = node(DeepTag::Fn, vec![params_node, body]);
            let amenability = chelis_pred::classify_predicate(&fn_node);
            meta_entries.push(M::Invariant(
                InvariantPredicate::try_from_expression(fn_node).expect("one-binder invariant"),
            ));
            let amenability = match amenability {
                chelis_pred::PredAmenability::Linear => Amenability::Linear,
                chelis_pred::PredAmenability::Polynomial => Amenability::Polynomial,
                chelis_pred::PredAmenability::Transcendental => Amenability::Transcendental,
                chelis_pred::PredAmenability::Opaque => Amenability::Opaque,
            };
            meta_entries.push(M::InvariantAmenability(Spanned::new(amenability, sp())));
        }

        node_meta(DeepTag::Deftype, meta_with_entries(meta_entries), children)
    }
}

fn desugar_variant(variant: &Variant, explicit_params: &UnordSet<String>) -> deep::Expr {
    match &variant.fields {
        VariantFields::Positional(fields) => {
            let mut children = vec![sym(&variant.name)];
            for f in fields {
                children.push(desugar_declaration_type(f, explicit_params));
            }
            node(DeepTag::Variant, children)
        }
        VariantFields::Record(fields) => {
            let mut children = vec![sym(&variant.name)];
            for (field_name, field_ty) in fields {
                children.push(node(
                    DeepTag::Field,
                    vec![
                        sym(field_name),
                        desugar_declaration_type(field_ty, explicit_params),
                    ],
                ));
            }
            node(DeepTag::Variant, children)
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl DesugarCtx {
    fn desugar_expr(&self, expr: &Expr) -> deep::Expr {
        self.desugar_expr_with_scope(expr, &[])
    }

    fn desugar_grad(
        &self,
        f: &Expr,
        wrt: Option<&[String]>,
        local_fn_params: &[String],
    ) -> deep::Expr {
        let desugared_fn = self.desugar_expr_with_scope(f, local_fn_params);
        let Some(wrt) = wrt else {
            return node(DeepTag::Grad, vec![desugared_fn]);
        };

        let key = f as *const Expr as usize;
        let indices = self
            .resolved_grad_indices
            .iter()
            .find_map(|(candidate, indices)| (*candidate == key).then(|| indices.clone()))
            .expect("fallible selector resolution runs before Deep construction");

        let mut variables = wrt.iter().map(|name| {
            VariableRef::new(
                Spanned::new(name.clone(), sp()),
                deep::Metadata::default(),
                sp(),
            )
            .expect("variable target")
        });
        let first = variables.next().expect("grad has a nonempty target list");
        let wrt_meta = if wrt.len() == 1 {
            WrtTargets::Variable(first)
        } else {
            WrtTargets::Tuple(
                VariableTuple::new(deep::Metadata::default(), first, variables.collect(), sp())
                    .expect("variable tuple"),
            )
        };
        let index_expr = if indices.len() == 1 {
            node_meta(
                DeepTag::Lit,
                meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
                vec![int(indices[0])],
            )
        } else {
            node(
                DeepTag::Tuple,
                indices
                    .into_iter()
                    .map(|index| {
                        node_meta(
                            DeepTag::Lit,
                            meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
                            vec![int(index)],
                        )
                    })
                    .collect(),
            )
        };

        node_meta(
            DeepTag::Grad,
            meta_with_entries(vec![M::Wrt(wrt_meta)]),
            vec![desugared_fn, index_expr],
        )
    }

    fn desugar_expr_with_scope(&self, expr: &Expr, local_fn_params: &[String]) -> deep::Expr {
        let desugared = match expr {
            Expr::Lit(lit, _) => desugar_literal(lit),
            Expr::Var(name, _) => dvar(name),
            Expr::Constructor(name, _) => dvar(name),
            Expr::List(items, _) => desugar_list_literal(
                &items
                    .iter()
                    .map(|item| self.desugar_expr_with_scope(item, local_fn_params))
                    .collect::<Vec<_>>(),
            ),
            Expr::Record(name, fields, _) => {
                let mut children = vec![sym(name)];
                for (field, value) in fields {
                    children.push(node(
                        DeepTag::Kv,
                        vec![
                            sym(field),
                            self.desugar_expr_with_scope(value, local_fn_params),
                        ],
                    ));
                }
                node(DeepTag::Record, children)
            }
            Expr::RecordUpdate(base, fields, _) => {
                let mut children = vec![self.desugar_expr_with_scope(base, local_fn_params)];
                for (field, value) in fields {
                    children.push(node(
                        DeepTag::Kv,
                        vec![
                            sym(field),
                            self.desugar_expr_with_scope(value, local_fn_params),
                        ],
                    ));
                }
                node(DeepTag::RecordUpdate, children)
            }
            Expr::Access(target, field, _) => node(
                DeepTag::Access,
                vec![
                    self.desugar_expr_with_scope(target, local_fn_params),
                    sym(field),
                ],
            ),
            Expr::TupleGet(target, index, _) => node(
                DeepTag::TupleGet,
                vec![
                    self.desugar_expr_with_scope(target, local_fn_params),
                    node_meta(
                        DeepTag::Lit,
                        meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
                        vec![deep::Expr::Atom(deep::Atom::Int(*index), sp())],
                    ),
                ],
            ),

            Expr::Apply(func, args, _) => self.desugar_apply(func, args, local_fn_params),

            Expr::Binary(op, lhs, rhs, _) => {
                // Every operator keeps its authored operand order
                // (spec/02-surf-syntax.md section 2): Deep application
                // evaluates arguments left to right, so a swap here would
                // reorder operand effects and traps (chelis#1180).
                let op_name = binop_name(*op);
                node(
                    DeepTag::App,
                    vec![
                        dvar(op_name),
                        self.desugar_expr_with_scope(lhs, local_fn_params),
                        self.desugar_expr_with_scope(rhs, local_fn_params),
                    ],
                )
            }

            // P10 parses every negative spelling as unary minus. The one
            // magnitude outside the positive i64 range is stored in the
            // Surf AST as a signed-minimum sentinel; fold that sentinel back
            // to the representable Deep literal instead of applying `neg`
            // twice or overflowing in Rust.
            Expr::Unary(UnaryOp::Neg, operand, _) if is_i64_min_magnitude_sentinel(operand) => {
                self.desugar_expr_with_scope(operand, local_fn_params)
            }

            Expr::Unary(op, operand, _) => {
                let op_name = match op {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Not => "not",
                };
                node(
                    DeepTag::App,
                    vec![
                        dvar(op_name),
                        self.desugar_expr_with_scope(operand, local_fn_params),
                    ],
                )
            }

            Expr::Pipe(head, stages, _) => {
                let mut children = vec![self.desugar_expr_with_scope(head, local_fn_params)];
                children.extend(
                    stages
                        .iter()
                        .map(|expr| self.desugar_pipe_stage(expr, local_fn_params)),
                );
                node(DeepTag::Pipe, children)
            }

            Expr::If(cond, then_e, else_e, _) => node(
                DeepTag::If,
                vec![
                    self.desugar_expr_with_scope(cond, local_fn_params),
                    self.desugar_expr_with_scope(then_e, local_fn_params),
                    self.desugar_expr_with_scope(else_e, local_fn_params),
                ],
            ),

            Expr::Match(scrutinee, arms, _) => {
                let mut children = vec![self.desugar_expr_with_scope(scrutinee, local_fn_params)];
                for arm in arms {
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                        .unwrap_or_else(|| bare_list(vec![]));
                    children.push(node(
                        DeepTag::Arm,
                        vec![
                            desugar_pattern(&arm.pattern),
                            guard,
                            self.desugar_expr_with_scope(&arm.body, local_fn_params),
                        ],
                    ));
                }
                node(DeepTag::Match, children)
            }

            Expr::Lambda(params, body, _) => {
                let param_names: Vec<deep::Expr> = params
                    .iter()
                    .map(|param| {
                        let annotation = param
                            .ty
                            .as_ref()
                            .map(|ty| self.desugar_body_annotation_type(ty));
                        desugar_param_with_annotation(param, annotation)
                    })
                    .collect();
                let params_node = node(DeepTag::Params, param_names);
                let lambda_params = params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<Vec<_>>();
                node(
                    DeepTag::Fn,
                    vec![
                        params_node,
                        self.desugar_expr_with_scope(body, &lambda_params),
                    ],
                )
            }

            Expr::Tuple(elems, _) if elems.is_empty() => node_meta(
                DeepTag::Lit,
                meta_with_type(node(DeepTag::TUnit, vec![])),
                vec![bare_list(vec![])],
            ),
            Expr::Tuple(elems, _) => node(
                DeepTag::Tuple,
                elems
                    .iter()
                    .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                    .collect(),
            ),

            Expr::Cast(e, prec, mode, _) => {
                // Normalize before choosing literal adoption as well as the
                // target node: both denote the same primitive under §P10a.
                let prec = canonical_primitive_name(prec).unwrap_or(prec);
                // Position 4 (spec §P10b / §5.6): first argument of a
                // `cast(literal, p)` expression. When the inner is a
                // bare list literal, narrow numeric entries to `p` and
                // produce a tensor literal. The outer `cast` then
                // becomes a no-op precision-confirm at the type level
                // (tensor[N, p] cast to p), which `infer_cast` accepts
                // because the precision matches.
                //
                // Issue #308: the same position-4 rule applies to a bare
                // scalar numeric literal. `cast(1.1, f64)` binds the
                // decimal `1.1` AT f64 — it is NOT "narrow to the §5.3
                // f32 default, then widen", which materializes the
                // f32-truncation signature `1.100000023841858` in every
                // value lane that honors the lit's type meta (the IR
                // Const lowering, the runtime evaluator). Suffixed
                // literals (spec §5.5) keep their explicit suffix
                // binding; `cast(1.1f32, f64)` still means "widen this
                // f32 value". A float literal under an integer target
                // keeps its default float source because a decimal cannot
                // bind at an integer type; the checked cast then requires
                // the value to be integral (spec/04 [04-NUM-14]).
                //
                // The [05-OP-6] truncating rung takes NONE of this: its
                // target is an integer width and its source must stay a
                // float, so adopting a literal at the target would turn
                // `cast_trunc([1.9], i32)` into an i32 tensor and
                // make the truncating cast a type error on its own
                // argument.
                let binder = self.current_type_binder(prec);
                // `binder` is consulted again below for the unbounded case, so
                // the bound is cloned out rather than moved.
                let binder_is_unbounded = matches!(binder, Some(None));
                let binder_is_declared = binder.is_some();
                let binder_bound = binder.flatten();
                let inner = match e.as_ref() {
                    _ if *mode == CastMode::Trunc => {
                        self.desugar_expr_with_scope(e, local_fn_params)
                    }
                    Expr::List(items, _) => {
                        self.desugar_list_as_tensor_literal(items, prec, local_fn_params)
                    }
                    other => {
                        let unsuffixed = is_unsuffixed_surf_numeric_literal(other);
                        // A signed direct `lit` is the unambiguous carrier only
                        // when the Surf syntax is eligible for adoption, or when
                        // it proves the narrow literal-source rejection for an
                        // unbounded binder. Ordinary suffixed casts keep their
                        // authored unary-minus application shape.
                        let needs_signed_literal = unsuffixed || binder_is_unbounded;
                        let ordinary = needs_signed_literal
                            .then(|| canonical_signed_cast_literal(other))
                            .flatten()
                            .map(|literal| attach_span_metadata(literal, expr_span(other)))
                            .unwrap_or_else(|| {
                                self.desugar_expr_with_scope(other, local_fn_params)
                            });
                        let adopted = classify_literal_source(&ordinary).and_then(|source| {
                            if scalar_literal_source_adopts_binder_target(
                                source,
                                binder_bound,
                                unsuffixed,
                            ) {
                                adopted_scalar_literal_source(source, prec, DeepTag::TVar)
                            } else if scalar_literal_source_adopts_cast_target(
                                source, prec, unsuffixed,
                            ) {
                                adopted_scalar_literal_source(source, prec, DeepTag::TPrim)
                            } else {
                                None
                            }
                        });
                        adopted
                            .map(|literal| attach_span_metadata(literal, expr_span(other)))
                            .unwrap_or(ordinary)
                    }
                };
                // Every declared binder is a `t-var`; only a bound permits
                // literal adoption. The checker rejects unbounded targets.
                let target = if binder_is_declared {
                    node(DeepTag::TVar, vec![sym(prec)])
                } else {
                    // A name that is not a primitive is passed through, so
                    // the checker still surfaces its unknown-primitive
                    // diagnostic rather than this arm inventing one.
                    node(DeepTag::TPrim, vec![sym(prec)])
                };
                let mut children = vec![inner, target];
                if let Some(selector) = mode.deep_selector() {
                    children.push(sym(selector));
                }
                node(DeepTag::Cast, children)
            }

            Expr::Grad(f, wrt, _) => self.desugar_grad(f, wrt.as_deref(), local_fn_params),

            Expr::Vmap(f, axis, _) => {
                let axis_node = node_meta(
                    DeepTag::Lit,
                    meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
                    vec![deep::Expr::Atom(deep::Atom::Int(axis.unwrap_or(0)), sp())],
                );
                node(
                    DeepTag::Vmap,
                    vec![self.desugar_expr_with_scope(f, local_fn_params), axis_node],
                )
            }

            Expr::Jit(f, _) => node(
                DeepTag::Jit,
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Realize(f, _) => node(
                DeepTag::Realize,
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Copy(f, _) => node(
                DeepTag::Copy,
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Borrow(f, _) => node(
                DeepTag::Borrow,
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::WithDevice(device, body, _) => node_meta(
                DeepTag::HandleEffect,
                meta_with_entries(vec![M::Effect(Spanned::new(EffectKind::Resource, sp()))]),
                vec![
                    self.desugar_expr_with_scope(device, local_fn_params),
                    self.desugar_expr_with_scope(body, local_fn_params),
                ],
            ),
            Expr::Par(exprs, _) => node(
                DeepTag::Par,
                exprs
                    .iter()
                    .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                    .collect(),
            ),
            Expr::Do(exprs, _) => node(
                DeepTag::Block,
                exprs
                    .iter()
                    .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                    .collect(),
            ),
            Expr::Quote(expr, _) => node(
                DeepTag::Quote,
                vec![self.desugar_expr_with_scope(expr, local_fn_params)],
            ),
            Expr::Unquote(expr, _) => node(
                DeepTag::Unquote,
                vec![self.desugar_expr_with_scope(expr, local_fn_params)],
            ),
            Expr::Splice(expr, _) => node(
                DeepTag::Splice,
                vec![self.desugar_expr_with_scope(expr, local_fn_params)],
            ),

            Expr::Annotate(e, ty, _) => {
                // Type annotation pushed into metadata of the desugared expression
                let desugared = self.desugar_expr_with_scope(e, local_fn_params);
                inject_type_metadata(desugared, self.desugar_body_annotation_type(ty))
            }

            Expr::Block(bindings, final_expr, _) => {
                if bindings.is_empty() {
                    self.desugar_expr_with_scope(final_expr, local_fn_params)
                } else {
                    self.desugar_let_bindings(
                        bindings,
                        final_expr,
                        self.desugar_expr_with_scope(final_expr, local_fn_params),
                    )
                }
            }
        };
        attach_span_metadata(desugared, expr_span(expr))
    }
}

fn tuple_index_expr(target: deep::Expr, index: i64) -> deep::Expr {
    node(
        DeepTag::TupleGet,
        vec![
            target,
            node_meta(
                DeepTag::Lit,
                meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
                vec![deep::Expr::Atom(deep::Atom::Int(index), sp())],
            ),
        ],
    )
}

fn bind_name_value(name: &str, name_span: Span, value: deep::Expr, body: deep::Expr) -> deep::Expr {
    let bind_node = node(
        DeepTag::Bind,
        vec![
            deep::Expr::Atom(deep::Atom::Name(name.to_string()), name_span),
            value,
        ],
    );
    node(DeepTag::Let, vec![bind_node, body])
}

/// Synthesized destructure bind (Linearity-F2).  Marks the `bind`
/// node with `destructure: true` in its meta-map so the linearity
/// checker can distinguish destructure components (and the
/// synthesized `__chelis_tmpN` intermediates that carry them) from
/// regular `let` bindings.  Each such bind introduces exactly one
/// name, and the checker marks that name as a destructured component
/// (`LinearScope::mark_destructured`): use-after-consume on a
/// component is an error because implicit Copy insertion does not
/// apply to it — tuple-get produces a fresh owned value, not an
/// aliased borrow.  Per chelis#1200 the marker scopes to the names
/// it introduces, never to the enclosing block.
fn bind_destructure_value(name: &str, value: deep::Expr, body: deep::Expr) -> deep::Expr {
    let bind_node = node_meta(
        DeepTag::Bind,
        meta_with_entries(vec![M::Destructure(Present::new(sp()))]),
        vec![sym(name), value],
    );
    node(DeepTag::Let, vec![bind_node, body])
}

fn destructure_pattern(
    pattern: &LetPattern,
    source_name: &str,
    body: deep::Expr,
    next_tmp: &std::cell::Cell<usize>,
    bindings: &[LetBinding],
    authored_body: &Expr,
) -> deep::Expr {
    match pattern {
        LetPattern::Var(name, _) => bind_destructure_value(name, dvar(source_name), body),
        LetPattern::Wildcard(_) => body,
        LetPattern::Tuple(parts, _) => {
            let mut out = body;
            for (index, part) in parts.iter().enumerate().rev() {
                let tuple_value = tuple_index_expr(dvar(source_name), index as i64);
                let tmp_name = fresh_destructure_temp(bindings, authored_body, next_tmp);
                out = destructure_pattern(part, &tmp_name, out, next_tmp, bindings, authored_body);
                out = bind_destructure_value(&tmp_name, tuple_value, out);
            }
            out
        }
    }
}

/// Mint a `__chelis_tmpN` that has not been minted before by this
/// `DesugarCtx` and that the authored source does not already mention.
///
/// `next_tmp` is the context-wide counter, read and written on every
/// mint rather than snapshotted, because `desugar_let_bindings` recurses
/// into nested blocks mid-loop: a snapshot would let the inner block
/// re-mint names the outer block had already taken. See
/// `DesugarCtx::next_destructure_temp` for what the collision cost.
fn fresh_destructure_temp(
    bindings: &[LetBinding],
    body: &Expr,
    next_tmp: &std::cell::Cell<usize>,
) -> String {
    loop {
        let candidate = format!("__chelis_tmp{}", next_tmp.get());
        next_tmp.set(next_tmp.get() + 1);
        let mentioned = bindings.iter().any(|binding| {
            let_pattern_mentions_name(&binding.pattern, &candidate)
                || binding
                    .ty
                    .as_ref()
                    .is_some_and(|ty| type_mentions_name(ty, &candidate))
                || expr_mentions_name(&binding.value, &candidate)
        }) || expr_mentions_name(body, &candidate);
        if !mentioned {
            return candidate;
        }
    }
}

impl DesugarCtx {
    fn desugar_let_bindings(
        &self,
        bindings: &[LetBinding],
        authored_body: &Expr,
        body: deep::Expr,
    ) -> deep::Expr {
        let mut out = body;
        let next_tmp = &self.next_destructure_temp;
        for binding in bindings.iter().rev() {
            match &binding.pattern {
                LetPattern::Var(name, binding_span) => {
                    // Position 1 (spec §P10b / §5.6) at block scope:
                    // `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]` inside
                    // a block uses the same contextual rule as the
                    // top-level form. Module-level LetDef and
                    // block-level let bindings are both let-bindings
                    // per §5.6 enumerated position 1.
                    let value = match (
                        binding.ty.as_ref().and_then(tensor_element_prim_name),
                        &binding.value,
                    ) {
                        (Some(prec), Expr::List(items, _)) => {
                            self.desugar_list_as_tensor_literal(items, &prec, &[])
                        }
                        _ if binding.ty.is_none()
                            && is_bare_numeric_tensor_literal(&binding.value) =>
                        {
                            self.desugar_default_tensor_binding(&binding.value)
                        }
                        _ => self.desugar_expr(&binding.value),
                    };
                    if let Some(ty) = &binding.ty {
                        let value = with_metadata_value(
                            inject_type_metadata(value, self.desugar_body_annotation_type(ty)),
                            M::SurfBindingType(Spanned::new(
                                BindingTypeOrigin::Explicit,
                                type_expr_span(ty),
                            )),
                        );
                        out = bind_name_value(name, *binding_span, value, out);
                    } else {
                        let value = if let Expr::Annotate(_, ty, _) = &binding.value {
                            with_metadata_value(
                                value,
                                M::SurfBindingType(Spanned::new(
                                    BindingTypeOrigin::Explicit,
                                    type_expr_span(ty),
                                )),
                            )
                        } else if has_type_metadata(&value) {
                            with_metadata_value(
                                value,
                                M::SurfBindingType(Spanned::new(
                                    BindingTypeOrigin::Inferred,
                                    *binding_span,
                                )),
                            )
                        } else {
                            value
                        };
                        out = bind_name_value(name, *binding_span, value, out);
                    }
                }
                pattern => {
                    let temp_name = fresh_destructure_temp(bindings, authored_body, next_tmp);
                    let value = self.desugar_expr(&binding.value);
                    out = destructure_pattern(
                        pattern,
                        &temp_name,
                        out,
                        next_tmp,
                        bindings,
                        authored_body,
                    );
                    out = bind_destructure_value(&temp_name, value, out);
                }
            }
        }
        out
    }
}

fn desugar_literal(lit: &Literal) -> deep::Expr {
    match lit {
        Literal::Int(n) => node_meta(
            DeepTag::Lit,
            meta_with_type(node(DeepTag::TPrim, vec![sym("i32")])),
            vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
        ),
        Literal::Float(f) => node_meta(
            DeepTag::Lit,
            meta_with_type(node(DeepTag::TPrim, vec![sym("f32")])),
            vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
        ),
        // Typed-suffix literals (spec/02-surf-syntax.md §P10a /
        // spec/04-type-system.md §5.5): bind at exactly the suffix
        // precision with no inference, no widening, no narrowing. The
        // type metadata is the user-facing contract.
        Literal::TypedInt(n, suffix) => {
            let prim = suffix.t_prim_name();
            let float_typed = matches!(prim, "f32" | "f64" | "bf16" | "f16");
            let ty = node(DeepTag::TPrim, vec![sym(prim)]);
            let meta = if float_typed {
                let style = (*suffix == LiteralSuffix::F32).then_some(LiteralStyle::Explicit);
                meta_with_integer_float_type(ty, style)
            } else if *suffix == LiteralSuffix::I32 {
                numeric_literal_meta(ty, LiteralStyle::Explicit)
            } else {
                meta_with_type(ty)
            };
            node_meta(
                DeepTag::Lit,
                meta,
                vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
            )
        }
        Literal::TypedFloat(f, suffix) => {
            let ty = node(DeepTag::TPrim, vec![sym(suffix.t_prim_name())]);
            let meta = if *suffix == LiteralSuffix::F32 {
                numeric_literal_meta(ty, LiteralStyle::Explicit)
            } else {
                meta_with_type(ty)
            };
            node_meta(
                DeepTag::Lit,
                meta,
                vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
            )
        }
        Literal::Bool(b) => node_meta(
            DeepTag::Lit,
            meta_with_type(node(DeepTag::TPrim, vec![sym("bool")])),
            vec![deep::Expr::Atom(deep::Atom::Bool(*b), sp())],
        ),
        Literal::Str(s) => node_meta(
            DeepTag::Lit,
            meta_with_type(node(DeepTag::TPrim, vec![sym("string")])),
            vec![deep::Expr::Atom(deep::Atom::Str(s.clone()), sp())],
        ),
    }
}

fn canonical_signed_cast_literal(expr: &Expr) -> Option<deep::Expr> {
    let Expr::Unary(UnaryOp::Neg, inner, _) = expr else {
        return None;
    };
    let literal = match inner.as_ref() {
        Expr::Lit(Literal::Int(value), _) => Literal::Int(fold_unary_minus_int(*value)),
        Expr::Lit(Literal::Float(value), _) => Literal::Float(-*value),
        Expr::Lit(Literal::TypedInt(value, suffix), _) => {
            Literal::TypedInt(fold_unary_minus_int(*value), *suffix)
        }
        Expr::Lit(Literal::TypedFloat(value, suffix), _) => Literal::TypedFloat(-*value, *suffix),
        _ => return None,
    };
    Some(desugar_literal(&literal))
}

fn is_unsuffixed_surf_numeric_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Lit(Literal::Int(_) | Literal::Float(_), _))
        || matches!(
            expr,
            Expr::Unary(UnaryOp::Neg, inner, _)
                if matches!(inner.as_ref(), Expr::Lit(Literal::Int(_) | Literal::Float(_), _))
        )
}

/// Position 4 (spec §5.6 / §P10b) admission test for a bare scalar
/// literal under `cast(literal, p)` (issue #308). An unsuffixed numeric
/// literal adopts the cast target when the binding is meaningful:
///
///   * float literal + float target (`f32`/`f64`/`bf16`/`f16`) — the
///     decimal binds at `p` (single rounding, no round-trip through the
///     §5.3 f32 default);
///   * int literal + integer target (`i8`..`i64`) — the value binds
///     at `p`, which is what makes the documented out-of-i32-range
///     escape hatch `cast(N, i64)` actually work (and routes the
///     i8/i16 forms through `infer_lit`'s contextual range check);
///   * int literal + float target — the integer binds at `p` exactly.
///
/// Everything else keeps the §5.3 default-then-convert behavior:
/// suffixed literals bind at their suffix (§5.5), float→integer keeps
/// truncation semantics, and bool/string targets are not numeric
/// binding precisions.
/// [02-P10b] binder-target literal adoption. Float literals require `Float`
/// or `Numeric`; integer literals also admit `Int`. Unbounded binders cannot
/// adopt and remain checker-rejected cast targets under [04-DTYPE-2].
fn scalar_literal_source_adopts_binder_target(
    source: LiteralSource<'_>,
    bound: Option<chelis_deep::DtypeBound>,
    unsuffixed: bool,
) -> bool {
    unsuffixed
        && bound.is_some_and(|bound| {
            matches!(
                source.bound_fit(&bound),
                LiteralFamilyFit::Fits | LiteralFamilyFit::IntegerOutOfRange
            )
        })
}

fn scalar_literal_source_adopts_cast_target(
    source: LiteralSource<'_>,
    prec: &str,
    unsuffixed: bool,
) -> bool {
    if !unsuffixed {
        return false;
    }
    let float_target = matches!(prec, "f32" | "f64" | "bf16" | "f16");
    let int_target = matches!(prec, "i8" | "i16" | "i32" | "i64");
    match source.numeric_atom() {
        Some(DeepAtom::Float(_)) => float_target,
        Some(DeepAtom::Int(_)) => float_target || int_target,
        _ => false,
    }
}

/// Build the adopted-literal Deep node for a scalar literal under
/// `cast(literal, p)`. Mirrors `desugar_tensor_literal_item`'s lit
/// construction. Exact Surf unary syntax is already a signed direct `lit`.
/// Only called for a syntactically adopting source; the checker diagnoses an
/// integer that cannot fit every member of a family bound.
fn adopted_scalar_literal_source(
    source: LiteralSource<'_>,
    prec: &str,
    target_tag: DeepTag,
) -> Option<deep::Expr> {
    let ty = node(target_tag, vec![sym(prec)]);
    match source.numeric_atom()? {
        DeepAtom::Int(value) => {
            let meta =
                if target_tag == DeepTag::TPrim && matches!(prec, "f32" | "f64" | "bf16" | "f16") {
                    meta_with_integer_float_type(ty, Some(LiteralStyle::Unsuffixed))
                } else {
                    numeric_literal_meta(ty, LiteralStyle::Unsuffixed)
                };
            Some(node_meta(
                DeepTag::Lit,
                meta,
                vec![deep::Expr::Atom(DeepAtom::Int(*value), sp())],
            ))
        }
        DeepAtom::Float(value) => Some(node_meta(
            DeepTag::Lit,
            numeric_literal_meta(ty, LiteralStyle::Unsuffixed),
            vec![deep::Expr::Atom(DeepAtom::Float(*value), sp())],
        )),
        _ => None,
    }
}

fn desugar_list_literal(items: &[deep::Expr]) -> deep::Expr {
    let mut out = dvar("Nil");
    for item in items.iter().rev() {
        out = node(DeepTag::App, vec![dvar("Cons"), item.clone(), out]);
    }
    out
}

// ---------------------------------------------------------------------------
// A binding without a declared List/tensor context uses the tensor default
// only for a nonempty numeric bracket literal. Arbitrary computed collections,
// empty lists, bool and string lists retain their own existing List behavior.
fn is_bare_numeric_tensor_literal(expr: &Expr) -> bool {
    fn numeric_leaf_or_list(expr: &Expr) -> bool {
        match expr {
            Expr::List(items, _) => !items.is_empty() && items.iter().all(numeric_leaf_or_list),
            Expr::Lit(
                Literal::Int(_)
                | Literal::Float(_)
                | Literal::TypedInt(_, _)
                | Literal::TypedFloat(_, _),
                _,
            ) => true,
            Expr::Unary(UnaryOp::Neg, inner, _) => matches!(
                inner.as_ref(),
                Expr::Lit(
                    Literal::Int(_)
                        | Literal::Float(_)
                        | Literal::TypedInt(_, _)
                        | Literal::TypedFloat(_, _),
                    _
                )
            ),
            _ => false,
        }
    }
    matches!(expr, Expr::List(_, _)) && numeric_leaf_or_list(expr)
}

// Contextual tensor-literal inference (spec §P10b / §5.6)
// ---------------------------------------------------------------------------
//
// When a tensor literal `[e1, e2, ...]` appears in a position with a known
// element type, the numeric literals in the body adopt that element type
// instead of the §5.3 / §P10 literal default (i32 for integer literals,
// f32 for float literals).
//
// The closed set of "known-element-type" positions is exactly four,
// per spec §5.6:
//
//   1. RHS of a `let`-binding whose declared type is a tensor type
//      `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
//   2. Argument position of a call whose callee has a declared signature
//      with a tensor parameter at that position
//      `f(xs)` where `f : tensor[3, f64] -> ...`
//   3. Body expression of a function with a declared return type that is
//      a tensor type, when the body is itself a tensor literal
//   4. First argument of an explicit `cast(literal, p)`
//
// Outside this closed set, numeric literals fall back to the §5.3 / §P10
// defaults; this is the WS-0 / D1 default and is implemented by
// `desugar_literal` above.
//
// The helpers here transform `Expr::List` into a `to_tensor` call wrapping
// a `Cons/Nil` chain whose numeric literal entries carry the contextual
// element type instead of the default. Non-literal entries (variables,
// function calls, ...) pass through unchanged; type unification at
// `Cons`/`to_tensor` will reject them if their inferred type does not
// match the contextual element type.
//
// Future agents reading this code: do NOT silently extend the closed
// set. Adding new positions (e.g. "any context where a tensor type might
// be inferred backward") is a spec change requiring an amendment to
// §5.6 / §P10b.

impl DesugarCtx {
    /// Desugar `items` as the body of a contextual tensor literal whose
    /// element type is `prec_name` (a precision name like `"f64"` or
    /// `"i32"`). Numeric literals in `items` are emitted with
    /// `(lit {type: (t-prim {} <prec_name>)} value)` instead of the
    /// default i32/f32. Non-literal entries are desugared normally.
    /// The chain is wrapped in `to_tensor` so type inference resolves
    /// the result as a tensor.
    fn desugar_list_as_tensor_literal(
        &self,
        items: &[Expr],
        prec_name: &str,
        local_fn_params: &[String],
    ) -> deep::Expr {
        let desugared_items: Vec<deep::Expr> = items
            .iter()
            .map(|item| self.desugar_tensor_literal_item(item, prec_name, local_fn_params))
            .collect();
        let list = desugar_list_literal(&desugared_items);
        node(DeepTag::App, vec![dvar("to_tensor"), list])
    }

    /// An unannotated numeric binding is a tensor at each leaf's authored
    /// dtype/default. Do not apply contextual adoption to explicit suffixes.
    fn desugar_default_tensor_binding(&self, value: &Expr) -> deep::Expr {
        attach_span_metadata(
            node(
                DeepTag::App,
                vec![dvar("to_tensor"), self.desugar_expr(value)],
            ),
            expr_span(value),
        )
    }

    /// Desugar a single entry of a contextual tensor literal. Numeric
    /// literals are narrowed to the contextual element prim. Nested
    /// `Expr::List` entries (rank > 1) recurse with the same element
    /// type. Anything else falls back to the standard expression
    /// desugarer; the type checker will validate compatibility via
    /// the `Cons` element-type unification path.
    fn desugar_tensor_literal_item(
        &self,
        item: &Expr,
        prec_name: &str,
        local_fn_params: &[String],
    ) -> deep::Expr {
        match item {
            Expr::Lit(Literal::Int(n), _) => {
                let float_typed = matches!(prec_name, "f32" | "f64" | "bf16" | "f16");
                let ty = node(DeepTag::TPrim, vec![sym(prec_name)]);
                let meta = if float_typed {
                    meta_with_integer_float_type(ty, Some(LiteralStyle::Unsuffixed))
                } else {
                    numeric_literal_meta(ty, LiteralStyle::Unsuffixed)
                };
                node_meta(
                    DeepTag::Lit,
                    meta,
                    vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
                )
            }
            Expr::Lit(Literal::Float(f), _) => node_meta(
                DeepTag::Lit,
                numeric_literal_meta(
                    node(DeepTag::TPrim, vec![sym(prec_name)]),
                    LiteralStyle::Unsuffixed,
                ),
                vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
            ),
            // RT-2 fixup P2: the surface parser turns `-128` into
            // `Unary(Neg, Lit(Int(128)))`. In a contextual tensor
            // literal position we fold the sign into the literal so
            // the WS-A0 D1 / WS-A0 D1-extension range checks see the
            // user-facing value (`-128` for i8) rather than the
            // raw inner literal (`128`, which overflows i8 max).
            // The same applies to negative float literals.
            Expr::Unary(UnaryOp::Neg, inner, _) => match inner.as_ref() {
                Expr::Lit(Literal::Int(n), _) => {
                    let value = fold_unary_minus_int(*n);
                    let float_typed = matches!(prec_name, "f32" | "f64" | "bf16" | "f16");
                    let ty = node(DeepTag::TPrim, vec![sym(prec_name)]);
                    let meta = if float_typed {
                        meta_with_integer_float_type(ty, Some(LiteralStyle::Unsuffixed))
                    } else {
                        numeric_literal_meta(ty, LiteralStyle::Unsuffixed)
                    };
                    node_meta(
                        DeepTag::Lit,
                        meta,
                        vec![deep::Expr::Atom(deep::Atom::Int(value), sp())],
                    )
                }
                Expr::Lit(Literal::Float(f), _) => node_meta(
                    DeepTag::Lit,
                    numeric_literal_meta(
                        node(DeepTag::TPrim, vec![sym(prec_name)]),
                        LiteralStyle::Unsuffixed,
                    ),
                    vec![deep::Expr::Atom(deep::Atom::Float(-*f), sp())],
                ),
                // Non-literal `neg` operand falls through to the
                // standard desugar; the type checker will validate
                // the resulting expression's type against the
                // contextual element type via the Cons unification
                // path.
                _ => self.desugar_expr_with_scope(item, local_fn_params),
            },
            // Nested list — rank-N contextual tensor literal.
            Expr::List(nested_items, _) => {
                // The inner list is itself a contextual tensor literal
                // body: numeric literals at every depth adopt the same
                // element type. We do NOT wrap each inner level in
                // to_tensor (only the outermost wrap is needed).
                let inner_items: Vec<deep::Expr> = nested_items
                    .iter()
                    .map(|n| self.desugar_tensor_literal_item(n, prec_name, local_fn_params))
                    .collect();
                desugar_list_literal(&inner_items)
            }
            // Anything else: normal desugar. Type unification at Cons
            // will catch a mismatch.
            other => self.desugar_expr_with_scope(other, local_fn_params),
        }
    }
}

fn is_i64_min_magnitude_sentinel(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Lit(Literal::Int(i64::MIN), _)
            | Expr::Lit(Literal::TypedInt(i64::MIN, LiteralSuffix::I64), _)
    )
}

fn fold_unary_minus_int(value: i64) -> i64 {
    value.checked_neg().unwrap_or(i64::MIN)
}

impl DesugarCtx {
    fn desugar_apply(&self, func: &Expr, args: &[Expr], local_fn_params: &[String]) -> deep::Expr {
        // Position 2 (spec §P10b / §5.6): if the callee is a top-level
        // function with a declared signature whose i-th parameter is a
        // tensor type with element prim P, and the i-th argument is a
        // bare list literal, narrow the literal entries to P.
        //
        // The callee must be a `Var` for the lookup to apply — function
        // values from local bindings, partial applications, and lambda
        // returns do not carry a declared signature at desugar time.
        // The type checker still validates non-direct-call positions
        // through the standard `Cons`/`to_tensor` element-type
        // unification path; the contextual narrowing here is the
        // ergonomic affordance for the named-callee case.
        let callee_param_prec: Option<&Vec<Option<String>>> =
            if let Expr::Var(callee_name, _) = func {
                self.top_level_fn_tensor_param_prec.get(callee_name)
            } else {
                None
            };

        let desugared_args: Vec<deep::Expr> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let prec = callee_param_prec
                    .and_then(|v| v.get(i))
                    .and_then(|opt| opt.as_deref());
                match (prec, arg) {
                    (Some(prec_name), Expr::List(items, _)) => {
                        self.desugar_list_as_tensor_literal(items, prec_name, local_fn_params)
                    }
                    _ => self.desugar_expr_with_scope(arg, local_fn_params),
                }
            })
            .collect();

        let mut children = vec![self.desugar_expr_with_scope(func, local_fn_params)];
        children.extend(desugared_args);
        node(DeepTag::App, children)
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
        BinOp::Gt => "gt",
        BinOp::Le => "lte",
        BinOp::Ge => "gte",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

// ---------------------------------------------------------------------------
// Type Expressions
// ---------------------------------------------------------------------------

/// Desugar a type with no declared dim params or quantified type vars
/// (module-level context).
fn desugar_type(ty: &TypeExpr) -> deep::Expr {
    desugar_type_with_scope(ty, &UnordSet::new(), &UnordSet::new())
}

/// Desugar a `deftype` field or `typealias` body against that declaration's
/// exact explicit parameter list. Unlike a function signature's single-letter
/// variable-use encoding, an unlisted name is a concrete symbolic axis
/// (`d-name`), matching spec/02 §P15's zero-parameter alias examples. Listed
/// names remain unkinded declaration binders and are emitted according to
/// their position (`t-var`, `d-var`, or precision `t-var`).
fn desugar_declaration_type(ty: &TypeExpr, explicit_params: &UnordSet<String>) -> deep::Expr {
    desugar_type_with_scope_mode(ty, explicit_params, explicit_params, false)
}

/// Desugar a sig against its complete explicit binder list.
fn desugar_sig_type(ty: &TypeExpr, declared: &UnordSet<String>) -> deep::Expr {
    desugar_type_with_scope(ty, declared, declared)
}

/// Desugar a type with declared dimension parameters and a set of
/// in-scope quantified type variable names. The contextual rule for the
/// tensor precision slot lives here:
///
/// - A primitive name (`f32`, `i32`, ...) becomes `(t-prim {} <name>)`.
/// - A name found in `tvar_set` becomes `(t-var {} <name>)`.
/// - Any other name in the precision slot is encoded as `(t-prim {} <name>)`
///   so the type checker reports the unknown primitive spelling through the
///   shared `Prim::parse_name` rejection path.
fn desugar_type_with_scope(
    ty: &TypeExpr,
    dim_vars: &UnordSet<String>,
    tvar_set: &UnordSet<String>,
) -> deep::Expr {
    desugar_type_with_scope_mode(ty, dim_vars, tvar_set, true)
}

fn desugar_type_with_scope_mode(
    ty: &TypeExpr,
    dim_vars: &UnordSet<String>,
    tvar_set: &UnordSet<String>,
    single_letter_dim_vars: bool,
) -> deep::Expr {
    let desugared = match ty {
        TypeExpr::DimensionLiteral(value, _) => node(DeepTag::DLit, vec![int(value.value())]),
        TypeExpr::Named(name, _) => {
            // The contextual rule for type-name positions:
            //
            // - A primitive name (`f32`, `i32`, ...) is a `t-prim`.
            // - A name that appears in the enclosing quantifier set
            //   (`tvar_set` — the declaration's explicit `[..]` clause) is a
            //   quantified type variable and
            //   becomes `(t-var {} <name>)` REGARDLESS of case. The
            //   `[..]` clause is the authoritative, unkinded source per
            //   `spec/02-surf-syntax.md` §P4b, so a name the user
            //   explicitly bound there overrides the lexical
            //   case-split. This is what threads a general type
            //   variable (e.g. `def f[n, P](.., f: .. -> P -> ..)`)
            //   through arrow argument positions so it can unify at the
            //   call site (chelis#293). Without this, an uppercase
            //   quantifier name was misclassified as a rigid ADT and
            //   every call site failed with `type mismatch: P vs ..`.
            // - A name reserved under §1.1.1 names no type at all and
            //   stays `(t-prim {} <name>)` so the checker's rejection
            //   fires. It outranks the quantifier set, exactly as the
            //   primitive arm does (chelis#1593).
            // - Otherwise a PascalCase name is an ADT and every other spelling
            //   stays `t-prim`, so the closed primitive resolver reports an
            //   unknown dtype rather than inventing a binder.
            if name == "unit" {
                node(DeepTag::TUnit, vec![])
            } else if let Some(canonical) = canonical_primitive_name(name) {
                node(DeepTag::TPrim, vec![sym(canonical)])
            } else if is_reserved_dtype_name(name) {
                // Above `tvar_set` rather than below it, because §5.8.1 states
                // the rule on the category: a reserved spelling names no type
                // variable in any type position, and an explicit `[..]` clause
                // does not rebind it. §P4b's clause overrides the
                // PascalCase-vs-snake_case case-split, which is a different
                // rule; the primitive arm above already outranks a binder for
                // the same reason. Below `tvar_set`, `def f[u8](x: u8) -> u8`
                // still scored 1.0.
                //
                // Mapped to NOTHING, unlike chelis#1587's `i8`..`i64` above:
                // those are input spellings for active primitives and
                // normalise, these name no primitive at all and
                // `Prim::parse_name` must keep failing on them. Same class,
                // opposite repair.
                node(DeepTag::TPrim, vec![sym(name)])
            } else if tvar_set.contains(name.as_str()) {
                node(DeepTag::TVar, vec![sym(name)])
            } else if name.starts_with(|c: char| c.is_uppercase()) {
                node(DeepTag::TAdt, vec![sym(name)])
            } else {
                node(DeepTag::TPrim, vec![sym(name)])
            }
        }

        // A bare `..r` reaching here (outside a tensor dim list) is not a
        // valid standalone type, but desugar defensively to the rank node so
        // the match stays exhaustive; validation rejects the misuse upstream.
        TypeExpr::RankSpread(name, _) => node(DeepTag::DRank, vec![sym(name)]),
        TypeExpr::Tensor(dims, precision, _) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|d| match d {
                    TypeExpr::DimensionLiteral(value, _) => {
                        node(DeepTag::DLit, vec![int(value.value())])
                    }
                    TypeExpr::Named(n, _) if n.parse::<i64>().is_ok() => {
                        node(DeepTag::DLit, vec![int(n.parse::<i64>().unwrap())])
                    }
                    TypeExpr::Named(n, _) if n == "*" => node(DeepTag::DName, vec![sym("*")]),
                    // Declared dim param → always d-var (polymorphic)
                    TypeExpr::Named(n, _) if dim_vars.contains(n.as_str()) => {
                        node(DeepTag::DVar, vec![sym(n)])
                    }
                    // Single lowercase letter → d-var (heuristic fallback)
                    TypeExpr::Named(n, _)
                        if single_letter_dim_vars
                            && n.len() == 1
                            && n.starts_with(|c: char| c.is_lowercase()) =>
                    {
                        node(DeepTag::DVar, vec![sym(n)])
                    }
                    // Everything else → d-name (concrete)
                    TypeExpr::Named(n, _) => node(DeepTag::DName, vec![sym(n)]),
                    // `..r` rank-variable spread → (d-rank {} r). May be
                    // interleaved with concrete anchors (Tier-3); a multi-letter
                    // anchor such as `seq` desugars to `d-name` above, which is
                    // what the name-preserving split matches on.
                    TypeExpr::RankSpread(n, _) => node(DeepTag::DRank, vec![sym(n)]),
                    _ => node(
                        DeepTag::DVar,
                        vec![desugar_type_with_scope_mode(
                            d,
                            dim_vars,
                            tvar_set,
                            single_letter_dim_vars,
                        )],
                    ),
                })
                .collect();
            // Explicit precision-binder rule (spec/02-surf-syntax.md P4b):
            // the declaration's one explicit binder scope applies to every
            // type position, so a listed precision name is a t-var in both
            // the signature and body. An unlisted name stays t-prim and the
            // type checker validates it against the closed primitive set.
            let prec_node = match canonical_primitive_name(precision) {
                Some(canonical) => node(DeepTag::TPrim, vec![sym(canonical)]),
                // A §1.1.1 reserved spelling outranks the quantifier set here
                // for the reason it does in the scalar arm above: an explicit
                // `[..]` clause does not rebind a name the language reserved.
                // Without this row `def f[u8](x: tensor[3, u8])` scored 1.0
                // even after the scalar arm was repaired (chelis#1593).
                None if is_reserved_dtype_name(precision) => {
                    node(DeepTag::TPrim, vec![sym(precision)])
                }
                None if tvar_set.contains(precision.as_str()) => {
                    node(DeepTag::TVar, vec![sym(precision)])
                }
                // Not a primitive and not quantified: still `t-prim`, so the
                // checker surfaces its unknown-primitive diagnostic.
                None => node(DeepTag::TPrim, vec![sym(precision)]),
            };
            children.push(with_structural_span(prec_node, precision.span()));
            node(DeepTag::TTensor, children)
        }

        TypeExpr::Arrow(params, ret, _) => {
            let mut children: Vec<deep::Expr> = params
                .iter()
                .map(|p| {
                    desugar_type_with_scope_mode(p, dim_vars, tvar_set, single_letter_dim_vars)
                })
                .collect();
            children.push(desugar_type_with_scope_mode(
                ret,
                dim_vars,
                tvar_set,
                single_letter_dim_vars,
            ));
            node(DeepTag::TFn, children)
        }

        TypeExpr::Ref(inner, _) => node(
            DeepTag::TRef,
            vec![desugar_type_with_scope_mode(
                inner,
                dim_vars,
                tvar_set,
                single_letter_dim_vars,
            )],
        ),

        TypeExpr::App(name, args, _) => {
            let mut children = vec![sym(name)];
            children.extend(args.iter().map(|a| {
                desugar_type_with_scope_mode(a, dim_vars, tvar_set, single_letter_dim_vars)
            }));
            node(DeepTag::TAdt, children)
        }

        TypeExpr::Tuple(elems, _) if elems.is_empty() => node(DeepTag::TUnit, vec![]),
        TypeExpr::Tuple(elems, _) => node(
            DeepTag::TTuple,
            elems
                .iter()
                .map(|e| {
                    desugar_type_with_scope_mode(e, dim_vars, tvar_set, single_letter_dim_vars)
                })
                .collect(),
        ),

        TypeExpr::Infer(_) => node(DeepTag::TVar, vec![sym("_")]),
    };
    with_structural_span(desugared, type_expr_span(ty))
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

fn desugar_pattern(pat: &Pattern) -> deep::Expr {
    match pat {
        Pattern::Wildcard(_) => node(DeepTag::PatWild, vec![]),
        Pattern::Var(name, _) => node(DeepTag::PatVar, vec![sym(name)]),
        Pattern::Lit(lit, _) => {
            // pat-lit contains the raw literal value, NOT a typed (lit ...) node.
            // Typed-suffix patterns desugar to the bare value: pattern matching
            // does not enforce the suffix dtype at the pattern level (the
            // checker reconciles it via the surrounding scrutinee type).
            let val = match lit {
                Literal::Int(n) | Literal::TypedInt(n, _) => {
                    deep::Expr::Atom(deep::Atom::Int(*n), sp())
                }
                Literal::Float(f) | Literal::TypedFloat(f, _) => {
                    deep::Expr::Atom(deep::Atom::Float(*f), sp())
                }
                Literal::Bool(b) => deep::Expr::Atom(deep::Atom::Bool(*b), sp()),
                Literal::Str(s) => deep::Expr::Atom(deep::Atom::Str(s.clone()), sp()),
            };
            node(DeepTag::PatLit, vec![val])
        }
        Pattern::Constructor(name, sub_pats, _) => {
            let mut children = vec![sym(name)];
            children.extend(sub_pats.iter().map(desugar_pattern));
            node(DeepTag::PatCtor, children)
        }
        Pattern::Tuple(pats, _) => node(
            DeepTag::PatTuple,
            pats.iter().map(desugar_pattern).collect(),
        ),
        Pattern::Record(name, fields, _) => {
            let mut children = vec![sym(name)];
            for (field_name, field_pat) in fields {
                children.push(node(
                    DeepTag::Kv,
                    vec![sym(field_name), desugar_pattern(field_pat)],
                ));
            }
            node(DeepTag::PatRecord, children)
        }
        Pattern::As(name, inner, _) => {
            node(DeepTag::PatAs, vec![sym(name), desugar_pattern(inner)])
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
        assert_eq!(result, "(lit {type: (t-prim {} i32)} 42)");
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

    #[test]
    fn property_contracts_desugar_to_repeatable_metadata() {
        let source = r#"@property reflected forall(x: f32):
  x == x
  with contract = "std.normal_cdf.reflection"
  with contract = "std.normal_cdf.range"
"#;
        let decls = crate::parser::parse_str(source).expect("parse");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let contracts = deep
            .iter()
            .find_map(|expr| match expr {
                deep::Expr::Node(node, _) if node.tag() == DeepTag::Def => {
                    node.meta().property_contracts()
                }
                _ => None,
            })
            .expect("property_contracts metadata")
            .values()
            .iter()
            .map(|value| value.value().as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            contracts,
            vec!["std.normal_cdf.reflection", "std.normal_cdf.range"]
        );
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
    fn test_gt_keeps_authored_operand_order() {
        // chelis#1180: `a > b` desugars to the `gt` builtin with the
        // authored operand order. The old operand-swapped `cmplt(b, a)`
        // form evaluated the right operand's effects and traps first.
        let expr = Expr::Binary(BinOp::Gt, Box::new(tvar("a")), Box::new(tvar("b")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} gt) (var {} a) (var {} b))"
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
    fn test_grouped_apply_chain_preserves_call_result_application() {
        let inner = Expr::Apply(Box::new(tvar("f")), vec![tvar("x")], s());
        let outer = Expr::Apply(Box::new(inner), vec![tvar("y")], s());
        assert_eq!(
            print_expr(&desugar_expr(&outer)),
            "(app {} (app {} (var {} f) (var {} x)) (var {} y))"
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

    #[test]
    fn test_pipe_stage_call_desugars_to_unary_lambda() {
        let expr = Expr::Pipe(
            Box::new(tvar("x")),
            vec![Expr::Apply(Box::new(tvar("add")), vec![tvar("y")], s())],
            s(),
        );
        let rendered = print_expr(&desugar_expr(&expr));
        assert!(rendered.contains("(var {} x)"));
        assert!(rendered.contains("(params {} __chelis_pipe)"));
        assert!(rendered.contains("(app {} (var {} add) (var {} __chelis_pipe) (var {} y))"));
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
            "(match {}\n  (var {} x)\n  (arm {} (pat-ctor {} Some (pat-var {} y)) () (var {} y))\n  (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} i32)} 0)))"
        );
    }

    // --- Block bindings ---

    #[test]
    fn test_block_binding() {
        let expr = Expr::Block(
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
            "(let {}\n  (bind {}\n    x\n    (lit {surf_binding_type: \"inferred\", type: (t-prim {} i32)} 1))\n  (var {} x))"
        );
    }

    #[test]
    fn test_block_tuple_destructuring() {
        let expr = Expr::Block(
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
        assert!(result.contains("__chelis_tmp0"));
        assert!(result.contains("(var {} triple)"));
        assert!(result.contains("(lit {type: (t-prim {} i32)} 0)"));
        assert!(result.contains("(lit {type: (t-prim {} i32)} 1)"));
        assert!(result.contains("(lit {type: (t-prim {} i32)} 2)"));
        assert_eq!(result.matches("(tuple-get {}").count(), 3);
        assert!(result.contains("(var {} __chelis_tmp0)"));
        assert!(result.contains("(var {} c)"));
        assert!(!result.contains("(bind {} _ "));
    }

    // --- Lambda ---

    #[test]
    fn test_lambda() {
        let expr = Expr::Lambda(vec![param("x", None)], Box::new(tvar("x")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(fn {}\n  (params {} x)\n  (var {} x))"
        );
    }

    // --- Fun def (no types) ---

    #[test]
    fn test_fun_def() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            type_binders: vec![],
            params: vec![param("x", None)],
            ret_ty: None,
            effects: None,
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
            type_binders: vec![],
            params: vec![param("x", Some(named_ty("f32")))],
            ret_ty: Some(named_ty("f32")),
            effects: None,
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
            "(def {} f (fn {} (params {} (x {type: (t-var {} _)})) (var {} x)))"
        );
    }

    // --- Fun def with dim params ---

    #[test]
    fn test_fun_def_with_dim_params() {
        // def transpose[batch, hidden](x: tensor[batch, hidden, f32]): tensor[hidden, batch, f32] = x
        // batch and hidden should be d-var (polymorphic), NOT d-name
        let decl = Decl::FunDef {
            name: "transpose".to_string(),
            type_binders: vec![
                TypeBinder::unbounded("batch"),
                TypeBinder::unbounded("hidden"),
            ],
            params: vec![param(
                "x",
                Some(TypeExpr::Tensor(
                    vec![named_ty("batch"), named_ty("hidden")],
                    TensorPrecision::new("f32", s()),
                    s(),
                )),
            )],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("hidden"), named_ty("batch")],
                TensorPrecision::new("f32", s()),
                s(),
            )),
            effects: None,
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

    #[test]
    fn def_quantifier_general_tvar_through_arrow_is_t_var() {
        // chelis#293: a def generic over a general type variable declared in
        // the explicit `[..]` quantifier list, threaded through a
        // function-typed parameter, must desugar that name to `(t-var {} P)`
        // — NOT `(t-adt {} P)` — even though it is uppercase. The `[..]`
        // clause is the authoritative, unkinded quantifier source
        // (spec/02-surf-syntax.md §P4b), so it overrides the lexical
        // case-split.
        //
        // def apply_resid[n, P](
        //     x: tensor[n, f32],
        //     inner_p: P,
        //     f: tensor[n, f32] -> P -> tensor[n, f32],
        // ) -> tensor[n, f32] = add(x, f(x, inner_p))
        let decl = Decl::FunDef {
            name: "apply_resid".to_string(),
            type_binders: vec![TypeBinder::unbounded("n"), TypeBinder::unbounded("P")],
            params: vec![
                param(
                    "x",
                    Some(TypeExpr::Tensor(
                        vec![named_ty("n")],
                        TensorPrecision::new("f32", s()),
                        s(),
                    )),
                ),
                param("inner_p", Some(named_ty("P"))),
                param(
                    "f",
                    Some(TypeExpr::Arrow(
                        vec![
                            TypeExpr::Tensor(
                                vec![named_ty("n")],
                                TensorPrecision::new("f32", s()),
                                s(),
                            ),
                            named_ty("P"),
                        ],
                        Box::new(TypeExpr::Tensor(
                            vec![named_ty("n")],
                            TensorPrecision::new("f32", s()),
                            s(),
                        )),
                        s(),
                    )),
                ),
            ],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("n")],
                TensorPrecision::new("f32", s()),
                s(),
            )),
            effects: None,
            body: Expr::Apply(
                Box::new(tvar("add")),
                vec![
                    tvar("x"),
                    Expr::Apply(Box::new(tvar("f")), vec![tvar("x"), tvar("inner_p")], s()),
                ],
                s(),
            ),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2, "expected defsig + def, got: {nodes:?}");
        let sig = &nodes[0];
        // The general type variable `P` must be a t-var everywhere it
        // appears in the signature — the bare `inner_p` annotation AND the
        // arrow parameter — so it can unify at the call site.
        assert!(
            sig.contains("(t-var {} P)"),
            "expected `P` to desugar to (t-var {{}} P) in the signature, got:\n{sig}"
        );
        assert!(
            !sig.contains("(t-adt {} P)"),
            "general type var `P` from the `[..]` quantifier list must NOT be a \
             rigid ADT (chelis#293), got:\n{sig}"
        );
        // `n` is a declared dim param and must stay a d-var.
        assert!(
            sig.contains("(d-var {} n)"),
            "expected `n` to desugar to (d-var {{}} n), got:\n{sig}"
        );
    }

    #[test]
    fn def_uppercase_name_not_in_quantifier_stays_adt() {
        // Negative control for the chelis#293 fix: an uppercase type name
        // that is NOT in the `[..]` quantifier list keeps the lexical
        // case-split and stays a `(t-adt {} Activation)`. Only names the
        // user explicitly bound in `[..]` are promoted to type variables.
        //
        // def run[n](x: tensor[n, f32], a: Activation) -> tensor[n, f32] = x
        let decl = Decl::FunDef {
            name: "run".to_string(),
            type_binders: vec![TypeBinder::unbounded("n")],
            params: vec![
                param(
                    "x",
                    Some(TypeExpr::Tensor(
                        vec![named_ty("n")],
                        TensorPrecision::new("f32", s()),
                        s(),
                    )),
                ),
                param("a", Some(named_ty("Activation"))),
            ],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("n")],
                TensorPrecision::new("f32", s()),
                s(),
            )),
            effects: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        let sig = &nodes[0];
        assert!(
            sig.contains("(t-adt {} Activation)"),
            "an uppercase name NOT in the quantifier list must stay an ADT, got:\n{sig}"
        );
        assert!(
            !sig.contains("(t-var {} Activation)"),
            "an unquantified ADT name must not be promoted to a type variable, got:\n{sig}"
        );
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
        assert_eq!(nodes[0], "(def {} x (lit {type: (t-prim {} i32)} 42))");
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
        let expr = Expr::Grad(Box::new(tvar("f")), None, s());
        assert_eq!(
            print_expr(&DesugarCtx::default().desugar_expr(&expr)),
            "(grad {} (var {} f))"
        );
    }

    #[test]
    fn test_grad_with_wrt() {
        let expr = Expr::Grad(
            Box::new(tvar("loss")),
            Some(vec!["w".to_string(), "b".to_string()]),
            s(),
        );
        let Expr::Grad(target, _, _) = &expr else {
            unreachable!()
        };
        let ctx = DesugarCtx {
            resolved_grad_indices: vec![(target.as_ref() as *const Expr as usize, vec![1, 2])],
            ..DesugarCtx::default()
        };
        let actual = print_expr(&ctx.desugar_expr(&expr))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            actual,
            "(grad {wrt: (tuple {} (var {} w) (var {} b))} (var {} loss) (tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} i32)} 2)))"
        );
    }

    #[test]
    fn test_cast() {
        let expr = Expr::Cast(
            Box::new(tvar("x")),
            "bf16".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {} (var {} x) (t-prim {} bf16))"
        );
    }

    // --- Position 4 (spec §5.6 / §P10b) for bare scalar literals ---
    //
    // `cast(literal, p)` binds the literal AT `p`, not at the §5.3
    // default narrowed-then-converted. Issue #308: `cast(1.1, f64)`
    // previously desugared to `(cast (lit {type: f32} 1.1) f64)`, so
    // every value lane that honors the lit's type meta materialized
    // f32(1.1) and then widened — the f32-truncation signature
    // `1.100000023841858` instead of exact f64 `1.1`.

    #[test]
    fn cast_of_float_literal_adopts_target_precision() {
        let expr = Expr::Cast(
            Box::new(float_lit(1.1)),
            "f64".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {}\n  (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} f64)} 1.1)\n  (t-prim {} f64))"
        );
    }

    #[test]
    fn cast_of_negative_float_literal_folds_sign_and_adopts() {
        // Mirror of the RT-2 P2 sign-fold in contextual tensor literals:
        // the parser produces `Unary(Neg, Lit(1.1))` for `-1.1`.
        let expr = Expr::Cast(
            Box::new(Expr::Unary(UnaryOp::Neg, Box::new(float_lit(1.1)), s())),
            "f64".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {}\n  (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} f64)} -1.1)\n  (t-prim {} f64))"
        );
    }

    #[test]
    fn cast_of_int_literal_adopts_integer_target() {
        // The documented §5.3 escape hatch for out-of-i32-range
        // literals: `cast(3000000000, i64)` must bind the literal at
        // i64 so `infer_lit` does not range-check it against i32.
        let expr = Expr::Cast(
            Box::new(int_lit(3_000_000_000)),
            "i64".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {}\n  (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} i64)} 3000000000)\n  (t-prim {} i64))"
        );
    }

    #[test]
    fn cast_of_int_literal_adopts_float_target() {
        let expr = Expr::Cast(
            Box::new(int_lit(5)),
            "f64".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {}\n  (lit {literal_source: integer, surf_literal_style: \"unsuffixed\", type: (t-prim {} f64)} 5)\n  (t-prim {} f64))"
        );
    }

    #[test]
    fn cast_of_float_literal_to_integer_does_not_adopt() {
        // Negative parity: a float literal cannot "adopt" an integer
        // type — `cast(1.9, i32)` keeps the §5.3 f32 default on the
        // literal, so the checked cast Domain-traps on the fractional value.
        let expr = Expr::Cast(
            Box::new(float_lit(1.9)),
            "i32".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {} (lit {type: (t-prim {} f32)} 1.9) (t-prim {} i32))"
        );
    }

    #[test]
    fn cast_of_suffixed_literal_does_not_adopt() {
        // Negative parity: a typed-suffix literal binds at exactly its
        // suffix precision (spec §5.5); `cast(1.1f32, f64)` means
        // "widen this f32 value", not "re-bind the decimal at f64".
        let expr = Expr::Cast(
            Box::new(Expr::Lit(Literal::TypedFloat(1.1, LiteralSuffix::F32), s())),
            "f64".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {}\n  (lit {surf_literal_style: \"explicit\", type: (t-prim {} f32)} 1.1)\n  (t-prim {} f64))"
        );
    }

    #[test]
    fn cast_of_literal_to_bool_does_not_adopt() {
        // Negative parity: bool is not a numeric binding precision for
        // a numeric literal; keep the default-typed literal + cast.
        let expr = Expr::Cast(
            Box::new(int_lit(1)),
            "bool".to_string(),
            CastMode::Checked,
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {} (lit {type: (t-prim {} i32)} 1) (t-prim {} bool))"
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
        // Outside an explicit declaration binder list, a lowercase unknown
        // spelling remains a primitive candidate for checker-owned rejection.
        assert_eq!(print_expr(&desugar_type(&named_ty("a"))), "(t-prim {} a)");
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
            TensorPrecision::new("f32", s()),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_tensor_with_literal_dims() {
        let ty = TypeExpr::Tensor(
            vec![named_ty("32"), named_ty("784")],
            TensorPrecision::new("f32", s()),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_tensor_rank_spread() {
        // `tensor[..r, f32]` desugars to a sole `(d-rank {} r)` dim node.
        let ty = TypeExpr::Tensor(
            vec![TypeExpr::RankSpread("r".to_string(), s())],
            TensorPrecision::new("f32", s()),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-rank {} r) (t-prim {} f32))"
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
            opaque: false,
            invariant: None,
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
            opaque: false,
            invariant: None,
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
        assert_eq!(nodes[0], "(import-all {surf_path: \"Foo\"} foo)");
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
        assert_eq!(nodes[0], "(import {surf_path: \"Foo\"} foo (a b))");
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
            "(pat-record {} Adam (kv {} lr (pat-var {} lr)) (kv {} eps (pat-var {} eps)))"
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
