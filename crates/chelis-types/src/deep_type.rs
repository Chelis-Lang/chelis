//! Fail-closed resolution of Deep type and dimension expressions.
//!
//! This is the single boundary that may translate Deep type syntax into the
//! checker's internal [`Type`]. Resolution is context-sensitive: declarations
//! choose how names become binders, while ordinary source annotations are
//! closed. A failure reports exactly once and returns an [`ErrorWitness`]
//! instead of inventing a variable, wildcard, or partial type.

use std::collections::{HashMap, HashSet};

use chelis_deep::ast as deep;

use crate::adt::AdtRegistry;
use crate::errors::{CheckError, CheckErrorKind, ErrorWitness, report_witness};
use crate::types::{Dim, DimVar, Prim, RankVar, TensorPrec, Type, TypeVar, VarGen};

/// A type that crossed the Deep syntax boundary without a silent fallback.
/// The tuple field is private so callers cannot assert resolution without
/// going through [`DeepTypeResolver::resolve`].
#[derive(Debug, Clone)]
pub(crate) struct ResolvedDeepType(Type);

impl ResolvedDeepType {
    pub(crate) fn into_type(self) -> Type {
        self.0
    }
}

/// The checker surface that owns a type expression. Included in diagnostics
/// so malformed nested syntax points to the boundary that interpreted it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TypeUseSite {
    DeftypeField,
    TypeAliasBody,
    Defsig,
    Annotation,
    CastTarget,
    CompilerMetadata,
}

impl TypeUseSite {
    fn label(self) -> &'static str {
        match self {
            Self::DeftypeField => "deftype field",
            Self::TypeAliasBody => "typealias body",
            Self::Defsig => "defsig",
            Self::Annotation => "type annotation",
            Self::CastTarget => "cast target",
            Self::CompilerMetadata => "compiler-generated type metadata",
        }
    }
}

/// Which names a Deep type expression may bind.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BinderMode<'a> {
    /// User input outside a binder declaration. Only `_` is a legal inference
    /// hole; other `t-var` / `d-var` / `d-rank` names are unbound.
    ClosedInput,
    /// A `deftype` or `typealias` parameter list explicitly names every legal
    /// type, dimension, and rank binder.
    Explicit(&'a HashSet<String>),
    /// A `defsig` implicitly quantifies each named type/dimension/rank variable.
    ImplicitGeneric,
    /// Metadata emitted by a checked compiler pass may carry generated names.
    TrustedCompilerMetadata,
}

/// Nominal names and arities visible while resolving a check unit. This is
/// deliberately transient and never enters the serde-backed checker context.
#[derive(Debug, Clone, Default)]
pub(crate) struct KnownTypeHeaders {
    arities: HashMap<String, usize>,
}

impl KnownTypeHeaders {
    pub(crate) fn from_registry(registry: &AdtRegistry) -> Self {
        let mut headers = Self::default();
        // Checker-native nominal surfaces that are typed structurally by
        // builtin rules instead of carrying registry variants.
        for (name, arity) in [("Dict", 2), ("Result", 2), ("String", 0)] {
            headers.arities.insert(name.to_string(), arity);
        }
        for (name, definition) in &registry.defs {
            headers
                .arities
                .insert(name.clone(), definition.type_params.len());
        }
        for (name, definition) in &registry.aliases {
            headers
                .arities
                .insert(name.clone(), definition.params.len());
        }
        headers
    }

    pub(crate) fn insert(&mut self, name: impl Into<String>, arity: usize) {
        self.arities.entry(name.into()).or_insert(arity);
    }

    fn arity(&self, name: &str) -> Option<usize> {
        self.arities.get(name).copied()
    }
}

/// Stateful resolver for one binder scope. Sharing an instance across field
/// types or signature components preserves repeated variable identity.
pub(crate) struct DeepTypeResolver<'a> {
    use_site: TypeUseSite,
    binder_mode: BinderMode<'a>,
    headers: &'a KnownTypeHeaders,
    vg: &'a mut VarGen,
    errors: &'a mut Vec<CheckError>,
    type_vars: HashMap<String, TypeVar>,
    dim_vars: HashMap<String, DimVar>,
    rank_vars: HashMap<String, RankVar>,
}

impl<'a> DeepTypeResolver<'a> {
    pub(crate) fn new(
        use_site: TypeUseSite,
        binder_mode: BinderMode<'a>,
        headers: &'a KnownTypeHeaders,
        vg: &'a mut VarGen,
        errors: &'a mut Vec<CheckError>,
    ) -> Self {
        let mut resolver = Self {
            use_site,
            binder_mode,
            headers,
            vg,
            errors,
            type_vars: HashMap::new(),
            dim_vars: HashMap::new(),
            rank_vars: HashMap::new(),
        };
        if let BinderMode::Explicit(names) = binder_mode {
            // Nominal parameters are type arguments even when their occurrence
            // in a field is dimension- or rank-kinded.
            for name in names {
                resolver
                    .type_vars
                    .insert(name.clone(), resolver.vg.fresh_tvar());
            }
        }
        resolver
    }

    pub(crate) fn resolve(&mut self, expr: &deep::Expr) -> Result<ResolvedDeepType, ErrorWitness> {
        self.resolve_type(expr).map(ResolvedDeepType)
    }

    pub(crate) fn type_var(&self, name: &str) -> Option<TypeVar> {
        self.type_vars.get(name).copied()
    }

    pub(crate) fn type_vars(&self) -> Vec<TypeVar> {
        let mut vars: Vec<_> = self.type_vars.values().copied().collect();
        vars.sort_by_key(|var| var.0);
        vars
    }

    pub(crate) fn dim_vars(&self) -> Vec<DimVar> {
        let mut vars: Vec<_> = self.dim_vars.values().copied().collect();
        vars.sort_by_key(|var| var.0);
        vars
    }

    pub(crate) fn rank_vars(&self) -> Vec<RankVar> {
        let mut vars: Vec<_> = self.rank_vars.values().copied().collect();
        vars.sort_by_key(|var| var.0);
        vars
    }

    fn resolve_type(&mut self, expr: &deep::Expr) -> Result<Type, ErrorWitness> {
        let deep::Expr::List(list, _) = expr else {
            return Err(self.malformed(format!(
                "{} must use a canonical Deep type form; bare `{}` is not a type",
                self.use_site.label(),
                render_expr(expr)
            )));
        };
        let (tag, children) = self.type_form(list)?;
        match tag {
            "t-prim" => {
                let name = self.one_symbol(tag, children)?;
                Prim::parse_name(name).map(Type::Prim).ok_or_else(|| {
                    self.type_error(format!(
                        "unknown primitive type `{name}` in {}",
                        self.use_site.label()
                    ))
                })
            }
            "t-var" => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_type_var(name).map(Type::Var)
            }
            "t-fn" => {
                if children.is_empty() {
                    return Err(self.malformed(format!(
                        "malformed `t-fn` in {}: expected at least a return type",
                        self.use_site.label()
                    )));
                }
                let mut parts = Vec::with_capacity(children.len());
                for child in children {
                    parts.push(self.resolve_type(child)?);
                }
                let ret = parts.pop().expect("non-empty checked above");
                Ok(Type::Fn(parts, Box::new(ret)))
            }
            "t-ref" => {
                self.exact_arity(tag, children, 1)?;
                Ok(Type::Ref(Box::new(self.resolve_type(&children[0])?)))
            }
            "t-tensor" => {
                if children.is_empty() {
                    return Err(self.malformed(format!(
                        "malformed `t-tensor` in {}: expected dimensions followed by a precision",
                        self.use_site.label()
                    )));
                }
                let mut dims = Vec::with_capacity(children.len().saturating_sub(1));
                for child in &children[..children.len() - 1] {
                    dims.push(self.resolve_dim(child)?);
                }
                let precision = match self.resolve_type(&children[children.len() - 1])? {
                    Type::Prim(prim) => TensorPrec::Concrete(prim),
                    Type::Var(var) => TensorPrec::Var(var),
                    other => {
                        return Err(self.type_error(format!(
                            "tensor precision in {} must be `t-prim` or a legal `t-var`, got `{other}`",
                            self.use_site.label()
                        )));
                    }
                };
                Ok(Type::Tensor(dims, precision))
            }
            "t-adt" => {
                let Some(name) = children.first().and_then(symbol_name) else {
                    return Err(self.malformed(format!(
                        "malformed `t-adt` in {}: expected a nominal type name followed by type arguments",
                        self.use_site.label()
                    )));
                };
                let Some(expected) = self.headers.arity(name) else {
                    return Err(self.type_error(format!(
                        "unknown nominal type `{name}` in {}",
                        self.use_site.label()
                    )));
                };
                let actual = children.len() - 1;
                if actual != expected {
                    return Err(self.type_error(format!(
                        "nominal type `{name}` in {} expects {expected} type argument(s), got {actual}",
                        self.use_site.label()
                    )));
                }
                let mut args = Vec::with_capacity(actual);
                for child in &children[1..] {
                    args.push(self.resolve_type(child)?);
                }
                Ok(Type::Adt(name.to_string(), args))
            }
            "t-tuple" => {
                let mut elements = Vec::with_capacity(children.len());
                for child in children {
                    elements.push(self.resolve_type(child)?);
                }
                Ok(Type::Tuple(elements))
            }
            "t-unit" => {
                self.exact_arity(tag, children, 0)?;
                Ok(Type::Unit)
            }
            other => Err(self.malformed(format!(
                "unknown Deep type tag `{other}` in {}",
                self.use_site.label()
            ))),
        }
    }

    fn resolve_dim(&mut self, expr: &deep::Expr) -> Result<Dim, ErrorWitness> {
        let deep::Expr::List(list, _) = expr else {
            return Err(self.malformed(format!(
                "tensor dimension in {} must use a canonical Deep dimension form, got `{}`",
                self.use_site.label(),
                render_expr(expr)
            )));
        };
        let (tag, children) = self.type_form(list)?;
        match tag {
            "d-name" => {
                let name = self.one_symbol(tag, children)?;
                if name == "*" {
                    Ok(Dim::Wildcard)
                } else {
                    Ok(Dim::Name(name.to_string()))
                }
            }
            "d-var" => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_dim_var(name).map(Dim::Var)
            }
            "d-rank" => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_rank_var(name).map(Dim::Rank)
            }
            "d-lit" => {
                self.exact_arity(tag, children, 1)?;
                match &children[0] {
                    deep::Expr::Atom(deep::Atom::Int(value), _) => Ok(Dim::Lit(*value)),
                    _ => Err(self.malformed(format!(
                        "malformed `d-lit` in {}: expected one integer child",
                        self.use_site.label()
                    ))),
                }
            }
            other => Err(self.malformed(format!(
                "unknown Deep dimension tag `{other}` in {}",
                self.use_site.label()
            ))),
        }
    }

    fn resolve_type_var(&mut self, name: &str) -> Result<TypeVar, ErrorWitness> {
        if name == "_" {
            return self
                .allows_hole()
                .then(|| self.vg.fresh_tvar())
                .ok_or_else(|| self.unbound("type", name));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("type", name));
        }
        Ok(*self
            .type_vars
            .entry(name.to_string())
            .or_insert_with(|| self.vg.fresh_tvar()))
    }

    fn resolve_dim_var(&mut self, name: &str) -> Result<DimVar, ErrorWitness> {
        if name == "_" {
            return self
                .allows_hole()
                .then(|| self.vg.fresh_dvar())
                .ok_or_else(|| self.unbound("dimension", name));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("dimension", name));
        }
        Ok(*self
            .dim_vars
            .entry(name.to_string())
            .or_insert_with(|| self.vg.fresh_dvar()))
    }

    fn resolve_rank_var(&mut self, name: &str) -> Result<RankVar, ErrorWitness> {
        if name == "_" {
            return self
                .allows_hole()
                .then(|| self.vg.fresh_rvar())
                .ok_or_else(|| self.unbound("rank", name));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("rank", name));
        }
        Ok(*self
            .rank_vars
            .entry(name.to_string())
            .or_insert_with(|| self.vg.fresh_rvar()))
    }

    fn allows_name(&self, name: &str) -> bool {
        match self.binder_mode {
            BinderMode::ClosedInput => false,
            BinderMode::Explicit(names) => names.contains(name),
            BinderMode::ImplicitGeneric | BinderMode::TrustedCompilerMetadata => true,
        }
    }

    fn allows_hole(&self) -> bool {
        !matches!(self.binder_mode, BinderMode::Explicit(_))
    }

    fn type_form<'b>(
        &mut self,
        list: &'b deep::List,
    ) -> Result<(&'b str, &'b [deep::Expr]), ErrorWitness> {
        let Some(tag) = list.elements.first().and_then(symbol_name) else {
            return Err(self.malformed(format!(
                "malformed Deep type form in {}: expected a tag symbol",
                self.use_site.label()
            )));
        };
        if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
            return Err(self.malformed(format!(
                "malformed `{tag}` in {}: the metadata map must be present at element 1",
                self.use_site.label()
            )));
        }
        Ok((tag, &list.elements[2..]))
    }

    fn one_symbol<'b>(
        &mut self,
        tag: &str,
        children: &'b [deep::Expr],
    ) -> Result<&'b str, ErrorWitness> {
        self.exact_arity(tag, children, 1)?;
        symbol_name(&children[0]).ok_or_else(|| {
            self.malformed(format!(
                "malformed `{tag}` in {}: expected one symbol child",
                self.use_site.label()
            ))
        })
    }

    fn exact_arity(
        &mut self,
        tag: &str,
        children: &[deep::Expr],
        expected: usize,
    ) -> Result<(), ErrorWitness> {
        if children.len() == expected {
            Ok(())
        } else {
            Err(self.malformed(format!(
                "malformed `{tag}` in {}: expected {expected} child(ren), got {}",
                self.use_site.label(),
                children.len()
            )))
        }
    }

    fn unbound(&mut self, kind: &str, name: &str) -> ErrorWitness {
        self.type_error(format!(
            "undeclared {kind} variable `{name}` in {}",
            self.use_site.label()
        ))
    }

    fn malformed(&mut self, message: String) -> ErrorWitness {
        report_witness(
            self.errors,
            CheckError::new(CheckErrorKind::MalformedForm, message, vec![]),
        )
    }

    fn type_error(&mut self, message: String) -> ErrorWitness {
        report_witness(
            self.errors,
            CheckError::new(CheckErrorKind::TypeMismatch, message, vec![]),
        )
    }
}

fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some(name),
        _ => None,
    }
}

fn render_expr(expr: &deep::Expr) -> String {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => name.clone(),
        deep::Expr::Atom(atom, _) => format!("{atom:?}"),
        deep::Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .unwrap_or("<list>")
            .to_string(),
        deep::Expr::Map(_, _) => "<metadata-map>".to_string(),
        deep::Expr::MetaExpr(_, _) => "<metadata-expression>".to_string(),
    }
}
