//! Fail-closed resolution of Deep type and dimension expressions.
//!
//! This is the single boundary that may translate Deep type syntax into the
//! checker's internal [`Type`]. Resolution is context-sensitive: declarations
//! choose how names become binders, while ordinary source annotations are
//! closed. A failure reports exactly once and returns an [`ErrorWitness`]
//! instead of inventing a variable, wildcard, or partial type.

use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::ast as deep;

use crate::adt::AdtRegistry;
use crate::errors::{CheckError, CheckErrorKind, ErrorWitness, report_witness};
use crate::session::DiagnosticSink;
use crate::types::{
    Dim, DimVar, NominalArg, NominalParamKind, Prim, RankVar, TensorPrec, Type, TypeVar, VarGen,
};

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

/// Cast-target syntax after it has crossed the same canonical-form and arity
/// checks as every other Deep type consumer. Bare primitive spellings remain a
/// compatibility surface; canonical `t-prim` records whether the historical
/// zero-arity nominal spelling is eligible for the opacity checks.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedCastTarget {
    PrimitiveSpelling { name: String, canonical: bool },
    Type(ResolvedDeepType),
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

/// Source location owned by one type-resolution root. Explicit producer span
/// metadata wins; otherwise the structural AST range is retained and exposed
/// through a stable `source:<start>..<end>` identifier as well as the byte
/// offset. The value is copied out of the AST so resolver reuse is safe.
#[derive(Debug, Clone, Default)]
pub(crate) struct TypeDiagnosticLocation {
    span_offset: Option<usize>,
    span_id: Option<String>,
}

impl TypeDiagnosticLocation {
    pub(crate) fn from_expr(expr: &deep::Expr) -> Option<Self> {
        let structural = expr.span();
        let explicit_id = expr.span_id().map(str::to_string);
        let span_offset = explicit_id
            .as_deref()
            .and_then(span_offset_from_id)
            .or_else(|| (structural.len > 0).then_some(structural.offset));
        let span_id = explicit_id.or_else(|| {
            (structural.len > 0)
                .then(|| format!("source:{}..{}", structural.offset, structural.end()))
        });
        (span_offset.is_some() || span_id.is_some()).then_some(Self {
            span_offset,
            span_id,
        })
    }

    pub(crate) fn attach(&self, mut error: CheckError) -> CheckError {
        if error.span_offset.is_none() {
            error.span_offset = self.span_offset;
        }
        if error.span_id.is_none() {
            error.span_id.clone_from(&self.span_id);
        }
        error
    }
}

fn span_offset_from_id(span_id: &str) -> Option<usize> {
    span_id
        .rfind(':')
        .map(|index| &span_id[index + 1..])
        .unwrap_or(span_id)
        .split_once("..")
        .and_then(|(start, _)| start.parse::<usize>().ok())
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
    /// A nominal declaration whose parameter kinds were fixed before any
    /// declaration body was resolved.
    ExplicitKinds(&'a HashMap<String, NominalParamKind>),
    /// A `defsig` implicitly quantifies each named type/dimension/rank variable.
    ImplicitGeneric,
    /// Metadata emitted by a checked compiler pass may carry generated names.
    TrustedCompilerMetadata,
}

/// Nominal declaration headers visible while resolving one check unit.
///
/// This is the explicit, runtime-only half of declaration state. It may
/// contain self/forward headers whose bodies have not yet validated; only
/// successfully resolved declarations enter [`AdtRegistry::defs`] or
/// [`AdtRegistry::aliases`]. The registry's serde representation skips this
/// environment and reconstructs it from those validated definitions before a
/// later check.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct TypeResolutionEnv {
    headers: HashMap<String, Vec<NominalParamKind>>,
}

impl TypeResolutionEnv {
    pub(crate) fn from_registry(registry: &AdtRegistry) -> Self {
        let mut headers = Self::default();
        // Checker-native nominal surfaces that are typed structurally by
        // builtin rules instead of carrying registry variants.
        for (name, arity) in [("Dict", 2), ("Result", 2), ("String", 0)] {
            headers
                .headers
                .insert(name.to_string(), vec![NominalParamKind::Type; arity]);
        }
        for (name, definition) in &registry.defs {
            let kinds = if definition.param_kinds.len() == definition.type_params.len() {
                definition.param_kinds.clone()
            } else {
                vec![NominalParamKind::Type; definition.type_params.len()]
            };
            headers.headers.insert(name.clone(), kinds);
        }
        for (name, definition) in &registry.aliases {
            let kinds = if definition.param_kinds.len() == definition.params.len() {
                definition.param_kinds.clone()
            } else {
                vec![NominalParamKind::Type; definition.params.len()]
            };
            headers.headers.insert(name.clone(), kinds);
        }
        headers
    }

    pub(crate) fn insert(&mut self, name: impl Into<String>, kinds: Vec<NominalParamKind>) {
        self.headers.entry(name.into()).or_insert(kinds);
    }

    pub(crate) fn extend_from(&mut self, other: &Self) {
        for (name, kinds) in &other.headers {
            self.headers.insert(name.clone(), kinds.clone());
        }
    }

    pub(crate) fn param_kinds(&self, name: &str) -> Option<&[NominalParamKind]> {
        self.headers.get(name).map(Vec::as_slice)
    }
}

/// Stateful resolver for one binder scope. Sharing an instance across field
/// types or signature components preserves repeated variable identity.
pub(crate) struct DeepTypeResolver<'resolver, 'session, 'binders> {
    use_site: TypeUseSite,
    binder_mode: BinderMode<'binders>,
    headers: &'resolver TypeResolutionEnv,
    vg: &'resolver mut VarGen,
    errors: &'resolver mut DiagnosticSink<'session>,
    type_vars: HashMap<String, TypeVar>,
    dim_vars: HashMap<String, DimVar>,
    rank_vars: HashMap<String, RankVar>,
    owner_location: Option<TypeDiagnosticLocation>,
    resolution_location: Option<TypeDiagnosticLocation>,
    current_location: Option<TypeDiagnosticLocation>,
}

impl<'resolver, 'session, 'binders> DeepTypeResolver<'resolver, 'session, 'binders> {
    pub(crate) fn new(
        use_site: TypeUseSite,
        binder_mode: BinderMode<'binders>,
        headers: &'resolver TypeResolutionEnv,
        vg: &'resolver mut VarGen,
        errors: &'resolver mut DiagnosticSink<'session>,
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
            owner_location: None,
            resolution_location: None,
            current_location: None,
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
        if let BinderMode::ExplicitKinds(kinds) = binder_mode {
            for (name, kind) in kinds {
                match kind {
                    NominalParamKind::Type => {
                        resolver
                            .type_vars
                            .insert(name.clone(), resolver.vg.fresh_tvar());
                    }
                    NominalParamKind::Dimension => {
                        resolver
                            .dim_vars
                            .insert(name.clone(), resolver.vg.fresh_dvar());
                    }
                }
            }
        }
        resolver
    }

    /// Provide the source construct that owns this resolver use site. The
    /// resolved type expression remains the preferred location; this owner is
    /// the fallback for synthesized children such as a Surf cast target.
    pub(crate) fn with_diagnostic_owner(mut self, owner: &deep::Expr) -> Self {
        self.owner_location = TypeDiagnosticLocation::from_expr(owner);
        self
    }

    pub(crate) fn resolve(&mut self, expr: &deep::Expr) -> Result<ResolvedDeepType, ErrorWitness> {
        self.begin_resolution(expr);
        self.resolve_type(expr).map(ResolvedDeepType)
    }

    /// Resolve cast-target syntax through this boundary before semantic cast
    /// classification. In particular, canonical `t-prim` uses `one_symbol`,
    /// so an extra child cannot be ignored by a cast-only fast path.
    pub(crate) fn resolve_cast_target(
        &mut self,
        expr: &deep::Expr,
    ) -> Result<ResolvedCastTarget, ErrorWitness> {
        self.begin_resolution(expr);
        if let Some(name) = symbol_name(expr) {
            return Ok(ResolvedCastTarget::PrimitiveSpelling {
                name: name.to_string(),
                canonical: false,
            });
        }
        if matches!(expr, deep::Expr::List(_, _) | deep::Expr::Node(_, _)) {
            let (form_tag, tag, children) = self.type_form_expr(expr)?;
            if form_tag == Some(DeepTag::TPrim) {
                let name = self.one_symbol(tag, children)?;
                // Reserved primitive spellings cannot be captured by a
                // declaration-local type variable with the same name. This
                // matters for the deferred `f8e4m3` spelling: treating it as
                // the signature's quantified result would bypass the owning
                // §1.1.1 rejection and let the build lane accept the program.
                // Ordinary authored binders such as `p` still take the
                // symbolic path below.
                if Prim::parse_name(name).is_some() {
                    return Ok(ResolvedCastTarget::PrimitiveSpelling {
                        name: name.to_string(),
                        canonical: true,
                    });
                }
                if let Some(var) = self.type_vars.get(name).copied() {
                    return Ok(ResolvedCastTarget::Type(ResolvedDeepType(Type::Var(var))));
                }
                return Ok(ResolvedCastTarget::PrimitiveSpelling {
                    name: name.to_string(),
                    canonical: true,
                });
            }
        }
        self.resolve(expr).map(ResolvedCastTarget::Type)
    }

    pub(crate) fn type_var(&self, name: &str) -> Option<TypeVar> {
        self.type_vars.get(name).copied()
    }

    pub(crate) fn dim_var(&self, name: &str) -> Option<DimVar> {
        self.dim_vars.get(name).copied()
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

    pub(crate) fn diagnostic_location(&self) -> Option<TypeDiagnosticLocation> {
        self.current_location
            .clone()
            .or_else(|| self.resolution_location.clone())
            .or_else(|| self.owner_location.clone())
    }

    fn begin_resolution(&mut self, expr: &deep::Expr) {
        self.resolution_location =
            TypeDiagnosticLocation::from_expr(expr).or_else(|| self.owner_location.clone());
        self.current_location = self.resolution_location.clone();
    }

    fn enter_expr(&mut self, expr: &deep::Expr) {
        self.current_location = TypeDiagnosticLocation::from_expr(expr)
            .or_else(|| self.resolution_location.clone())
            .or_else(|| self.owner_location.clone());
    }

    fn resolve_type(&mut self, expr: &deep::Expr) -> Result<Type, ErrorWitness> {
        self.enter_expr(expr);
        let (form_tag, tag, children) = self.type_form_expr(expr)?;
        match form_tag {
            Some(DeepTag::TPrim) => {
                let name = self.one_symbol(tag, children)?;
                Prim::parse_name(name).map(Type::Prim).ok_or_else(|| {
                    self.type_error(format!(
                        "unknown primitive type `{name}` in {}",
                        self.use_site.label()
                    ))
                })
            }
            Some(DeepTag::TVar) => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_type_var(name).map(Type::Var)
            }
            Some(DeepTag::TFn) => {
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
            Some(DeepTag::TRef) => {
                self.exact_arity(tag, children, 1)?;
                Ok(Type::Ref(Box::new(self.resolve_type(&children[0])?)))
            }
            Some(DeepTag::TTensor) => {
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
            Some(DeepTag::TAdt) => {
                let Some(name) = children.first().and_then(symbol_name) else {
                    return Err(self.malformed(format!(
                        "malformed `t-adt` in {}: expected a nominal type name followed by type arguments",
                        self.use_site.label()
                    )));
                };
                let Some(param_kinds) = self.headers.param_kinds(name) else {
                    return Err(self.type_error(format!(
                        "unknown nominal type `{name}` in {}",
                        self.use_site.label()
                    )));
                };
                let actual = children.len() - 1;
                if actual != param_kinds.len() {
                    return Err(self.type_error(format!(
                        "nominal type `{name}` in {} expects {} argument(s), got {actual}",
                        self.use_site.label(),
                        param_kinds.len()
                    )));
                }
                let mut args = Vec::with_capacity(actual);
                for (index, (child, kind)) in children[1..].iter().zip(param_kinds).enumerate() {
                    let (child_tag, _, child_children) = self.type_form_expr(child)?;
                    match kind {
                        NominalParamKind::Type => {
                            if matches!(
                                child_tag,
                                Some(
                                    DeepTag::DName | DeepTag::DVar | DeepTag::DLit | DeepTag::DRank
                                )
                            ) {
                                return Err(self.type_error(format!(
                                    "nominal type `{name}` argument {} expects a type, got a dimension",
                                    index + 1
                                )));
                            }
                            args.push(NominalArg::Type(self.resolve_type(child)?));
                        }
                        NominalParamKind::Dimension => {
                            let dim = match child_tag {
                                Some(DeepTag::DName | DeepTag::DVar | DeepTag::DLit) => {
                                    self.resolve_dim(child)?
                                }
                                // Surf cannot know the target header while desugaring
                                // `Frame[n]`, so symbolic nominal arguments arrive as
                                // `t-var`; the checker-owned header gives them their
                                // dimension meaning here.
                                Some(DeepTag::TVar) => {
                                    let child_name = self.one_symbol("t-var", child_children)?;
                                    Dim::Var(self.resolve_dim_var(child_name)?)
                                }
                                _ => {
                                    return Err(self.type_error(format!(
                                        "nominal type `{name}` argument {} expects a dimension, got a type",
                                        index + 1
                                    )));
                                }
                            };
                            args.push(NominalArg::Dimension(dim));
                        }
                    }
                }
                if param_kinds.contains(&NominalParamKind::Dimension) {
                    Ok(Type::KindedAdt(name.to_string(), args))
                } else {
                    Ok(Type::Adt(
                        name.to_string(),
                        args.into_iter()
                            .map(|argument| match argument {
                                NominalArg::Type(ty) => ty,
                                NominalArg::Dimension(_) => {
                                    unreachable!("type-only header produced a dimension argument")
                                }
                            })
                            .collect(),
                    ))
                }
            }
            Some(DeepTag::TTuple) => {
                let mut elements = Vec::with_capacity(children.len());
                for child in children {
                    elements.push(self.resolve_type(child)?);
                }
                Ok(Type::Tuple(elements))
            }
            Some(DeepTag::TUnit) => {
                self.exact_arity(tag, children, 0)?;
                Ok(Type::Unit)
            }
            _ => Err(self.malformed(format!(
                "unknown Deep type tag `{tag}` in {}",
                self.use_site.label()
            ))),
        }
    }

    fn resolve_dim(&mut self, expr: &deep::Expr) -> Result<Dim, ErrorWitness> {
        self.enter_expr(expr);
        let (form_tag, tag, children) = self.type_form_expr(expr)?;
        match form_tag {
            Some(DeepTag::DName) => {
                let name = self.one_symbol(tag, children)?;
                if name == "*" {
                    Ok(Dim::Wildcard)
                } else {
                    Ok(Dim::Name(name.to_string()))
                }
            }
            Some(DeepTag::DVar) => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_dim_var(name).map(Dim::Var)
            }
            Some(DeepTag::DRank) => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_rank_var(name).map(Dim::Rank)
            }
            Some(DeepTag::DLit) => {
                self.exact_arity(tag, children, 1)?;
                match &children[0] {
                    deep::Expr::Atom(deep::Atom::Int(value), _) => Ok(Dim::Lit(*value)),
                    _ => Err(self.malformed(format!(
                        "malformed `d-lit` in {}: expected one integer child",
                        self.use_site.label()
                    ))),
                }
            }
            _ => Err(self.malformed(format!(
                "unknown Deep dimension tag `{tag}` in {}",
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
            BinderMode::ExplicitKinds(kinds) => kinds.contains_key(name),
            BinderMode::ImplicitGeneric | BinderMode::TrustedCompilerMetadata => true,
        }
    }

    fn allows_hole(&self) -> bool {
        !matches!(self.binder_mode, BinderMode::Explicit(_))
    }

    /// Decode-once (chelis#731 Phase 3): the decoded tag drives dispatch;
    /// the string is the diagnostic spelling. `None` with a symbol string
    /// is the raw-string boundary (a lenient-parsed unknown head), which
    /// the dispatch rejects with the same unknown-tag diagnostic as any
    /// non-type vocabulary tag.
    fn type_form<'b>(
        &mut self,
        list: &'b deep::List,
    ) -> Result<(Option<DeepTag>, &'b str, &'b [deep::Expr]), ErrorWitness> {
        let (tag, tag_str) = match list.elements.first() {
            Some(deep::Expr::Atom(deep::Atom::Tag(tag), _)) => (Some(*tag), tag.as_str()),
            Some(deep::Expr::Atom(deep::Atom::Name(name), _)) => (None, name.as_str()),
            _ => {
                return Err(self.malformed(format!(
                    "malformed Deep type form in {}: expected a tag symbol",
                    self.use_site.label()
                )));
            }
        };
        if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
            return Err(self.malformed(format!(
                "malformed `{tag_str}` in {}: the metadata map must be present at element 1",
                self.use_site.label()
            )));
        }
        Ok((tag, tag_str, &list.elements[2..]))
    }

    fn type_form_expr<'b>(
        &mut self,
        expr: &'b deep::Expr,
    ) -> Result<(Option<DeepTag>, &'b str, &'b [deep::Expr]), ErrorWitness> {
        match expr {
            deep::Expr::Node(node, _) => {
                Ok((Some(node.tag()), node.tag().as_str(), node.children_slice()))
            }
            deep::Expr::List(list, _) => self.type_form(list),
            deep::Expr::UnknownForm(data) => {
                Ok((None, data.head.as_str(), data.children.as_slice()))
            }
            _ => Err(self.malformed(format!(
                "{} must use a canonical Deep type form; bare `{}` is not a type",
                self.use_site.label(),
                render_expr(expr)
            ))),
        }
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
        let error = CheckError::new(CheckErrorKind::MalformedForm, message, vec![]);
        let error = self
            .diagnostic_location()
            .map_or(error.clone(), |location| location.attach(error));
        report_witness(self.errors, error)
    }

    fn type_error(&mut self, message: String) -> ErrorWitness {
        let error = CheckError::new(CheckErrorKind::TypeMismatch, message, vec![]);
        let error = self
            .diagnostic_location()
            .map_or(error.clone(), |location| location.attach(error));
        report_witness(self.errors, error)
    }
}

fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn render_expr(expr: &deep::Expr) -> String {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(name), _) => name.clone(),
        deep::Expr::Atom(atom, _) => format!("{atom:?}"),
        deep::Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .unwrap_or("<list>")
            .to_string(),
        deep::Expr::Map(_, _) => "<metadata-map>".to_string(),
        deep::Expr::MetaExpr(_, _) => "<metadata-expression>".to_string(),
        deep::Expr::Node(node, _) => node.tag().as_str().to_string(),
        deep::Expr::BareList(_, _) => "(...)".to_string(),
        deep::Expr::UnknownForm(data) => format!("({})", data.head),
    }
}
