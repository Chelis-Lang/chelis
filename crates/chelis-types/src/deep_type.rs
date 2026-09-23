//! Fail-closed resolution of Deep type and dimension expressions.
//!
//! This is the single boundary that may translate Deep type syntax into the
//! checker's internal [`Type`]. Resolution is context-sensitive: declarations
//! choose how names become binders, while ordinary source annotations are
//! closed. A failure returns an [`ErrorWitness`], never a successful partial
//! type. Private rejected-signature frames preserve valid constraints for body
//! checks. Known constructors inspect all sibling components before failure.

mod recovery;
pub(crate) use recovery::RejectedSignatureType;

use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::ast as deep;

use crate::adt::AdtRegistry;
use crate::env::DeclarationBinderIdentities;
use crate::errors::{CheckError, CheckErrorKind, ErrorWitness, report_witness};
use crate::session::{DeclarationDiagnosticOwner, DeclarationTypeDiagnosticClass, DiagnosticSink};
use crate::types::{
    Dim, DimVar, NominalArg, NominalParamKind, Prim, RankVar, TensorPrec, Type, TypeVar,
    TypeVarRestriction, VarGen,
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

/// Source location owned by one type-resolution node. Explicit producer span
/// metadata wins; otherwise the structural AST range is retained and exposed
/// through a stable `source:<start>..<end>` identifier as well as the byte
/// offset. A nested node with an explicitly unknown `0..0` span remains
/// unknown rather than inheriting its parent's location.
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

    pub(crate) fn stable_key(&self) -> (Option<usize>, Option<&str>) {
        (self.span_offset, self.span_id.as_deref())
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
    /// References inside a declaration reuse its instantiated type,
    /// dimension, and rank binders. Allocating a fresh variable here would
    /// discard [04-INF-6] identity (and a type binder's [04-DTYPE-2] bound).
    Lexical(&'a DeclarationBinderIdentities),
    /// A nominal declaration whose parameter kinds were fixed before any
    /// declaration body was resolved.
    ExplicitKinds(&'a UnordMap<String, NominalParamKind>),
    /// A `defsig` may use only names in its explicit binder list. The list is
    /// unkinded: position determines whether one spelling denotes a type,
    /// dimension, or rank variable.
    ExplicitGeneric(&'a UnordSet<String>),
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
    headers: UnordMap<String, Vec<NominalParamKind>>,
}

/// Checker-native nominal containers whose payloads are their type arguments.
/// Other nominal shapes come from the ADT registry, including List and Option.
pub(crate) const NATIVE_NOMINAL_HEADERS: &[(&str, usize)] =
    &[("Dict", 2), ("Result", 2), ("String", 0)];

impl TypeResolutionEnv {
    pub(crate) fn from_registry(registry: &AdtRegistry) -> Self {
        let mut headers = Self::default();
        // Checker-native nominal surfaces that are typed structurally by
        // builtin rules instead of carrying registry variants.
        for &(name, arity) in NATIVE_NOMINAL_HEADERS {
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
        for (name, kinds) in other.headers.to_sorted() {
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
    type_vars: UnordMap<String, TypeVar>,
    dim_vars: UnordMap<String, DimVar>,
    rank_vars: UnordMap<String, RankVar>,
    /// Declaration identity shared by name-resolution diagnostics across one
    /// standalone signature and its matching inline/body annotations.
    declaration_diagnostic_owner: Option<DeclarationDiagnosticOwner>,
    /// Declared dtype-family bounds, keyed by binder name
    /// (`spec/04-type-system.md` §5.9 [04-DTYPE-2]). Empty for every
    /// declaration that declares no bound.
    dtype_bounds: UnordMap<String, TypeVarRestriction>,
    /// Bounds actually attached to a resolved variable, in first-occurrence
    /// order. The caller installs them on the substitution so generalization
    /// re-quantifies them onto the declaration's scheme.
    installed_bounds: Vec<(TypeVar, TypeVarRestriction)>,
    owner_location: Option<TypeDiagnosticLocation>,
    resolution_location: Option<TypeDiagnosticLocation>,
    current_location: Option<TypeDiagnosticLocation>,
    /// True while resolving the trailing precision child of a `t-tensor`.
    /// That slot has its own reserved-name diagnostic in
    /// `validate_tensor_precisions_in_program`, so the `t-prim` arm must not
    /// emit a second one (chelis#1593).
    resolving_tensor_precision: bool,
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
            type_vars: UnordMap::new(),
            dim_vars: UnordMap::new(),
            rank_vars: UnordMap::new(),
            declaration_diagnostic_owner: None,
            dtype_bounds: UnordMap::new(),
            installed_bounds: Vec::new(),
            owner_location: None,
            resolution_location: None,
            current_location: None,
            resolving_tensor_precision: false,
        };
        if let BinderMode::Lexical(identities) = binder_mode {
            resolver.type_vars = identities.type_vars.clone();
            resolver.dim_vars = identities.dim_vars.clone();
            resolver.rank_vars = identities.rank_vars.clone();
        }
        if let BinderMode::ExplicitKinds(kinds) = binder_mode {
            for (name, kind) in kinds.to_sorted() {
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

    pub(crate) fn with_declaration_diagnostic_owner(
        mut self,
        owner: Option<&DeclarationDiagnosticOwner>,
    ) -> Self {
        self.declaration_diagnostic_owner = owner.cloned();
        self
    }

    /// Declare dtype-family bounds for this scope's binders
    /// (`spec/04-type-system.md` §5.9).
    pub(crate) fn with_dtype_bounds(
        mut self,
        bounds: UnordMap<String, TypeVarRestriction>,
    ) -> Self {
        self.dtype_bounds = bounds;
        self
    }

    /// The bounds attached to resolved variables, after checking that every
    /// declared bound actually reached a binder.
    ///
    /// [04-DTYPE-2] makes an unused bound a declaration error rather than a
    /// no-op: a bound naming a binder the type never mentions is a typo whose
    /// silent acceptance would leave the intended variable unconstrained,
    /// which is the exact failure this rule exists to prevent.
    pub(crate) fn finish_dtype_bounds(
        &mut self,
    ) -> Result<Vec<(TypeVar, TypeVarRestriction)>, ErrorWitness> {
        let unused: Vec<&String> = self
            .dtype_bounds
            .to_sorted()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| !self.type_vars.contains_key(name.as_str()))
            .collect();
        if let Some(name) = unused.first().map(|name| (*name).clone()) {
            let family = self
                .dtype_bounds
                .get(&name)
                .expect("the unused name came from this map")
                .family_name();
            return Err(self.type_error(format!(
                "binder `{name}` is bounded by dtype family `{family}` but does not occur in {}",
                self.use_site.label()
            )));
        }
        Ok(std::mem::take(&mut self.installed_bounds))
    }

    /// Retain the usable bounds even if a different declared bound is invalid.
    pub(crate) fn resolved_dtype_bounds(&self) -> Vec<(TypeVar, TypeVarRestriction)> {
        self.installed_bounds.clone()
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

    /// Parse a parameter annotation as a real constraint or an inference hole.
    /// A whole-slot hole adds no constraint. The caller supplies its declared
    /// slot, or a fresh body-local variable when no signature exists.
    pub(crate) fn resolve_parameter(
        &mut self,
        expr: &deep::Expr,
    ) -> Result<Option<ResolvedDeepType>, ErrorWitness> {
        self.begin_resolution(expr);
        let (form, tag, children) = self.type_form_expr(expr)?;
        if form == Some(DeepTag::TVar) && self.one_symbol(tag, children)? == "_" {
            return if self.allows_hole() {
                Ok(None)
            } else {
                Err(self.unbound("type", "_"))
            };
        }
        self.recover_parts(form, tag, children)
            .into_result()
            .map(|ty| Some(ResolvedDeepType(ty)))
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
        if matches!(expr, deep::Expr::Node(_, _)) {
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

    /// The source name bound to each type variable this resolver minted for
    /// an authored binder, as `TypeVar -> name` (chelis#260 Site 2 and
    /// chelis#1486, [04-INF-6]).
    ///
    /// `type_vars` above answers "which variable is `t`"; this answers the
    /// inverse, "which source name is `?N`", which is what a diagnostic
    /// reporting on an inference identity needs.
    ///
    /// The type twin of [`Self::dim_var_names`], and the exact set of
    /// AUTHORED binders: `type_vars` is populated only by
    /// [`Self::resolve_type_var`] on a named `t-var`, so an inference hole
    /// (`(t-var {} _)`, which mints a fresh variable and stores nothing) is
    /// absent by construction. That distinction is [04-INF-5] versus
    /// [04-INF-6]: a hole is not a binder and is never rigid.
    ///
    /// These are the PRE-generalization variables, so a consumer reporting on
    /// an instantiated signature composes this with the instantiation's
    /// original-to-fresh mapping, exactly as the dimension side does.
    pub(crate) fn type_var_names(&self) -> UnordMap<TypeVar, String> {
        // `to_sorted` rather than an unordered walk, for the reason
        // `dim_var_names` gives: hash order must not reach observable
        // compiler behavior (chelis#1444).
        self.type_vars
            .to_sorted()
            .into_iter()
            .map(|(name, tv)| (*tv, name.clone()))
            .collect()
    }

    pub(crate) fn type_vars(&self) -> Vec<TypeVar> {
        let mut vars = self
            .type_vars
            .to_sorted()
            .into_iter()
            .map(|(_, var)| *var)
            .collect::<Vec<_>>();
        vars.sort_by_key(|var| var.0);
        vars
    }

    pub(crate) fn dim_vars(&self) -> Vec<DimVar> {
        let mut vars = self
            .dim_vars
            .to_sorted()
            .into_iter()
            .map(|(_, var)| *var)
            .collect::<Vec<_>>();
        vars.sort_by_key(|var| var.0);
        vars
    }

    pub(crate) fn rank_vars(&self) -> Vec<RankVar> {
        let mut vars = self
            .rank_vars
            .to_sorted()
            .into_iter()
            .map(|(_, var)| *var)
            .collect::<Vec<_>>();
        vars.sort_by_key(|var| var.0);
        vars
    }

    /// All role-specific identities introduced while resolving a declaration
    /// signature. The binder list is unkinded, so one source name may have
    /// independent identities in more than one map.
    pub(crate) fn binder_identities(&self) -> DeclarationBinderIdentities {
        DeclarationBinderIdentities {
            type_vars: self.type_vars.clone(),
            dim_vars: self.dim_vars.clone(),
            rank_vars: self.rank_vars.clone(),
        }
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
        // `Some(default)` is an intentional unknown-location sentinel. It
        // prevents `diagnostic_location` from falling through to the root or
        // owner after a legacy child explicitly says its token span is unknown.
        self.current_location = Some(TypeDiagnosticLocation::from_expr(expr).unwrap_or_default());
    }

    fn resolve_type(&mut self, expr: &deep::Expr) -> Result<Type, ErrorWitness> {
        let (form, tag, children) = self.type_form_expr(expr)?;
        self.recover_parts(form, tag, children).into_result()
    }

    fn resolve_atomic_type(
        &mut self,
        form_tag: Option<DeepTag>,
        tag: &str,
        children: &[deep::Expr],
    ) -> Result<Type, ErrorWitness> {
        match form_tag {
            Some(DeepTag::TPrim) => {
                let name = self.one_symbol(tag, children)?;
                if let Some(prim) = Prim::parse_name(name) {
                    if prim.is_admissible_active() {
                        return Ok(Type::Prim(prim));
                    }
                    // chelis#1606: an internal Prim variant does not imply
                    // an admitted source type. Resolve every type position
                    // through the same [04-DTYPE-1] rejection.
                    let tensor = self.resolving_tensor_precision;
                    let diagnostic = f8e4m3_diagnostic(tensor);
                    return Err(self.report_rejected_primitive_diagnostic_once(name, diagnostic));
                }
                // chelis#1593: this is the one boundary every type position
                // crosses, so a name reserved under §1.1.1 is reported the
                // same way everywhere -- the defsig, the annotation, the
                // `deftype` field, the `typealias` body, and a hand-written
                // `.dp`. The generic message below names the wrong defect for
                // a reserved spelling: it says the name is unknown, when the
                // language reserved it and rejects it. The predicate pair is
                // the tensor precision slot's, so the two positions recognise
                // exactly the same names.
                //
                // `resolving_tensor_precision` selects the surface word, so
                // the precision slot says "tensor element" and everywhere else
                // says "scalar". It used to SUPPRESS this arm in the precision
                // slot and leave `validate_tensor_precisions_in_program` as the
                // only voice; that left `tensor[..., u8]` still receiving the
                // generic "unknown primitive type" message this arm exists to
                // replace (round 1, P3). The validator's own reserved-name
                // branches are gone, so this is the single voice for both.
                let tensor = self.resolving_tensor_precision;
                if let Some(diagnostic) = retired_integer_diagnostic(name, "deep", tensor)
                    .or_else(|| unsigned_family_diagnostic(name, tensor))
                    .or_else(|| deferred_family_diagnostic(name, tensor))
                {
                    return Err(self.report_rejected_primitive_diagnostic_once(name, diagnostic));
                }
                let nearest = nearest_active_dtype(name);
                let diagnostic = CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "unknown primitive type `{name}` in {}",
                        self.use_site.label()
                    ),
                    vec![format!(
                        "replace `{name}` with nearest active dtype `{nearest}`, or declare `{name}` in the signature binder list if it is intentionally generic"
                    )],
                );
                Err(self.report_unknown_primitive_diagnostic_once(name, diagnostic))
            }
            Some(DeepTag::TVar) => {
                let name = self.one_symbol(tag, children)?;
                self.resolve_type_var(name).map(Type::Var)
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
        // chelis#1593 round 1. `spec/04-type-system.md` §5.8.1 says a spelling
        // [04-DTYPE-1] rejects names no type variable in ANY type position, and
        // Deep is a representation of the same public language, so the rule has
        // to hold on this carrier too. Repairing only the Surf desugarer left a
        // hand-written `(t-var {} u8)` scoring 1.0 with an empty error vector,
        // and `chelis surf` printed it back as `def f[u8](x: u8) -> u8 = x`,
        // which the same build rejects: one program, two verdicts, depending on
        // which spelling it arrived in. `_` is checked first and stays a hole.
        let tensor = self.resolving_tensor_precision;
        if let Some(diagnostic) = retired_integer_diagnostic(name, "deep", tensor)
            .or_else(|| unsigned_family_diagnostic(name, tensor))
            .or_else(|| deferred_family_diagnostic(name, tensor))
        {
            return Err(self.report_rejected_primitive_diagnostic_once(name, diagnostic));
        }
        if Prim::parse_name(name).is_some_and(|prim| prim.is_admissible_active()) {
            return Err(self.type_error_with_suggestions(
                format!(
                    "active primitive `{name}` cannot be rebound as a type variable in {}",
                    self.use_site.label()
                ),
                vec![format!(
                    "write `(t-prim {{}} {name})` and remove `{name}` from the `defsig` binder list"
                )],
            ));
        }
        if matches!(
            self.binder_mode,
            BinderMode::ExplicitKinds(kinds)
                if kinds.get(name) == Some(&NominalParamKind::Dimension)
        ) {
            return Err(self.type_error(format!(
                "nominal parameter `{name}` has Dimension kind and cannot be used in a type slot in {}",
                self.use_site.label()
            )));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("type", name));
        }
        if matches!(self.binder_mode, BinderMode::Lexical(..)) && !self.type_vars.contains_key(name)
        {
            return Err(self.unbound("type", name));
        }
        let variable = *self
            .type_vars
            .entry(name.to_string())
            .or_insert_with(|| self.vg.fresh_tvar());
        // [04-DTYPE-2]: the bound belongs to the binder, so it is recorded on
        // first occurrence and every later occurrence of the same name reuses
        // the same bounded variable.
        if let Some(restriction) = self.dtype_bounds.get(name).copied()
            && !self
                .installed_bounds
                .iter()
                .any(|(bound, _)| *bound == variable)
        {
            self.installed_bounds.push((variable, restriction));
        }
        Ok(variable)
    }

    fn resolve_dim_var(&mut self, name: &str) -> Result<DimVar, ErrorWitness> {
        if name == "_" {
            return self
                .allows_hole()
                .then(|| self.vg.fresh_dvar())
                .ok_or_else(|| self.unbound("dimension", name));
        }
        if matches!(
            self.binder_mode,
            BinderMode::ExplicitKinds(kinds)
                if kinds.get(name) == Some(&NominalParamKind::Type)
        ) {
            return Err(self.type_error(format!(
                "nominal parameter `{name}` has Type kind and cannot be used in a dimension slot in {}",
                self.use_site.label()
            )));
        }
        if self.dtype_bounds.contains_key(name) {
            return Err(self.bounded_binder_misuse(name, "a dimension slot"));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("dimension", name));
        }
        if matches!(self.binder_mode, BinderMode::Lexical(..)) && !self.dim_vars.contains_key(name)
        {
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
        if matches!(
            self.binder_mode,
            BinderMode::ExplicitKinds(kinds) if kinds.contains_key(name)
        ) {
            return Err(self.type_error(format!(
                "nominal parameter `{name}` cannot be used as a rank spread in {}; nominal parameters have only Type or Dimension kind",
                self.use_site.label()
            )));
        }
        if self.dtype_bounds.contains_key(name) {
            return Err(self.bounded_binder_misuse(name, "a rank spread"));
        }
        if !self.allows_name(name) {
            return Err(self.unbound("rank", name));
        }
        if matches!(self.binder_mode, BinderMode::Lexical(..)) && !self.rank_vars.contains_key(name)
        {
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
            BinderMode::Lexical(identities) => identities.contains_name(name),
            BinderMode::ExplicitKinds(kinds) => kinds.contains_key(name),
            BinderMode::ExplicitGeneric(names) => names.contains(name),
            BinderMode::TrustedCompilerMetadata => true,
        }
    }

    fn allows_hole(&self) -> bool {
        !matches!(self.binder_mode, BinderMode::ExplicitKinds(_))
    }

    /// Decode-once (chelis#731 Phase 3): the decoded tag drives dispatch;
    /// the string is the diagnostic spelling. `None` with a head string is
    /// the raw-string boundary (an undecodable head), which the dispatch
    /// rejects with the same unknown-tag diagnostic as any non-type
    /// vocabulary tag.
    fn type_form_expr<'b>(
        &mut self,
        expr: &'b deep::Expr,
    ) -> Result<(Option<DeepTag>, &'b str, &'b [deep::Expr]), ErrorWitness> {
        match expr {
            deep::Expr::Node(node, _) => {
                Ok((Some(node.tag()), node.tag().as_str(), node.children_slice()))
            }
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

    /// A dtype family names a set of element types, so a bounded binder can
    /// never stand for an extent or a run of extents ([04-DTYPE-2]).
    fn bounded_binder_misuse(&mut self, name: &str, position: &str) -> ErrorWitness {
        let family = self
            .dtype_bounds
            .get(name)
            .map_or("a dtype family", |restriction| restriction.family_name());
        self.type_error(format!(
            "binder `{name}` is bounded by dtype family `{family}` and cannot be used as {position} in {}",
            self.use_site.label()
        ))
    }

    fn unbound(&mut self, kind: &str, name: &str) -> ErrorWitness {
        let class = match kind {
            "type" => DeclarationTypeDiagnosticClass::UndeclaredTypeVariable,
            "dimension" => DeclarationTypeDiagnosticClass::UndeclaredDimensionVariable,
            "rank" => DeclarationTypeDiagnosticClass::UndeclaredRankVariable,
            _ => {
                return self.type_error(format!(
                    "undeclared {kind} variable `{name}` in {}",
                    self.use_site.label()
                ));
            }
        };
        self.report_declaration_type_diagnostic_once(
            name,
            class,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "undeclared {kind} variable `{name}` in {}",
                    self.use_site.label()
                ),
                vec![],
            ),
            false,
        )
    }

    fn malformed(&mut self, message: String) -> ErrorWitness {
        let error = CheckError::new(CheckErrorKind::MalformedForm, message, vec![]);
        let error = self
            .diagnostic_location()
            .map_or(error.clone(), |location| location.attach(error));
        report_witness(self.errors, error)
    }

    fn type_error(&mut self, message: String) -> ErrorWitness {
        self.type_error_with_suggestions(message, vec![])
    }

    fn type_error_with_suggestions(
        &mut self,
        message: String,
        suggestions: Vec<String>,
    ) -> ErrorWitness {
        let error = CheckError::new(CheckErrorKind::TypeMismatch, message, suggestions);
        let error = self
            .diagnostic_location()
            .map_or(error.clone(), |location| location.attach(error));
        report_witness(self.errors, error)
    }

    fn report_primitive_diagnostic(&mut self, diagnostic: CheckError) -> ErrorWitness {
        let diagnostic = self
            .diagnostic_location()
            .map_or(diagnostic.clone(), |location| location.attach(diagnostic));
        report_witness(self.errors, diagnostic)
    }

    fn report_rejected_primitive_diagnostic_once(
        &mut self,
        name: &str,
        diagnostic: CheckError,
    ) -> ErrorWitness {
        self.report_declaration_type_diagnostic_once(
            name,
            DeclarationTypeDiagnosticClass::RejectedPrimitive,
            diagnostic,
            false,
        )
    }

    fn report_unknown_primitive_diagnostic_once(
        &mut self,
        name: &str,
        diagnostic: CheckError,
    ) -> ErrorWitness {
        self.report_declaration_type_diagnostic_once(
            name,
            DeclarationTypeDiagnosticClass::UnknownPrimitive,
            diagnostic,
            true,
        )
    }

    fn report_declaration_type_diagnostic_once(
        &mut self,
        name: &str,
        class: DeclarationTypeDiagnosticClass,
        diagnostic: CheckError,
        record_unknown_site: bool,
    ) -> ErrorWitness {
        let location = self.diagnostic_location();
        let declaration_owner = self.declaration_diagnostic_owner.clone();
        if let Some(owner) = &declaration_owner
            && let Some(witness) = self.errors.declaration_type_witness(owner, name, class)
        {
            if record_unknown_site {
                self.record_unknown_primitive_site(location.as_ref(), name, witness);
            }
            return witness;
        }
        let witness = self.report_primitive_diagnostic(diagnostic);
        if record_unknown_site {
            self.record_unknown_primitive_site(location.as_ref(), name, witness);
        }
        if let Some(owner) = declaration_owner {
            self.errors
                .record_declaration_type_witness(owner, name.to_string(), class, witness);
        }
        witness
    }

    fn record_unknown_primitive_site(
        &mut self,
        location: Option<&TypeDiagnosticLocation>,
        name: &str,
        witness: ErrorWitness,
    ) {
        let Some(location) = location else {
            return;
        };
        let (span_offset, span_id) = location.stable_key();
        self.errors.record_unknown_primitive_site_witness(
            span_offset,
            span_id.map(str::to_string),
            name.to_string(),
            witness,
        );
    }
}

const ACTIVE_DTYPE_NAMES: &[&str] = &[
    "f32", "f64", "bf16", "f16", "i8", "i16", "i32", "i64", "bool",
];

fn nearest_active_dtype(name: &str) -> &'static str {
    match name {
        "float" | "float32" | "fp32" => "f32",
        "double" | "float64" | "fp64" => "f64",
        "half" | "float16" | "fp16" => "f16",
        _ => ACTIVE_DTYPE_NAMES
            .iter()
            .copied()
            .min_by_key(|candidate| edit_distance(name.as_bytes(), candidate.as_bytes()))
            .expect("active dtype vocabulary is nonempty"),
    }
}

fn edit_distance(left: &[u8], right: &[u8]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_byte) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_byte) in right.iter().enumerate() {
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + usize::from(left_byte != right_byte));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

// ---------------------------------------------------------------------------
// Reserved-but-rejected dtype spellings (`spec/04-type-system.md` §1.1.1)
//
// These answer the same question `Prim::parse_name` does -- which spellings
// name an active primitive -- for the names that deliberately name none, so
// they belong beside the resolver that asks it. Both the inference passes and
// `resolve_type` above need them, and `infer` already depends on this module,
// so keeping them under `infer` would make the two modules circular
// (chelis#1593).
// ---------------------------------------------------------------------------

/// True if `name` is one of the unsigned integer dtype names reserved
/// as deferred by `spec/04-type-system.md` §1.1.1 (§1.1.2 names the
/// `uint*` spellings canonical; the short `u*` spellings are not
/// reserved). Covers both the short form (`u8`/`u16`/`u32`/`u64`) and
/// the canonical `uint*` family that LLMs and cross-language users
/// tend to write.
pub(crate) fn is_unsigned_dtype_name(name: &str) -> bool {
    matches!(
        name,
        "u8" | "u16" | "u32" | "u64" | "uint8" | "uint16" | "uint32" | "uint64"
    )
}

/// Build a §1.1.1 diagnostic for an unsigned dtype name appearing as a
/// cast target or a tensor element type. Returns `None` for non-unsigned
/// names so call sites can short-circuit with `&&`.
pub(crate) fn unsigned_family_diagnostic(name: &str, tensor: bool) -> Option<CheckError> {
    if !is_unsigned_dtype_name(name) {
        return None;
    }
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, i8, i16, i32, i64";
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `{name}` as a {surface} dtype: unsigned integer types \
             are deferred per spec/04-type-system.md §1.1.1 (canonical \
             spelling uint8/uint16/uint32/uint64 per §1.1.2; active set: \
             {active_set})"
        ),
        vec![format!(
            "spec/04-type-system.md §1.1.2 names uint8/uint16/uint32/uint64 \
             canonical and rejects all of them, the short u8/u16/u32/u64 \
             forms included, under [04-DTYPE-1]; §1.1.1 records their \
             declared arithmetic width. [04-DTYPE-1] requires a primitive \
             type position to name an active primitive, so pick one of \
             {active_set}"
        )],
    ))
}

/// Reserved non-unsigned numeric names from spec/04 §1.1.1.
/// The internal `Prim::F8e4m3` variant is also reserved as a type-variable name.
pub(crate) fn is_deferred_dtype_name(name: &str) -> bool {
    matches!(
        name,
        "f8e4m3"
            | "f8e5m2"
            | "int4"
            | "uint4"
            | "complex64"
            | "complex128"
            | "decimal128"
            | "decimal256"
    )
}

pub(crate) fn is_retired_integer_dtype_name(name: &str) -> bool {
    matches!(name, "int8" | "int16" | "int32" | "int64")
}

/// Active primitive names and every reserved, retired, or deferred dtype
/// spelling are not declaration binders. An unknown intentional name such as
/// `float32` remains available for explicit generic use.
pub(crate) fn is_forbidden_dtype_binder_name(name: &str) -> bool {
    name == "unit"
        || Prim::parse_name(name).is_some()
        || is_unsigned_dtype_name(name)
        || is_deferred_dtype_name(name)
        || is_retired_integer_dtype_name(name)
}

pub(crate) fn retired_integer_diagnostic(
    name: &str,
    _carrier: &str,
    tensor: bool,
) -> Option<CheckError> {
    if !is_retired_integer_dtype_name(name) {
        return None;
    }
    let canonical = match name {
        "int8" => "i8",
        "int16" => "i16",
        "int32" => "i32",
        "int64" => "i64",
        _ => unreachable!("retired spelling checked above"),
    };
    let surface = if tensor { "tensor element" } else { "scalar" };
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use retired spelling `{name}` as a {surface} dtype; the canonical \
             spelling is `{canonical}`; run `chelis migrate surf --from 0.18 <path>` \
             for `.ch` or `chelis migrate deep --from 0.18 <path>` for `.dp`"
        ),
        vec![format!(
            "the versioned migration rewrites `{name}` to `{canonical}` without \
             admitting the retired spelling at normal compiler ingress"
        )],
    ))
}

/// Build a §1.1.1 diagnostic for a reserved-but-deferred dtype name
/// appearing as a cast target or a tensor element type. Returns `None`
/// for other names so call sites can short-circuit.
pub(crate) fn deferred_family_diagnostic(name: &str, tensor: bool) -> Option<CheckError> {
    if name == "f8e4m3" {
        return Some(f8e4m3_diagnostic(tensor));
    }
    if !is_deferred_dtype_name(name) {
        return None;
    }
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, i8, i16, i32, i64";
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `{name}` as a {surface} dtype: {name} is reserved \
             but deferred per spec/04-type-system.md §1.1.1 (active set: \
             {active_set})"
        ),
        vec![format!(
            "spec/04-type-system.md §1.1.1 records the deferral rationale \
             and {name}'s declared arithmetic width; pick one of \
             {active_set} until it activates"
        )],
    ))
}

/// Preserve the existing FP8 diagnostic phrase at the shared type boundary.
fn f8e4m3_diagnostic(tensor: bool) -> CheckError {
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, i8, i16, i32, i64";
    CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `f8e4m3` as a {surface} dtype: f8e4m3 is deferred per \
             spec/04-type-system.md §1.1.1 and is not part of the active \
             numeric primitive set ({active_set})"
        ),
        vec![format!(
            "f8e4m3 has no active backend in this cycle; pick one of \
             {active_set}, or see spec/04-type-system.md §1.1.1 for the \
             deferral rationale"
        )],
    )
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
        deep::Expr::Map(_, _) => "<metadata-map>".to_string(),
        deep::Expr::MetaExpr(_, _) => "<metadata-expression>".to_string(),
        deep::Expr::Node(node, _) => node.tag().as_str().to_string(),
        deep::Expr::BareList(_, _) => "(...)".to_string(),
        deep::Expr::UnknownForm(data) => format!("({})", data.head),
    }
}
