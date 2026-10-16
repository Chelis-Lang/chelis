//! ADT (Algebraic Data Type) registry for the Chelis type checker.
//!
//! Processes `deftype` Deep nodes to extract constructor type signatures.

use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};
use std::collections::BTreeMap;

use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};

use crate::deep_type::{BinderMode, DeepTypeResolver, TypeResolutionEnv, TypeUseSite};
use crate::errors::ErrorWitness;
use crate::session::DiagnosticSink;
use crate::types::*;

/// Information about a single variant of an ADT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantInfo {
    pub name: String,
    /// (field_name, field_type) -- name is None for positional args.
    pub fields: Vec<(Option<String>, Type)>,
}

/// An ADT definition extracted from a `deftype` node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdtDef {
    pub name: String,
    pub type_params: Vec<String>,
    /// Checker-owned kind for each nominal parameter, in source order.
    #[serde(default)]
    pub param_kinds: Vec<NominalParamKind>,
    /// The fresh `TypeVar`s allocated for the type-kinded subset of
    /// `type_params`, in source order with dimension-kinded parameters
    /// omitted. Variant field types reference these vars, so storing them
    /// lets call sites instantiate a
    /// SPECIFIC ADT's constructor without going through the
    /// name-keyed env (where same-named constructors from colliding
    /// ADTs overwrite each other, chelis#148).
    #[serde(default)]
    pub param_vars: Vec<TypeVar>,
    /// Registration-time type/dimension variables aligned with
    /// `type_params` and `param_kinds`.
    #[serde(default)]
    pub param_args: Vec<NominalArg>,
    pub variants: Vec<VariantInfo>,
    /// True when the `deftype` carried `opaque: true` metadata
    /// (RFC D-CHECK): construction and inspection are checker-gated
    /// to the defining module.
    pub opaque: bool,
    /// Module identity recorded at `deftype` registration: the
    /// lexical `(module ...)` key, or the reef internal-name stem for
    /// package-linked declarations. `None` for top-level declarations
    /// outside any module (illegal for opaque types, D-CHECK).
    pub defining_module: Option<String>,
}

/// A type alias definition extracted from a `typealias` node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAliasDef {
    pub params: Vec<String>,
    #[serde(default)]
    pub param_kinds: Vec<NominalParamKind>,
    pub param_vars: Vec<TypeVar>,
    #[serde(default)]
    pub param_args: Vec<NominalArg>,
    pub body: Type,
}

/// Shape of a constructor call site, used by
/// [`AdtRegistry::lookup_variant_preferring_shape`] to disambiguate
/// same-named variants across colliding ADTs (chelis#148).
///
/// `Positional` covers `Ctor(arg1, arg2)` form (lowered to `app`);
/// `Record` covers `Ctor { field1: ..., field2: ... }` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallShape {
    Positional,
    Record,
}

/// Registry of all ADT definitions and type aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdtRegistry {
    pub defs: BTreeMap<String, AdtDef>,
    pub aliases: BTreeMap<String, TypeAliasDef>,
    /// Runtime declaration-header scope for the current check. It is kept
    /// separate from validated definitions because self/forward names must be
    /// visible while their bodies are resolving. Rejected declarations never
    /// enter `defs`/`aliases`, and this provisional scope is deliberately not
    /// serialized into a reusable checker context.
    #[serde(skip, default)]
    pub(crate) resolution_env: TypeResolutionEnv,
}

/// Spec/04 section 8.4.1: whether `ty` carries a key, given the data types
/// already known to carry one. A type variable carries none here: it is a
/// field of a generic data type, and what it is instantiated at decides.
fn type_carries_key(ty: &Type, carriers: &UnordSet<String>) -> bool {
    match ty {
        Type::Prim(prim) => *prim == Prim::Key,
        Type::Tensor(_, precision) => precision.as_concrete() == Some(Prim::Key),
        Type::Ref(inner) => type_carries_key(inner, carriers),
        Type::Tuple(elements) => elements
            .iter()
            .any(|element| type_carries_key(element, carriers)),
        Type::Adt(name, arguments) => {
            carriers.contains(name)
                || arguments
                    .iter()
                    .any(|argument| type_carries_key(argument, carriers))
        }
        Type::KindedAdt(name, arguments) => {
            carriers.contains(name)
                || arguments.iter().any(|argument| match argument {
                    NominalArg::Type(argument) => type_carries_key(argument, carriers),
                    NominalArg::Dimension(_) => false,
                })
        }
        Type::Fn(_, _) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

impl Default for AdtRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AdtRegistry {
    pub fn new() -> Self {
        #[allow(clippy::default_constructed_unit_structs)]
        AdtRegistry {
            aliases: BTreeMap::new(),
            defs: BTreeMap::new(),
            resolution_env: TypeResolutionEnv::default(),
        }
    }

    pub(crate) fn resolution_env(&self) -> &TypeResolutionEnv {
        &self.resolution_env
    }

    /// Spec/04 section 8.4.1: the registered data types one of whose variant
    /// fields carries a key, as the least fixed point over every registered
    /// definition, so a data type that holds another key-carrying data type
    /// is found whatever order they were declared in.
    pub(crate) fn key_carrying_adts(&self) -> UnordSet<String> {
        let mut carriers = UnordSet::new();
        loop {
            let mut grew = false;
            for (name, def) in &self.defs {
                if carriers.contains(name) {
                    continue;
                }
                let carries = def.variants.iter().any(|variant| {
                    variant
                        .fields
                        .iter()
                        .any(|(_, field)| type_carries_key(field, &carriers))
                });
                if carries {
                    carriers.insert(name.clone());
                    grew = true;
                }
            }
            if !grew {
                return carriers;
            }
        }
    }

    pub(crate) fn install_resolution_env(&mut self, resolution_env: TypeResolutionEnv) {
        self.resolution_env = resolution_env;
    }

    /// Register an ADT from a deftype Deep node.
    /// `children` should be the children after tag+metadata: name, type_params_list, variant...
    /// `opaque` is the `opaque: true` metadata flag and
    /// `defining_module` the module identity computed by the caller
    /// (RFC D-CHECK); both are recorded on the [`AdtDef`].
    /// Returns constructor schemes to add to the type environment.
    pub(crate) fn register_deftype(
        &mut self,
        children: &[deep::Expr],
        vg: &mut VarGen,
        headers: &TypeResolutionEnv,
        errors: &mut DiagnosticSink<'_>,
        opaque: bool,
        defining_module: Option<String>,
    ) -> Result<Vec<(String, Scheme)>, ErrorWitness> {
        // children[0] = name (symbol)
        // children[1] = type params list like (a) or (a b) -- a bare list of symbols wrapped in parens
        // children[2..] = variant nodes
        if children.is_empty() {
            return Ok(vec![]);
        }

        let name = match &children[0] {
            deep::Expr::Atom(deep::Atom::Name(s), _) => s.clone(),
            _ => return Ok(vec![]),
        };

        // Parse type parameters -- could be (a b) as a list, or just individual symbols
        let mut type_params: Vec<String> = Vec::new();
        let variant_start;
        if children.len() > 1 {
            match &children[1] {
                deep::Expr::Node(node, _)
                    if matches!(node.tag(), DeepTag::Variant | DeepTag::Field) =>
                {
                    variant_start = 1;
                }
                deep::Expr::BareList(elements, _) => {
                    for element in elements {
                        if let deep::Expr::Atom(deep::Atom::Name(name), _) = element {
                            type_params.push(name.clone());
                        }
                    }
                    variant_start = 2;
                }
                _ => {
                    variant_start = 1;
                }
            }
        } else {
            variant_start = 1;
        }

        let param_kinds = headers
            .param_kinds(&name)
            .map(<[NominalParamKind]>::to_vec)
            .unwrap_or_else(|| vec![NominalParamKind::Type; type_params.len()]);
        let explicit_params: UnordMap<String, NominalParamKind> = type_params
            .iter()
            .cloned()
            .zip(param_kinds.iter().copied())
            .collect();
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::DeftypeField,
            BinderMode::ExplicitKinds(&explicit_params),
            headers,
            vg,
            errors,
        );

        // Parse variants
        let mut variants = Vec::new();
        let variant_children = if variant_start < children.len() {
            &children[variant_start..]
        } else {
            &[]
        };

        for variant_expr in variant_children {
            let vchildren = match variant_expr.carrier() {
                deep::ExprCarrier::DecodedNode(DeepTag::Variant, _, children) => children,
                deep::ExprCarrier::DecodedNode(_, _, _)
                | deep::ExprCarrier::StructuralList(_)
                | deep::ExprCarrier::UndecodableHead(_, _, _)
                | deep::ExprCarrier::Atom(_)
                | deep::ExprCarrier::MetadataMap(_)
                | deep::ExprCarrier::MetadataExpression(_) => continue,
            };
            if vchildren.is_empty() {
                continue;
            }

            let vname = match &vchildren[0] {
                deep::Expr::Atom(deep::Atom::Name(s), _) => s.clone(),
                _ => continue,
            };

            // Remaining children are either field definitions or positional type args.
            // Field types are expanded through the alias registry so a field declared
            // with a transparent alias (`type EffectRow = List[Effect]`) is stored and
            // unified as its expansion. Any alias the field references must already be
            // registered; `collect_declarations` registers all `typealias` decls before
            // any `deftype` so forward references (alias declared after the deftype that
            // uses it) resolve too.
            let mut fields: Vec<(Option<String>, Type)> = Vec::new();
            for field_expr in &vchildren[1..] {
                match field_expr.carrier() {
                    deep::ExprCarrier::DecodedNode(DeepTag::Field, _, fchildren) => {
                        if fchildren.len() >= 2 {
                            let fname = match &fchildren[0] {
                                deep::Expr::Atom(deep::Atom::Name(s), _) => s.clone(),
                                _ => continue,
                            };
                            let ftype =
                                self.expand_aliases(&resolver.resolve(&fchildren[1])?.into_type());
                            fields.push((Some(fname), ftype));
                        }
                    }
                    deep::ExprCarrier::DecodedNode(_, _, _)
                    | deep::ExprCarrier::StructuralList(_)
                    | deep::ExprCarrier::UndecodableHead(_, _, _)
                    | deep::ExprCarrier::Atom(_)
                    | deep::ExprCarrier::MetadataMap(_)
                    | deep::ExprCarrier::MetadataExpression(_) => {
                        // Positional type argument
                        let ftype = self.expand_aliases(&resolver.resolve(field_expr)?.into_type());
                        fields.push((None, ftype));
                    }
                }
            }

            variants.push(VariantInfo {
                name: vname,
                fields,
            });
        }

        let all_tvars = resolver.type_vars();
        let all_dvars = resolver.dim_vars();
        let all_rvars = resolver.rank_vars();
        let param_args: Vec<NominalArg> = type_params
            .iter()
            .zip(&param_kinds)
            .map(|(param, kind)| match kind {
                NominalParamKind::Type => NominalArg::Type(Type::Var(
                    resolver
                        .type_var(param)
                        .expect("type-kinded nominal parameter is pre-bound"),
                )),
                NominalParamKind::Dimension => NominalArg::Dimension(Dim::Var(
                    resolver
                        .dim_var(param)
                        .expect("dimension-kinded nominal parameter is pre-bound"),
                )),
            })
            .collect();
        let param_vars = param_args
            .iter()
            .filter_map(|argument| match argument {
                NominalArg::Type(Type::Var(var)) => Some(*var),
                _ => None,
            })
            .collect();
        let adt_type = if param_kinds.contains(&NominalParamKind::Dimension) {
            Type::KindedAdt(name.clone(), param_args.clone())
        } else {
            Type::Adt(
                name.clone(),
                param_args
                    .iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => ty.clone(),
                        NominalArg::Dimension(_) => unreachable!("type-only nominal header"),
                    })
                    .collect(),
            )
        };
        let constructor_schemes = variants
            .iter()
            .map(|variant| {
                let ctor_type = if variant.fields.is_empty() {
                    adt_type.clone()
                } else {
                    Type::Fn(
                        variant
                            .fields
                            .iter()
                            .map(|(_, field_type)| field_type.clone())
                            .collect(),
                        Box::new(adt_type.clone()),
                    )
                };
                (
                    variant.name.clone(),
                    Scheme {
                        result_origin: ResultOrigin::aggregate(
                            &ctor_type, &all_tvars, &all_dvars, &all_rvars, vg,
                        ),
                        constraints: vec![],
                        tvars: all_tvars.clone(),
                        tvar_restrictions: vec![],
                        dvars: all_dvars.clone(),
                        rvars: all_rvars.clone(),
                        body: ctor_type,
                    },
                )
            })
            .collect();
        self.defs.insert(
            name.clone(),
            AdtDef {
                name,
                type_params,
                param_kinds,
                param_vars,
                param_args,
                variants,
                opaque,
                defining_module,
            },
        );

        Ok(constructor_schemes)
    }

    /// Look up an ADT definition by name.
    pub fn lookup(&self, name: &str) -> Option<&AdtDef> {
        self.defs.get(name)
    }

    /// If `name` is already bound (by a `deftype`, `typealias`, or
    /// prelude ADT registered earlier in this program), return the
    /// human-readable kind of the existing definition so the caller
    /// can emit a `DuplicateDefinition` diagnostic. Returns `None` if
    /// the name is free.
    ///
    /// Type names share one flat string-keyed namespace here, so a
    /// `deftype Foo` cannot coexist with either a second `deftype Foo`
    /// or a `typealias Foo = ...` — map insertion is last-write-
    /// wins and silently corrupts the registry otherwise.
    pub fn existing_kind(&self, name: &str) -> Option<&'static str> {
        if self.defs.contains_key(name) {
            Some("deftype")
        } else if self.aliases.contains_key(name) {
            Some("typealias")
        } else {
            None
        }
    }

    /// Get all variant names for an ADT (for exhaustiveness checking).
    pub fn variant_names(&self, adt_name: &str) -> Option<Vec<String>> {
        self.defs
            .get(adt_name)
            .map(|d| d.variants.iter().map(|v| v.name.clone()).collect())
    }

    /// Look up a variant by constructor name across all ADTs.
    /// Returns the ADT name and variant info.
    pub fn lookup_variant(&self, ctor_name: &str) -> Option<(&str, &VariantInfo)> {
        for (adt_name, def) in &self.defs {
            for variant in &def.variants {
                if variant.name == ctor_name {
                    return Some((adt_name.as_str(), variant));
                }
            }
        }
        None
    }

    /// Look up an imported or qualified constructor by its unique terminal segment.
    pub fn lookup_variant_terminal_unique(&self, ctor_name: &str) -> Option<(&str, &VariantInfo)> {
        let mut matches = self.defs.iter().flat_map(|(adt_name, def)| {
            def.variants.iter().filter_map(move |variant| {
                terminal_name_matches(&variant.name, ctor_name)
                    .then_some((adt_name.as_str(), variant))
            })
        });
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    /// Look up a variant by constructor name, preferring the variant whose
    /// field-naming style matches `call_shape`. Resolves chelis#148-class
    /// collisions where two ADTs in different deps export constructors with
    /// the same unqualified name but different shapes (e.g.
    /// School.Data.Dataset.IntCol is a record-style constructor;
    /// Coral.Frame.Column.IntCol is a positional/tuple constructor). When
    /// the caller's call syntax is positional, return the positional
    /// variant; when it's record-style, return the record variant.
    /// Falls back to the first match if no shape-preferred variant exists.
    ///
    /// Candidates are sorted by ADT name before the shape filter, so
    /// dispatch is deterministic across runs even when multiple variants
    /// of the same shape collide. Without the sort, `self.defs.iter()`
    /// (formerly a hash map) leaks iteration-order non-determinism into the choice
    /// of "first match" in both the shape-match and the fallback path.
    pub fn lookup_variant_preferring_shape(
        &self,
        ctor_name: &str,
        call_shape: CallShape,
    ) -> Option<(&str, &VariantInfo)> {
        let mut candidates: Vec<(&str, &VariantInfo)> = self
            .defs
            .iter()
            .flat_map(|(adt_name, def)| {
                def.variants.iter().filter_map(move |variant| {
                    (variant.name == ctor_name).then_some((adt_name.as_str(), variant))
                })
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by_key(|(a, _)| *a);
        let want_named = matches!(call_shape, CallShape::Record);
        let shape_match = candidates.iter().find(|(_, v)| {
            !v.fields.is_empty()
                && v.fields
                    .iter()
                    .all(|(name, _)| name.is_some() == want_named)
        });
        shape_match.copied().or_else(|| candidates.first().copied())
    }

    /// Register a type alias: `typealias Name[params] = Type`.
    pub fn register_alias(
        &mut self,
        name: String,
        params: Vec<String>,
        param_kinds: Vec<NominalParamKind>,
        param_args: Vec<NominalArg>,
        body: Type,
    ) {
        let param_vars = param_args
            .iter()
            .filter_map(|argument| match argument {
                NominalArg::Type(Type::Var(var)) => Some(*var),
                _ => None,
            })
            .collect();
        self.aliases.insert(
            name,
            TypeAliasDef {
                params,
                param_kinds,
                param_vars,
                param_args,
                body,
            },
        );
    }

    /// Resolve a type alias definition by name. Returns None if not an alias.
    pub fn resolve_alias(&self, name: &str) -> Option<&TypeAliasDef> {
        self.aliases.get(name)
    }

    /// Exact alias identity, or the existing unique linked terminal identity.
    /// An ordinary nominal declaration always wins over a terminal alias.
    pub fn resolve_alias_named(&self, name: &str) -> Option<(&str, &TypeAliasDef)> {
        if let Some((name, alias)) = self.aliases.get_key_value(name) {
            return Some((name, alias));
        }
        if self.lookup(name).is_some() {
            return None;
        }
        let mut aliases = self
            .aliases
            .iter()
            .filter(|(key, _)| terminal_name_matches(key, name));
        let (name, alias) = aliases.next()?;
        aliases.next().is_none().then_some((name.as_str(), alias))
    }

    /// Expand transparent aliases using declaration-owned parameter identities.
    /// Actual checked argument syntax/metadata is retained; no fresh validation
    /// variable becomes an output binder. The input remains the origin owner.
    pub fn normalize_alias_syntax(&self, expr: &deep::Expr) -> Result<deep::Expr, String> {
        fn expand(
            registry: &AdtRegistry,
            headers: &TypeResolutionEnv,
            expr: &deep::Expr,
            visiting: &mut UnordMap<String, usize>,
            constructor_depth: usize,
        ) -> Result<deep::Expr, String> {
            match expr {
                deep::Expr::MetaExpr(meta, span) => {
                    let mut meta = meta.clone();
                    meta.expr = Box::new(expand(
                        registry,
                        headers,
                        &meta.expr,
                        visiting,
                        constructor_depth,
                    )?);
                    Ok(deep::Expr::MetaExpr(meta, *span))
                }
                deep::Expr::Node(node, span) => {
                    if node.tag() == DeepTag::TAdt {
                        let [name, arguments @ ..] = node.children_slice() else {
                            return Err("nominal type has no name".into());
                        };
                        let deep::Expr::Atom(deep::Atom::Name(name), _) = name else {
                            return Err("nominal type name is not a symbol".into());
                        };
                        if let Some((canonical, alias)) = registry.resolve_alias_named(name) {
                            let arguments = arguments
                                .iter()
                                .map(|argument| {
                                    expand(registry, headers, argument, visiting, constructor_depth)
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut children = vec![deep::Expr::Atom(
                                deep::Atom::Name(canonical.to_owned()),
                                *span,
                            )];
                            children.extend(arguments.iter().cloned());
                            let application = deep::Expr::node(
                                DeepTag::TAdt,
                                node.meta().clone(),
                                children,
                                *span,
                            );
                            let kinds = crate::session::checked_alias_argument_kinds(
                                headers,
                                &application,
                            )?;
                            if kinds.len() != arguments.len() {
                                return Err(
                                    "checked alias resolver lost argument kind alignment".into()
                                );
                            }
                            let actuals = arguments
                                .into_iter()
                                .zip(kinds)
                                .map(|(argument, kind)| match kind {
                                    NominalParamKind::Type => NominalActual::Type(argument),
                                    NominalParamKind::Dimension => {
                                        // The canonical resolver disambiguates a Surf
                                        // t-var argument by the declaration's kind.
                                        let argument = match &argument {
                                            deep::Expr::Node(node, span)
                                                if node.tag() == DeepTag::TVar =>
                                            {
                                                deep::Expr::node(
                                                    DeepTag::DVar,
                                                    node.meta().clone(),
                                                    node.children_slice().to_vec(),
                                                    *span,
                                                )
                                            }
                                            _ => argument,
                                        };
                                        NominalActual::Dimension(argument)
                                    }
                                })
                                .collect::<Vec<_>>();
                            let (types, dimensions) = nominal_argument_bindings(&alias.param_args, &actuals)
                                .ok_or_else(|| format!("alias `{canonical}` lost declaration parameter kind/arity alignment"))?;
                            if let Some(depth) = visiting.get(canonical) {
                                // Transparent aliases form a finite graph. A
                                // value constructor makes recursion productive;
                                // retain its canonical applied edge instead of
                                // unfolding it (possibly with growing arguments).
                                if constructor_depth > *depth {
                                    return Ok(application);
                                }
                                return Err(format!(
                                    "unproductive alias cycle through `{canonical}`"
                                ));
                            }
                            visiting.insert(canonical.to_owned(), constructor_depth);
                            let materialized = crate::infer::type_to_deep_expr_with_arguments(
                                &alias.body,
                                &types,
                                &dimensions,
                            )?;
                            let result = expand(
                                registry,
                                headers,
                                &materialized,
                                visiting,
                                constructor_depth,
                            );
                            visiting.remove(canonical);
                            return result;
                        }
                        if headers.param_kinds(name).is_none() {
                            return Err(format!("unknown or ambiguous nominal type `{name}`"));
                        }
                    }
                    let child_depth = constructor_depth
                        + usize::from(matches!(
                            node.tag(),
                            DeepTag::TTuple | DeepTag::TFn | DeepTag::TAdt
                        ));
                    let children = node
                        .children_slice()
                        .iter()
                        .map(|child| expand(registry, headers, child, visiting, child_depth))
                        .collect::<Result<_, _>>()?;
                    Ok(deep::Expr::node(
                        node.tag(),
                        node.meta().clone(),
                        children,
                        *span,
                    ))
                }
                _ => Ok(expr.clone()),
            }
        }
        let headers = TypeResolutionEnv::for_normalization(self)?;
        expand(self, &headers, expr, &mut UnordMap::new(), 0)
    }

    /// Instantiate a type alias with the given type arguments.
    pub fn instantiate_alias(&self, name: &str, args: &[Type]) -> Option<Type> {
        let args = args
            .iter()
            .cloned()
            .map(NominalArg::Type)
            .collect::<Vec<_>>();
        self.instantiate_nominal_alias(name, &args)
    }

    /// Instantiate a type alias whose parameters may be ordinary types or
    /// dimensions. The declaration-owned parameter arguments and the applied
    /// arguments must agree positionally in kind.
    pub fn instantiate_nominal_alias(&self, name: &str, args: &[NominalArg]) -> Option<Type> {
        let alias = self.aliases.get(name)?;
        let (type_subst, dim_subst) = nominal_substitutions(&alias.param_args, args)?;
        Some(substitute_nominal_type(
            &alias.body,
            &type_subst,
            &dim_subst,
        ))
    }

    /// Recursively expand every registered type alias inside `ty`, leaving
    /// non-alias `Adt`, tuple, and function structure intact. This is the
    /// transparency rule from `spec/02-surf-syntax.md` ("Aliases are
    /// transparent — expanded during desugaring") applied to a stored type.
    ///
    /// Used at `deftype` registration so the constructor schemes bound into
    /// the type environment carry the expanded field type (e.g. a field
    /// declared `EffectRow` where `type EffectRow = List[Effect]` is stored
    /// as `List[Effect]`), not the opaque alias `Adt` node. Without this,
    /// constructor application unifies the supplied `List[Effect]` argument
    /// against the unexpanded `EffectRow` alias and reports a spurious
    /// `EffectRow vs List` mismatch.
    ///
    /// Mirrors the logic of `resolve_type_aliases` in `infer.rs` but lives
    /// here as a method so it can read `self.aliases` immutably from within
    /// `register_deftype`'s `&mut self` borrow. The `seen` set guards against
    /// infinite recursion on a (mutually) recursive alias chain, matching the
    /// `infer.rs` guard.
    pub fn expand_aliases(&self, ty: &Type) -> Type {
        let mut seen = UnordSet::new();
        self.expand_aliases_inner(ty, &mut seen)
    }

    fn expand_aliases_inner(&self, ty: &Type, seen: &mut UnordSet<String>) -> Type {
        match ty {
            Type::Adt(name, args) => {
                let resolved_args: Vec<Type> = args
                    .iter()
                    .map(|arg| self.expand_aliases_inner(arg, seen))
                    .collect();

                if seen.contains(name) {
                    return Type::Adt(name.clone(), resolved_args);
                }

                if let Some(expanded) = self.instantiate_alias(name, &resolved_args) {
                    seen.insert(name.clone());
                    let resolved = self.expand_aliases_inner(&expanded, seen);
                    seen.remove(name);
                    resolved
                } else {
                    Type::Adt(name.clone(), resolved_args)
                }
            }
            Type::KindedAdt(name, args) => {
                let resolved_args = args
                    .iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => {
                            NominalArg::Type(self.expand_aliases_inner(ty, seen))
                        }
                        NominalArg::Dimension(dim) => NominalArg::Dimension(dim.clone()),
                    })
                    .collect::<Vec<_>>();

                if seen.contains(name) {
                    return Type::KindedAdt(name.clone(), resolved_args);
                }

                if let Some(expanded) = self.instantiate_nominal_alias(name, &resolved_args) {
                    seen.insert(name.clone());
                    let resolved = self.expand_aliases_inner(&expanded, seen);
                    seen.remove(name);
                    resolved
                } else {
                    Type::KindedAdt(name.clone(), resolved_args)
                }
            }
            Type::Fn(args, ret) => Type::Fn(
                args.iter()
                    .map(|a| self.expand_aliases_inner(a, seen))
                    .collect(),
                Box::new(self.expand_aliases_inner(ret, seen)),
            ),
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| self.expand_aliases_inner(t, seen))
                    .collect(),
            ),
            _ => ty.clone(),
        }
    }
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

pub(crate) fn substitute_alias_type(ty: &Type, subst: &UnordMap<TypeVar, Type>) -> Type {
    substitute_nominal_type(ty, subst, &UnordMap::new())
}

enum NominalActual<T, D> {
    Type(T),
    Dimension(D),
}

fn nominal_argument_bindings<T: Clone, D: Clone>(
    parameters: &[NominalArg],
    arguments: &[NominalActual<T, D>],
) -> Option<(UnordMap<TypeVar, T>, UnordMap<DimVar, D>)> {
    if parameters.len() != arguments.len() {
        return None;
    }
    let mut types = UnordMap::new();
    let mut dimensions = UnordMap::new();
    for (parameter, argument) in parameters.iter().zip(arguments) {
        match (parameter, argument) {
            (NominalArg::Type(Type::Var(parameter)), NominalActual::Type(argument)) => {
                types.insert(*parameter, argument.clone());
            }
            (NominalArg::Dimension(Dim::Var(parameter)), NominalActual::Dimension(argument)) => {
                dimensions.insert(*parameter, argument.clone());
            }
            _ => return None,
        }
    }
    Some((types, dimensions))
}

pub(crate) fn nominal_substitutions(
    parameters: &[NominalArg],
    arguments: &[NominalArg],
) -> Option<(UnordMap<TypeVar, Type>, UnordMap<DimVar, Dim>)> {
    let arguments = arguments
        .iter()
        .map(|argument| match argument {
            NominalArg::Type(ty) => NominalActual::Type(ty.clone()),
            NominalArg::Dimension(dim) => NominalActual::Dimension(dim.clone()),
        })
        .collect::<Vec<_>>();
    nominal_argument_bindings(parameters, &arguments)
}

pub(crate) fn substitute_nominal_type(
    ty: &Type,
    type_subst: &UnordMap<TypeVar, Type>,
    dim_subst: &UnordMap<DimVar, Dim>,
) -> Type {
    match ty {
        Type::Var(tv) => type_subst.get(tv).cloned().unwrap_or(Type::Var(*tv)),
        Type::Ref(inner) => Type::Ref(Box::new(substitute_nominal_type(
            inner, type_subst, dim_subst,
        ))),
        Type::Tensor(dims, precision) => {
            let precision = match precision {
                TensorPrec::Var(var) => match type_subst.get(var) {
                    Some(Type::Prim(prim)) => TensorPrec::Concrete(*prim),
                    Some(Type::Var(var)) => TensorPrec::Var(*var),
                    _ => TensorPrec::Var(*var),
                },
                TensorPrec::Concrete(prim) => TensorPrec::Concrete(*prim),
            };
            Type::Tensor(
                dims.iter()
                    .map(|dim| match dim {
                        Dim::Var(var) => dim_subst.get(var).cloned().unwrap_or_else(|| dim.clone()),
                        _ => dim.clone(),
                    })
                    .collect(),
                precision,
            )
        }
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|arg| substitute_nominal_type(arg, type_subst, dim_subst))
                .collect(),
            Box::new(substitute_nominal_type(ret, type_subst, dim_subst)),
        ),
        Type::Adt(name, args) => Type::Adt(
            name.clone(),
            args.iter()
                .map(|arg| substitute_nominal_type(arg, type_subst, dim_subst))
                .collect(),
        ),
        Type::KindedAdt(name, args) => Type::KindedAdt(
            name.clone(),
            args.iter()
                .map(|argument| match argument {
                    NominalArg::Type(ty) => {
                        NominalArg::Type(substitute_nominal_type(ty, type_subst, dim_subst))
                    }
                    NominalArg::Dimension(Dim::Var(var)) => {
                        NominalArg::Dimension(dim_subst.get(var).cloned().unwrap_or(Dim::Var(*var)))
                    }
                    NominalArg::Dimension(dim) => NominalArg::Dimension(dim.clone()),
                })
                .collect(),
        ),
        Type::Tuple(items) => Type::Tuple(
            items
                .iter()
                .map(|item| substitute_nominal_type(item, type_subst, dim_subst))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

#[cfg(test)]
mod alias_syntax_materialization_tests {
    use super::*;

    fn parse(source: &str) -> deep::Expr {
        chelis_deep::parser::parse_str(source).unwrap().remove(0)
    }
    fn registry() -> AdtRegistry {
        let mut registry = AdtRegistry::default();
        registry.register_alias(
            "Contract".into(),
            vec!["n".into()],
            vec![NominalParamKind::Dimension],
            vec![NominalArg::Dimension(Dim::Var(DimVar(77)))],
            Type::Fn(
                vec![Type::Ref(Box::new(Type::Tensor(
                    vec![Dim::Wildcard],
                    TensorPrec::Concrete(Prim::F64),
                )))],
                Box::new(Type::Tensor(
                    vec![Dim::Var(DimVar(77))],
                    TensorPrec::Concrete(Prim::Bool),
                )),
            ),
        );
        registry.register_alias(
            "Identity".into(),
            vec!["T".into()],
            vec![NominalParamKind::Type],
            vec![NominalArg::Type(Type::Var(TypeVar(77)))],
            Type::Var(TypeVar(77)),
        );
        registry
    }

    #[test]
    fn alias_syntax_preserves_borrow_full_result_and_actual_argument_metadata() {
        let registry = registry();
        let actual = parse("(t-adt {} Contract (d-lit {span: \"source_extent\"} 2))");
        let expanded = registry.normalize_alias_syntax(&actual).unwrap();
        let deep::Expr::Node(function, _) = &expanded else {
            panic!("function");
        };
        assert_eq!(function.tag(), DeepTag::TFn);
        assert_eq!(function.children_slice()[0].tag(), Some(DeepTag::TRef));
        let deep::Expr::Node(result, _) = &function.children_slice()[1] else {
            panic!("result");
        };
        let deep::Expr::Node(application, _) = &actual else {
            panic!("application");
        };
        assert_eq!(
            &result.children_slice()[0],
            &application.children_slice()[1]
        );
        let borrowed = parse(
            "(t-adt {} Identity (t-ref {span: \"borrowed_actual\"} (t-tensor {} (d-lit {} 2) (t-prim {} f32))))",
        );
        let deep::Expr::Node(application, _) = &borrowed else {
            panic!("application");
        };
        assert_eq!(
            registry.normalize_alias_syntax(&borrowed).unwrap(),
            application.children_slice()[1]
        );
        let nested = parse(
            "(t-adt {} Identity (t-ref {span: \"nested_borrow\"} (t-adt {} Contract (d-lit {} 2))))",
        );
        let expanded = registry.normalize_alias_syntax(&nested).unwrap();
        let deep::Expr::Node(reference, _) = &expanded else {
            panic!("nested reference");
        };
        assert_eq!(reference.tag(), DeepTag::TRef);
        assert_eq!(reference.children_slice()[0].tag(), Some(DeepTag::TFn));
        let collision = parse("(t-adt {} Identity (t-var {span: \"caller_t77\"} t77))");
        let deep::Expr::Node(application, _) = &collision else {
            panic!("application");
        };
        assert_eq!(
            registry.normalize_alias_syntax(&collision).unwrap(),
            application.children_slice()[1]
        );
    }

    #[test]
    fn alias_syntax_rejects_kind_arity_unknown_and_error_recovery() {
        let mut registry = registry();
        for source in [
            "(t-adt {} Contract)",
            "(t-adt {} Contract (d-lit {} 2) (d-lit {} 3))",
            "(t-adt {} Contract (t-prim {} f64))",
            "(t-adt {} Identity (d-lit {} 2))",
            "(t-adt {} Unknown)",
        ] {
            assert!(
                registry.normalize_alias_syntax(&parse(source)).is_err(),
                "{source}"
            );
        }
        registry.register_alias(
            "Broken".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            // Simulate rejected metadata at a serialized ingress; the required
            // materializer must refuse it, not recover a successful hole.
            serde_json::from_str::<Type>(r#"{"Error":null}"#).unwrap(),
        );
        assert!(
            registry
                .normalize_alias_syntax(&parse("(t-adt {} Broken)"))
                .is_err()
        );
    }

    #[test]
    fn alias_syntax_preserves_scoped_actual_names_and_rejects_cycles_or_ambiguous_aliases() {
        let mut registry = registry();
        let actual = parse("(t-adt {} Contract (d-name {span: \"enclosing_n\"} n))");
        let deep::Expr::Node(application, _) = &actual else {
            panic!("application");
        };
        let expanded = registry.normalize_alias_syntax(&actual).unwrap();
        let deep::Expr::Node(function, _) = &expanded else {
            panic!("function");
        };
        let deep::Expr::Node(result, _) = &function.children_slice()[1] else {
            panic!("result");
        };
        assert_eq!(
            &result.children_slice()[0],
            &application.children_slice()[1]
        );
        registry.register_alias(
            "A".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Ref(Box::new(Type::Adt("B".into(), Vec::new()))),
        );
        registry.register_alias(
            "B".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Adt("A".into(), Vec::new()),
        );
        assert!(
            registry
                .normalize_alias_syntax(&parse("(t-adt {} A)"))
                .is_err()
        );
        let contract = registry.resolve_alias("Contract").unwrap().clone();
        registry.aliases.remove("Contract");
        registry
            .aliases
            .insert("Left__Contract".into(), contract.clone());
        registry.aliases.insert("Right__Contract".into(), contract);
        assert!(
            registry
                .normalize_alias_syntax(&parse("(t-adt {} Pkg__Contract (d-lit {} 2))"))
                .is_err()
        );
    }

    // Spec/02 alias transparency and [04-INF-6] apply to every complete
    // registry, including public registration and serialized checked contexts.
    fn registry_normalization_context() -> AdtRegistry {
        let mut registry = AdtRegistry::default();
        registry.register_alias(
            "Function".into(),
            vec!["T".into(), "n".into()],
            vec![NominalParamKind::Type, NominalParamKind::Dimension],
            vec![
                NominalArg::Type(Type::Var(TypeVar(77))),
                NominalArg::Dimension(Dim::Var(DimVar(99))),
            ],
            Type::Fn(
                vec![Type::Ref(Box::new(Type::Var(TypeVar(77))))],
                Box::new(Type::Tensor(
                    vec![Dim::Var(DimVar(99))],
                    TensorPrec::Concrete(Prim::Bool),
                )),
            ),
        );
        registry
    }

    fn assert_registry_context_full_signature(
        registry: &AdtRegistry,
        actual: &deep::Expr,
    ) -> deep::Expr {
        let normalized = registry.normalize_alias_syntax(actual).unwrap();
        let deep::Expr::Node(application, _) = actual else {
            panic!("nominal application");
        };
        let deep::Expr::Node(function, _) = &normalized else {
            panic!("normalized function");
        };
        assert_eq!(function.tag(), DeepTag::TFn);
        let deep::Expr::Node(borrow, _) = &function.children_slice()[0] else {
            panic!("borrowed parameter");
        };
        assert_eq!(borrow.tag(), DeepTag::TRef);
        assert_eq!(
            &borrow.children_slice()[0],
            &application.children_slice()[1],
            "actual checked type syntax and metadata remain intact"
        );
        let deep::Expr::Node(result, _) = &function.children_slice()[1] else {
            panic!("independent result");
        };
        assert_eq!(result.tag(), DeepTag::TTensor);
        assert_eq!(
            &result.children_slice()[0],
            &application.children_slice()[2],
            "actual checked dimension syntax and metadata remain intact"
        );
        assert_eq!(
            result.children_slice()[1],
            crate::infer::type_to_deep_expr(&Type::Prim(Prim::Bool)),
            "result precision is independent of the input type argument"
        );
        normalized
    }

    #[test]
    fn registry_normalization_context_public_registration_retains_full_signature() {
        let registry = registry_normalization_context();
        assert!(registry.resolution_env().param_kinds("Function").is_none());
        let actual = parse(
            "(t-adt {span: \"caller_adapter\"} Function (t-tensor {span: \"caller_tensor\"} (d-name {} *) (t-prim {span: \"caller_precision\"} f32)) (d-lit {span: \"caller_extent\"} 2))",
        );
        assert_registry_context_full_signature(&registry, &actual);
    }

    #[test]
    fn registry_normalization_context_serde_roundtrip_retains_arguments_and_borrows() {
        let registry = registry_normalization_context();
        let restored: AdtRegistry =
            serde_json::from_slice(&serde_json::to_vec(&registry).unwrap()).unwrap();
        assert!(restored.resolution_env().param_kinds("Function").is_none());
        let actual = parse(
            "(t-adt {} Function (t-ref {span: \"caller_borrow\"} (t-tensor {} (d-lit {} 4) (t-prim {} f64))) (d-name {span: \"caller_extent\"} n))",
        );
        let original = assert_registry_context_full_signature(&registry, &actual);
        let roundtrip = assert_registry_context_full_signature(&restored, &actual);
        assert_eq!(original, roundtrip);
        let collision = parse(
            "(t-adt {} Function (t-var {span: \"caller_type_identity\"} t77) (d-var {span: \"caller_dimension_identity\"} d99))",
        );
        assert_registry_context_full_signature(&restored, &collision);
    }

    #[test]
    fn registry_normalization_context_rejects_wrong_arguments_and_unknown_names() {
        let registry = registry_normalization_context();
        for source in [
            "(t-adt {} Function)",
            "(t-adt {} Function (t-prim {} f32))",
            "(t-adt {} Function (t-prim {} f32) (d-lit {} 2) (d-lit {} 3))",
            "(t-adt {} Function (d-lit {} 2) (d-lit {} 3))",
            "(t-adt {} Function (t-prim {} f32) (t-prim {} f64))",
            "(t-adt {} Missing)",
        ] {
            assert!(
                registry.normalize_alias_syntax(&parse(source)).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn registry_normalization_context_rejects_malformed_stored_alignment() {
        let baseline = registry_normalization_context();
        let actual = parse("(t-adt {} Function (t-prim {} f32) (d-lit {} 2))");
        for mutation in 0..5 {
            let mut registry = baseline.clone();
            let alias = registry.aliases.get_mut("Function").unwrap();
            match mutation {
                0 => alias.param_kinds.clear(),
                1 => assert!(alias.param_args.pop().is_some()),
                2 => alias.param_args[0] = NominalArg::Dimension(Dim::Var(DimVar(77))),
                3 => alias.param_args[1] = NominalArg::Type(Type::Var(TypeVar(99))),
                4 => alias.params.push("extra".into()),
                _ => unreachable!(),
            }
            assert!(
                registry.normalize_alias_syntax(&actual).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn registry_normalization_context_preserves_nominal_precedence_and_rejects_ambiguity() {
        let mut registry = registry_normalization_context();
        let alias = registry.aliases.remove("Function").unwrap();
        registry
            .aliases
            .insert("Left__Function".into(), alias.clone());
        registry.aliases.insert("Right__Function".into(), alias);
        registry.defs.insert(
            "Function".into(),
            AdtDef {
                name: "Function".into(),
                type_params: Vec::new(),
                param_kinds: Vec::new(),
                param_vars: Vec::new(),
                param_args: Vec::new(),
                variants: Vec::new(),
                opaque: false,
                defining_module: None,
            },
        );
        let nominal = parse("(t-adt {span: \"ordinary_nominal\"} Function)");
        assert_eq!(registry.normalize_alias_syntax(&nominal).unwrap(), nominal);
        assert!(
            registry
                .normalize_alias_syntax(&parse(
                    "(t-adt {} Pkg__Function (t-prim {} f32) (d-lit {} 2))"
                ))
                .is_err()
        );
    }

    #[test]
    fn alias_normalization_preserves_productive_recursion_as_a_finite_graph() {
        let mut registry = registry();
        registry.register_alias(
            "R".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Tuple(vec![
                Type::Prim(Prim::Int64),
                Type::Adt("R".into(), Vec::new()),
            ]),
        );
        registry.register_alias(
            "Grow".into(),
            vec!["a".into()],
            vec![NominalParamKind::Type],
            vec![NominalArg::Type(Type::Var(TypeVar(17)))],
            Type::Tuple(vec![
                Type::Var(TypeVar(17)),
                Type::Adt(
                    "Grow".into(),
                    vec![Type::Tuple(vec![
                        Type::Var(TypeVar(17)),
                        Type::Var(TypeVar(17)),
                    ])],
                ),
            ]),
        );
        for registry in [
            registry.clone(),
            serde_json::from_slice::<AdtRegistry>(&serde_json::to_vec(&registry).unwrap()).unwrap(),
        ] {
            assert_eq!(
                chelis_deep::printer::print_expr_flat(
                    &registry
                        .normalize_alias_syntax(&parse("(t-adt {} R)"))
                        .unwrap()
                ),
                "(t-tuple {} (t-prim {} i64) (t-adt {} R))",
            );
            assert_eq!(
                chelis_deep::printer::print_expr_flat(
                    &registry
                        .normalize_alias_syntax(&parse("(t-adt {} Grow (t-prim {} i64))"))
                        .unwrap()
                ),
                "(t-tuple {} (t-prim {} i64) (t-adt {} Grow (t-tuple {} (t-prim {} i64) (t-prim {} i64))))",
            );
        }
    }

    #[test]
    fn registry_normalization_context_rejects_cycles_recovery_and_provisional_only_names() {
        let mut registry = registry_normalization_context();
        registry.register_alias(
            "A".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Ref(Box::new(Type::Adt("B".into(), Vec::new()))),
        );
        registry.register_alias(
            "B".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Type::Adt("A".into(), Vec::new()),
        );
        registry.register_alias(
            "Broken".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            serde_json::from_str::<Type>(r#"{"Error":null}"#).unwrap(),
        );
        registry
            .resolution_env
            .insert("OnlyProvisional", Vec::new());
        for name in ["A", "Broken", "OnlyProvisional"] {
            assert!(
                registry
                    .normalize_alias_syntax(&parse(&format!("(t-adt {{}} {name})")))
                    .is_err(),
                "{name}"
            );
        }
    }
}
