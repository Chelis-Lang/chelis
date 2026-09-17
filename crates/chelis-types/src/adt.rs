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
                deep::Expr::List(list, _) => {
                    // Could be (a b) or (variant ...) -- check if first elem is a variant tag
                    if let Some(tag) = list.tag() {
                        if tag == DeepTag::Variant || tag == DeepTag::Field {
                            // No type params, this is already a variant
                            variant_start = 1;
                        } else {
                            // Type params list: elements are symbols
                            for el in &list.elements {
                                if let deep::Expr::Atom(deep::Atom::Name(s), _) = el {
                                    type_params.push(s.clone());
                                }
                            }
                            variant_start = 2;
                        }
                    } else {
                        // Elements might be bare symbols for type params
                        for el in &list.elements {
                            if let deep::Expr::Atom(deep::Atom::Name(s), _) = el {
                                type_params.push(s.clone());
                            }
                        }
                        variant_start = 2;
                    }
                }
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
            if let Some((DeepTag::Variant, vchildren)) = stamped_parts(variant_expr) {
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
                    match stamped_parts(field_expr) {
                        Some((DeepTag::Field, fchildren)) => {
                            if fchildren.len() >= 2 {
                                let fname = match &fchildren[0] {
                                    deep::Expr::Atom(deep::Atom::Name(s), _) => s.clone(),
                                    _ => continue,
                                };
                                let ftype = self
                                    .expand_aliases(&resolver.resolve(&fchildren[1])?.into_type());
                                fields.push((Some(fname), ftype));
                            }
                        }
                        _ => {
                            // Positional type argument
                            let ftype =
                                self.expand_aliases(&resolver.resolve(field_expr)?.into_type());
                            fields.push((None, ftype));
                        }
                    }
                }

                variants.push(VariantInfo {
                    name: vname,
                    fields,
                });
            }
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

// Transitional E5b adapter; `Expr::carrier` owns physical-carrier decoding.
fn stamped_parts(expr: &deep::Expr) -> Option<(DeepTag, &[deep::Expr])> {
    match expr.carrier() {
        deep::ExprCarrier::DecodedNode(tag, _, children) => Some((tag, children)),
        deep::ExprCarrier::StructuralList(_)
        | deep::ExprCarrier::UndecodableHead(_, _, _)
        | deep::ExprCarrier::Atom(_)
        | deep::ExprCarrier::MetadataMap(_)
        | deep::ExprCarrier::MetadataExpression(_)
        | deep::ExprCarrier::MalformedLegacyList(_) => None,
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

pub(crate) fn nominal_substitutions(
    parameters: &[NominalArg],
    arguments: &[NominalArg],
) -> Option<(UnordMap<TypeVar, Type>, UnordMap<DimVar, Dim>)> {
    if parameters.len() != arguments.len() {
        return None;
    }
    let mut type_subst = UnordMap::new();
    let mut dim_subst = UnordMap::new();
    for (parameter, argument) in parameters.iter().zip(arguments) {
        match (parameter, argument) {
            (NominalArg::Type(Type::Var(parameter)), NominalArg::Type(argument)) => {
                type_subst.insert(*parameter, argument.clone());
            }
            (NominalArg::Dimension(Dim::Var(parameter)), NominalArg::Dimension(argument)) => {
                dim_subst.insert(*parameter, argument.clone());
            }
            _ => return None,
        }
    }
    Some((type_subst, dim_subst))
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
