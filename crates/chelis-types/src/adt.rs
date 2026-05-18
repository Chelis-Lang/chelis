//! ADT (Algebraic Data Type) registry for the Chelis type checker.
//!
//! Processes `deftype` Deep nodes to extract constructor type signatures.

use std::collections::HashMap;

use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};

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
    pub variants: Vec<VariantInfo>,
}

/// A type alias definition extracted from a `typealias` node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAliasDef {
    pub params: Vec<String>,
    pub param_vars: Vec<TypeVar>,
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
    pub defs: HashMap<String, AdtDef>,
    pub aliases: HashMap<String, TypeAliasDef>,
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
            aliases: HashMap::new(),
            defs: HashMap::new(),
        }
    }

    /// Register an ADT from a deftype Deep node.
    /// `children` should be the children after tag+metadata: name, type_params_list, variant...
    /// Returns constructor schemes to add to the type environment.
    pub fn register_deftype(
        &mut self,
        children: &[deep::Expr],
        vg: &mut VarGen,
    ) -> Vec<(String, Scheme)> {
        // children[0] = name (symbol)
        // children[1] = type params list like (a) or (a b) -- a bare list of symbols wrapped in parens
        // children[2..] = variant nodes
        if children.is_empty() {
            return vec![];
        }

        let name = match &children[0] {
            deep::Expr::Atom(deep::Atom::Symbol(s), _) => s.clone(),
            _ => return vec![],
        };

        // Parse type parameters -- could be (a b) as a list, or just individual symbols
        let mut type_params: Vec<String> = Vec::new();
        let variant_start;
        if children.len() > 1 {
            match &children[1] {
                deep::Expr::List(list, _) => {
                    // Could be (a b) or (variant ...) -- check if first elem is a variant tag
                    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) =
                        list.elements.first()
                    {
                        if tag == "variant" || tag == "field" {
                            // No type params, this is already a variant
                            variant_start = 1;
                        } else {
                            // Type params list: elements are symbols
                            for el in &list.elements {
                                if let deep::Expr::Atom(deep::Atom::Symbol(s), _) = el {
                                    type_params.push(s.clone());
                                }
                            }
                            variant_start = 2;
                        }
                    } else {
                        // Elements might be bare symbols for type params
                        for el in &list.elements {
                            if let deep::Expr::Atom(deep::Atom::Symbol(s), _) = el {
                                type_params.push(s.clone());
                            }
                        }
                        variant_start = 2;
                    }
                }
                _ => {
                    variant_start = 1;
                }
            }
        } else {
            variant_start = 1;
        }

        // Build a mapping from type param names to fresh TypeVars
        let mut param_map: HashMap<String, TypeVar> = HashMap::new();
        for p in &type_params {
            param_map.insert(p.clone(), vg.fresh_tvar());
        }

        // Parse variants
        let mut variants = Vec::new();
        let mut constructor_schemes = Vec::new();

        let variant_children = if variant_start < children.len() {
            &children[variant_start..]
        } else {
            &[]
        };

        for variant_expr in variant_children {
            if let deep::Expr::List(list, _) = variant_expr
                && get_tag(list) == Some("variant")
            {
                let vchildren = list_children(list);
                if vchildren.is_empty() {
                    continue;
                }

                let vname = match &vchildren[0] {
                    deep::Expr::Atom(deep::Atom::Symbol(s), _) => s.clone(),
                    _ => continue,
                };

                // Remaining children are either field definitions or positional type args
                let mut fields: Vec<(Option<String>, Type)> = Vec::new();
                for field_expr in &vchildren[1..] {
                    match field_expr {
                        deep::Expr::List(flist, _) if get_tag(flist) == Some("field") => {
                            let fchildren = list_children(flist);
                            if fchildren.len() >= 2 {
                                let fname = match &fchildren[0] {
                                    deep::Expr::Atom(deep::Atom::Symbol(s), _) => s.clone(),
                                    _ => continue,
                                };
                                let ftype =
                                    deep_type_to_type_with_params(&fchildren[1], &param_map);
                                fields.push((Some(fname), ftype));
                            }
                        }
                        _ => {
                            // Positional type argument
                            let ftype = deep_type_to_type_with_params(field_expr, &param_map);
                            fields.push((None, ftype));
                        }
                    }
                }

                // Build constructor type
                let adt_type = Type::Adt(
                    name.clone(),
                    type_params
                        .iter()
                        .map(|p| Type::Var(*param_map.get(p).unwrap()))
                        .collect(),
                );

                let ctor_type = if fields.is_empty() {
                    // Nullary constructor: just the ADT type
                    adt_type.clone()
                } else {
                    // Constructor function: field types -> ADT type
                    let arg_types: Vec<Type> = fields.iter().map(|(_, t)| t.clone()).collect();
                    Type::Fn(arg_types, Box::new(adt_type.clone()))
                };

                let all_tvars: Vec<TypeVar> = param_map.values().copied().collect();
                let scheme = Scheme {
                    tvars: all_tvars,
                    dvars: vec![],
                    body: ctor_type,
                };

                constructor_schemes.push((vname.clone(), scheme));

                variants.push(VariantInfo {
                    name: vname,
                    fields,
                });
            }
        }

        self.defs.insert(
            name.clone(),
            AdtDef {
                name,
                type_params,
                variants,
            },
        );

        constructor_schemes
    }

    /// Look up an ADT definition by name.
    pub fn lookup(&self, name: &str) -> Option<&AdtDef> {
        self.defs.get(name)
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
    /// (HashMap) leaks iteration-order non-determinism into the choice
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
        param_vars: Vec<TypeVar>,
        body: Type,
    ) {
        self.aliases.insert(
            name,
            TypeAliasDef {
                params,
                param_vars,
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
        let alias = self.aliases.get(name)?;
        if alias.param_vars.len() != args.len() {
            return None;
        }

        let subst: HashMap<TypeVar, Type> = alias
            .param_vars
            .iter()
            .copied()
            .zip(args.iter().cloned())
            .collect();

        Some(substitute_alias_type(&alias.body, &subst))
    }
}

/// Helper: get tag string from a Deep List.
fn get_tag(list: &deep::List) -> Option<&str> {
    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = list.elements.first() {
        Some(tag.as_str())
    } else {
        None
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

/// Helper: get children (elements after tag and metadata) from a Deep List.
fn list_children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn substitute_alias_type(ty: &Type, subst: &HashMap<TypeVar, Type>) -> Type {
    match ty {
        Type::Var(tv) => subst.get(tv).cloned().unwrap_or(Type::Var(*tv)),
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|arg| substitute_alias_type(arg, subst))
                .collect(),
            Box::new(substitute_alias_type(ret, subst)),
        ),
        Type::Adt(name, args) => Type::Adt(
            name.clone(),
            args.iter()
                .map(|arg| substitute_alias_type(arg, subst))
                .collect(),
        ),
        Type::Tuple(items) => Type::Tuple(
            items
                .iter()
                .map(|item| substitute_alias_type(item, subst))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

/// Convert a Deep type expression to internal Type, resolving type param names.
fn deep_type_to_type_with_params(expr: &deep::Expr, param_map: &HashMap<String, TypeVar>) -> Type {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let children = list_children(list);
            match tag {
                "t-prim" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) = children.first() {
                        Prim::parse_name(name)
                            .map(Type::Prim)
                            .unwrap_or(Type::Error)
                    } else {
                        Type::Error
                    }
                }
                "t-var" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) = children.first() {
                        if let Some(&tv) = param_map.get(name.as_str()) {
                            Type::Var(tv)
                        } else {
                            Type::Error
                        }
                    } else {
                        Type::Error
                    }
                }
                "t-fn" => {
                    if children.is_empty() {
                        return Type::Error;
                    }
                    let args: Vec<Type> = children[..children.len() - 1]
                        .iter()
                        .map(|c| deep_type_to_type_with_params(c, param_map))
                        .collect();
                    let ret =
                        deep_type_to_type_with_params(&children[children.len() - 1], param_map);
                    Type::Fn(args, Box::new(ret))
                }
                "t-tensor" => {
                    if children.is_empty() {
                        return Type::Error;
                    }
                    let prec_expr = &children[children.len() - 1];
                    // Per WS-A5 (spec/04-type-system.md §5.8) the precision
                    // slot may be either a concrete primitive or a type
                    // variable (within a sig). Translate both shapes; any
                    // other shape is an ill-formed tensor.
                    let prec = match deep_type_to_type_with_params(prec_expr, param_map) {
                        Type::Prim(p) => TensorPrec::Concrete(p),
                        Type::Var(v) => TensorPrec::Var(v),
                        _ => return Type::Error,
                    };
                    let dims: Vec<Dim> = children[..children.len() - 1]
                        .iter()
                        .filter_map(deep_dim)
                        .collect();
                    Type::Tensor(dims, prec)
                }
                "t-adt" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) = children.first() {
                        let args: Vec<Type> = children[1..]
                            .iter()
                            .map(|c| deep_type_to_type_with_params(c, param_map))
                            .collect();
                        Type::Adt(name.clone(), args)
                    } else {
                        Type::Error
                    }
                }
                "t-tuple" => {
                    let elems: Vec<Type> = children
                        .iter()
                        .map(|c| deep_type_to_type_with_params(c, param_map))
                        .collect();
                    Type::Tuple(elems)
                }
                "t-unit" => Type::Unit,
                _ => Type::Error,
            }
        }
        _ => Type::Error,
    }
}

/// Parse a dimension expression from Deep AST.
fn deep_dim(expr: &deep::Expr) -> Option<Dim> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let children = list_children(list);
            match tag {
                "d-name" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) = children.first() {
                        if name == "*" {
                            Some(Dim::Wildcard)
                        } else {
                            Some(Dim::Name(name.clone()))
                        }
                    } else {
                        None
                    }
                }
                "d-var" => {
                    // Dimension variables in ADT context aren't common, treat as wildcard
                    Some(Dim::Wildcard)
                }
                "d-lit" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = children.first() {
                        Some(Dim::Lit(*n))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}
