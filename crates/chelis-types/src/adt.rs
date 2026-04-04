//! ADT (Algebraic Data Type) registry for the Chelis type checker.
//!
//! Processes `deftype` Deep nodes to extract constructor type signatures.

use std::collections::HashMap;

use chelis_deep::ast as deep;

use crate::types::*;

/// Information about a single variant of an ADT.
pub struct VariantInfo {
    pub name: String,
    /// (field_name, field_type) -- name is None for positional args.
    pub fields: Vec<(Option<String>, Type)>,
}

/// An ADT definition extracted from a `deftype` node.
pub struct AdtDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<VariantInfo>,
}

/// Registry of all ADT definitions.
pub struct AdtRegistry {
    pub defs: HashMap<String, AdtDef>,
}

impl Default for AdtRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AdtRegistry {
    pub fn new() -> Self {
        AdtRegistry {
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
}

/// Helper: get tag string from a Deep List.
fn get_tag(list: &deep::List) -> Option<&str> {
    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = list.elements.first() {
        Some(tag.as_str())
    } else {
        None
    }
}

/// Helper: get children (elements after tag and metadata) from a Deep List.
fn list_children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
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
                    let prec = match deep_type_to_type_with_params(prec_expr, param_map) {
                        Type::Prim(p) => p,
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
