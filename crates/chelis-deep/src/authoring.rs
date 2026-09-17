//! Structural query and edit helpers for Deep authoring tools.
//!
//! This module is deliberately text-surface agnostic: callers parse Deep text
//! before entering and print canonical Deep after leaving. The helpers operate
//! on the AST so cascade tools can prove what they touched instead of relying
//! on string search.

use std::collections::{BTreeMap, BTreeSet};

use crate::tag::DeepTag;
use crate::{
    Atom, DeepPath, Expr, List, Metadata, PathSegment, ResolveError, Span, function_body, printer,
    resolve_function,
};

const MODULE_DECLS_START: usize = 3;
const DEF_NAME_INDEX: usize = 2;
const DEF_VALUE_INDEX: usize = 3;
const FN_PARAMS_INDEX: usize = 2;

/// Recursively bridge stamped vocabulary nodes to the legacy `List` carrier
/// used by the mutating authoring implementation while preserving structural
/// `BareList` role. Read-only authoring APIs must consume `Node` directly and
/// therefore never call this adapter.
fn normalize_for_mutation(exprs: &[Expr]) -> Vec<Expr> {
    exprs.iter().map(normalize_expr_for_mutation).collect()
}

fn normalize_expr_for_mutation(expr: &Expr) -> Expr {
    match expr {
        Expr::Node(node, span) => {
            let list = node.to_list(*span);
            let elements = list
                .elements
                .iter()
                .map(normalize_expr_for_mutation)
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::BareList(elements, span) => Expr::BareList(
            elements.iter().map(normalize_expr_for_mutation).collect(),
            *span,
        ),
        Expr::List(list, span) => Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(normalize_expr_for_mutation)
                    .collect(),
            },
            *span,
        ),
        Expr::Map(map, span) => Expr::Map(
            map.map_expressions(&mut |v, _| normalize_expr_for_mutation(v))
                .expect("normalization preserves payloads"),
            *span,
        ),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            crate::ast::MetaExpr {
                metadata: meta
                    .metadata
                    .map_expressions(&mut |v, _| normalize_expr_for_mutation(v))
                    .expect("normalization preserves payloads"),
                expr: Box::new(normalize_expr_for_mutation(&meta.expr)),
            },
            *span,
        ),
        Expr::UnknownForm(data) => {
            let mut elements = vec![
                Expr::Atom(Atom::Name(data.head.clone()), data.span),
                Expr::Map(
                    data.meta
                        .map_expressions(&mut |v, _| normalize_expr_for_mutation(v))
                        .expect("normalization preserves payloads"),
                    data.span,
                ),
            ];
            elements.extend(data.children.iter().map(normalize_expr_for_mutation));
            Expr::List(List { elements }, data.span)
        }
        Expr::Atom(..) => expr.clone(),
    }
}

/// Borrowed canonical-node view shared by the legacy and successor carriers.
///
/// `children` uses semantic indexing: child 0 is the first element after the
/// tag and metadata map.  No `List` is reconstructed for a `Node`.
#[derive(Clone, Copy)]
struct NodeView<'a> {
    tag: DeepTag,
    meta: Option<&'a Metadata>,
    children: &'a [Expr],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleOutline {
    pub module_name: String,
    pub exports: Vec<String>,
    pub functions: Vec<FunctionOutline>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionOutline {
    pub name: String,
    pub qualified_name: String,
    pub params: Vec<String>,
    pub has_defsig: bool,
    pub def_deep: String,
    pub defsig_deep: Option<String>,
    pub body_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub caller: String,
    pub callee: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct References {
    pub symbol: String,
    pub references: Vec<Reference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallGraph {
    pub edges: Vec<Reference>,
}

#[derive(Debug, Clone)]
pub struct RenameReport {
    pub module: Vec<Expr>,
    pub renamed_def: Expr,
    pub renamed_defsig: Option<Expr>,
    pub renamed_references: usize,
    pub residual_old_references: usize,
}

#[derive(Debug, Clone)]
pub struct ReplaceFunctionReport {
    pub module: Vec<Expr>,
    pub replaced_def: Expr,
    pub replaced_defsig: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct ChangeSignatureReport {
    pub module: Vec<Expr>,
    pub changed_def: Expr,
    pub changed_defsig: Expr,
    pub rewritten_calls: usize,
    pub stale_calls: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoringError {
    #[error("{0}")]
    Metadata(#[from] crate::metadata::MetadataError),
    #[error("no module declaration found")]
    NoModule,
    #[error("expected exactly one module but found {count}")]
    MultipleModules { count: usize },
    #[error("{0}")]
    Resolve(#[from] ResolveError),
    #[error("{0}")]
    InvalidDecl(String),
    #[error("function `{name}` already exists")]
    DuplicateFunction { name: String },
    #[error("preimage mismatch for `{target}`")]
    PreimageMismatch { target: String },
    #[error("cascade incomplete for `{target}`: expected {expected} rewrites, applied {actual}")]
    CascadeIncomplete {
        target: String,
        expected: usize,
        actual: usize,
    },
}

pub fn outline(module_exprs: &[Expr]) -> Result<ModuleOutline, AuthoringError> {
    let module = single_module_expr(module_exprs)?;
    let module_name = module_name_view(module).unwrap_or_default().to_string();
    let exports = module_decls_view(module)
        .iter()
        .find_map(export_names)
        .unwrap_or_default();

    let defsig_names = defsig_map_view(module);
    let functions = module_decls_view(module)
        .iter()
        .filter_map(|decl| {
            let def = tagged_node_view(decl, DeepTag::Def)?;
            if !def_is_function_view(def) {
                return None;
            }
            let name = decl_name_view(def)?.to_string();
            let defsig = defsig_names.get(&name);
            let qualified_name = qualify(&module_name, &name);
            Some(FunctionOutline {
                params: fn_params_from_def_view(def),
                has_defsig: defsig.is_some(),
                def_deep: printer::print_expr(decl),
                defsig_deep: defsig.map(printer::print_expr),
                body_path: render_path(&DeepPath::body()),
                name,
                qualified_name,
            })
        })
        .collect();

    Ok(ModuleOutline {
        module_name,
        exports,
        functions,
    })
}

pub fn references(module_exprs: &[Expr], symbol: &str) -> Result<References, AuthoringError> {
    let module = single_module_expr(module_exprs)?;
    let module_name = module_name_view(module).unwrap_or_default().to_string();
    let bare = bare_name(symbol);
    let top_level_names = top_level_function_names_view(module);
    let mut references = Vec::new();

    for decl in module_decls_view(module) {
        let Some(def) = tagged_node_view(decl, DeepTag::Def) else {
            continue;
        };
        if !def_is_function_view(def) {
            continue;
        }
        let Some(caller_name) = decl_name_view(def) else {
            continue;
        };
        let caller = qualify(&module_name, caller_name);
        let mut metadata_scope: BTreeSet<String> = property_quantifier_names_from_def_view(def)
            .into_iter()
            .collect();
        collect_metadata_symbol_references_view(
            def,
            bare,
            &top_level_names,
            &mut metadata_scope,
            &caller,
            &mut references,
        );
        let Some(body) = function_body_view(def) else {
            continue;
        };
        let mut scope: BTreeSet<String> = fn_params_from_def_view(def).into_iter().collect();
        collect_symbol_references(
            body,
            bare,
            &top_level_names,
            &mut scope,
            "body".to_string(),
            &caller,
            &mut references,
        );
    }

    Ok(References {
        symbol: qualify_symbol_like(&module_name, symbol),
        references,
    })
}

pub fn call_graph(module_exprs: &[Expr]) -> Result<CallGraph, AuthoringError> {
    let module = single_module_expr(module_exprs)?;
    let module_name = module_name_view(module).unwrap_or_default().to_string();
    let top_level_names = top_level_function_names_view(module);
    let mut edges = Vec::new();

    for decl in module_decls_view(module) {
        let Some(def) = tagged_node_view(decl, DeepTag::Def) else {
            continue;
        };
        if !def_is_function_view(def) {
            continue;
        }
        let Some(caller_name) = decl_name_view(def) else {
            continue;
        };
        let caller = qualify(&module_name, caller_name);
        let mut metadata_scope: BTreeSet<String> = property_quantifier_names_from_def_view(def)
            .into_iter()
            .collect();
        collect_metadata_call_edges_view(
            def,
            &top_level_names,
            &mut metadata_scope,
            &caller,
            &module_name,
            &mut edges,
        );
        let Some(body) = function_body_view(def) else {
            continue;
        };
        let mut scope: BTreeSet<String> = fn_params_from_def_view(def).into_iter().collect();
        collect_call_edges(
            body,
            &top_level_names,
            &mut scope,
            "body".to_string(),
            &caller,
            &module_name,
            &mut edges,
        );
    }

    Ok(CallGraph { edges })
}

pub fn rename_function(
    module_exprs: &[Expr],
    function_name: &str,
    new_name: &str,
) -> Result<RenameReport, AuthoringError> {
    let module_exprs = normalize_for_mutation(module_exprs);
    let resolved = resolve_function(&module_exprs, function_name)?;
    validate_new_symbol(new_name)?;
    let old_name = bare_name(&resolved.qualified_name).to_string();
    let module = single_module(&module_exprs)?;
    if top_level_function_names(module).contains(new_name) {
        return Err(AuthoringError::DuplicateFunction {
            name: new_name.to_string(),
        });
    }
    let expected = references(&module_exprs, &old_name)?.references.len();
    let old_top_level_names = top_level_function_names(module);

    let mut rewritten = module_exprs.to_vec();
    let module = single_module_mut(&mut rewritten)?;
    let module_name = module_name(module).unwrap_or_default().to_string();

    for decl in module.elements.iter_mut().skip(MODULE_DECLS_START) {
        if let Some(list) = tagged_list_mut(decl, DeepTag::Def) {
            if decl_name(list) == Some(old_name.as_str()) {
                set_decl_name(list, new_name);
            }
        } else if let Some(list) = tagged_list_mut(decl, DeepTag::Defsig)
            && decl_name(list) == Some(old_name.as_str())
        {
            set_decl_name(list, new_name);
        }
        rename_export_symbol(decl, &old_name, new_name);
    }

    let mut renamed_references = 0;
    for decl in module.elements.iter_mut().skip(MODULE_DECLS_START) {
        let Some(def) = tagged_list_mut(decl, DeepTag::Def) else {
            continue;
        };
        if !def_is_function(def) {
            continue;
        }
        let fn_params = fn_params_from_def_list(def);
        let mut metadata_scope: BTreeSet<String> = property_quantifier_names_from_def_list(def)
            .into_iter()
            .collect();
        renamed_references += rename_metadata_references(
            def,
            &old_name,
            new_name,
            &old_top_level_names,
            &mut metadata_scope,
        );
        let mut scope: BTreeSet<String> = fn_params.into_iter().collect();
        if let Some(body) = function_body_mut_expr(decl) {
            renamed_references +=
                rename_unshadowed_vars(body, &old_name, new_name, &old_top_level_names, &mut scope);
        }
    }

    let residual_old_references = unshadowed_symbol_occurrences(&rewritten, &old_name)?;
    if residual_old_references != 0 || renamed_references != expected {
        return Err(AuthoringError::CascadeIncomplete {
            target: old_name,
            expected,
            actual: renamed_references,
        });
    }

    let renamed_def = find_function_def(&rewritten, &qualify(&module_name, new_name))?;
    let renamed_defsig = find_function_defsig(&rewritten, new_name);
    Ok(RenameReport {
        module: rewritten,
        renamed_def,
        renamed_defsig,
        renamed_references,
        residual_old_references,
    })
}

pub fn replace_function(
    module_exprs: &[Expr],
    function_name: &str,
    new_decls: &[Expr],
) -> Result<ReplaceFunctionReport, AuthoringError> {
    let module_exprs = normalize_for_mutation(module_exprs);
    let new_decls = normalize_for_mutation(new_decls);
    let resolved = resolve_function(&module_exprs, function_name)?;
    let old_name = bare_name(&resolved.qualified_name).to_string();
    let mut parsed = parse_function_bundle(&new_decls, Some(&old_name))?;
    let old_def = find_function_def(&module_exprs, function_name)?;
    parsed.def = parsed.def.try_inherit_extensions(&old_def)?;
    if let (Some(new), Some(old)) = (
        &mut parsed.defsig,
        find_function_defsig(&module_exprs, &old_name),
    ) {
        *new = new.clone().try_inherit_extensions(&old)?;
    }
    for decl in &mut parsed.ordered {
        *decl = if tagged_list(decl, DeepTag::Def).is_some() {
            parsed.def.clone()
        } else {
            parsed
                .defsig
                .clone()
                .expect("bundle signatures have a signature")
        };
    }

    let mut rewritten = module_exprs.to_vec();
    let module = single_module_mut(&mut rewritten)?;
    let decls_start = MODULE_DECLS_START;
    let mut def_index = None;
    let mut defsig_indices = Vec::new();
    for (index, decl) in module.elements.iter().enumerate().skip(decls_start) {
        if let Some(list) = tagged_list(decl, DeepTag::Def)
            && decl_name(list) == Some(old_name.as_str())
        {
            def_index = Some(index);
        }
        if let Some(list) = tagged_list(decl, DeepTag::Defsig)
            && decl_name(list) == Some(old_name.as_str())
        {
            defsig_indices.push(index);
        }
    }
    let Some(def_index) = def_index else {
        return Err(AuthoringError::Resolve(ResolveError::FunctionNotFound {
            searched: function_name.to_string(),
            name: old_name,
        }));
    };

    for index in defsig_indices.iter().rev() {
        module.elements.remove(*index);
    }
    let removed_before_def = defsig_indices
        .iter()
        .filter(|index| **index < def_index)
        .count();
    let adjusted_def_index = def_index - removed_before_def;
    module.elements.remove(adjusted_def_index);
    for (offset, decl) in parsed.ordered.iter().cloned().enumerate() {
        module.elements.insert(adjusted_def_index + offset, decl);
    }

    Ok(ReplaceFunctionReport {
        module: rewritten,
        replaced_def: parsed.def,
        replaced_defsig: parsed.defsig,
    })
}

pub fn change_signature(
    module_exprs: &[Expr],
    function_name: &str,
    new_defsig: &Expr,
    new_params: &Expr,
    argument_order: &[String],
    param_renames: &[(String, String)],
) -> Result<ChangeSignatureReport, AuthoringError> {
    let module_exprs = normalize_for_mutation(module_exprs);
    let new_defsig = normalize_expr_for_mutation(new_defsig);
    let new_params = normalize_expr_for_mutation(new_params);
    let resolved = resolve_function(&module_exprs, function_name)?;
    let target_name = bare_name(&resolved.qualified_name).to_string();
    validate_new_defsig(&new_defsig, &target_name)?;
    validate_params_node(&new_params)?;
    let old_params = function_params(&module_exprs, function_name)?;
    let old_param_set: BTreeSet<String> = old_params.iter().cloned().collect();
    validate_argument_order(argument_order, &old_param_set)?;
    let expected_calls = references(&module_exprs, &target_name)?.references.len();
    let rename_map: BTreeMap<String, String> = param_renames.iter().cloned().collect();

    let mut rewritten = module_exprs.to_vec();
    let module = single_module_mut(&mut rewritten)?;
    let top_level_names = top_level_function_names(module);
    let mut replaced_defsig = false;
    let mut rewritten_calls = 0;

    for decl in module.elements.iter_mut().skip(MODULE_DECLS_START) {
        if let Some(list) = tagged_list(decl, DeepTag::Defsig)
            && decl_name(list) == Some(target_name.as_str())
        {
            *decl = new_defsig.clone().try_inherit_extensions(decl)?;
            replaced_defsig = true;
            continue;
        }

        let is_target_def = tagged_list(decl, DeepTag::Def)
            .and_then(decl_name)
            .is_some_and(|name| name == target_name);
        if is_target_def {
            replace_def_params(decl, new_params.clone())?;
            if !rename_map.is_empty()
                && let Some(body) = function_body_mut_expr(decl)
            {
                rename_local_vars(body, &rename_map, &mut BTreeSet::new());
            }
        }
    }

    if !replaced_defsig {
        return Err(AuthoringError::InvalidDecl(format!(
            "function `{target_name}` has no defsig to replace"
        )));
    }

    for decl in module.elements.iter_mut().skip(MODULE_DECLS_START) {
        let Some(def) = tagged_list_mut(decl, DeepTag::Def) else {
            continue;
        };
        if !def_is_function(def) {
            continue;
        }
        let fn_params = fn_params_from_def_list(def);
        let mut metadata_scope: BTreeSet<String> = property_quantifier_names_from_def_list(def)
            .into_iter()
            .collect();
        rewritten_calls += rewrite_metadata_direct_calls(
            def,
            &target_name,
            &old_params,
            argument_order,
            &top_level_names,
            &mut metadata_scope,
        )?;
        let mut scope: BTreeSet<String> = fn_params.into_iter().collect();
        if let Some(body) = function_body_mut_expr(decl) {
            rewritten_calls += rewrite_direct_calls(
                body,
                &target_name,
                &old_params,
                argument_order,
                &top_level_names,
                &mut scope,
            )?;
        }
    }

    if rewritten_calls != expected_calls {
        return Err(AuthoringError::CascadeIncomplete {
            target: target_name,
            expected: expected_calls,
            actual: rewritten_calls,
        });
    }

    let stale_calls =
        stale_direct_calls(&rewritten, &resolved.qualified_name, argument_order.len())?;
    if stale_calls != 0 {
        return Err(AuthoringError::CascadeIncomplete {
            target: resolved.qualified_name,
            expected: expected_calls,
            actual: rewritten_calls.saturating_sub(stale_calls),
        });
    }

    Ok(ChangeSignatureReport {
        changed_def: find_function_def(&rewritten, function_name)?,
        changed_defsig: find_function_defsig(&rewritten, &target_name).ok_or_else(|| {
            AuthoringError::InvalidDecl(format!("function `{target_name}` has no defsig"))
        })?,
        module: rewritten,
        rewritten_calls,
        stale_calls,
    })
}

pub fn function_params(
    module_exprs: &[Expr],
    function_name: &str,
) -> Result<Vec<String>, AuthoringError> {
    let resolved = resolve_function(module_exprs, function_name)?;
    let module = single_module(module_exprs)?;
    let def = module
        .elements
        .get(MODULE_DECLS_START + resolved.decl_index)
        .and_then(|expr| tagged_list(expr, DeepTag::Def))
        .ok_or_else(|| {
            AuthoringError::Resolve(ResolveError::FunctionNotFound {
                searched: function_name.to_string(),
                name: bare_name(&resolved.qualified_name).to_string(),
            })
        })?;
    Ok(fn_params_from_def_list(def))
}

pub fn unshadowed_symbol_occurrences(
    module_exprs: &[Expr],
    symbol: &str,
) -> Result<usize, AuthoringError> {
    Ok(references(module_exprs, symbol)?.references.len())
}

fn collect_metadata_symbol_references_view(
    def: NodeView<'_>,
    symbol: &str,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
    caller: &str,
    out: &mut Vec<Reference>,
) {
    let Some(meta) = def.meta else {
        return;
    };
    meta.visit_syntax(&mut |key, value| {
        let key = key.spelling();
        collect_symbol_references(
            value,
            symbol,
            top_level_names,
            scope,
            format!("metadata.{key}"),
            caller,
            out,
        );
    });
}

fn collect_metadata_call_edges_view(
    def: NodeView<'_>,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
    caller: &str,
    module_name: &str,
    out: &mut Vec<Reference>,
) {
    let Some(meta) = def.meta else {
        return;
    };
    meta.visit_syntax(&mut |key, value| {
        let key = key.spelling();
        collect_call_edges(
            value,
            top_level_names,
            scope,
            format!("metadata.{key}"),
            caller,
            module_name,
            out,
        );
    });
}

fn rename_metadata_references(
    def: &mut List,
    old_name: &str,
    new_name: &str,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> usize {
    let Some(Expr::Map(meta, _)) = def.elements.get_mut(1) else {
        return 0;
    };
    let mut count = 0;
    edit_metadata(meta, &mut |value| {
        count += rename_unshadowed_vars(value, old_name, new_name, top_level_names, scope);
    });
    count
}

fn rewrite_metadata_direct_calls(
    def: &mut List,
    target_name: &str,
    old_params: &[String],
    argument_order: &[String],
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> Result<usize, AuthoringError> {
    let Some(Expr::Map(meta, _)) = def.elements.get_mut(1) else {
        return Ok(0);
    };
    let mut count = 0;
    *meta = meta.try_map_expressions::<AuthoringError>(&mut |value, _| {
        let mut value = value.clone();
        count += rewrite_direct_calls(
            &mut value,
            target_name,
            old_params,
            argument_order,
            top_level_names,
            scope,
        )?;
        Ok(value)
    })?;
    Ok(count)
}

fn count_stale_metadata_calls(
    def: &List,
    target: &str,
    expected_arity: usize,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> usize {
    let Some(Expr::Map(meta, _)) = def.elements.get(1) else {
        return 0;
    };
    let mut count = 0;
    meta.visit_syntax(&mut |_, value| {
        count += count_stale_calls(value, target, expected_arity, top_level_names, scope);
    });
    count
}

fn collect_call_edges(
    expr: &Expr,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
    path: String,
    caller: &str,
    module_name: &str,
    out: &mut Vec<Reference>,
) {
    if let Some((callee, _args)) = app_name_and_args(expr)
        && top_level_names.contains(callee)
        && !scope.contains(callee)
    {
        out.push(Reference {
            caller: caller.to_string(),
            callee: qualify(module_name, callee),
            path: format!("{path}.0"),
        });
    }
    walk_children(expr, scope, path, |child, scope, child_path| {
        collect_call_edges(
            child,
            top_level_names,
            scope,
            child_path,
            caller,
            module_name,
            out,
        )
    });
}

fn collect_symbol_references(
    expr: &Expr,
    symbol: &str,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
    path: String,
    caller: &str,
    out: &mut Vec<Reference>,
) {
    if let Some(name) = var_name(expr)
        && name == symbol
        && !scope.contains(name)
    {
        out.push(Reference {
            caller: caller.to_string(),
            callee: symbol.to_string(),
            path: path.clone(),
        });
    }
    let _ = top_level_names;
    walk_children(expr, scope, path, |child, scope, child_path| {
        collect_symbol_references(
            child,
            symbol,
            top_level_names,
            scope,
            child_path,
            caller,
            out,
        )
    });
}

fn rename_unshadowed_vars(
    expr: &mut Expr,
    old_name: &str,
    new_name: &str,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> usize {
    let mut count = 0;
    if top_level_names.contains(old_name)
        && !scope.contains(old_name)
        && let Some(name) = var_name_mut(expr)
        && name == old_name
    {
        *name = new_name.to_string();
        count += 1;
    }
    walk_children_mut(expr, scope, &mut |child, scope| {
        count += rename_unshadowed_vars(child, old_name, new_name, top_level_names, scope);
    });
    count
}

fn rewrite_direct_calls(
    expr: &mut Expr,
    target_name: &str,
    old_params: &[String],
    argument_order: &[String],
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> Result<usize, AuthoringError> {
    let mut count = 0;
    if top_level_names.contains(target_name)
        && !scope.contains(target_name)
        && let Some(list) = app_list_mut(expr)
    {
        let is_target = list
            .elements
            .get(2)
            .and_then(var_name)
            .is_some_and(|name| name == target_name);
        if is_target {
            let old_args: Vec<Expr> = list.elements.iter().skip(3).cloned().collect();
            let mut old_by_name = BTreeMap::new();
            for (name, arg) in old_params.iter().zip(old_args) {
                old_by_name.insert(name.clone(), arg);
            }
            let mut new_elements = list.elements[..3].to_vec();
            for name in argument_order {
                let arg = old_by_name.get(name).ok_or_else(|| {
                    AuthoringError::InvalidDecl(format!(
                        "argument_order references unknown old parameter `{name}`"
                    ))
                })?;
                new_elements.push(arg.clone());
            }
            list.elements = new_elements;
            count += 1;
        }
    }
    walk_children_mut(expr, scope, &mut |child, scope| {
        if let Ok(child_count) = rewrite_direct_calls(
            child,
            target_name,
            old_params,
            argument_order,
            top_level_names,
            scope,
        ) {
            count += child_count;
        }
    });
    Ok(count)
}

fn stale_direct_calls(
    module_exprs: &[Expr],
    function_name: &str,
    expected_arity: usize,
) -> Result<usize, AuthoringError> {
    let bare = bare_name(function_name);
    let module = single_module(module_exprs)?;
    let top_level_names = top_level_function_names(module);
    let mut stale = 0;
    for decl in module.elements.iter().skip(MODULE_DECLS_START) {
        let Some(def) = tagged_list(decl, DeepTag::Def) else {
            continue;
        };
        if !def_is_function(def) {
            continue;
        }
        let mut metadata_scope: BTreeSet<String> = property_quantifier_names_from_def_list(def)
            .into_iter()
            .collect();
        stale += count_stale_metadata_calls(
            def,
            bare,
            expected_arity,
            &top_level_names,
            &mut metadata_scope,
        );
        let Some(body) = function_body(decl) else {
            continue;
        };
        let mut scope: BTreeSet<String> = fn_params_from_def_list(def).into_iter().collect();
        stale += count_stale_calls(body, bare, expected_arity, &top_level_names, &mut scope);
    }
    Ok(stale)
}

fn count_stale_calls(
    expr: &Expr,
    target: &str,
    expected_arity: usize,
    top_level_names: &BTreeSet<String>,
    scope: &mut BTreeSet<String>,
) -> usize {
    let mut count = 0;
    if let Some((callee, args)) = app_name_and_args(expr)
        && callee == target
        && top_level_names.contains(callee)
        && !scope.contains(callee)
        && args.len() != expected_arity
    {
        count += 1;
    }
    walk_children(expr, scope, String::new(), |child, scope, _| {
        count += count_stale_calls(child, target, expected_arity, top_level_names, scope);
    });
    count
}

fn rename_local_vars(
    expr: &mut Expr,
    renames: &BTreeMap<String, String>,
    shadowed: &mut BTreeSet<String>,
) {
    if let Some(name) = var_name_mut(expr)
        && !shadowed.contains(name)
        && let Some(new_name) = renames.get(name)
    {
        *name = new_name.clone();
    }
    walk_children_mut(expr, shadowed, &mut |child, shadowed| {
        rename_local_vars(child, renames, shadowed);
    });
}

fn walk_children<F>(expr: &Expr, scope: &mut BTreeSet<String>, path: String, mut f: F)
where
    F: FnMut(&Expr, &mut BTreeSet<String>, String),
{
    match expr {
        Expr::Map(meta, _) => {
            meta.visit_syntax(&mut |key, value| {
                let key = key.spelling();
                f(value, scope, format!("{path}.{key}"));
            });
            return;
        }
        Expr::MetaExpr(meta, _) => {
            meta.metadata.visit_syntax(&mut |key, value| {
                let key = key.spelling();
                f(value, scope, format!("{path}.{key}"));
            });
            f(&meta.expr, scope, format!("{path}.expr"));
            return;
        }
        Expr::Atom(..) => return,
        Expr::BareList(children, _) => {
            for (index, child) in children.iter().enumerate() {
                f(child, scope, format!("{path}.{index}"));
            }
            return;
        }
        Expr::UnknownForm(data) => {
            data.meta.visit_syntax(&mut |key, value| {
                let key = key.spelling();
                f(value, scope, format!("{path}.meta.{key}"));
            });
            for (index, child) in data.children.iter().enumerate() {
                f(child, scope, format!("{path}.{index}"));
            }
            return;
        }
        Expr::List(..) | Expr::Node(..) => {}
    }

    let Some(node) = node_view(expr) else {
        // chelis#1136 red-team M1: a `List` whose head is not in the
        // closed tag vocabulary (an unknown form - the shape the
        // chelis#1088 stamped ingress deliberately preserves) is still
        // TRAVERSED, mirroring the pre-NodeView walker and the mutating
        // twin. Skipping it silently drops references inside unknown
        // forms and desynchronizes rename's cascade accounting.
        if let Expr::List(list, _) = expr {
            if let Some(meta) = list.elements.get(1) {
                f(meta, scope, format!("{path}.meta"));
            }
            for (index, child) in list.elements.iter().enumerate().skip(2) {
                f(child, scope, format!("{path}.{index}"));
            }
        }
        return;
    };
    if let Some(meta) = node.meta {
        meta.visit_syntax(&mut |key, value| {
            let key = key.spelling();
            f(value, scope, format!("{path}.meta.{key}"));
        });
    }
    if node.tag == DeepTag::Fn {
        let added = node
            .children
            .first()
            .and_then(params_node_names)
            .unwrap_or_default();
        with_scope(scope, added, |scope| {
            if let Some(body) = node.children.get(1) {
                f(body, scope, format!("{path}.1"));
            }
        });
        return;
    }
    if node.tag == DeepTag::Let {
        if let Some(bind) = node.children.first() {
            let mut inserted = Vec::new();
            if let Some(bind) = tagged_node_view(bind, DeepTag::Bind) {
                if let Some(meta) = bind.meta {
                    meta.visit_syntax(&mut |key, value| {
                        let key = key.spelling();
                        f(value, scope, format!("{path}.0.meta.{key}"));
                    });
                }
                for pair_start in (0..bind.children.len()).step_by(2) {
                    if let Some(name_expr) = bind.children.get(pair_start) {
                        f(name_expr, scope, format!("{path}.0.{pair_start}"));
                    }
                    let has_value = if let Some(value) = bind.children.get(pair_start + 1) {
                        f(value, scope, format!("{path}.0.{}", pair_start + 1));
                        true
                    } else {
                        false
                    };
                    if has_value
                        && let Some(name) = bind.children.get(pair_start).and_then(symbol)
                        && scope.insert(name.to_string())
                    {
                        inserted.push(name.to_string());
                    }
                }
            } else {
                f(bind, scope, format!("{path}.0"));
            }
            if let Some(body) = node.children.get(1) {
                f(body, scope, format!("{path}.1"));
            }
            for name in inserted {
                scope.remove(&name);
            }
        }
        return;
    }
    if node.tag == DeepTag::Arm {
        let Some(pattern) = node.children.first() else {
            return;
        };
        f(pattern, scope, format!("{path}.0"));
        let added = crate::pattern_binder_names(pattern);
        with_scope(scope, added, |scope| {
            for (semantic_index, child) in node.children.iter().enumerate().skip(1) {
                f(child, scope, format!("{path}.{semantic_index}"));
            }
        });
        return;
    }
    for (semantic_index, child) in node.children.iter().enumerate() {
        f(child, scope, format!("{path}.{semantic_index}"));
    }
}

fn walk_children_mut<F>(expr: &mut Expr, scope: &mut BTreeSet<String>, f: &mut F)
where
    F: FnMut(&mut Expr, &mut BTreeSet<String>),
{
    let list = match expr {
        Expr::Map(meta, _) => {
            edit_metadata(meta, &mut |value| f(value, scope));
            return;
        }
        Expr::MetaExpr(meta, _) => {
            edit_metadata(&mut meta.metadata, &mut |value| f(value, scope));
            f(&mut meta.expr, scope);
            return;
        }
        Expr::List(list, _) => list,
        Expr::Atom(..) => return,
        Expr::Node(..) => {
            // Bridge: convert Node to List in place so mutable traversal works (#908).
            // Node lacks mutable child access, so we destructure into List form.
            let placeholder = Expr::Atom(Atom::Bool(false), Span::new(0, 0));
            match std::mem::replace(expr, placeholder) {
                Expr::Node(node, span) => {
                    *expr = Expr::List(node.to_list(span), span);
                    walk_children_mut(expr, scope, f);
                }
                _ => unreachable!(),
            }
            return;
        }
        // chelis#1087: mirror the read-only walker above — skipping these
        // silently drops references inside transitional variants and
        // desynchronizes rename's cascade accounting.
        Expr::BareList(children, _) => {
            for child in children.iter_mut() {
                f(child, scope);
            }
            return;
        }
        Expr::UnknownForm(data) => {
            edit_metadata(&mut data.meta, &mut |value| f(value, scope));
            for child in &mut data.children {
                f(child, scope);
            }
            return;
        }
    };
    let tag = list_tag(list);
    if let Some(meta) = list.elements.get_mut(1) {
        f(meta, scope);
    }
    if tag == Some(DeepTag::Fn) {
        let added = list
            .elements
            .get(FN_PARAMS_INDEX)
            .and_then(params_node_names)
            .unwrap_or_default();
        with_scope(scope, added, |scope| {
            if let Some(body) = list.elements.get_mut(3) {
                f(body, scope);
            }
        });
        return;
    }
    if tag == Some(DeepTag::Let) {
        let mut inserted = Vec::new();
        if let Some(bind) = list.elements.get_mut(2) {
            if let Some(bind) = tagged_list_mut(bind, DeepTag::Bind) {
                if let Some(meta) = bind.elements.get_mut(1) {
                    f(meta, scope);
                }
                let mut pair_start = 2;
                while pair_start < bind.elements.len() {
                    let name = bind
                        .elements
                        .get(pair_start)
                        .and_then(symbol)
                        .map(str::to_string);
                    if let Some(name_expr) = bind.elements.get_mut(pair_start) {
                        f(name_expr, scope);
                    }
                    let has_value = if let Some(value) = bind.elements.get_mut(pair_start + 1) {
                        f(value, scope);
                        true
                    } else {
                        false
                    };
                    if has_value
                        && let Some(name) = name
                        && scope.insert(name.clone())
                    {
                        inserted.push(name);
                    }
                    pair_start += 2;
                }
            } else {
                f(bind, scope);
            }
        }
        if let Some(body) = list.elements.get_mut(3) {
            f(body, scope);
        }
        for name in inserted {
            scope.remove(&name);
        }
        return;
    }
    if tag == Some(DeepTag::Arm) {
        let added = list
            .elements
            .get(2)
            .map(crate::pattern_binder_names)
            .unwrap_or_default();
        if let Some(pattern) = list.elements.get_mut(2) {
            f(pattern, scope);
        }
        with_scope(scope, added, |scope| {
            for child in list.elements.iter_mut().skip(3) {
                f(child, scope);
            }
        });
        return;
    }
    for child in list.elements.iter_mut().skip(2) {
        f(child, scope);
    }
}

fn with_scope<R, F>(scope: &mut BTreeSet<String>, added: Vec<String>, f: F) -> R
where
    F: FnOnce(&mut BTreeSet<String>) -> R,
{
    let mut inserted = Vec::new();
    for name in added {
        if scope.insert(name.clone()) {
            inserted.push(name);
        }
    }
    let result = f(scope);
    for name in inserted {
        scope.remove(&name);
    }
    result
}

struct ParsedFunctionBundle {
    ordered: Vec<Expr>,
    def: Expr,
    defsig: Option<Expr>,
}

fn parse_function_bundle(
    exprs: &[Expr],
    required_name: Option<&str>,
) -> Result<ParsedFunctionBundle, AuthoringError> {
    if !(1..=2).contains(&exprs.len()) {
        return Err(AuthoringError::InvalidDecl(format!(
            "function bundle must contain exactly one `(def ...)` and optional `(defsig ...)`, got {} expressions",
            exprs.len()
        )));
    }
    let mut def = None;
    let mut defsig = None;
    for expr in exprs {
        match expr_tag(expr) {
            Some(DeepTag::Def) => {
                if def.is_some() {
                    return Err(AuthoringError::InvalidDecl(
                        "function bundle contains more than one `(def ...)`".to_string(),
                    ));
                }
                let list = tagged_list(expr, DeepTag::Def).expect("tag checked");
                let name = decl_name(list).ok_or_else(|| {
                    AuthoringError::InvalidDecl("new `(def ...)` has no name".to_string())
                })?;
                if required_name.is_some_and(|required| required != name) {
                    return Err(AuthoringError::InvalidDecl(format!(
                        "new `(def ...)` names `{name}`, expected `{}`",
                        required_name.unwrap()
                    )));
                }
                if !def_is_function(list) {
                    return Err(AuthoringError::InvalidDecl(
                        "replacement `(def ...)` must define a function".to_string(),
                    ));
                }
                def = Some(expr.clone());
            }
            Some(DeepTag::Defsig) => {
                if defsig.is_some() {
                    return Err(AuthoringError::InvalidDecl(
                        "function bundle contains more than one `(defsig ...)`".to_string(),
                    ));
                }
                let list = tagged_list(expr, DeepTag::Defsig).expect("tag checked");
                let name = decl_name(list).ok_or_else(|| {
                    AuthoringError::InvalidDecl("new `(defsig ...)` has no name".to_string())
                })?;
                if required_name.is_some_and(|required| required != name) {
                    return Err(AuthoringError::InvalidDecl(format!(
                        "new `(defsig ...)` names `{name}`, expected `{}`",
                        required_name.unwrap()
                    )));
                }
                defsig = Some(expr.clone());
            }
            Some(tag) => {
                return Err(AuthoringError::InvalidDecl(format!(
                    "function bundle cannot contain `({} ...)`",
                    tag.as_str()
                )));
            }
            None => {
                return Err(AuthoringError::InvalidDecl(
                    "function bundle entries must be lists".to_string(),
                ));
            }
        }
    }
    let Some(def) = def else {
        return Err(AuthoringError::InvalidDecl(
            "function bundle must contain one `(def ...)`".to_string(),
        ));
    };
    Ok(ParsedFunctionBundle {
        ordered: exprs.to_vec(),
        def,
        defsig,
    })
}

fn validate_new_defsig(expr: &Expr, required_name: &str) -> Result<(), AuthoringError> {
    let Some(list) = tagged_list(expr, DeepTag::Defsig) else {
        return Err(AuthoringError::InvalidDecl(
            "`new_defsig` must be one `(defsig ...)` expression".to_string(),
        ));
    };
    if decl_name(list) != Some(required_name) {
        return Err(AuthoringError::InvalidDecl(format!(
            "`new_defsig` must name `{required_name}`"
        )));
    }
    Ok(())
}

fn validate_params_node(expr: &Expr) -> Result<(), AuthoringError> {
    if tagged_list(expr, DeepTag::Params).is_none() {
        return Err(AuthoringError::InvalidDecl(
            "`new_params` must be one `(params ...)` expression".to_string(),
        ));
    }
    Ok(())
}

fn validate_argument_order(
    argument_order: &[String],
    old_param_set: &BTreeSet<String>,
) -> Result<(), AuthoringError> {
    let mut seen = BTreeSet::new();
    for name in argument_order {
        if !seen.insert(name) {
            return Err(AuthoringError::InvalidDecl(format!(
                "argument_order contains duplicate old parameter `{name}`"
            )));
        }
    }
    let argument_set: BTreeSet<String> = argument_order.iter().cloned().collect();
    if argument_order.len() != old_param_set.len() || argument_set != *old_param_set {
        return Err(AuthoringError::InvalidDecl(
            "argument_order must be a permutation of the old parameter names".to_string(),
        ));
    }
    Ok(())
}

fn validate_new_symbol(name: &str) -> Result<(), AuthoringError> {
    if name.is_empty()
        || name.contains('.')
        || !name
            .chars()
            .all(|ch| ch == '_' || ch == '-' || ch.is_ascii_alphanumeric())
    {
        return Err(AuthoringError::InvalidDecl(format!(
            "`{name}` is not a valid bare Deep symbol for this tool"
        )));
    }
    Ok(())
}

fn replace_def_params(def: &mut Expr, new_params: Expr) -> Result<(), AuthoringError> {
    let Some(def_list) = tagged_list_mut(def, DeepTag::Def) else {
        return Err(AuthoringError::InvalidDecl(
            "target declaration is not a `(def ...)`".to_string(),
        ));
    };
    let Some(Expr::List(fn_list, _)) = def_list.elements.get_mut(DEF_VALUE_INDEX) else {
        return Err(AuthoringError::InvalidDecl(
            "target `(def ...)` has no `(fn ...)` child".to_string(),
        ));
    };
    if list_tag(fn_list) != Some(DeepTag::Fn) {
        return Err(AuthoringError::InvalidDecl(
            "target `(def ...)` has no `(fn ...)` child".to_string(),
        ));
    }
    if fn_list.elements.len() <= FN_PARAMS_INDEX {
        return Err(AuthoringError::InvalidDecl(
            "target `(fn ...)` has no params node".to_string(),
        ));
    }
    fn_list.elements[FN_PARAMS_INDEX] =
        new_params.try_inherit_extensions(&fn_list.elements[FN_PARAMS_INDEX])?;
    Ok(())
}

fn find_function_def(module_exprs: &[Expr], function_name: &str) -> Result<Expr, AuthoringError> {
    let resolved = resolve_function(module_exprs, function_name)?;
    let module = single_module(module_exprs)?;
    module
        .elements
        .get(MODULE_DECLS_START + resolved.decl_index)
        .cloned()
        .ok_or_else(|| {
            AuthoringError::InvalidDecl(format!(
                "resolved function `{function_name}` had no declaration"
            ))
        })
}

fn find_function_defsig(module_exprs: &[Expr], function_name: &str) -> Option<Expr> {
    let module = single_module(module_exprs).ok()?;
    let bare = bare_name(function_name);
    module
        .elements
        .iter()
        .skip(MODULE_DECLS_START)
        .find_map(|decl| {
            let list = tagged_list(decl, DeepTag::Defsig)?;
            (decl_name(list) == Some(bare)).then(|| decl.clone())
        })
}

fn top_level_function_names(module: &List) -> BTreeSet<String> {
    module
        .elements
        .iter()
        .skip(MODULE_DECLS_START)
        .filter_map(|decl| {
            let def = tagged_list(decl, DeepTag::Def)?;
            def_is_function(def).then(|| decl_name(def).map(str::to_string))?
        })
        .collect()
}

fn export_names(expr: &Expr) -> Option<Vec<String>> {
    let export = tagged_node_view(expr, DeepTag::Export)?;
    Some(
        export
            .children
            .iter()
            .filter_map(symbol)
            .map(str::to_string)
            .collect(),
    )
}

fn rename_export_symbol(expr: &mut Expr, old_name: &str, new_name: &str) {
    let Some(list) = tagged_list_mut(expr, DeepTag::Export) else {
        return;
    };
    for element in list.elements.iter_mut().skip(2) {
        if let Expr::Atom(Atom::Name(name), _) = element
            && name == old_name
        {
            *name = new_name.to_string();
        }
    }
}

fn fn_params_from_def_list(def: &List) -> Vec<String> {
    def.elements
        .get(DEF_VALUE_INDEX)
        .and_then(tagged_fn_params)
        .unwrap_or_default()
}

fn property_quantifier_names_from_def_list(def: &List) -> Vec<String> {
    let Some(Expr::Map(meta, _)) = def.elements.get(1) else {
        return Vec::new();
    };
    meta.property_quantifiers()
        .map(|v| {
            v.values()
                .iter()
                .map(|v| v.name().value().clone())
                .collect()
        })
        .unwrap_or_default()
}

fn tagged_fn_params(expr: &Expr) -> Option<Vec<String>> {
    let function = tagged_node_view(expr, DeepTag::Fn)?;
    function.children.first().and_then(params_node_names)
}

fn params_node_names(expr: &Expr) -> Option<Vec<String>> {
    let params = tagged_node_view(expr, DeepTag::Params)?;
    Some(
        params
            .children
            .iter()
            .filter_map(param_name)
            .map(str::to_string)
            .collect(),
    )
}

fn param_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(list, _) => list.elements.first().and_then(symbol),
        Expr::BareList(elements, _) => elements.first().and_then(symbol),
        _ => symbol(expr),
    }
}

fn function_body_mut_expr(def: &mut Expr) -> Option<&mut Expr> {
    let def_list = tagged_list_mut(def, DeepTag::Def)?;
    let Expr::List(fn_list, _) = def_list.elements.get_mut(DEF_VALUE_INDEX)? else {
        return None;
    };
    if list_tag(fn_list) != Some(DeepTag::Fn) {
        return None;
    }
    fn_list.elements.get_mut(3)
}

fn app_name_and_args(expr: &Expr) -> Option<(&str, &[Expr])> {
    let app = tagged_node_view(expr, DeepTag::App)?;
    let callee = app.children.first().and_then(var_name)?;
    Some((callee, app.children.get(1..).unwrap_or_default()))
}

fn app_list_mut(expr: &mut Expr) -> Option<&mut List> {
    tagged_list_mut(expr, DeepTag::App)
}

fn var_name(expr: &Expr) -> Option<&str> {
    tagged_node_view(expr, DeepTag::Var)?
        .children
        .first()
        .and_then(symbol)
}

fn var_name_mut(expr: &mut Expr) -> Option<&mut String> {
    let list = tagged_list_mut(expr, DeepTag::Var)?;
    match list.elements.get_mut(2)? {
        Expr::Atom(Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn node_view(expr: &Expr) -> Option<NodeView<'_>> {
    match expr.carrier() {
        crate::ExprCarrier::DecodedNode(tag, metadata, children) => Some(NodeView {
            tag,
            meta: Some(metadata),
            children,
        }),
        crate::ExprCarrier::StructuralList(_)
        | crate::ExprCarrier::UndecodableHead(_, _, _)
        | crate::ExprCarrier::Atom(_)
        | crate::ExprCarrier::MetadataMap(_)
        | crate::ExprCarrier::MetadataExpression(_)
        | crate::ExprCarrier::MalformedLegacyList(_) => None,
    }
}

fn tagged_node_view(expr: &Expr, expected: DeepTag) -> Option<NodeView<'_>> {
    let node = node_view(expr)?;
    (node.tag == expected).then_some(node)
}

fn single_module_expr(exprs: &[Expr]) -> Result<&Expr, AuthoringError> {
    let modules: Vec<&Expr> = exprs
        .iter()
        .filter(|expr| tagged_node_view(expr, DeepTag::Module).is_some())
        .collect();
    match modules.as_slice() {
        [] => Err(AuthoringError::NoModule),
        [module] => Ok(module),
        many => Err(AuthoringError::MultipleModules { count: many.len() }),
    }
}

fn module_name_view(module: &Expr) -> Option<&str> {
    tagged_node_view(module, DeepTag::Module)?
        .children
        .first()
        .and_then(symbol)
}

fn module_decls_view(module: &Expr) -> &[Expr] {
    tagged_node_view(module, DeepTag::Module)
        .and_then(|module| module.children.get(1..))
        .unwrap_or_default()
}

fn decl_name_view(decl: NodeView<'_>) -> Option<&str> {
    decl.children.first().and_then(symbol)
}

fn def_is_function_view(def: NodeView<'_>) -> bool {
    def.children
        .get(1)
        .is_some_and(|value| tagged_node_view(value, DeepTag::Fn).is_some())
}

fn function_body_view(def: NodeView<'_>) -> Option<&Expr> {
    let function = tagged_node_view(def.children.get(1)?, DeepTag::Fn)?;
    function.children.get(1)
}

fn fn_params_from_def_view(def: NodeView<'_>) -> Vec<String> {
    def.children
        .get(1)
        .and_then(tagged_fn_params)
        .unwrap_or_default()
}

fn property_quantifier_names_from_def_view(def: NodeView<'_>) -> Vec<String> {
    def.meta
        .and_then(Metadata::property_quantifiers)
        .map(|v| {
            v.values()
                .iter()
                .map(|v| v.name().value().clone())
                .collect()
        })
        .unwrap_or_default()
}

fn top_level_function_names_view(module: &Expr) -> BTreeSet<String> {
    module_decls_view(module)
        .iter()
        .filter_map(|decl| {
            let def = tagged_node_view(decl, DeepTag::Def)?;
            def_is_function_view(def).then(|| decl_name_view(def).map(str::to_string))?
        })
        .collect()
}

fn defsig_map_view(module: &Expr) -> BTreeMap<String, Expr> {
    module_decls_view(module)
        .iter()
        .filter_map(|decl| {
            let defsig = tagged_node_view(decl, DeepTag::Defsig)?;
            Some((decl_name_view(defsig)?.to_string(), decl.clone()))
        })
        .collect()
}

fn single_module(exprs: &[Expr]) -> Result<&List, AuthoringError> {
    let modules: Vec<&List> = exprs
        .iter()
        .filter_map(|expr| tagged_list(expr, DeepTag::Module))
        .collect();
    match modules.as_slice() {
        [] => Err(AuthoringError::NoModule),
        [module] => Ok(module),
        many => Err(AuthoringError::MultipleModules { count: many.len() }),
    }
}

fn single_module_mut(exprs: &mut [Expr]) -> Result<&mut List, AuthoringError> {
    let count = exprs
        .iter()
        .filter(|expr| tagged_list(expr, DeepTag::Module).is_some())
        .count();
    if count == 0 {
        return Err(AuthoringError::NoModule);
    }
    if count > 1 {
        return Err(AuthoringError::MultipleModules { count });
    }
    exprs
        .iter_mut()
        .find_map(|expr| tagged_list_mut(expr, DeepTag::Module))
        .ok_or(AuthoringError::NoModule)
}

fn expr_tag(expr: &Expr) -> Option<DeepTag> {
    any_list(expr).and_then(list_tag)
}

fn tagged_list(expr: &Expr, tag: DeepTag) -> Option<&List> {
    let list = any_list(expr)?;
    (list_tag(list) == Some(tag)).then_some(list)
}

fn tagged_list_mut(expr: &mut Expr, tag: DeepTag) -> Option<&mut List> {
    let list = any_list_mut(expr)?;
    (list_tag(list) == Some(tag)).then_some(list)
}

fn any_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn any_list_mut(expr: &mut Expr) -> Option<&mut List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn list_tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn module_name(list: &List) -> Option<&str> {
    list.elements.get(2).and_then(symbol)
}

fn decl_name(list: &List) -> Option<&str> {
    list.elements.get(DEF_NAME_INDEX).and_then(symbol)
}

fn set_decl_name(list: &mut List, new_name: &str) {
    if let Some(Expr::Atom(Atom::Name(name), _)) = list.elements.get_mut(DEF_NAME_INDEX) {
        *name = new_name.to_string();
    }
}

fn def_is_function(list: &List) -> bool {
    matches!(
        list.elements.get(DEF_VALUE_INDEX),
        Some(expr) if tagged_list(expr, DeepTag::Fn).is_some()
    )
}

fn symbol(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(value), _) => Some(value.as_str()),
        _ => None,
    }
}

fn bare_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

fn qualify(module_name: &str, bare: &str) -> String {
    if module_name.is_empty() {
        bare.to_string()
    } else {
        format!("{module_name}.{bare}")
    }
}

fn qualify_symbol_like(module_name: &str, symbol: &str) -> String {
    if symbol.contains('.') {
        symbol.to_string()
    } else {
        qualify(module_name, symbol)
    }
}

fn render_path(path: &DeepPath) -> String {
    path.segments()
        .iter()
        .map(|segment| match segment {
            PathSegment::Body => "body".to_string(),
            PathSegment::Child(index) => index.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn edit_metadata(meta: &mut Metadata, f: &mut impl FnMut(&mut Expr)) {
    *meta = meta
        .map_expressions(&mut |value, _| {
            let mut value = value.clone();
            f(&mut value);
            value
        })
        .expect("authoring transformation preserves metadata shape");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnknownFormData;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    /// A tree exercising both transitional variants, with material at every
    /// position a walker can skip: a BareList element, an UnknownForm
    /// metadata value, an UnknownForm child, and a nested BareList interior.
    /// Deliberately Node-free: the mutating walker bridges `Node` to `List`
    /// in place, which would make the two visit logs differ by
    /// representation rather than by coverage.
    fn transitional_fixture() -> Expr {
        Expr::BareList(
            vec![
                Expr::Atom(Atom::Name("a".to_string()), sp()),
                Expr::UnknownForm(Box::new(UnknownFormData {
                    head: "mystery".to_string(),
                    meta: {
                        let mut metadata = Metadata::default();
                        metadata
                            .extensions_mut()
                            .insert("note".into(), crate::ExtensionData::parse("m").unwrap())
                            .unwrap();
                        metadata
                    },
                    children: vec![
                        Expr::Atom(Atom::Name("b".to_string()), sp()),
                        Expr::BareList(vec![Expr::Atom(Atom::Name("c".to_string()), sp())], sp()),
                    ],
                    span: sp(),
                })),
            ],
            sp(),
        )
    }

    fn read_walk(expr: &Expr, out: &mut Vec<String>) {
        fn go(expr: &Expr, scope: &mut BTreeSet<String>, out: &mut Vec<String>) {
            walk_children(expr, scope, String::new(), |child, scope, _path| {
                out.push(format!("{child:?}"));
                go(child, scope, out);
            });
        }
        go(expr, &mut BTreeSet::new(), out);
    }

    fn mut_walk(expr: &mut Expr, out: &mut Vec<String>) {
        fn go(expr: &mut Expr, scope: &mut BTreeSet<String>, out: &mut Vec<String>) {
            let mut f = |child: &mut Expr, scope: &mut BTreeSet<String>| {
                out.push(format!("{child:?}"));
                go(child, scope, out);
            };
            walk_children_mut(expr, scope, &mut f);
        }
        go(expr, &mut BTreeSet::new(), out);
    }

    /// chelis#1087: the mutating walker must traverse exactly what the
    /// read-only walker traverses over the transitional variants. Before
    /// the fix it returned early on both, so `rename_local_vars` silently
    /// skipped references the read-only cascade accounting had counted.
    #[test]
    fn walk_children_mut_matches_walk_children_over_transitional_variants() {
        let fixture = transitional_fixture();
        let mut read_visits = Vec::new();
        read_walk(&fixture, &mut read_visits);

        let mut mutable = fixture.clone();
        let mut mut_visits = Vec::new();
        mut_walk(&mut mutable, &mut mut_visits);

        assert_eq!(
            read_visits, mut_visits,
            "the walkers must visit the same children in the same order"
        );
        assert!(
            read_visits.iter().any(|visit| visit.contains("\"c\"")),
            "traversal reaches the nested BareList interior: {read_visits:?}"
        );
        assert_eq!(
            mutable, fixture,
            "a no-op visitor leaves the tree unchanged"
        );
    }
}
