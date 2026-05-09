use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use chelis_deep::Span as DeepSpan;
use chelis_surf::ast::{Decl, Expr, ImportKind, LetPattern, MatchArm, Param, Pattern, TypeExpr};
use chelis_tide::compiler;
use chelis_tide::schema::{CheckRequest, Diagnostic as ApiDiagnostic, SourceKind};
use chelis_types::BUILTIN_NAMES;
use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, Location, MarkedString,
    Position, Range, Url,
};

#[derive(Debug, Clone)]
pub struct DocumentState {
    pub uri: Url,
    pub text: String,
    pub analysis: DocumentAnalysis,
}

#[derive(Debug, Clone)]
pub struct DocumentAnalysis {
    pub source_kind: SourceKind,
    pub diagnostics: Vec<Diagnostic>,
    pub score: Option<f64>,
    pub score_stale: bool,
    pub module_name: Option<String>,
    pub deep_view: Option<String>,
    pub definitions: Vec<Definition>,
    pub references: Vec<Reference>,
    pub completions: Vec<VisibleName>,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub name: String,
    pub range: Range,
    pub hover: String,
    pub target: DefinitionTarget,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub name: String,
    pub range: Range,
    pub hover: String,
    pub target: DefinitionTarget,
}

#[derive(Debug, Clone)]
pub struct VisibleName {
    pub name: String,
    pub detail: String,
    pub kind: CompletionItemKind,
    pub visible_in: Range,
}

#[derive(Debug, Clone)]
pub enum DefinitionTarget {
    CurrentDocument(Range),
    ModuleFile { module: String },
    ImportedName { module: String, name: String },
    Builtin,
}

#[derive(Debug, Clone)]
struct LocalBinding {
    name: String,
    definition: Range,
    visible_in: Range,
    hover: String,
}

#[derive(Debug, Clone)]
struct TopLevelSymbol {
    name: String,
    range: Range,
    hover: String,
    kind: CompletionItemKind,
}

#[derive(Debug, Clone, Default)]
struct TopLevelIndex {
    defs: HashMap<String, TopLevelSymbol>,
    exports: HashSet<String>,
    has_explicit_exports: bool,
    module_name: Option<String>,
    imports: HashMap<String, String>,
}

pub fn analyze_document(uri: &Url, text: &str) -> DocumentAnalysis {
    let source_kind = source_kind_from_uri(uri);
    match source_kind {
        SourceKind::Deep => analyze_deep_document(text),
        SourceKind::Surf => analyze_surf_document(text),
    }
}

pub fn hover_markdown(
    state: &DocumentState,
    position: Position,
) -> Option<tower_lsp::lsp_types::Hover> {
    let offset = position_to_offset(&state.text, position)?;
    if let Some(reference) = state
        .analysis
        .references
        .iter()
        .find(|reference| range_contains_offset(&state.text, reference.range, offset))
    {
        return Some(tower_lsp::lsp_types::Hover {
            contents: tower_lsp::lsp_types::HoverContents::Array(vec![MarkedString::String(
                reference.hover.clone(),
            )]),
            range: Some(reference.range),
        });
    }

    state
        .analysis
        .definitions
        .iter()
        .find(|definition| range_contains_offset(&state.text, definition.range, offset))
        .map(|definition| tower_lsp::lsp_types::Hover {
            contents: tower_lsp::lsp_types::HoverContents::Array(vec![MarkedString::String(
                definition.hover.clone(),
            )]),
            range: Some(definition.range),
        })
}

pub fn completion_items(state: &DocumentState, position: Position) -> Vec<CompletionItem> {
    let Some(offset) = position_to_offset(&state.text, position) else {
        return Vec::new();
    };

    let mut items = BTreeMap::new();
    for visible in &state.analysis.completions {
        if range_contains_offset(&state.text, visible.visible_in, offset) {
            items
                .entry(visible.name.clone())
                .or_insert_with(|| CompletionItem {
                    label: visible.name.clone(),
                    detail: Some(visible.detail.clone()),
                    kind: Some(visible.kind),
                    ..CompletionItem::default()
                });
        }
    }
    items.into_values().collect()
}

pub fn definition_location(
    state: &DocumentState,
    position: Position,
    workspace_root: Option<&Path>,
) -> Option<Location> {
    let offset = position_to_offset(&state.text, position)?;
    let target = state
        .analysis
        .references
        .iter()
        .find(|reference| range_contains_offset(&state.text, reference.range, offset))
        .map(|reference| &reference.target)
        .or_else(|| {
            state
                .analysis
                .definitions
                .iter()
                .find(|definition| range_contains_offset(&state.text, definition.range, offset))
                .map(|definition| &definition.target)
        })?;
    resolve_target_to_location(target, &state.uri, workspace_root)
}

pub fn deep_view(state: &DocumentState, _selection: Option<Range>) -> Result<String, String> {
    state.analysis.deep_view.clone().ok_or_else(|| {
        "Deep view is only available for Surf documents that parse successfully".to_string()
    })
}

pub fn document_status(state: &DocumentState) -> serde_json::Value {
    serde_json::json!({
        "score": state.analysis.score,
        "stale": state.analysis.score_stale,
        "source_kind": match state.analysis.source_kind {
            SourceKind::Surf => "surf",
            SourceKind::Deep => "deep",
        }
    })
}

fn analyze_surf_document(text: &str) -> DocumentAnalysis {
    let decls = match chelis_surf::parser::parse_str(text) {
        Ok(decls) => decls,
        Err(err) => {
            return DocumentAnalysis {
                source_kind: SourceKind::Surf,
                diagnostics: vec![parse_error_diagnostic(
                    text,
                    err.to_string(),
                    parse_error_offset(text, &err),
                )],
                score: None,
                score_stale: true,
                module_name: None,
                deep_view: None,
                definitions: Vec::new(),
                references: Vec::new(),
                completions: builtin_completions(full_document_range(text)),
            };
        }
    };

    let top_level = build_top_level_index(text, &decls);
    let mut definitions = Vec::new();
    let mut references = Vec::new();
    let mut completions = builtin_completions(full_document_range(text));

    for symbol in top_level.defs.values() {
        definitions.push(Definition {
            name: symbol.name.clone(),
            range: symbol.range,
            hover: symbol.hover.clone(),
            target: DefinitionTarget::CurrentDocument(symbol.range),
        });
        completions.push(VisibleName {
            name: symbol.name.clone(),
            detail: symbol.hover.clone(),
            kind: symbol.kind,
            visible_in: full_document_range(text),
        });
    }
    for (name, module) in &top_level.imports {
        completions.push(VisibleName {
            name: name.clone(),
            detail: format!("imported from {module}"),
            kind: CompletionItemKind::MODULE,
            visible_in: full_document_range(text),
        });
    }

    for decl in &decls {
        collect_decl_symbols(
            text,
            decl,
            &top_level,
            &mut Vec::new(),
            &mut definitions,
            &mut references,
            &mut completions,
        );
    }

    let envelope = compiler::result_envelope(compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: text.to_string(),
    }));
    let (score, score_stale, diagnostics) = match envelope {
        chelis_tide::schema::ApiEnvelope::Success(success) => {
            let result = success.result;
            (
                Some(result.score),
                false,
                diagnostics_from_api(text, &result.errors, first_decl_range(text, &decls)),
            )
        }
        chelis_tide::schema::ApiEnvelope::Failure(failure) => (
            None,
            true,
            diagnostics_from_api(text, &failure.errors, first_decl_range(text, &decls)),
        ),
    };

    let deep_view = Some(chelis_deep::printer::print_canonical(
        &chelis_surf::desugar::desugar_program(&decls),
    ));

    DocumentAnalysis {
        source_kind: SourceKind::Surf,
        diagnostics,
        score,
        score_stale,
        module_name: top_level.module_name,
        deep_view,
        definitions,
        references,
        completions,
    }
}

fn analyze_deep_document(text: &str) -> DocumentAnalysis {
    let diagnostics = match chelis_deep::parser::parse_str(text) {
        Ok(_) => Vec::new(),
        Err(err) => vec![Diagnostic {
            range: range_for_offset(text, parse_error_offset_deep(&err)),
            severity: Some(DiagnosticSeverity::ERROR),
            message: err.to_string(),
            source: Some("chelis".to_string()),
            ..Diagnostic::default()
        }],
    };

    DocumentAnalysis {
        source_kind: SourceKind::Deep,
        diagnostics,
        score: None,
        score_stale: false,
        module_name: None,
        deep_view: None,
        definitions: Vec::new(),
        references: Vec::new(),
        completions: Vec::new(),
    }
}

fn build_top_level_index(text: &str, decls: &[Decl]) -> TopLevelIndex {
    let mut index = TopLevelIndex::default();
    for decl in decls {
        build_top_level_index_decl(text, decl, &mut index);
    }
    index
}

fn build_top_level_index_decl(text: &str, decl: &Decl, index: &mut TopLevelIndex) {
    match decl {
        Decl::Module { name, decls, .. } => {
            index.module_name = Some(name.clone());
            for decl in decls {
                build_top_level_index_decl(text, decl, index);
            }
        }
        Decl::Import { module, kind, .. } => {
            if let ImportKind::Names(names) = kind {
                for name in names {
                    index.imports.insert(name.clone(), module.clone());
                }
            }
        }
        Decl::Sig { name, ty, span, .. } => {
            index.defs.insert(
                name.clone(),
                TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover: format!("sig {}: {}", name, format_type_expr(ty)),
                    kind: CompletionItemKind::FUNCTION,
                },
            );
        }
        Decl::TypeAlias {
            name,
            params,
            ty,
            span,
        } => {
            index.defs.insert(
                name.clone(),
                TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover: format!(
                        "type {}{} = {}",
                        name,
                        format_type_params(params),
                        format_type_expr(ty)
                    ),
                    kind: CompletionItemKind::STRUCT,
                },
            );
        }
        Decl::TypeDef {
            name,
            params,
            variants,
            span,
        } => {
            let mut hover = format!("type {}{}", name, format_type_params(params));
            if !variants.is_empty() {
                let body = variants
                    .iter()
                    .map(|variant| variant.name.clone())
                    .collect::<Vec<_>>()
                    .join(" | ");
                hover.push_str(&format!(" = {body}"));
            }
            index.defs.insert(
                name.clone(),
                TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover,
                    kind: CompletionItemKind::ENUM,
                },
            );
        }
        Decl::FunDef {
            name,
            params,
            ret_ty,
            span,
            ..
        } => {
            let entry = index
                .defs
                .entry(name.clone())
                .or_insert_with(|| TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover: format!(
                        "def {}({}){}",
                        name,
                        format_params(params),
                        ret_ty
                            .as_ref()
                            .map(|ret_ty| format!(": {}", format_type_expr(ret_ty)))
                            .unwrap_or_default()
                    ),
                    kind: CompletionItemKind::FUNCTION,
                });
            if entry.hover.starts_with("sig ") {
                entry.range = range_for_span(text, *span);
            }
        }
        Decl::LetDef { name, ty, span, .. } => {
            index.defs.insert(
                name.clone(),
                TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover: ty
                        .as_ref()
                        .map(|ty| format!("{name}: {}", format_type_expr(ty)))
                        .unwrap_or_else(|| name.clone()),
                    kind: CompletionItemKind::VARIABLE,
                },
            );
        }
        Decl::MacroDef { name, span, .. } => {
            index.defs.insert(
                name.clone(),
                TopLevelSymbol {
                    name: name.clone(),
                    range: range_for_span(text, *span),
                    hover: format!("macro {name}(...)"),
                    kind: CompletionItemKind::FUNCTION,
                },
            );
        }
        Decl::Export { names, .. } => {
            index.has_explicit_exports = true;
            for name in names {
                index.exports.insert(name.clone());
            }
        }
        Decl::Dim { .. } => {}
    }
}

fn collect_decl_symbols(
    text: &str,
    decl: &Decl,
    top_level: &TopLevelIndex,
    locals: &mut Vec<LocalBinding>,
    definitions: &mut Vec<Definition>,
    references: &mut Vec<Reference>,
    completions: &mut Vec<VisibleName>,
) {
    match decl {
        Decl::Module { decls, .. } => {
            for child in decls {
                collect_decl_symbols(
                    text,
                    child,
                    top_level,
                    locals,
                    definitions,
                    references,
                    completions,
                );
            }
        }
        Decl::FunDef {
            name: _,
            params,
            body,
            span,
            ..
        } => {
            let body_range = range_for_expr(text, body);
            let start_len = locals.len();
            for param in params {
                let def_range = range_for_span(text, param.span);
                let hover = format!(
                    "param {}{}",
                    param.name,
                    param
                        .ty
                        .as_ref()
                        .map(|ty| format!(": {}", format_type_expr(ty)))
                        .unwrap_or_default()
                );
                locals.push(LocalBinding {
                    name: param.name.clone(),
                    definition: def_range,
                    visible_in: body_range,
                    hover: hover.clone(),
                });
                definitions.push(Definition {
                    name: param.name.clone(),
                    range: def_range,
                    hover: hover.clone(),
                    target: DefinitionTarget::CurrentDocument(def_range),
                });
                completions.push(VisibleName {
                    name: param.name.clone(),
                    detail: hover,
                    kind: CompletionItemKind::VARIABLE,
                    visible_in: body_range,
                });
            }
            collect_expr_symbols(
                text,
                body,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            locals.truncate(start_len);
            let fun_range = range_for_span(text, *span);
            definitions.push(Definition {
                name: function_name_from_decl(decl).to_string(),
                range: fun_range,
                hover: top_level
                    .defs
                    .get(function_name_from_decl(decl))
                    .map(|symbol| symbol.hover.clone())
                    .unwrap_or_else(|| function_name_from_decl(decl).to_string()),
                target: DefinitionTarget::CurrentDocument(fun_range),
            });
        }
        Decl::LetDef {
            name,
            value,
            span,
            ty,
        } => {
            collect_expr_symbols(
                text,
                value,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            let range = range_for_span(text, *span);
            let hover = ty
                .as_ref()
                .map(|ty| format!("{name}: {}", format_type_expr(ty)))
                .unwrap_or_else(|| name.clone());
            definitions.push(Definition {
                name: name.clone(),
                range,
                hover,
                target: DefinitionTarget::CurrentDocument(range),
            });
        }
        _ => {}
    }
}

fn collect_expr_symbols(
    text: &str,
    expr: &Expr,
    top_level: &TopLevelIndex,
    locals: &mut Vec<LocalBinding>,
    references: &mut Vec<Reference>,
    definitions: &mut Vec<Definition>,
    completions: &mut Vec<VisibleName>,
) {
    match expr {
        Expr::Var(name, span) => {
            let range = range_for_span(text, *span);
            let (target, hover) = resolve_name(name, range.start, top_level, locals);
            references.push(Reference {
                name: name.clone(),
                range,
                hover,
                target,
            });
        }
        Expr::Apply(func, args, _) => {
            collect_expr_symbols(
                text,
                func,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            for arg in args {
                collect_expr_symbols(
                    text,
                    arg,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::List(items, _) => {
            for item in items {
                collect_expr_symbols(
                    text,
                    item,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::Record(_, fields, _) => {
            for (_, value) in fields {
                collect_expr_symbols(
                    text,
                    value,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::Access(base, _, _)
        | Expr::TupleGet(base, _, _)
        | Expr::Borrow(base, _)
        | Expr::Unary(_, base, _)
        | Expr::Grad(base, _, _)
        | Expr::Jit(base, _)
        | Expr::Realize(base, _)
        | Expr::Copy(base, _)
        | Expr::Cast(base, _, _)
        | Expr::Annotate(base, _, _) => {
            collect_expr_symbols(
                text,
                base,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::WithSeed(body, seed, _) => {
            collect_expr_symbols(
                text,
                body,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            collect_expr_symbols(
                text,
                seed,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::WithDevice(body, device, _) => {
            collect_expr_symbols(
                text,
                body,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            collect_expr_symbols(
                text,
                device,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::Binary(_, lhs, rhs, _) => {
            collect_expr_symbols(
                text,
                lhs,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            collect_expr_symbols(
                text,
                rhs,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::Pipe(lhs, stages, _) => {
            collect_expr_symbols(
                text,
                lhs,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            for stage in stages {
                collect_expr_symbols(
                    text,
                    stage,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::If(cond, then_branch, else_branch, _) => {
            collect_expr_symbols(
                text,
                cond,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            collect_expr_symbols(
                text,
                then_branch,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            collect_expr_symbols(
                text,
                else_branch,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::Match(value, arms, _) => {
            collect_expr_symbols(
                text,
                value,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            for arm in arms {
                collect_match_arm_symbols(
                    text,
                    arm,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::Block(bindings, body, _) => {
            let start_len = locals.len();
            for binding in bindings {
                collect_expr_symbols(
                    text,
                    &binding.value,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
                let pattern_defs = pattern_definitions(text, &binding.pattern, body);
                for (name, def_range, visible_in) in pattern_defs {
                    let hover = binding
                        .ty
                        .as_ref()
                        .map(|ty| format!("{name}: {}", format_type_expr(ty)))
                        .unwrap_or_else(|| name.clone());
                    locals.push(LocalBinding {
                        name: name.clone(),
                        definition: def_range,
                        visible_in,
                        hover: hover.clone(),
                    });
                    definitions.push(Definition {
                        name: name.clone(),
                        range: def_range,
                        hover: hover.clone(),
                        target: DefinitionTarget::CurrentDocument(def_range),
                    });
                    completions.push(VisibleName {
                        name,
                        detail: hover,
                        kind: CompletionItemKind::VARIABLE,
                        visible_in,
                    });
                }
            }
            collect_expr_symbols(
                text,
                body,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            locals.truncate(start_len);
        }
        Expr::Lambda(params, body, _) => {
            let start_len = locals.len();
            let body_range = range_for_expr(text, body);
            for param in params {
                let def_range = range_for_span(text, param.span);
                let hover = format!(
                    "param {}{}",
                    param.name,
                    param
                        .ty
                        .as_ref()
                        .map(|ty| format!(": {}", format_type_expr(ty)))
                        .unwrap_or_default()
                );
                locals.push(LocalBinding {
                    name: param.name.clone(),
                    definition: def_range,
                    visible_in: body_range,
                    hover: hover.clone(),
                });
                definitions.push(Definition {
                    name: param.name.clone(),
                    range: def_range,
                    hover: hover.clone(),
                    target: DefinitionTarget::CurrentDocument(def_range),
                });
                completions.push(VisibleName {
                    name: param.name.clone(),
                    detail: hover,
                    kind: CompletionItemKind::VARIABLE,
                    visible_in: body_range,
                });
            }
            collect_expr_symbols(
                text,
                body,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
            locals.truncate(start_len);
        }
        Expr::Tuple(items, _) | Expr::Par(items, _) => {
            for item in items {
                collect_expr_symbols(
                    text,
                    item,
                    top_level,
                    locals,
                    references,
                    definitions,
                    completions,
                );
            }
        }
        Expr::Vmap(expr, _, _) => {
            collect_expr_symbols(
                text,
                expr,
                top_level,
                locals,
                references,
                definitions,
                completions,
            );
        }
        Expr::Lit(_, _) | Expr::Constructor(_, _) => {}
    }
}

fn collect_match_arm_symbols(
    text: &str,
    arm: &MatchArm,
    top_level: &TopLevelIndex,
    locals: &mut Vec<LocalBinding>,
    references: &mut Vec<Reference>,
    definitions: &mut Vec<Definition>,
    completions: &mut Vec<VisibleName>,
) {
    let start_len = locals.len();
    for (name, range, visible_in) in pattern_bindings(text, &arm.pattern, arm.body.clone()) {
        let hover = format!("match binding {name}");
        locals.push(LocalBinding {
            name: name.clone(),
            definition: range,
            visible_in,
            hover: hover.clone(),
        });
        definitions.push(Definition {
            name: name.clone(),
            range,
            hover: hover.clone(),
            target: DefinitionTarget::CurrentDocument(range),
        });
        completions.push(VisibleName {
            name,
            detail: hover,
            kind: CompletionItemKind::VARIABLE,
            visible_in,
        });
    }
    if let Some(guard) = &arm.guard {
        collect_expr_symbols(
            text,
            guard,
            top_level,
            locals,
            references,
            definitions,
            completions,
        );
    }
    collect_expr_symbols(
        text,
        &arm.body,
        top_level,
        locals,
        references,
        definitions,
        completions,
    );
    locals.truncate(start_len);
}

fn pattern_definitions(
    text: &str,
    pattern: &LetPattern,
    body: &Expr,
) -> Vec<(String, Range, Range)> {
    let visible_in = range_for_expr(text, body);
    match pattern {
        LetPattern::Var(name, span) => {
            vec![(name.clone(), range_for_span(text, *span), visible_in)]
        }
        LetPattern::Tuple(items, _) => items
            .iter()
            .flat_map(|item| pattern_definitions(text, item, body))
            .collect(),
        LetPattern::Wildcard(_) => Vec::new(),
    }
}

fn pattern_bindings(text: &str, pattern: &Pattern, body: Expr) -> Vec<(String, Range, Range)> {
    let visible_in = range_for_expr(text, &body);
    match pattern {
        Pattern::Var(name, span) | Pattern::As(name, _, span) => {
            vec![(name.clone(), range_for_span(text, *span), visible_in)]
        }
        Pattern::Constructor(_, patterns, _) | Pattern::Tuple(patterns, _) => patterns
            .iter()
            .flat_map(|pattern| pattern_bindings(text, pattern, body.clone()))
            .collect(),
        Pattern::Record(_, fields, _) => fields
            .iter()
            .flat_map(|(_, pattern)| pattern_bindings(text, pattern, body.clone()))
            .collect(),
        Pattern::Wildcard(_) | Pattern::Lit(_, _) => Vec::new(),
    }
}

fn resolve_name(
    name: &str,
    position: Position,
    top_level: &TopLevelIndex,
    locals: &[LocalBinding],
) -> (DefinitionTarget, String) {
    for binding in locals.iter().rev() {
        if binding.name == name && position_in_range(position, binding.visible_in) {
            return (
                DefinitionTarget::CurrentDocument(binding.definition),
                binding.hover.clone(),
            );
        }
    }

    if let Some(symbol) = top_level.defs.get(name) {
        return (
            DefinitionTarget::CurrentDocument(symbol.range),
            symbol.hover.clone(),
        );
    }

    if let Some(module) = top_level.imports.get(name) {
        return (
            DefinitionTarget::ImportedName {
                module: module.clone(),
                name: name.to_string(),
            },
            format!("imported from {module}"),
        );
    }

    if BUILTIN_NAMES.contains(&name) {
        return (DefinitionTarget::Builtin, format!("builtin {name}"));
    }

    (DefinitionTarget::Builtin, name.to_string())
}

fn diagnostics_from_api(
    text: &str,
    diagnostics: &[ApiDiagnostic],
    fallback: Option<Range>,
) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| Diagnostic {
            range: diagnostic
                .span
                .map(|span| range_for_span(text, DeepSpan::new(span.offset, span.len)))
                .or(fallback)
                .unwrap_or_else(|| full_document_range(text)),
            severity: Some(severity(diagnostic.severity)),
            message: diagnostic.message.clone(),
            source: Some("chelis".to_string()),
            ..Diagnostic::default()
        })
        .collect()
}

fn parse_error_diagnostic(text: &str, message: String, offset: usize) -> Diagnostic {
    Diagnostic {
        range: range_for_offset(text, offset),
        severity: Some(DiagnosticSeverity::ERROR),
        message,
        source: Some("chelis".to_string()),
        ..Diagnostic::default()
    }
}

fn severity(value: f64) -> DiagnosticSeverity {
    if value >= 0.8 {
        DiagnosticSeverity::ERROR
    } else if value >= 0.6 {
        DiagnosticSeverity::WARNING
    } else {
        DiagnosticSeverity::INFORMATION
    }
}

fn resolve_target_to_location(
    target: &DefinitionTarget,
    current_uri: &Url,
    workspace_root: Option<&Path>,
) -> Option<Location> {
    match target {
        DefinitionTarget::CurrentDocument(range) => Some(Location {
            uri: current_uri.clone(),
            range: *range,
        }),
        DefinitionTarget::ModuleFile { module } => {
            let path = module_path(workspace_root?, module)?;
            let uri = Url::from_file_path(path).ok()?;
            Some(Location {
                uri,
                range: Range::default(),
            })
        }
        DefinitionTarget::ImportedName { module, name } => {
            let path = module_path(workspace_root?, module)?;
            let contents = fs::read_to_string(&path).ok()?;
            let decls = chelis_surf::parser::parse_str(&contents).ok()?;
            let index = build_top_level_index(&contents, &decls);
            if !symbol_is_public(&index, name) {
                return None;
            }
            let symbol = index.defs.get(name)?;
            Some(Location {
                uri: Url::from_file_path(path).ok()?,
                range: symbol.range,
            })
        }
        DefinitionTarget::Builtin => None,
    }
}

fn symbol_is_public(index: &TopLevelIndex, name: &str) -> bool {
    if !index.has_explicit_exports {
        return true;
    }
    index.exports.contains(name)
}

fn module_path(root: &Path, module: &str) -> Option<PathBuf> {
    let mut path = root.to_path_buf();
    for segment in module.split('.') {
        path.push(segment.to_ascii_lowercase());
    }
    path.set_extension("ch");
    path.exists().then_some(path)
}

fn function_name_from_decl(decl: &Decl) -> &str {
    match decl {
        Decl::FunDef { name, .. } => name,
        _ => "",
    }
}

fn builtin_completions(visible_in: Range) -> Vec<VisibleName> {
    let mut items = BUILTIN_NAMES
        .iter()
        .map(|name| VisibleName {
            name: (*name).to_string(),
            detail: format!("builtin {name}"),
            kind: CompletionItemKind::FUNCTION,
            visible_in,
        })
        .collect::<Vec<_>>();
    items.extend([
        keyword_completion("def", visible_in),
        keyword_completion("type", visible_in),
        keyword_completion("match", visible_in),
        keyword_completion("fn", visible_in),
        keyword_completion("module", visible_in),
        keyword_completion("import", visible_in),
    ]);
    items
}

fn keyword_completion(name: &str, visible_in: Range) -> VisibleName {
    VisibleName {
        name: name.to_string(),
        detail: "keyword".to_string(),
        kind: CompletionItemKind::KEYWORD,
        visible_in,
    }
}

pub fn source_kind_from_uri(uri: &Url) -> SourceKind {
    match uri.path().rsplit('.').next() {
        Some("dp") => SourceKind::Deep,
        _ => SourceKind::Surf,
    }
}

pub fn full_document_range(text: &str) -> Range {
    let end = offset_to_position(text, text.len());
    Range::new(Position::new(0, 0), end)
}

fn first_decl_range(text: &str, decls: &[Decl]) -> Option<Range> {
    decls.first().map(|decl| match decl {
        Decl::Module { span, .. }
        | Decl::Import { span, .. }
        | Decl::Sig { span, .. }
        | Decl::Dim { span, .. }
        | Decl::TypeDef { span, .. }
        | Decl::TypeAlias { span, .. }
        | Decl::FunDef { span, .. }
        | Decl::LetDef { span, .. }
        | Decl::Export { span, .. }
        | Decl::MacroDef { span, .. } => range_for_span(text, *span),
    })
}

fn parse_error_offset(text: &str, err: &chelis_surf::parser::ParseError) -> usize {
    match err {
        chelis_surf::parser::ParseError::Lex(_) => 0,
        chelis_surf::parser::ParseError::UnexpectedEof => text.len(),
        chelis_surf::parser::ParseError::Expected { offset, .. }
        | chelis_surf::parser::ParseError::NonAssocChain { offset } => *offset,
    }
}

fn parse_error_offset_deep(err: &chelis_deep::parser::ParseError) -> usize {
    match err {
        chelis_deep::parser::ParseError::Lex(_) => 0,
        chelis_deep::parser::ParseError::UnexpectedEof { offset }
        | chelis_deep::parser::ParseError::Expected { offset, .. }
        | chelis_deep::parser::ParseError::EmptyList { offset } => *offset,
        chelis_deep::parser::ParseError::ForbiddenSpanChar { value_offset, .. } => *value_offset,
    }
}

fn range_for_expr(text: &str, expr: &Expr) -> Range {
    let span = match expr {
        Expr::Lit(_, span)
        | Expr::Var(_, span)
        | Expr::Constructor(_, span)
        | Expr::Apply(_, _, span)
        | Expr::List(_, span)
        | Expr::Record(_, _, span)
        | Expr::Access(_, _, span)
        | Expr::TupleGet(_, _, span)
        | Expr::Binary(_, _, _, span)
        | Expr::Unary(_, _, span)
        | Expr::Pipe(_, _, span)
        | Expr::If(_, _, _, span)
        | Expr::Match(_, _, span)
        | Expr::Lambda(_, _, span)
        | Expr::Tuple(_, span)
        | Expr::Cast(_, _, span)
        | Expr::Grad(_, _, span)
        | Expr::Vmap(_, _, span)
        | Expr::Jit(_, span)
        | Expr::Realize(_, span)
        | Expr::Copy(_, span)
        | Expr::Borrow(_, span)
        | Expr::WithSeed(_, _, span)
        | Expr::WithDevice(_, _, span)
        | Expr::Par(_, span)
        | Expr::Annotate(_, _, span)
        | Expr::Block(_, _, span) => *span,
    };
    range_for_span(text, span)
}

fn range_for_span(text: &str, span: DeepSpan) -> Range {
    let start = offset_to_position(text, span.offset);
    let end = offset_to_position(text, span.end());
    Range::new(start, end)
}

fn range_for_offset(text: &str, offset: usize) -> Range {
    let position = offset_to_position(text, offset);
    Range::new(position, position)
}

pub fn offset_to_position(text: &str, offset: usize) -> Position {
    let clamped = offset.min(text.len());
    let mut line = 0u32;
    let mut column = 0u32;
    let mut seen = 0usize;
    for ch in text.chars() {
        if seen >= clamped {
            break;
        }
        let len = ch.len_utf8();
        if seen + len > clamped {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 0;
        } else {
            column += ch.len_utf16() as u32;
        }
        seen += len;
    }
    Position::new(line, column)
}

pub fn position_to_offset(text: &str, position: Position) -> Option<usize> {
    let mut line = 0u32;
    let mut column = 0u32;
    let mut offset = 0usize;
    for ch in text.chars() {
        if line == position.line && column == position.character {
            return Some(offset);
        }
        if ch == '\n' {
            if line == position.line {
                return Some(offset);
            }
            line += 1;
            column = 0;
            offset += 1;
            continue;
        }
        if line == position.line {
            column += ch.len_utf16() as u32;
            if column > position.character {
                return Some(offset);
            }
        }
        offset += ch.len_utf8();
    }
    if line == position.line {
        Some(offset)
    } else {
        None
    }
}

fn range_contains_offset(text: &str, range: Range, offset: usize) -> bool {
    let Some(start) = position_to_offset(text, range.start) else {
        return false;
    };
    let Some(end) = position_to_offset(text, range.end) else {
        return false;
    };
    start <= offset && offset <= end
}

fn position_in_range(position: Position, range: Range) -> bool {
    (position.line > range.start.line
        || (position.line == range.start.line && position.character >= range.start.character))
        && (position.line < range.end.line
            || (position.line == range.end.line && position.character <= range.end.character))
}

fn format_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            if let Some(ty) = &param.ty {
                format!("{}: {}", param.name, format_type_expr(ty))
            } else {
                param.name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_type_params(params: &[String]) -> String {
    if params.is_empty() {
        String::new()
    } else {
        format!("[{}]", params.join(", "))
    }
}

fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(name, _) => name.clone(),
        TypeExpr::Tensor(items, precision, _) => {
            let inner = items
                .iter()
                .map(format_type_expr)
                .chain([precision.clone()])
                .collect::<Vec<_>>()
                .join(", ");
            format!("tensor[{inner}]")
        }
        TypeExpr::Arrow(args, ret, _) => {
            let args = args
                .iter()
                .map(format_type_expr)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({args}) -> {}", format_type_expr(ret))
        }
        TypeExpr::Ref(inner, _) => format!("&{}", format_type_expr(inner)),
        TypeExpr::App(name, args, _) => {
            let args = args
                .iter()
                .map(format_type_expr)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{name}[{args}]")
        }
        TypeExpr::Tuple(items, _) => {
            let items = items
                .iter()
                .map(format_type_expr)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({items})")
        }
        TypeExpr::Infer(_) => "_".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn surf_uri() -> Url {
        Url::parse("file:///tmp/test.ch").expect("uri")
    }

    #[test]
    fn surf_analysis_reports_parse_error_range() {
        let analysis = analyze_document(&surf_uri(), "def f(x: = x\n");
        assert_eq!(analysis.source_kind, SourceKind::Surf);
        assert_eq!(analysis.diagnostics.len(), 1);
        assert_eq!(
            analysis.diagnostics[0].severity,
            Some(DiagnosticSeverity::ERROR)
        );
    }

    #[test]
    fn completions_include_builtins_and_locals() {
        let text = "def f(x: f32): f32 = {\n  y = x\n  add(y, x)\n}\n";
        let state = DocumentState {
            uri: surf_uri(),
            text: text.to_string(),
            analysis: analyze_document(&surf_uri(), text),
        };
        let items = completion_items(&state, Position::new(2, 4));
        let labels = items.into_iter().map(|item| item.label).collect::<Vec<_>>();
        assert!(labels.iter().any(|label| label == "add"));
    }

    #[test]
    fn completions_include_imported_names() {
        let text = "module Main\nimport Foo.Bar (baz)\ndef f(x: f32): f32 = b\n";
        let state = DocumentState {
            uri: surf_uri(),
            text: text.to_string(),
            analysis: analyze_document(&surf_uri(), text),
        };
        let items = completion_items(&state, Position::new(2, 24));
        let labels = items.into_iter().map(|item| item.label).collect::<Vec<_>>();
        assert!(labels.iter().any(|label| label == "baz"));
    }

    #[test]
    fn definition_resolves_workspace_imports() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("foo")).expect("mkdir");
        fs::write(
            root.join("foo/bar.ch"),
            "module Foo.Bar\nexport (baz)\ndef baz(x: f32): f32 = x\n",
        )
        .expect("write");
        let uri = Url::from_file_path(root.join("main.ch")).expect("uri");
        let text = "module Main\nimport Foo.Bar (baz)\ndef use_it(x: f32): f32 = baz(x)\n";
        let state = DocumentState {
            uri: uri.clone(),
            text: text.to_string(),
            analysis: analyze_document(&uri, text),
        };
        let position = Position::new(2, 27);
        let location = definition_location(&state, position, Some(root)).expect("location");
        assert!(location.uri.path().ends_with("/foo/bar.ch"));
    }
}
