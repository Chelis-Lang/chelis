use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct ExpansionOptions {
    pub max_iterations: usize,
    pub load_std_prelude: bool,
}

impl Default for ExpansionOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            load_std_prelude: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpandedProgram {
    exprs: Vec<Expr>,
    expansions: usize,
}

impl ExpandedProgram {
    pub fn exprs(&self) -> &[Expr] {
        &self.exprs
    }

    pub fn into_exprs(self) -> Vec<Expr> {
        self.exprs
    }

    pub fn expansions(&self) -> usize {
        self.expansions
    }
}

#[derive(Debug, Error)]
pub enum ExpansionError {
    #[error("macro expansion limit exceeded after {limit} expansions")]
    ExpansionLimitExceeded { limit: usize },

    #[error("malformed macro definition: {message}")]
    MalformedDefinition { message: String },

    #[error("failed to load standard macro prelude: {message}")]
    PreludeLoad { message: String },
}

#[derive(Debug, Clone)]
struct MacroDef {
    name: String,
    params: Vec<String>,
    body: Expr,
}

pub fn expand_program(
    exprs: &[Expr],
    options: &ExpansionOptions,
) -> Result<ExpandedProgram, ExpansionError> {
    let prelude = if options.load_std_prelude {
        standard_prelude_macros()?
    } else {
        HashMap::new()
    };
    let mut expander = Expander::new(options.max_iterations, prelude);
    let exprs = expander.expand_sequence(exprs, &HashMap::new())?;
    Ok(ExpandedProgram {
        exprs,
        expansions: expander.expansions,
    })
}

struct Expander {
    remaining_expansions: usize,
    hygiene_counter: usize,
    expansions: usize,
    prelude_macros: HashMap<String, MacroDef>,
}

impl Expander {
    fn new(limit: usize, prelude_macros: HashMap<String, MacroDef>) -> Self {
        Self {
            remaining_expansions: limit,
            hygiene_counter: 0,
            expansions: 0,
            prelude_macros,
        }
    }

    fn expand_sequence(
        &mut self,
        exprs: &[Expr],
        inherited_macros: &HashMap<String, MacroDef>,
    ) -> Result<Vec<Expr>, ExpansionError> {
        let mut macros = inherited_macros.clone();
        for (name, def) in &self.prelude_macros {
            macros.entry(name.clone()).or_insert_with(|| def.clone());
        }

        let mut user_macros = HashMap::new();
        for expr in exprs {
            if let Some(def) = extract_macro_def(expr)? {
                user_macros.insert(def.name.clone(), def);
            }
        }
        macros.extend(user_macros);

        let mut out = Vec::new();
        for expr in exprs {
            if extract_macro_def(expr)?.is_some() {
                continue;
            }
            out.push(self.expand_expr(expr, &macros, &Scope::default())?);
        }
        Ok(out)
    }

    fn expand_expr(
        &mut self,
        expr: &Expr,
        macros: &HashMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Expr, ExpansionError> {
        match expr {
            Expr::Atom(_, _) | Expr::Map(_, _) => Ok(expr.clone()),
            // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
            Expr::Node(node, span) => {
                let bridged = Expr::List(node.to_list(*span), *span);
                self.expand_expr(&bridged, macros, scope)
            }
            Expr::BareList(_, _) | Expr::UnknownForm(_) => Ok(expr.clone()),
            Expr::MetaExpr(meta, span) => Ok(Expr::MetaExpr(
                MetaExpr {
                    entries: meta.entries.clone(),
                    expr: Box::new(self.expand_expr(&meta.expr, macros, scope)?),
                },
                *span,
            )),
            Expr::List(list, span) => {
                if let Some(expanded) = self.try_expand_macro_call(list, macros, scope)? {
                    return self.expand_expr(&expanded, macros, scope);
                }

                let Some(tag) = get_tag(list) else {
                    return Ok(Expr::List(
                        List {
                            elements: list
                                .elements
                                .iter()
                                .map(|child| self.expand_expr(child, macros, scope))
                                .collect::<Result<Vec<_>, _>>()?,
                        },
                        *span,
                    ));
                };

                match tag {
                    DeepTag::Module => self.expand_module(list, macros, *span),
                    DeepTag::Fn => self.expand_fn(list, macros, scope, *span),
                    DeepTag::Let => self.expand_let(list, macros, scope, *span),
                    DeepTag::Match => self.expand_match(list, macros, scope, *span),
                    _ => Ok(Expr::List(
                        List {
                            elements: list
                                .elements
                                .iter()
                                .map(|child| self.expand_expr(child, macros, scope))
                                .collect::<Result<Vec<_>, _>>()?,
                        },
                        *span,
                    )),
                }
            }
        }
    }

    fn expand_module(
        &mut self,
        list: &List,
        macros: &HashMap<String, MacroDef>,
        span: Span,
    ) -> Result<Expr, ExpansionError> {
        let kids = children(list);
        if kids.is_empty() {
            return Ok(Expr::List(list.clone(), span));
        }
        let mut expanded = vec![
            list.elements[0].clone(),
            list.elements[1].clone(),
            kids[0].clone(),
        ];
        let body = self.expand_sequence(&kids[1..], macros)?;
        expanded.extend(body);
        Ok(Expr::List(List { elements: expanded }, span))
    }

    fn expand_fn(
        &mut self,
        list: &List,
        macros: &HashMap<String, MacroDef>,
        scope: &Scope,
        span: Span,
    ) -> Result<Expr, ExpansionError> {
        let mut elements = list.elements.clone();
        let kids = children(list);
        if kids.len() < 2 {
            return Ok(Expr::List(List { elements }, span));
        }

        let params_expr = kids[0].clone();
        let blocker_names = params_blockers(&params_expr);
        let fn_scope = scope.with_blockers(&blocker_names);
        elements[2] = params_expr;
        elements[3] = self.expand_expr(&kids[1], macros, &fn_scope)?;
        Ok(Expr::List(List { elements }, span))
    }

    fn expand_let(
        &mut self,
        list: &List,
        macros: &HashMap<String, MacroDef>,
        scope: &Scope,
        span: Span,
    ) -> Result<Expr, ExpansionError> {
        let kids = children(list);
        if kids.len() < 2 {
            return Ok(Expr::List(list.clone(), span));
        }
        let mut elements = list.elements.clone();
        if let Expr::List(bind_list, bind_span) = &kids[0] {
            let bind_kids = children(bind_list);
            let mut scope_for_values = scope.clone();
            let mut new_bind_children = Vec::new();
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                let name_expr = bind_kids[i].clone();
                let value_expr = self.expand_expr(&bind_kids[i + 1], macros, &scope_for_values)?;
                if let Some(name) = symbol_name(&bind_kids[i]) {
                    scope_for_values.add_blocker(name.to_string());
                }
                new_bind_children.push(name_expr);
                new_bind_children.push(value_expr);
                i += 2;
            }
            elements[2] = node_with_meta(
                DeepTag::Bind,
                list.elements[1].clone(),
                new_bind_children,
                *bind_span,
            );
            elements[3] = self.expand_expr(&kids[1], macros, &scope_for_values)?;
        }
        Ok(Expr::List(List { elements }, span))
    }

    fn expand_match(
        &mut self,
        list: &List,
        macros: &HashMap<String, MacroDef>,
        scope: &Scope,
        span: Span,
    ) -> Result<Expr, ExpansionError> {
        let kids = children(list);
        if kids.is_empty() {
            return Ok(Expr::List(list.clone(), span));
        }
        let mut elements = Vec::with_capacity(list.elements.len());
        elements.push(list.elements[0].clone());
        elements.push(list.elements[1].clone());
        elements.push(self.expand_expr(&kids[0], macros, scope)?);
        for arm in &kids[1..] {
            if let Expr::List(arm_list, arm_span) = arm
                && get_tag(arm_list) == Some(DeepTag::Arm)
            {
                let arm_kids = children(arm_list);
                if arm_kids.len() >= 3 {
                    let mut arm_scope = scope.clone();
                    arm_scope.add_blockers(pattern_binders(&arm_kids[0]));
                    let mut arm_elements = arm_list.elements.clone();
                    arm_elements[4] = self.expand_expr(&arm_kids[2], macros, &arm_scope)?;
                    if !is_unit_list(&arm_kids[1]) {
                        arm_elements[3] = self.expand_expr(&arm_kids[1], macros, &arm_scope)?;
                    }
                    elements.push(Expr::List(
                        List {
                            elements: arm_elements,
                        },
                        *arm_span,
                    ));
                    continue;
                }
            }
            elements.push(self.expand_expr(arm, macros, scope)?);
        }
        Ok(Expr::List(List { elements }, span))
    }

    fn try_expand_macro_call(
        &mut self,
        list: &List,
        macros: &HashMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Option<Expr>, ExpansionError> {
        let (name, args) = if internal_tag(list) == Some("macro-invoke") {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                return Ok(None);
            };
            (name.to_string(), kids[1..].to_vec())
        } else {
            match get_tag(list) {
                Some(DeepTag::App) => {
                    let kids = children(list);
                    if kids.is_empty() {
                        return Ok(None);
                    }
                    let Some(name) = var_name(&kids[0]) else {
                        return Ok(None);
                    };
                    (name.to_string(), kids[1..].to_vec())
                }
                Some(_) | None => return Ok(None),
            }
        };

        if scope.blocks(&name) {
            return Ok(None);
        }
        let Some(def) = macros.get(&name) else {
            return Ok(None);
        };
        self.consume_expansion_budget()?;
        let invocation = macro_source(&def.name, &args);
        let (placeholder_params, placeholder_args) =
            macro_arg_placeholders(def, &args, self.expansions);
        let placeholder_body = substitute_expr(&def.body, &placeholder_params, &HashSet::new());
        let hygienic = hygienize_expr(
            &placeholder_body,
            &mut self.hygiene_counter,
            &HashMap::new(),
        );
        let substituted = replace_placeholder_vars(&hygienic, &placeholder_args);
        Ok(Some(annotate_source_expr(&substituted, &invocation)))
    }

    fn consume_expansion_budget(&mut self) -> Result<(), ExpansionError> {
        if self.remaining_expansions == 0 {
            return Err(ExpansionError::ExpansionLimitExceeded {
                limit: self.expansions,
            });
        }
        self.remaining_expansions -= 1;
        self.expansions += 1;
        Ok(())
    }
}

fn macro_arg_placeholders(
    def: &MacroDef,
    args: &[Expr],
    expansion_id: usize,
) -> (HashMap<String, Expr>, HashMap<String, Expr>) {
    let mut used_symbols = HashSet::new();
    collect_symbols(&def.body, &mut used_symbols);

    let mut placeholder_params = HashMap::new();
    let mut placeholder_args = HashMap::new();
    for (idx, (param, arg)) in def.params.iter().zip(args.iter()).enumerate() {
        let placeholder = fresh_placeholder(idx, expansion_id, &mut used_symbols);
        placeholder_params.insert(param.clone(), var(&placeholder));
        placeholder_args.insert(placeholder, arg.clone());
    }
    (placeholder_params, placeholder_args)
}

fn fresh_placeholder(
    idx: usize,
    expansion_id: usize,
    used_symbols: &mut HashSet<String>,
) -> String {
    let mut attempt = 0;
    loop {
        let candidate = format!("__chelis_macro_arg_{expansion_id}_{idx}_{attempt}");
        if used_symbols.insert(candidate.clone()) {
            return candidate;
        }
        attempt += 1;
    }
}

fn replace_placeholder_vars(expr: &Expr, replacements: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            replace_placeholder_vars(&bridged, replacements)
        }
        Expr::BareList(_, _) | Expr::UnknownForm(_) => expr.clone(),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(replace_placeholder_vars(&meta.expr, replacements)),
            },
            *span,
        ),
        Expr::List(list, span) => {
            if get_tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
                && let Some(replacement) = replacements.get(name)
            {
                return replacement.clone();
            }
            Expr::List(
                List {
                    elements: list
                        .elements
                        .iter()
                        .map(|child| replace_placeholder_vars(child, replacements))
                        .collect(),
                },
                *span,
            )
        }
    }
}

fn collect_symbols(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Atom(Atom::Name(name), _) => {
            out.insert(name.clone());
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            collect_symbols(&bridged, out);
        }
        Expr::BareList(_, _) | Expr::UnknownForm(_) => {}
        Expr::MetaExpr(meta, _) => collect_symbols(&meta.expr, out),
        Expr::List(list, _) => {
            for element in &list.elements {
                collect_symbols(element, out);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Scope {
    blockers: HashSet<String>,
}

impl Scope {
    fn with_blockers(&self, blockers: &[String]) -> Self {
        let mut next = self.clone();
        next.add_blockers(blockers.to_vec());
        next
    }

    fn add_blocker(&mut self, name: String) {
        self.blockers.insert(name);
    }

    fn add_blockers<I>(&mut self, names: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.blockers.extend(names);
    }

    fn blocks(&self, name: &str) -> bool {
        self.blockers.contains(name)
    }
}

fn extract_macro_def(expr: &Expr) -> Result<Option<MacroDef>, ExpansionError> {
    let Expr::List(list, _) = expr else {
        return Ok(None);
    };
    if internal_tag(list) != Some("defmacro") {
        return Ok(None);
    }
    let kids = children(list);
    if kids.len() != 3 {
        return Err(ExpansionError::MalformedDefinition {
            message: "defmacro expects name, params, and body".to_string(),
        });
    }
    let Some(name) = symbol_name(&kids[0]) else {
        return Err(ExpansionError::MalformedDefinition {
            message: "defmacro name must be a symbol".to_string(),
        });
    };
    let Expr::List(params_list, _) = &kids[1] else {
        return Err(ExpansionError::MalformedDefinition {
            message: format!("defmacro `{name}` must use `(params {{}} ...)`"),
        });
    };
    if get_tag(params_list) != Some(DeepTag::Params) {
        return Err(ExpansionError::MalformedDefinition {
            message: format!("defmacro `{name}` must use `(params {{}} ...)`"),
        });
    }
    let params = children(params_list)
        .iter()
        .map(|param| {
            symbol_name(param)
                .map(|name| name.to_string())
                .ok_or_else(|| ExpansionError::MalformedDefinition {
                    message: format!("defmacro `{name}` params must be symbols"),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(MacroDef {
        name: name.to_string(),
        params,
        body: kids[2].clone(),
    }))
}

fn substitute_expr(
    expr: &Expr,
    params: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
) -> Expr {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            substitute_expr(&bridged, params, shadowed)
        }
        Expr::BareList(_, _) | Expr::UnknownForm(_) => expr.clone(),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(substitute_expr(&meta.expr, params, shadowed)),
            },
            *span,
        ),
        Expr::List(list, span) => {
            if get_tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
                && !shadowed.contains(name)
                && let Some(replacement) = params.get(name)
            {
                return replacement.clone();
            }

            match get_tag(list) {
                Some(DeepTag::Fn) => substitute_fn(list, params, shadowed, *span),
                Some(DeepTag::Let) => substitute_let(list, params, shadowed, *span),
                Some(DeepTag::Match) => substitute_match(list, params, shadowed, *span),
                _ => Expr::List(
                    List {
                        elements: list
                            .elements
                            .iter()
                            .map(|child| substitute_expr(child, params, shadowed))
                            .collect(),
                    },
                    *span,
                ),
            }
        }
    }
}

fn substitute_fn(
    list: &List,
    params: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
    span: Span,
) -> Expr {
    let mut elements = list.elements.clone();
    let kids = children(list);
    if kids.len() < 2 {
        return Expr::List(List { elements }, span);
    }
    let mut child_shadowed = shadowed.clone();
    for blocker in params_blockers(&kids[0]) {
        child_shadowed.insert(blocker);
    }
    elements[3] = substitute_expr(&kids[1], params, &child_shadowed);
    Expr::List(List { elements }, span)
}

fn substitute_let(
    list: &List,
    params: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
    span: Span,
) -> Expr {
    let mut elements = list.elements.clone();
    let kids = children(list);
    if kids.len() < 2 {
        return Expr::List(List { elements }, span);
    }
    let mut scope_for_values = shadowed.clone();
    if let Expr::List(bind_list, bind_span) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut new_bind_children = Vec::new();
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            new_bind_children.push(bind_kids[i].clone());
            new_bind_children.push(substitute_expr(
                &bind_kids[i + 1],
                params,
                &scope_for_values,
            ));
            if let Some(name) = symbol_name(&bind_kids[i]) {
                scope_for_values.insert(name.to_string());
            }
            i += 2;
        }
        elements[2] = node_with_meta(
            DeepTag::Bind,
            list.elements[1].clone(),
            new_bind_children,
            *bind_span,
        );
    }
    elements[3] = substitute_expr(&kids[1], params, &scope_for_values);
    Expr::List(List { elements }, span)
}

fn substitute_match(
    list: &List,
    params: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
    span: Span,
) -> Expr {
    let mut elements = Vec::with_capacity(list.elements.len());
    elements.push(list.elements[0].clone());
    elements.push(list.elements[1].clone());
    let kids = children(list);
    if kids.is_empty() {
        return Expr::List(List { elements }, span);
    }
    elements.push(substitute_expr(&kids[0], params, shadowed));
    for arm in &kids[1..] {
        if let Expr::List(arm_list, arm_span) = arm
            && get_tag(arm_list) == Some(DeepTag::Arm)
        {
            let arm_kids = children(arm_list);
            if arm_kids.len() >= 3 {
                let mut arm_shadowed = shadowed.clone();
                arm_shadowed.extend(pattern_binders(&arm_kids[0]));
                let mut arm_elements = arm_list.elements.clone();
                if !is_unit_list(&arm_kids[1]) {
                    arm_elements[3] = substitute_expr(&arm_kids[1], params, &arm_shadowed);
                }
                arm_elements[4] = substitute_expr(&arm_kids[2], params, &arm_shadowed);
                elements.push(Expr::List(
                    List {
                        elements: arm_elements,
                    },
                    *arm_span,
                ));
                continue;
            }
        }
        elements.push(substitute_expr(arm, params, shadowed));
    }
    Expr::List(List { elements }, span)
}

fn hygienize_expr(expr: &Expr, counter: &mut usize, env: &HashMap<String, String>) -> Expr {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            hygienize_expr(&bridged, counter, env)
        }
        Expr::BareList(_, _) | Expr::UnknownForm(_) => expr.clone(),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(hygienize_expr(&meta.expr, counter, env)),
            },
            *span,
        ),
        Expr::List(list, span) => {
            if get_tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
                && let Some(renamed) = env.get(name)
            {
                let mut elements = list.elements.clone();
                elements[2] = Expr::Atom(Atom::Name(renamed.clone()), children(list)[0].span());
                return Expr::List(List { elements }, *span);
            }
            match get_tag(list) {
                Some(DeepTag::Fn) => hygienize_fn(list, counter, env, *span),
                Some(DeepTag::Let) => hygienize_let(list, counter, env, *span),
                Some(DeepTag::Match) => hygienize_match(list, counter, env, *span),
                _ => Expr::List(
                    List {
                        elements: list
                            .elements
                            .iter()
                            .map(|child| hygienize_expr(child, counter, env))
                            .collect(),
                    },
                    *span,
                ),
            }
        }
    }
}

fn hygienize_fn(
    list: &List,
    counter: &mut usize,
    env: &HashMap<String, String>,
    span: Span,
) -> Expr {
    let mut elements = list.elements.clone();
    let kids = children(list);
    if kids.len() < 2 {
        return Expr::List(List { elements }, span);
    }
    let (params_expr, next_env) = hygienize_params_expr(&kids[0], counter, env);
    elements[2] = params_expr;
    elements[3] = hygienize_expr(&kids[1], counter, &next_env);
    Expr::List(List { elements }, span)
}

fn hygienize_let(
    list: &List,
    counter: &mut usize,
    env: &HashMap<String, String>,
    span: Span,
) -> Expr {
    let mut elements = list.elements.clone();
    let kids = children(list);
    if kids.len() < 2 {
        return Expr::List(List { elements }, span);
    }
    let mut scope_env = env.clone();
    if let Expr::List(bind_list, bind_span) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut new_bind_children = Vec::new();
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            let name = symbol_name(&bind_kids[i]).unwrap_or("_");
            let fresh = fresh_name(name, counter);
            new_bind_children.push(Expr::Atom(Atom::Name(fresh.clone()), bind_kids[i].span()));
            new_bind_children.push(hygienize_expr(&bind_kids[i + 1], counter, &scope_env));
            scope_env.insert(name.to_string(), fresh);
            i += 2;
        }
        elements[2] = node_with_meta(
            DeepTag::Bind,
            list.elements[1].clone(),
            new_bind_children,
            *bind_span,
        );
    }
    elements[3] = hygienize_expr(&kids[1], counter, &scope_env);
    Expr::List(List { elements }, span)
}

fn hygienize_match(
    list: &List,
    counter: &mut usize,
    env: &HashMap<String, String>,
    span: Span,
) -> Expr {
    let kids = children(list);
    let mut elements = Vec::with_capacity(list.elements.len());
    elements.push(list.elements[0].clone());
    elements.push(list.elements[1].clone());
    if kids.is_empty() {
        return Expr::List(List { elements }, span);
    }
    elements.push(hygienize_expr(&kids[0], counter, env));
    for arm in &kids[1..] {
        if let Expr::List(arm_list, arm_span) = arm
            && get_tag(arm_list) == Some(DeepTag::Arm)
        {
            let arm_kids = children(arm_list);
            if arm_kids.len() >= 3 {
                let (pattern, arm_env) = hygienize_pattern(&arm_kids[0], counter, env);
                let mut arm_elements = arm_list.elements.clone();
                arm_elements[2] = pattern;
                if !is_unit_list(&arm_kids[1]) {
                    arm_elements[3] = hygienize_expr(&arm_kids[1], counter, &arm_env);
                }
                arm_elements[4] = hygienize_expr(&arm_kids[2], counter, &arm_env);
                elements.push(Expr::List(
                    List {
                        elements: arm_elements,
                    },
                    *arm_span,
                ));
                continue;
            }
        }
        elements.push(hygienize_expr(arm, counter, env));
    }
    Expr::List(List { elements }, span)
}

fn hygienize_params_expr(
    expr: &Expr,
    counter: &mut usize,
    env: &HashMap<String, String>,
) -> (Expr, HashMap<String, String>) {
    let mut next_env = env.clone();
    let Expr::List(list, span) = expr else {
        return (expr.clone(), next_env);
    };
    if get_tag(list) != Some(DeepTag::Params) {
        return (expr.clone(), next_env);
    }
    let mut elements = vec![list.elements[0].clone(), list.elements[1].clone()];
    for param in children(list) {
        match param {
            Expr::Atom(Atom::Name(name), span) => {
                let fresh = fresh_name(name, counter);
                next_env.insert(name.clone(), fresh.clone());
                elements.push(Expr::Atom(Atom::Name(fresh), *span));
            }
            Expr::List(param_list, param_span) if param_list.elements.len() == 2 => {
                let Some(name) = symbol_name(&param_list.elements[0]) else {
                    elements.push(param.clone());
                    continue;
                };
                let fresh = fresh_name(name, counter);
                next_env.insert(name.to_string(), fresh.clone());
                elements.push(Expr::List(
                    List {
                        elements: vec![
                            Expr::Atom(Atom::Name(fresh), param_list.elements[0].span()),
                            param_list.elements[1].clone(),
                        ],
                    },
                    *param_span,
                ));
            }
            _ => elements.push(param.clone()),
        }
    }
    (Expr::List(List { elements }, *span), next_env)
}

fn hygienize_pattern(
    expr: &Expr,
    counter: &mut usize,
    env: &HashMap<String, String>,
) -> (Expr, HashMap<String, String>) {
    let Expr::List(list, span) = expr else {
        return (expr.clone(), env.clone());
    };
    let Some(tag) = get_tag(list) else {
        return (expr.clone(), env.clone());
    };
    let mut next_env = env.clone();
    match tag {
        DeepTag::PatVar => {
            let Some(name) = children(list).first().and_then(symbol_name) else {
                return (expr.clone(), env.clone());
            };
            let fresh = fresh_name(name, counter);
            next_env.insert(name.to_string(), fresh.clone());
            (
                node_with_meta(
                    DeepTag::PatVar,
                    list.elements[1].clone(),
                    vec![Expr::Atom(Atom::Name(fresh), children(list)[0].span())],
                    *span,
                ),
                next_env,
            )
        }
        DeepTag::PatAs => {
            let kids = children(list);
            if kids.len() < 2 {
                return (expr.clone(), env.clone());
            }
            let Some(name) = symbol_name(&kids[0]) else {
                return (expr.clone(), env.clone());
            };
            let fresh = fresh_name(name, counter);
            next_env.insert(name.to_string(), fresh.clone());
            let (inner, inner_env) = hygienize_pattern(&kids[1], counter, &next_env);
            (
                node_with_meta(
                    DeepTag::PatAs,
                    list.elements[1].clone(),
                    vec![Expr::Atom(Atom::Name(fresh), kids[0].span()), inner],
                    *span,
                ),
                inner_env,
            )
        }
        DeepTag::PatTuple | DeepTag::PatCtor => {
            let kids = children(list);
            let mut new_children = Vec::new();
            let mut working_env = env.clone();
            if tag == DeepTag::PatCtor && !kids.is_empty() {
                new_children.push(kids[0].clone());
                for child in &kids[1..] {
                    let (child_pat, child_env) = hygienize_pattern(child, counter, &working_env);
                    working_env = child_env;
                    new_children.push(child_pat);
                }
            } else {
                for child in kids {
                    let (child_pat, child_env) = hygienize_pattern(child, counter, &working_env);
                    working_env = child_env;
                    new_children.push(child_pat);
                }
            }
            (
                node_with_meta(tag, list.elements[1].clone(), new_children, *span),
                working_env,
            )
        }
        DeepTag::PatRecord => {
            let mut new_children = Vec::new();
            let kids = children(list);
            if kids.is_empty() {
                return (expr.clone(), env.clone());
            }
            new_children.push(kids[0].clone());
            let mut working_env = env.clone();
            for kv in &kids[1..] {
                if let Expr::List(kv_list, kv_span) = kv
                    && get_tag(kv_list) == Some(DeepTag::Kv)
                {
                    let kv_kids = children(kv_list);
                    if kv_kids.len() == 2 {
                        let (child_pat, child_env) =
                            hygienize_pattern(&kv_kids[1], counter, &working_env);
                        working_env = child_env;
                        new_children.push(node_with_meta(
                            DeepTag::Kv,
                            kv_list.elements[1].clone(),
                            vec![kv_kids[0].clone(), child_pat],
                            *kv_span,
                        ));
                        continue;
                    }
                }
                new_children.push(kv.clone());
            }
            (
                node_with_meta(
                    DeepTag::PatRecord,
                    list.elements[1].clone(),
                    new_children,
                    *span,
                ),
                working_env,
            )
        }
        _ => (expr.clone(), env.clone()),
    }
}

fn annotate_source_expr(expr: &Expr, invocation: &Expr) -> Expr {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => expr.clone(),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            annotate_source_expr(&bridged, invocation)
        }
        Expr::BareList(_, _) | Expr::UnknownForm(_) => expr.clone(),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(annotate_source_expr(&meta.expr, invocation)),
            },
            *span,
        ),
        Expr::List(list, span) => {
            let mut elements = list.elements.clone();
            if let Some(Expr::Map(meta, meta_span)) = elements.get_mut(1) {
                if !meta.entries.iter().any(|(key, _)| key == "source") {
                    meta.entries
                        .push(("source".to_string(), invocation.clone()));
                }
                elements[1] = Expr::Map(meta.clone(), *meta_span);
            }
            for child in elements.iter_mut().skip(2) {
                *child = annotate_source_expr(child, invocation);
            }
            Expr::List(List { elements }, *span)
        }
    }
}

fn params_blockers(expr: &Expr) -> Vec<String> {
    match expr {
        Expr::List(list, _) if get_tag(list) == Some(DeepTag::Params) => children(list)
            .iter()
            .filter_map(|param| match param {
                Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
                Expr::List(param_list, _) if param_list.elements.len() == 2 => {
                    symbol_name(&param_list.elements[0]).map(|name| name.to_string())
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn pattern_binders(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_binders(expr, &mut names);
    names
}

fn collect_pattern_binders(expr: &Expr, out: &mut Vec<String>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some(DeepTag::PatVar) => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                out.push(name.to_string());
            }
        }
        Some(DeepTag::PatAs) => {
            let kids = children(list);
            if let Some(name) = kids.first().and_then(symbol_name) {
                out.push(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_binders(inner, out);
            }
        }
        Some(DeepTag::PatTuple) => {
            for child in children(list) {
                collect_pattern_binders(child, out);
            }
        }
        Some(DeepTag::PatCtor) => {
            for child in children(list).iter().skip(1) {
                collect_pattern_binders(child, out);
            }
        }
        Some(DeepTag::PatRecord) => {
            for kv in children(list).iter().skip(1) {
                if let Expr::List(kv_list, _) = kv
                    && get_tag(kv_list) == Some(DeepTag::Kv)
                    && let Some(pattern) = children(kv_list).get(1)
                {
                    collect_pattern_binders(pattern, out);
                }
            }
        }
        _ => {}
    }
}

fn macro_source(name: &str, args: &[Expr]) -> Expr {
    let mut elements = vec![Expr::Atom(Atom::Name(name.to_string()), zero_span())];
    elements.extend(args.iter().cloned());
    Expr::List(List { elements }, zero_span())
}

fn standard_prelude_macros() -> Result<HashMap<String, MacroDef>, ExpansionError> {
    let mut defs = HashMap::new();
    for def in [
        prelude_linear_layer(),
        prelude_residual(),
        prelude_cross_entropy(),
    ] {
        defs.insert(def.name.clone(), def);
    }
    Ok(defs)
}

fn get_tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

/// Compiler-internal pre-expansion tags (`defmacro` / `macro-invoke`)
/// are deliberately outside the public vocabulary (spec/03 macro
/// boundary rule) and remain symbol-headed; this is the macro layer's
/// recorded raw-string entry point (checker_totality.md §C1.2). It
/// returns None for stamped vocabulary nodes by construction.
fn internal_tag(list: &List) -> Option<&str> {
    list.unknown_tag_symbol()
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some(DeepTag::Var) {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn is_unit_list(expr: &Expr) -> bool {
    matches!(expr, Expr::List(List { elements }, _) if elements.is_empty())
}

fn node_with_meta(tag: DeepTag, meta: Expr, children: Vec<Expr>, span: Span) -> Expr {
    let mut elements = vec![Expr::Atom(Atom::Tag(tag), span), meta];
    elements.extend(children);
    Expr::List(List { elements }, span)
}

fn zero_span() -> Span {
    Span::new(0, 0)
}

fn fresh_name(name: &str, counter: &mut usize) -> String {
    let fresh = format!("{name}_macro_{counter}");
    *counter += 1;
    fresh
}

fn prelude_linear_layer() -> MacroDef {
    MacroDef {
        name: "linear_layer".to_string(),
        params: vec!["x".to_string(), "w".to_string(), "b".to_string()],
        body: app(
            "add",
            vec![
                app("matmul", vec![var("x"), var("w")]),
                app("expand", vec![var("b"), int32_lit(0), var("batch")]),
            ],
        ),
    }
}

fn prelude_residual() -> MacroDef {
    MacroDef {
        name: "residual".to_string(),
        params: vec!["x".to_string(), "f".to_string()],
        body: app("add", vec![var("x"), app_expr(var("f"), vec![var("x")])]),
    }
}

fn prelude_cross_entropy() -> MacroDef {
    MacroDef {
        name: "cross_entropy".to_string(),
        params: vec!["logits".to_string(), "labels".to_string()],
        body: app(
            "mean",
            vec![
                app(
                    "neg",
                    vec![app(
                        "sum",
                        vec![
                            app(
                                "mul",
                                vec![
                                    var("labels"),
                                    app(
                                        "log",
                                        vec![app("softmax", vec![var("logits"), int32_lit(1)])],
                                    ),
                                ],
                            ),
                            int32_lit(1),
                        ],
                    )],
                ),
                int32_lit(0),
            ],
        ),
    }
}

fn app(name: &str, args: Vec<Expr>) -> Expr {
    app_expr(var(name), args)
}

fn app_expr(func: Expr, args: Vec<Expr>) -> Expr {
    let mut children = vec![func];
    children.extend(args);
    node(DeepTag::App, children)
}

fn var(name: &str) -> Expr {
    node(
        DeepTag::Var,
        vec![Expr::Atom(Atom::Name(name.to_string()), zero_span())],
    )
}

fn int32_lit(value: i64) -> Expr {
    node_with_meta(
        DeepTag::Lit,
        Expr::Map(
            MetaMap {
                entries: vec![(
                    "type".to_string(),
                    node(
                        DeepTag::TPrim,
                        vec![Expr::Atom(Atom::Name("int32".to_string()), zero_span())],
                    ),
                )],
            },
            zero_span(),
        ),
        vec![Expr::Atom(Atom::Int(value), zero_span())],
        zero_span(),
    )
}

fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    node_with_meta(tag, meta_empty(), children, zero_span())
}

fn meta_empty() -> Expr {
    Expr::Map(MetaMap::default(), zero_span())
}
