use chelis_deep::DeepTag;
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::Span;
use chelis_deep::annotations::{MacroSource, MetadataValue};
use chelis_deep::ast::{Atom, Expr, ExprCarrier, MetaExpr, Metadata, UnknownFormData};
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
    #[error("{0}")]
    Metadata(#[from] chelis_deep::metadata::MetadataError),
    #[error("macro expansion produced an invalid stamped node: {0}")]
    InvalidNode(#[from] chelis_deep::node::NodeError),
    #[error("macro expansion limit exceeded after {limit} expansions")]
    ExpansionLimitExceeded { limit: usize },

    #[error("malformed macro definition: {message}")]
    MalformedDefinition { message: String },

    #[error("failed to load standard macro prelude: {message}")]
    PreludeLoad { message: String },

    #[error(
        "`{declaration} {name}` collides with the standard prelude macro `{name}`: ordinary top-level `def`/`sig` declarations may not reuse a loaded standard-prelude macro name (spec/02-surf-syntax.md §P5b); rename the declaration or define a user macro when macro override is intended"
    )]
    StandardPreludeNameCollision {
        name: String,
        declaration: &'static str,
    },
}

#[derive(Debug, Clone)]
struct MacroDef {
    name: String,
    params: Vec<String>,
    body: Expr,
}

#[derive(Clone, Copy)]
struct MacroNode<'a> {
    expr: &'a Expr,
    tag: DeepTag,
    metadata: &'a Metadata,
    children: &'a [Expr],
}

impl<'a> MacroNode<'a> {
    fn new(expr: &'a Expr, tag: DeepTag, metadata: &'a Metadata, children: &'a [Expr]) -> Self {
        Self {
            expr,
            tag,
            metadata,
            children,
        }
    }
}

enum AdmittedSpecialForm<'a> {
    Fn {
        params_expr: &'a Expr,
        body: &'a Expr,
    },
    Let {
        bind_node: MacroNode<'a>,
        body: &'a Expr,
    },
}

#[derive(Clone, Copy)]
enum MacroParameterDisposition<'a> {
    Binder(&'a str),
    Invalid,
}

/// How a `params` child takes part in macro scoping. A bare name, an
/// annotated `(name {type: ...})` parameter, and the prefix metadata spelling
/// used for vocabulary names are all `fn` parameters. Hygiene renames them
/// and they block macro expansion of their name
/// (spec/02-surf-syntax.md §P5).
fn macro_parameter_disposition(expr: &Expr) -> MacroParameterDisposition<'_> {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Name(name)) => MacroParameterDisposition::Binder(name),
        ExprCarrier::StructuralList([Expr::Atom(Atom::Name(name), _), Expr::Map(_, _)]) => {
            MacroParameterDisposition::Binder(name)
        }
        ExprCarrier::MetadataExpression(meta) => match meta.expr.as_ref() {
            Expr::Atom(Atom::Name(name), _) => MacroParameterDisposition::Binder(name),
            _ => MacroParameterDisposition::Invalid,
        },
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => MacroParameterDisposition::Invalid,
    }
}

fn admitted_special_form(node: MacroNode<'_>) -> Option<AdmittedSpecialForm<'_>> {
    match (node.tag, node.children) {
        (DeepTag::Fn, [params_expr, body]) => {
            let ExprCarrier::DecodedNode(DeepTag::Params, _, params) = params_expr.carrier() else {
                return None;
            };
            if !params.iter().all(|param| {
                !matches!(
                    macro_parameter_disposition(param),
                    MacroParameterDisposition::Invalid
                )
            }) {
                return None;
            }
            Some(AdmittedSpecialForm::Fn { params_expr, body })
        }
        (DeepTag::Let, [bind_expr, body]) => {
            let ExprCarrier::DecodedNode(DeepTag::Bind, metadata, bind_children) =
                bind_expr.carrier()
            else {
                return None;
            };
            let (pairs, remainder) = bind_children.as_chunks::<2>();
            if !remainder.is_empty() || !pairs.iter().all(|[name, _]| symbol_name(name).is_some()) {
                return None;
            }
            Some(AdmittedSpecialForm::Let {
                bind_node: MacroNode::new(bind_expr, DeepTag::Bind, metadata, bind_children),
                body,
            })
        }
        _ => None,
    }
}

fn rebuild_macro_node(node: MacroNode<'_>, metadata: Metadata, children: Vec<Expr>) -> Expr {
    Expr::node(node.tag, metadata, children, node.expr.span())
}

fn try_rebuild_macro_node(
    node: MacroNode<'_>,
    metadata: Metadata,
    children: Vec<Expr>,
) -> Result<Expr, chelis_deep::node::NodeError> {
    Ok(Expr::Node(
        Box::new(chelis_deep::node::Node::try_new(
            node.tag, metadata, children,
        )?),
        node.expr.span(),
    ))
}

pub fn expand_program(
    exprs: &[Expr],
    options: &ExpansionOptions,
) -> Result<ExpandedProgram, ExpansionError> {
    let prelude = if options.load_std_prelude {
        standard_prelude_macros()?
    } else {
        UnordMap::new()
    };
    let mut expander = Expander::new(options.max_iterations, prelude);
    let exprs = expander.expand_sequence(exprs, &UnordMap::new())?;
    Ok(ExpandedProgram {
        exprs,
        expansions: expander.expansions,
    })
}

struct Expander {
    remaining_expansions: usize,
    hygiene_counter: usize,
    expansions: usize,
    prelude_macros: UnordMap<String, MacroDef>,
}

impl Expander {
    fn new(limit: usize, prelude_macros: UnordMap<String, MacroDef>) -> Self {
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
        inherited_macros: &UnordMap<String, MacroDef>,
    ) -> Result<Vec<Expr>, ExpansionError> {
        self.reject_standard_prelude_callable_collisions(exprs)?;

        let mut macros = inherited_macros.clone();
        for (name, def) in self.prelude_macros.to_sorted() {
            macros.entry(name.clone()).or_insert_with(|| def.clone());
        }

        let mut user_macros = UnordMap::new();
        for expr in exprs {
            if let Some(def) = extract_macro_def(expr)? {
                user_macros.insert(def.name.clone(), def);
            }
        }
        macros.merge(user_macros);

        let mut out = Vec::new();
        for expr in exprs {
            if extract_macro_def(expr)?.is_some() {
                continue;
            }
            out.push(self.expand_expr(expr, &macros, &Scope::default())?);
        }
        Ok(out)
    }

    /// Reject an ordinary declaration whose calls would be consumed by the
    /// loaded standard macro prelude before ordinary function resolution
    /// (spec/02-surf-syntax.md §P5b, chelis#672).
    ///
    /// The check reads the exact prelude map used by expansion, so adding or
    /// removing a standard macro changes collision detection in the same
    /// operation. An inline-annotated `def` desugars to `defsig` plus `def`;
    /// collect the def names first so its one diagnostic names the authored
    /// `def`, even when the synthesized `defsig` appears first.
    fn reject_standard_prelude_callable_collisions(
        &self,
        exprs: &[Expr],
    ) -> Result<(), ExpansionError> {
        if self.prelude_macros.is_empty() {
            return Ok(());
        }

        let def_names = exprs
            .iter()
            .filter_map(|expr| standard_prelude_decl_name(expr, DeepTag::Def, &self.prelude_macros))
            .collect::<UnordSet<_>>();

        for expr in exprs {
            let name = standard_prelude_decl_name(expr, DeepTag::Def, &self.prelude_macros)
                .or_else(|| {
                    standard_prelude_decl_name(expr, DeepTag::Defsig, &self.prelude_macros)
                });
            let Some(name) = name else {
                continue;
            };
            return Err(ExpansionError::StandardPreludeNameCollision {
                name: name.to_string(),
                declaration: if def_names.contains(name) {
                    "def"
                } else {
                    "sig"
                },
            });
        }
        Ok(())
    }

    fn expand_expr(
        &mut self,
        expr: &Expr,
        macros: &UnordMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Expr, ExpansionError> {
        match expr.carrier() {
            ExprCarrier::Atom(_) => Ok(expr.clone()),
            ExprCarrier::MetadataMap(meta) => Ok(Expr::Map(
                try_map_meta_entries(meta, |value| self.expand_expr(value, macros, scope))?,
                expr.span(),
            )),
            ExprCarrier::MetadataExpression(meta) => Ok(Expr::MetaExpr(
                MetaExpr {
                    metadata: try_map_meta_entries(&meta.metadata, |value| {
                        self.expand_expr(value, macros, scope)
                    })?,
                    expr: Box::new(self.expand_expr(&meta.expr, macros, scope)?),
                },
                expr.span(),
            )),
            ExprCarrier::StructuralList(elements) => Ok(Expr::BareList(
                elements
                    .iter()
                    .map(|child| self.expand_expr(child, macros, scope))
                    .collect::<Result<Vec<_>, _>>()?,
                expr.span(),
            )),
            ExprCarrier::UndecodableHead(_, _, _) => {
                try_map_unknown_form(unknown_form(expr), |child| {
                    self.expand_expr(child, macros, scope)
                })
            }
            ExprCarrier::DecodedNode(tag, metadata, children) => {
                let node = MacroNode::new(expr, tag, metadata, children);
                if let Some(expanded) = self.try_expand_macro_call(node, macros, scope)? {
                    return self.expand_expr(&expanded, macros, scope);
                }
                match tag {
                    DeepTag::Module => self.expand_module(node, macros),
                    DeepTag::Fn => self.expand_fn(node, macros, scope),
                    DeepTag::Let => self.expand_let(node, macros, scope),
                    DeepTag::Match => self.expand_match(node, macros, scope),
                    _ => try_rebuild_macro_node(
                        node,
                        try_map_meta_entries(metadata, |value| {
                            self.expand_expr(value, macros, scope)
                        })?,
                        children
                            .iter()
                            .map(|child| self.expand_expr(child, macros, scope))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .map_err(Into::into),
                }
            }
        }
    }

    fn expand_module(
        &mut self,
        node: MacroNode<'_>,
        macros: &UnordMap<String, MacroDef>,
    ) -> Result<Expr, ExpansionError> {
        let kids = node.children;
        if kids.is_empty() {
            return Ok(node.expr.clone());
        }
        let mut expanded = vec![kids[0].clone()];
        let body = self.expand_sequence(&kids[1..], macros)?;
        expanded.extend(body);
        try_rebuild_macro_node(node, node.metadata.clone(), expanded).map_err(Into::into)
    }

    fn expand_fn(
        &mut self,
        node: MacroNode<'_>,
        macros: &UnordMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Expr, ExpansionError> {
        let Some(AdmittedSpecialForm::Fn { params_expr, body }) = admitted_special_form(node)
        else {
            return Ok(node.expr.clone());
        };

        let blocker_names = params_blockers(params_expr);
        let fn_scope = scope.with_blockers(&blocker_names);
        try_rebuild_macro_node(
            node,
            node.metadata.clone(),
            vec![
                params_expr.clone(),
                self.expand_expr(body, macros, &fn_scope)?,
            ],
        )
        .map_err(Into::into)
    }

    fn expand_let(
        &mut self,
        node: MacroNode<'_>,
        macros: &UnordMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Expr, ExpansionError> {
        let Some(AdmittedSpecialForm::Let { bind_node, body }) = admitted_special_form(node) else {
            return Ok(node.expr.clone());
        };
        // Preserve the node, but do not clone the entire unvisited body
        // before replacing it with its expansion at every nested binding.
        let bind_kids = bind_node.children;
        let mut scope_for_values = scope.clone();
        let mut new_bind_children = Vec::with_capacity(bind_kids.len());
        for [name_expr, value_expr] in bind_kids.as_chunks::<2>().0 {
            let name = symbol_name(name_expr).expect("special-form admission checked binders");
            new_bind_children.push(name_expr.clone());
            new_bind_children.push(self.expand_expr(value_expr, macros, &scope_for_values)?);
            scope_for_values.add_blocker(name.to_string());
        }
        let children = vec![
            try_rebuild_macro_node(bind_node, bind_node.metadata.clone(), new_bind_children)?,
            self.expand_expr(body, macros, &scope_for_values)?,
        ];
        try_rebuild_macro_node(node, node.metadata.clone(), children).map_err(Into::into)
    }

    fn expand_match(
        &mut self,
        node: MacroNode<'_>,
        macros: &UnordMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Expr, ExpansionError> {
        let kids = node.children;
        if kids.is_empty() {
            return Ok(node.expr.clone());
        }
        let mut children = Vec::with_capacity(kids.len());
        children.push(self.expand_expr(&kids[0], macros, scope)?);
        for arm in &kids[1..] {
            if let ExprCarrier::DecodedNode(DeepTag::Arm, metadata, arm_children) = arm.carrier() {
                let arm_node = MacroNode::new(arm, DeepTag::Arm, metadata, arm_children);
                let arm_kids = arm_node.children;
                if arm_kids.len() >= 3 {
                    let mut arm_scope = scope.clone();
                    arm_scope.add_blockers(chelis_deep::pattern_binder_names(&arm_kids[0]));
                    let mut arm_children = arm_kids.to_vec();
                    arm_children[2] = self.expand_expr(&arm_kids[2], macros, &arm_scope)?;
                    if !is_unit_list(&arm_kids[1]) {
                        arm_children[1] = self.expand_expr(&arm_kids[1], macros, &arm_scope)?;
                    }
                    children.push(try_rebuild_macro_node(
                        arm_node,
                        arm_node.metadata.clone(),
                        arm_children,
                    )?);
                    continue;
                }
            }
            children.push(self.expand_expr(arm, macros, scope)?);
        }
        try_rebuild_macro_node(node, node.metadata.clone(), children).map_err(Into::into)
    }

    fn try_expand_macro_call(
        &mut self,
        node: MacroNode<'_>,
        macros: &UnordMap<String, MacroDef>,
        scope: &Scope,
    ) -> Result<Option<Expr>, ExpansionError> {
        if node.tag != DeepTag::App || node.children.is_empty() {
            return Ok(None);
        }
        let Some(name) = var_name(&node.children[0]) else {
            return Ok(None);
        };
        let name = name.to_string();
        let args = node.children[1..].to_vec();

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
        let placeholder_body = substitute_expr(&def.body, &placeholder_params, &UnordSet::new());
        let hygienic = hygienize_expr(
            &placeholder_body,
            &mut self.hygiene_counter,
            &UnordMap::new(),
        );
        let substituted = inherit_replaced_value_annotations(
            replace_placeholder_vars(&hygienic, &placeholder_args)?,
            node.expr,
        )?
        .try_inherit_extensions(node.expr)?;
        Ok(Some(annotate_source_expr(
            &substituted,
            &MacroSource::try_from_expression(&invocation)?,
        )))
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

/// A replaced expression owns its type obligation and Surf binding origin,
/// whether it is an invocation or a template parameter reference. If the new
/// root already owns either key, a one-expression `block` keeps both independent
/// metadata owners; the block has exactly the expression's value and effect.
fn inherit_replaced_value_annotations(
    mut replacement: Expr,
    owner: &Expr,
) -> Result<Expr, ExpansionError> {
    let ExprCarrier::DecodedNode(_, metadata, _) = owner.carrier() else {
        return Ok(replacement);
    };
    let annotations = metadata
        .values()
        .filter(|value| {
            matches!(
                value,
                MetadataValue::Type(_) | MetadataValue::SurfBindingType(_)
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    if annotations.is_empty() {
        return Ok(replacement);
    }
    if let Expr::Node(node, _) = &mut replacement {
        let mut metadata = node.meta().clone();
        let conflict = annotations.iter().any(|value| match value {
            MetadataValue::Type(_) => metadata.ty().is_some(),
            MetadataValue::SurfBindingType(_) => metadata.surf_binding_type().is_some(),
            _ => unreachable!("only invocation value annotations were collected"),
        });
        if !conflict {
            for value in annotations {
                metadata.insert(value)?;
            }
            node.try_replace_meta(metadata)?;
            return Ok(replacement);
        }
    }
    let metadata = Metadata::try_from_values(annotations)?;
    let span = replacement.span();
    Ok(Expr::Node(
        Box::new(chelis_deep::node::Node::try_new(
            DeepTag::Block,
            metadata,
            vec![replacement],
        )?),
        span,
    ))
}

fn macro_arg_placeholders(
    def: &MacroDef,
    args: &[Expr],
    expansion_id: usize,
) -> (UnordMap<String, Expr>, UnordMap<String, Expr>) {
    let mut used_symbols = UnordSet::new();
    collect_symbols(&def.body, &mut used_symbols);

    let mut placeholder_params = UnordMap::new();
    let mut placeholder_args = UnordMap::new();
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
    used_symbols: &mut UnordSet<String>,
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

fn replace_placeholder_vars(
    expr: &Expr,
    replacements: &UnordMap<String, Expr>,
) -> Result<Expr, ExpansionError> {
    Ok(match expr.carrier() {
        ExprCarrier::Atom(_) => expr.clone(),
        ExprCarrier::MetadataMap(meta) => Expr::Map(
            try_map_meta_entries(meta, |value| replace_placeholder_vars(value, replacements))?,
            expr.span(),
        ),
        ExprCarrier::MetadataExpression(meta) => Expr::MetaExpr(
            MetaExpr {
                metadata: try_map_meta_entries(&meta.metadata, |value| {
                    replace_placeholder_vars(value, replacements)
                })?,
                expr: Box::new(replace_placeholder_vars(&meta.expr, replacements)?),
            },
            expr.span(),
        ),
        ExprCarrier::StructuralList(elements) => Expr::BareList(
            elements
                .iter()
                .map(|child| replace_placeholder_vars(child, replacements))
                .collect::<Result<_, _>>()?,
            expr.span(),
        ),
        ExprCarrier::UndecodableHead(_, _, _) => {
            try_map_unknown_form(unknown_form(expr), |child| {
                replace_placeholder_vars(child, replacements)
            })?
        }
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::Var
                && let Some(name) = children.first().and_then(symbol_name)
                && let Some(replacement) = replacements.get(name)
            {
                return Ok(
                    inherit_replaced_value_annotations(replacement.clone(), expr)?
                        .try_inherit_extensions(expr)?,
                );
            }
            try_rebuild_macro_node(
                MacroNode::new(expr, tag, metadata, children),
                try_map_meta_entries(metadata, |value| {
                    replace_placeholder_vars(value, replacements)
                })?,
                children
                    .iter()
                    .map(|child| replace_placeholder_vars(child, replacements))
                    .collect::<Result<_, _>>()?,
            )?
        }
    })
}

fn collect_symbols(expr: &Expr, out: &mut UnordSet<String>) {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Name(name)) => {
            out.insert(name.clone());
        }
        ExprCarrier::Atom(_) => {}
        ExprCarrier::MetadataMap(meta) => {
            meta.visit_syntax(&mut |_, value| collect_symbols(value, out));
        }
        ExprCarrier::MetadataExpression(meta) => {
            meta.metadata
                .visit_syntax(&mut |_, value| collect_symbols(value, out));
            collect_symbols(&meta.expr, out);
        }
        ExprCarrier::StructuralList(elements) => {
            for element in elements {
                collect_symbols(element, out);
            }
        }
        ExprCarrier::UndecodableHead(_, metadata, children) => {
            metadata.visit_syntax(&mut |_, value| collect_symbols(value, out));
            for child in children {
                collect_symbols(child, out);
            }
        }
        ExprCarrier::DecodedNode(_, metadata, children) => {
            metadata.visit_syntax(&mut |_, value| collect_symbols(value, out));
            for child in children {
                collect_symbols(child, out);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Scope {
    blockers: UnordSet<String>,
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
    let kids = match expr.carrier() {
        ExprCarrier::UndecodableHead("defmacro", _, children) => children,
        ExprCarrier::StructuralList(elements)
            if bare_internal_tag(elements) == Some("defmacro") =>
        {
            elements.get(2..).unwrap_or(&[])
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return Ok(None),
    };
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
    let params_children = match kids[1].carrier() {
        ExprCarrier::DecodedNode(DeepTag::Params, _, children) => children,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => {
            return Err(ExpansionError::MalformedDefinition {
                message: format!("defmacro `{name}` must use `(params {{}} ...)`"),
            });
        }
    };
    let params = params_children
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
    params: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> Expr {
    match expr.carrier() {
        ExprCarrier::Atom(_) => expr.clone(),
        ExprCarrier::MetadataMap(meta) => Expr::Map(
            map_meta_entries(meta, |value| substitute_expr(value, params, shadowed)),
            expr.span(),
        ),
        ExprCarrier::MetadataExpression(meta) => Expr::MetaExpr(
            MetaExpr {
                metadata: map_meta_entries(&meta.metadata, |value| {
                    substitute_expr(value, params, shadowed)
                }),
                expr: Box::new(substitute_expr(&meta.expr, params, shadowed)),
            },
            expr.span(),
        ),
        ExprCarrier::StructuralList(elements) => Expr::BareList(
            elements
                .iter()
                .map(|child| substitute_expr(child, params, shadowed))
                .collect(),
            expr.span(),
        ),
        ExprCarrier::UndecodableHead(_, _, _) => map_unknown_form(unknown_form(expr), |child| {
            substitute_expr(child, params, shadowed)
        }),
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::Var
                && let Some(name) = children.first().and_then(symbol_name)
                && !shadowed.contains(name)
                && let Some(replacement) = params.get(name)
            {
                return inherit_replaced_value_annotations(replacement.clone(), expr)
                    .expect("macro parameter annotation remains valid on a placeholder")
                    .try_inherit_extensions(expr)
                    .expect("fresh macro placeholders have no conflicting extensions");
            }

            let node = MacroNode::new(expr, tag, metadata, children);
            match tag {
                DeepTag::Fn => substitute_fn(node, params, shadowed),
                DeepTag::Let => substitute_let(node, params, shadowed),
                DeepTag::Match => substitute_match(node, params, shadowed),
                _ => rebuild_macro_node(
                    node,
                    map_meta_entries(metadata, |value| substitute_expr(value, params, shadowed)),
                    children
                        .iter()
                        .map(|child| substitute_expr(child, params, shadowed))
                        .collect(),
                ),
            }
        }
    }
}

fn substitute_fn(
    node: MacroNode<'_>,
    params: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> Expr {
    let Some(AdmittedSpecialForm::Fn { params_expr, body }) = admitted_special_form(node) else {
        return node.expr.clone();
    };
    let mut child_shadowed = shadowed.clone();
    for blocker in params_blockers(params_expr) {
        child_shadowed.insert(blocker);
    }
    rebuild_macro_node(
        node,
        node.metadata.clone(),
        vec![
            params_expr.clone(),
            substitute_expr(body, params, &child_shadowed),
        ],
    )
}

fn substitute_let(
    node: MacroNode<'_>,
    params: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> Expr {
    let Some(AdmittedSpecialForm::Let { bind_node, body }) = admitted_special_form(node) else {
        return node.expr.clone();
    };
    let mut scope_for_values = shadowed.clone();
    let bind_kids = bind_node.children;
    let mut new_bind_children = Vec::with_capacity(bind_kids.len());
    for [name_expr, value_expr] in bind_kids.as_chunks::<2>().0 {
        let name = symbol_name(name_expr).expect("special-form admission checked binders");
        new_bind_children.push(name_expr.clone());
        new_bind_children.push(substitute_expr(value_expr, params, &scope_for_values));
        scope_for_values.insert(name.to_string());
    }
    rebuild_macro_node(
        node,
        node.metadata.clone(),
        vec![
            rebuild_macro_node(bind_node, bind_node.metadata.clone(), new_bind_children),
            substitute_expr(body, params, &scope_for_values),
        ],
    )
}

fn substitute_match(
    node: MacroNode<'_>,
    params: &UnordMap<String, Expr>,
    shadowed: &UnordSet<String>,
) -> Expr {
    let kids = node.children;
    let mut children = Vec::with_capacity(kids.len());
    if kids.is_empty() {
        return node.expr.clone();
    }
    children.push(substitute_expr(&kids[0], params, shadowed));
    for arm in &kids[1..] {
        if let ExprCarrier::DecodedNode(DeepTag::Arm, metadata, arm_children) = arm.carrier() {
            let arm_node = MacroNode::new(arm, DeepTag::Arm, metadata, arm_children);
            let arm_kids = arm_node.children;
            if arm_kids.len() >= 3 {
                let mut arm_shadowed = shadowed.clone();
                arm_shadowed.extend(chelis_deep::pattern_binder_names(&arm_kids[0]));
                let mut arm_children = arm_kids.to_vec();
                if !is_unit_list(&arm_kids[1]) {
                    arm_children[1] = substitute_expr(&arm_kids[1], params, &arm_shadowed);
                }
                arm_children[2] = substitute_expr(&arm_kids[2], params, &arm_shadowed);
                children.push(rebuild_macro_node(
                    arm_node,
                    arm_node.metadata.clone(),
                    arm_children,
                ));
                continue;
            }
        }
        children.push(substitute_expr(arm, params, shadowed));
    }
    rebuild_macro_node(node, node.metadata.clone(), children)
}

fn hygienize_expr(expr: &Expr, counter: &mut usize, env: &UnordMap<String, String>) -> Expr {
    match expr.carrier() {
        ExprCarrier::Atom(_) => expr.clone(),
        ExprCarrier::MetadataMap(meta) => Expr::Map(
            map_meta_entries(meta, |value| hygienize_expr(value, counter, env)),
            expr.span(),
        ),
        ExprCarrier::MetadataExpression(meta) => Expr::MetaExpr(
            MetaExpr {
                metadata: map_meta_entries(&meta.metadata, |value| {
                    hygienize_expr(value, counter, env)
                }),
                expr: Box::new(hygienize_expr(&meta.expr, counter, env)),
            },
            expr.span(),
        ),
        ExprCarrier::StructuralList(elements) => Expr::BareList(
            elements
                .iter()
                .map(|child| hygienize_expr(child, counter, env))
                .collect(),
            expr.span(),
        ),
        ExprCarrier::UndecodableHead(_, _, _) => map_unknown_form(unknown_form(expr), |child| {
            hygienize_expr(child, counter, env)
        }),
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            let node = MacroNode::new(expr, tag, metadata, children);
            if tag == DeepTag::Var
                && let Some(name) = children.first().and_then(symbol_name)
                && let Some(renamed) = env.get(name)
            {
                let mut renamed_children = children.to_vec();
                renamed_children[0] = Expr::Atom(Atom::Name(renamed.clone()), children[0].span());
                return rebuild_macro_node(node, metadata.clone(), renamed_children);
            }
            match tag {
                DeepTag::Fn => hygienize_fn(node, counter, env),
                DeepTag::Let => hygienize_let(node, counter, env),
                DeepTag::Match => hygienize_match(node, counter, env),
                _ => rebuild_macro_node(
                    node,
                    map_meta_entries(metadata, |value| hygienize_expr(value, counter, env)),
                    children
                        .iter()
                        .map(|child| hygienize_expr(child, counter, env))
                        .collect(),
                ),
            }
        }
    }
}

fn hygienize_fn(node: MacroNode<'_>, counter: &mut usize, env: &UnordMap<String, String>) -> Expr {
    let Some(AdmittedSpecialForm::Fn { params_expr, body }) = admitted_special_form(node) else {
        return node.expr.clone();
    };
    let (params_expr, next_env) = hygienize_params_expr(params_expr, counter, env);
    rebuild_macro_node(
        node,
        node.metadata.clone(),
        vec![params_expr, hygienize_expr(body, counter, &next_env)],
    )
}

fn hygienize_let(node: MacroNode<'_>, counter: &mut usize, env: &UnordMap<String, String>) -> Expr {
    let Some(AdmittedSpecialForm::Let { bind_node, body }) = admitted_special_form(node) else {
        return node.expr.clone();
    };
    let mut scope_env = env.clone();
    let bind_kids = bind_node.children;
    let mut new_bind_children = Vec::with_capacity(bind_kids.len());
    for [name_expr, value_expr] in bind_kids.as_chunks::<2>().0 {
        let name = symbol_name(name_expr).expect("special-form admission checked binders");
        let fresh = fresh_name(name, counter);
        new_bind_children.push(Expr::Atom(Atom::Name(fresh.clone()), name_expr.span()));
        new_bind_children.push(hygienize_expr(value_expr, counter, &scope_env));
        scope_env.insert(name.to_string(), fresh);
    }
    rebuild_macro_node(
        node,
        node.metadata.clone(),
        vec![
            rebuild_macro_node(bind_node, bind_node.metadata.clone(), new_bind_children),
            hygienize_expr(body, counter, &scope_env),
        ],
    )
}

fn hygienize_match(
    node: MacroNode<'_>,
    counter: &mut usize,
    env: &UnordMap<String, String>,
) -> Expr {
    let kids = node.children;
    let mut children = Vec::with_capacity(kids.len());
    if kids.is_empty() {
        return node.expr.clone();
    }
    children.push(hygienize_expr(&kids[0], counter, env));
    for arm in &kids[1..] {
        if let ExprCarrier::DecodedNode(DeepTag::Arm, metadata, arm_children) = arm.carrier() {
            let arm_node = MacroNode::new(arm, DeepTag::Arm, metadata, arm_children);
            let arm_kids = arm_node.children;
            if arm_kids.len() >= 3 {
                let (pattern, arm_env) = hygienize_pattern(&arm_kids[0], counter, env);
                let mut arm_children = arm_kids.to_vec();
                arm_children[0] = pattern;
                if !is_unit_list(&arm_kids[1]) {
                    arm_children[1] = hygienize_expr(&arm_kids[1], counter, &arm_env);
                }
                arm_children[2] = hygienize_expr(&arm_kids[2], counter, &arm_env);
                children.push(rebuild_macro_node(
                    arm_node,
                    arm_node.metadata.clone(),
                    arm_children,
                ));
                continue;
            }
        }
        children.push(hygienize_expr(arm, counter, env));
    }
    rebuild_macro_node(node, node.metadata.clone(), children)
}

fn hygienize_params_expr(
    expr: &Expr,
    counter: &mut usize,
    env: &UnordMap<String, String>,
) -> (Expr, UnordMap<String, String>) {
    let mut next_env = env.clone();
    let node = match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Params, metadata, children) => {
            MacroNode::new(expr, DeepTag::Params, metadata, children)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return (expr.clone(), next_env),
    };
    let dispositions = node
        .children
        .iter()
        .map(macro_parameter_disposition)
        .collect::<Vec<_>>();
    if dispositions
        .iter()
        .any(|disposition| matches!(disposition, MacroParameterDisposition::Invalid))
    {
        return (expr.clone(), next_env);
    }
    let mut children = Vec::with_capacity(node.children.len());
    for (param, disposition) in node.children.iter().zip(dispositions) {
        let MacroParameterDisposition::Binder(name) = disposition else {
            children.push(param.clone());
            continue;
        };
        let fresh = fresh_name(name, counter);
        next_env.insert(name.to_string(), fresh.clone());
        match param {
            Expr::Atom(Atom::Name(_), _) => {
                children.push(Expr::Atom(Atom::Name(fresh), param.span()));
            }
            Expr::BareList(elements, span) => {
                children.push(Expr::BareList(
                    vec![
                        Expr::Atom(Atom::Name(fresh), elements[0].span()),
                        elements[1].clone(),
                    ],
                    *span,
                ));
            }
            Expr::MetaExpr(meta, span) => {
                let Expr::Atom(Atom::Name(_), name_span) = meta.expr.as_ref() else {
                    unreachable!("admitted prefix metadata parameter has a name")
                };
                children.push(Expr::MetaExpr(
                    MetaExpr {
                        metadata: meta.metadata.clone(),
                        expr: Box::new(Expr::Atom(Atom::Name(fresh), *name_span)),
                    },
                    *span,
                ));
            }
            Expr::Atom(_, _) | Expr::Map(_, _) | Expr::Node(_, _) | Expr::UnknownForm(_) => {
                unreachable!("admitted macro parameter retains its exact source carrier")
            }
        }
    }
    (
        rebuild_macro_node(node, node.metadata.clone(), children),
        next_env,
    )
}

fn hygienize_pattern(
    expr: &Expr,
    counter: &mut usize,
    env: &UnordMap<String, String>,
) -> (Expr, UnordMap<String, String>) {
    let node = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            MacroNode::new(expr, tag, metadata, children)
        }
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return (expr.clone(), env.clone()),
    };
    let tag = node.tag;
    let mut next_env = env.clone();
    match tag {
        DeepTag::PatVar => {
            let Some(name) = node.children.first().and_then(symbol_name) else {
                return (expr.clone(), env.clone());
            };
            let fresh = fresh_name(name, counter);
            next_env.insert(name.to_string(), fresh.clone());
            (
                rebuild_macro_node(
                    node,
                    node.metadata.clone(),
                    vec![Expr::Atom(Atom::Name(fresh), node.children[0].span())],
                ),
                next_env,
            )
        }
        DeepTag::PatAs => {
            let kids = node.children;
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
                rebuild_macro_node(
                    node,
                    node.metadata.clone(),
                    vec![Expr::Atom(Atom::Name(fresh), kids[0].span()), inner],
                ),
                inner_env,
            )
        }
        DeepTag::PatTuple | DeepTag::PatCtor => {
            let kids = node.children;
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
                rebuild_macro_node(node, node.metadata.clone(), new_children),
                working_env,
            )
        }
        DeepTag::PatRecord => {
            let mut new_children = Vec::new();
            let kids = node.children;
            if kids.is_empty() {
                return (expr.clone(), env.clone());
            }
            new_children.push(kids[0].clone());
            let mut working_env = env.clone();
            for kv in &kids[1..] {
                if let ExprCarrier::DecodedNode(DeepTag::Kv, metadata, kv_children) = kv.carrier() {
                    let kv_node = MacroNode::new(kv, DeepTag::Kv, metadata, kv_children);
                    let kv_kids = kv_node.children;
                    if kv_kids.len() == 2 {
                        let (child_pat, child_env) =
                            hygienize_pattern(&kv_kids[1], counter, &working_env);
                        working_env = child_env;
                        new_children.push(rebuild_macro_node(
                            kv_node,
                            kv_node.metadata.clone(),
                            vec![kv_kids[0].clone(), child_pat],
                        ));
                        continue;
                    }
                }
                new_children.push(kv.clone());
            }
            (
                rebuild_macro_node(node, node.metadata.clone(), new_children),
                working_env,
            )
        }
        _ => (expr.clone(), env.clone()),
    }
}

fn annotate_source_expr(expr: &Expr, invocation: &MacroSource) -> Expr {
    match expr.carrier() {
        ExprCarrier::Atom(_) => expr.clone(),
        ExprCarrier::MetadataMap(meta) => Expr::Map(
            map_meta_entries(meta, |value| annotate_source_expr(value, invocation)),
            expr.span(),
        ),
        // An inline-annotated parameter `(x {type: T})` is one binder the
        // template wrote, and its map is that binder's metadata. Provenance
        // goes on the map, and, as for a node, the annotation's values (the
        // type syntax) are not stamped again.
        ExprCarrier::StructuralList(
            [
                name @ Expr::Atom(Atom::Name(_), _),
                Expr::Map(metadata, map_span),
            ],
        ) => {
            let mut metadata = metadata.clone();
            if metadata.source().is_none() {
                metadata
                    .insert(MetadataValue::Source(invocation.clone()))
                    .expect("source is absent");
            }
            Expr::BareList(
                vec![name.clone(), Expr::Map(metadata, *map_span)],
                expr.span(),
            )
        }
        ExprCarrier::StructuralList(elements) => Expr::BareList(
            elements
                .iter()
                .map(|child| annotate_source_expr(child, invocation))
                .collect(),
            expr.span(),
        ),
        ExprCarrier::UndecodableHead(_, _, _) => map_unknown_form(unknown_form(expr), |child| {
            annotate_source_expr(child, invocation)
        }),
        ExprCarrier::MetadataExpression(meta) => {
            if matches!(meta.expr.as_ref(), Expr::Atom(Atom::Name(_), _)) {
                let mut metadata = meta.metadata.clone();
                if metadata.source().is_none() {
                    metadata
                        .insert(MetadataValue::Source(invocation.clone()))
                        .expect("source is absent");
                }
                Expr::MetaExpr(
                    MetaExpr {
                        metadata,
                        expr: meta.expr.clone(),
                    },
                    expr.span(),
                )
            } else {
                Expr::MetaExpr(
                    MetaExpr {
                        metadata: map_meta_entries(&meta.metadata, |value| {
                            annotate_source_expr(value, invocation)
                        }),
                        expr: Box::new(annotate_source_expr(&meta.expr, invocation)),
                    },
                    expr.span(),
                )
            }
        }
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            let node = MacroNode::new(expr, tag, metadata, children);
            let mut metadata = metadata.clone();
            if metadata.source().is_none() {
                metadata
                    .insert(MetadataValue::Source(invocation.clone()))
                    .expect("source is absent");
            }
            rebuild_macro_node(
                node,
                metadata,
                children
                    .iter()
                    .map(|child| annotate_source_expr(child, invocation))
                    .collect(),
            )
        }
    }
}

fn params_blockers(expr: &Expr) -> Vec<String> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Params, _, children) => children
            .iter()
            .filter_map(|param| match macro_parameter_disposition(param) {
                MacroParameterDisposition::Binder(name) => Some(name.to_string()),
                MacroParameterDisposition::Invalid => None,
            })
            .collect(),
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => Vec::new(),
    }
}

/// The invocation `(name args...)` as the historical syntax record that
/// `source` metadata carries.
fn macro_source(name: &str, args: &[Expr]) -> Expr {
    let mut elements = vec![Expr::Atom(Atom::Name(name.to_string()), zero_span())];
    elements.extend(args.iter().cloned());
    Expr::BareList(elements, zero_span())
}

fn standard_prelude_macros() -> Result<UnordMap<String, MacroDef>, ExpansionError> {
    let mut defs = UnordMap::new();
    for def in [
        prelude_linear_layer(),
        prelude_residual(),
        prelude_cross_entropy(),
    ] {
        defs.insert(def.name.clone(), def);
    }
    Ok(defs)
}

/// Rebuild only live payloads; the typed visitor preserves structural roots,
/// dtype binder data, and historical source records.
fn map_meta_entries(meta: &Metadata, mut f: impl FnMut(&Expr) -> Expr) -> Metadata {
    meta.map_expressions(&mut |value, _| f(value))
        .expect("macro rewrite preserves metadata shape")
}
fn try_map_meta_entries(
    meta: &Metadata,
    mut f: impl FnMut(&Expr) -> Result<Expr, ExpansionError>,
) -> Result<Metadata, ExpansionError> {
    meta.try_map_expressions(&mut |value, _| f(value))
}

/// Rebuild an `UnknownForm`, applying `f` to every metadata value and every
/// child (chelis#1087). Macro-relevant material — an invocation, a captured
/// symbol, a binder — can sit in either position, so both recurse.
fn map_unknown_form(data: &UnknownFormData, mut f: impl FnMut(&Expr) -> Expr) -> Expr {
    Expr::UnknownForm(Box::new(UnknownFormData {
        head: data.head.clone(),
        meta: map_meta_entries(&data.meta, &mut f),
        children: data.children.iter().map(&mut f).collect(),
        span: data.span,
    }))
}

/// The `UnknownForm` behind an undecodable-head carrier, its only source.
fn unknown_form(expr: &Expr) -> &UnknownFormData {
    let Expr::UnknownForm(data) = expr else {
        unreachable!("an undecodable head is carried only by an UnknownForm");
    };
    data
}

/// Fallible twin of [`map_unknown_form`] for the expansion walk.
fn try_map_unknown_form(
    data: &UnknownFormData,
    mut f: impl FnMut(&Expr) -> Result<Expr, ExpansionError>,
) -> Result<Expr, ExpansionError> {
    let metadata = try_map_meta_entries(&data.meta, &mut f)?;
    let mut children = Vec::with_capacity(data.children.len());
    for child in &data.children {
        children.push(f(child)?);
    }
    Ok(Expr::UnknownForm(Box::new(UnknownFormData {
        head: data.head.clone(),
        meta: metadata,
        children,
        span: data.span,
    })))
}

/// Parsed compiler-internal forms at a structural syntax position remain a
/// `BareList`; recognize their raw head only at this recorded macro boundary.
fn bare_internal_tag(elements: &[Expr]) -> Option<&str> {
    let [Expr::Atom(Atom::Name(head), _), Expr::Map(_, _), ..] = elements else {
        return None;
    };
    DeepTag::parse(head).is_none().then_some(head.as_str())
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn standard_prelude_decl_name<'a>(
    expr: &'a Expr,
    expected: DeepTag,
    prelude: &UnordMap<String, MacroDef>,
) -> Option<&'a str> {
    let kids = match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) if tag == expected => children,
        ExprCarrier::MetadataExpression(meta) => {
            return standard_prelude_decl_name(&meta.expr, expected, prelude);
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => return None,
    };
    kids.first()
        .and_then(symbol_name)
        .filter(|name| prelude.contains_key(*name))
}

fn var_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            children.first().and_then(symbol_name)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

/// The empty `()` guard of an `arm`.
fn is_unit_list(expr: &Expr) -> bool {
    matches!(expr.carrier(), ExprCarrier::StructuralList([]))
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
                app("insert", vec![var("b"), int32_lit(0), var("batch")]),
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
    Expr::node(
        DeepTag::Lit,
        Metadata::from(MetadataValue::Type(
            chelis_deep::annotations::TypeSyntax::try_new(node(
                DeepTag::TPrim,
                vec![Expr::Atom(Atom::Name("i32".to_string()), zero_span())],
            ))
            .expect("i32 type syntax"),
        )),
        vec![Expr::Atom(Atom::Int(value), zero_span())],
        zero_span(),
    )
}

fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, Metadata::default(), children, zero_span())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An annotated `(name {})` parameter, the structural list a `params`
    /// node carries for a typed binder.
    fn typed_parameter(name: &str) -> Expr {
        Expr::BareList(
            vec![
                Expr::Atom(Atom::Name(name.to_string()), zero_span()),
                Expr::Map(Metadata::default(), zero_span()),
            ],
            zero_span(),
        )
    }

    /// Both typed parameter spellings block macro expansion of their names
    /// (spec/02-surf-syntax.md §P5). Negative control: a structural list that
    /// is not a name-and-annotations pair binds nothing.
    #[test]
    fn typed_parameter_is_a_macro_blocker() {
        let params = node(DeepTag::Params, vec![typed_parameter("typed_parameter")]);
        assert_eq!(
            params_blockers(&params),
            vec!["typed_parameter".to_string()]
        );

        let prefix_parameter = Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(Expr::Atom(Atom::Name("record".to_string()), zero_span())),
            },
            zero_span(),
        );
        assert_eq!(
            params_blockers(&node(DeepTag::Params, vec![prefix_parameter])),
            vec!["record".to_string()]
        );

        let malformed = Expr::BareList(
            vec![
                Expr::Atom(Atom::Name("not_a_parameter".to_string()), zero_span()),
                Expr::Map(Metadata::default(), zero_span()),
                Expr::Atom(Atom::Name("extra".to_string()), zero_span()),
            ],
            zero_span(),
        );
        assert!(params_blockers(&node(DeepTag::Params, vec![malformed])).is_empty());
    }

    /// Hygiene renames a macro-introduced typed parameter and keeps its
    /// annotated structural carrier.
    #[test]
    fn typed_parameter_is_hygienized_as_a_binder() {
        let params = node(DeepTag::Params, vec![typed_parameter("typed_parameter")]);
        let mut counter = 0;
        let (result, env) = hygienize_params_expr(&params, &mut counter, &UnordMap::new());
        assert_eq!(counter, 1);
        assert_eq!(
            env.get("typed_parameter").map(String::as_str),
            Some("typed_parameter_macro_0")
        );
        assert_eq!(
            result,
            node(
                DeepTag::Params,
                vec![typed_parameter("typed_parameter_macro_0")]
            )
        );
    }

    #[test]
    fn source_specific_parameters_do_not_freeze_fn_rewrite_passes() {
        // Each parameter with the params node hygiene leaves behind and the
        // fresh names it spends: both typed parameter spellings are binders.
        let parameters = [
            (
                typed_parameter("structural_parameter"),
                node(
                    DeepTag::Params,
                    vec![typed_parameter("structural_parameter_macro_0")],
                ),
                1,
            ),
            {
                let parameter = Expr::MetaExpr(
                    chelis_deep::MetaExpr {
                        metadata: Metadata::default(),
                        expr: Box::new(Expr::Atom(
                            Atom::Name("metadata_parameter".to_string()),
                            zero_span(),
                        )),
                    },
                    zero_span(),
                );
                let renamed = Expr::MetaExpr(
                    chelis_deep::MetaExpr {
                        metadata: Metadata::default(),
                        expr: Box::new(Expr::Atom(
                            Atom::Name("metadata_parameter_macro_0".to_string()),
                            zero_span(),
                        )),
                    },
                    zero_span(),
                );
                (parameter, node(DeepTag::Params, vec![renamed]), 1)
            },
        ];

        for (parameter, hygienized_params, fresh_names) in parameters {
            let params = node(DeepTag::Params, vec![parameter]);
            let mut macros = UnordMap::new();
            macros.insert(
                "rewrite".to_string(),
                MacroDef {
                    name: "rewrite".to_string(),
                    params: Vec::new(),
                    body: int32_lit(7),
                },
            );
            let function = node(
                DeepTag::Fn,
                vec![params.clone(), app("rewrite", Vec::new())],
            );
            let mut expander = Expander::new(10, UnordMap::new());
            let expanded = expander
                .expand_expr(&function, &macros, &Scope::default())
                .expect("valid source-specific parameter admits expansion");
            let ExprCarrier::DecodedNode(DeepTag::Fn, _, expanded_children) = expanded.carrier()
            else {
                panic!("expanded function retains its decoded carrier");
            };
            assert_eq!(
                expanded_children.first(),
                Some(&params),
                "expansion must preserve the source-specific parameter carrier"
            );
            assert!(
                matches!(
                    expanded_children.get(1).map(Expr::carrier),
                    Some(ExprCarrier::DecodedNode(
                        DeepTag::Lit,
                        _,
                        [Expr::Atom(Atom::Int(7), _)]
                    ))
                ),
                "a typed parameter carrier must not freeze body expansion: {expanded:?}"
            );

            let function = node(DeepTag::Fn, vec![params.clone(), var("value")]);
            let mut substitutions = UnordMap::new();
            substitutions.insert("value".to_string(), int32_lit(9));
            assert_eq!(
                substitute_expr(&function, &substitutions, &UnordSet::new()),
                node(DeepTag::Fn, vec![params.clone(), int32_lit(9)]),
                "a typed parameter carrier must not freeze substitution"
            );

            let function = node(DeepTag::Fn, vec![params.clone(), var("outer")]);
            let mut env = UnordMap::new();
            env.insert("outer".to_string(), "renamed".to_string());
            let mut counter = 0;
            assert_eq!(
                hygienize_expr(&function, &mut counter, &env),
                node(DeepTag::Fn, vec![hygienized_params, var("renamed")]),
                "a parameter carrier must not freeze outer hygiene"
            );
            assert_eq!(
                counter, fresh_names,
                "only a binder parameter spends a fresh name"
            );
        }
    }

    #[test]
    fn malformed_fn_and_let_shapes_are_opaque_to_every_rewrite_pass() {
        let rewrite_call = app("rewrite", vec![]);
        let mut macros = UnordMap::new();
        macros.insert(
            "rewrite".to_string(),
            MacroDef {
                name: "rewrite".to_string(),
                params: Vec::new(),
                body: int32_lit(7),
            },
        );

        let malformed_fn = node(DeepTag::Fn, vec![int32_lit(0), rewrite_call.clone()]);
        let malformed_let = node(
            DeepTag::Let,
            vec![
                node(DeepTag::Bind, vec![int32_lit(0), rewrite_call.clone()]),
                var("outer"),
            ],
        );
        let mut expander = Expander::new(10, UnordMap::new());
        assert_eq!(
            expander
                .expand_expr(&malformed_fn, &macros, &Scope::default())
                .expect("malformed fn remains opaque"),
            malformed_fn,
            "expansion must not rewrite a fn whose first child is not Params"
        );
        assert_eq!(
            expander
                .expand_expr(&malformed_let, &macros, &Scope::default())
                .expect("malformed let remains opaque"),
            malformed_let,
            "expansion must not rewrite a let with a non-name binder"
        );

        let mut substitutions = UnordMap::new();
        substitutions.insert("value".to_string(), int32_lit(9));
        let malformed_fn_params = node(
            DeepTag::Fn,
            vec![node(DeepTag::Params, vec![int32_lit(0)]), var("value")],
        );
        let malformed_let_owner = node(DeepTag::Let, vec![int32_lit(0), var("value")]);
        assert_eq!(
            substitute_expr(&malformed_fn_params, &substitutions, &UnordSet::new()),
            malformed_fn_params,
            "substitution must not enter a fn with a non-name parameter"
        );
        assert_eq!(
            substitute_expr(&malformed_let_owner, &substitutions, &UnordSet::new()),
            malformed_let_owner,
            "substitution must not enter a let whose first child is not Bind"
        );

        let mut env = UnordMap::new();
        env.insert("outer".to_string(), "renamed".to_string());
        let malformed_hygiene_fn = node(
            DeepTag::Fn,
            vec![node(DeepTag::Params, vec![int32_lit(0)]), var("outer")],
        );
        let malformed_hygiene_let = node(
            DeepTag::Let,
            vec![
                node(DeepTag::Bind, vec![int32_lit(0), var("outer")]),
                var("outer"),
            ],
        );
        let mut counter = 0;
        assert_eq!(
            hygienize_expr(&malformed_hygiene_fn, &mut counter, &env),
            malformed_hygiene_fn,
            "hygiene must not enter a fn with a non-name parameter"
        );
        assert_eq!(
            hygienize_expr(&malformed_hygiene_let, &mut counter, &env),
            malformed_hygiene_let,
            "hygiene must not synthesize a binder for an invalid let name"
        );
        assert_eq!(counter, 0, "invalid binders must not consume fresh names");
    }

    /// Collect every `let` binder name and every `var` reference, in
    /// pre-order, skipping metadata.
    fn let_binders_and_references(expr: &Expr, binders: &mut Vec<String>, refs: &mut Vec<String>) {
        let ExprCarrier::DecodedNode(tag, _, children) = expr.carrier() else {
            return;
        };
        match tag {
            DeepTag::Bind => {
                for (index, child) in children.iter().enumerate() {
                    if index % 2 == 0 {
                        binders.extend(symbol_name(child).map(str::to_string));
                    } else {
                        let_binders_and_references(child, binders, refs);
                    }
                }
            }
            DeepTag::Var => refs.extend(var_name(expr).map(str::to_string)),
            _ => {
                for child in children {
                    let_binders_and_references(child, binders, refs);
                }
            }
        }
    }

    /// chelis#1320 capture polarity, over stamped `let`/`bind` nodes: the
    /// `let` binder a macro body introduces is renamed together with its
    /// references, while the caller's own binder of the same name, and the
    /// argument that refers to it, are left alone.
    #[test]
    fn macro_introduced_let_binder_is_renamed_and_user_binder_is_not() {
        let program = chelis_deep::parser::parse_str(
            "(defmacro {} shadow (params {} x) \
               (let {} (bind {} tmp (var {} x)) \
                 (app {} (var {} add) (var {} tmp) (var {} x)))) \
             (def {} f (let {} (bind {} tmp (lit {} 1)) \
               (app {} (var {} shadow) (var {} tmp))))",
        )
        .expect("fixture parses");
        let options = ExpansionOptions {
            max_iterations: 10,
            load_std_prelude: false,
        };
        let expanded = expand_program(&program, &options).expect("expansion succeeds");
        assert_eq!(expanded.expansions(), 1);
        let [def] = expanded.exprs() else {
            panic!("one declaration survives expansion: {:?}", expanded.exprs());
        };

        let mut binders = Vec::new();
        let mut references = Vec::new();
        let_binders_and_references(def, &mut binders, &mut references);
        assert_eq!(
            binders,
            ["tmp", "tmp_macro_0"],
            "the user binder keeps its name; the macro binder is renamed"
        );
        assert_eq!(
            references,
            ["tmp", "add", "tmp_macro_0", "tmp"],
            "the macro body reads its renamed binder, and the argument still \
             reads the caller's binder"
        );
    }
}
