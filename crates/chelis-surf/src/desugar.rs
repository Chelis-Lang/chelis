//! Desugars Surface AST into Deep (s-expression) AST.
//!
//! Every Deep node is a 3-tuple: (tag {} children...)
//! where {} is an inline metadata map.

use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast as deep;

use crate::ast::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn desugar_program(decls: &[Decl]) -> Vec<deep::Expr> {
    let ctx = DesugarCtx::new(decls);
    decls
        .iter()
        .flat_map(|decl| ctx.desugar_decl(decl))
        .collect()
}

pub fn desugar_decl_only(decl: &Decl) -> Vec<deep::Expr> {
    DesugarCtx::default().desugar_decl(decl)
}

pub fn desugar_expr_only(expr: &Expr) -> deep::Expr {
    DesugarCtx::default().desugar_expr(expr)
}

#[derive(Default)]
struct DesugarCtx {
    top_level_fn_params: HashMap<String, Vec<String>>,
    /// Per-function tensor element types declared in the function's
    /// signature, indexed by parameter position. `None` for non-tensor
    /// parameters or parameters with no declared type.
    ///
    /// Used by the contextual tensor-literal inference rule
    /// (`spec/02-surf-syntax.md` §P10b, `spec/04-type-system.md` §5.6),
    /// position 2: the corresponding argument position of a call whose
    /// callee has a declared signature with a tensor parameter at that
    /// position.
    top_level_fn_tensor_param_prec: HashMap<String, Vec<Option<String>>>,
    /// Names that carry an explicit standalone `sig`/signature declaration
    /// (`Decl::Sig`). When a `def` of the same name also has inline
    /// annotations, `desugar_fun_def` would otherwise synthesize a second
    /// `defsig` filling every un-annotated position with a wildcard
    /// `(t-var {} _)`. That synthesized signature is last-write-wins in the
    /// type checker's defsig binding (`chelis-types` `collect_declarations`),
    /// so it silently overwrites the concrete explicit `sig`, dropping the
    /// body-vs-signature contract on the un-annotated positions
    /// (chelis#285). When an explicit sig exists, the synthesized one is
    /// strictly redundant and weaker, so we suppress it here.
    explicit_sig_names: HashSet<String>,
    /// Explicit effect clauses (`! { ... }`) declared on each `def`, keyed by
    /// name. The effect upper-bound check reads the declared effect set only
    /// from a `defsig`'s `t-fn` `eff` metadata
    /// (`chelis-effects::declared_effects_from_defsig`). A `def`'s clause
    /// reaches that check solely via the synthesized `defsig` — but
    /// `explicit_sig_names` now suppresses that synthesized `defsig`. So when
    /// an explicit `sig` declares no effects of its own, `Decl::Sig`
    /// desugaring inherits the same-named `def`'s clause from this map;
    /// otherwise suppression would silently drop the def's effect contract
    /// (chelis#285). Stored even for an empty `! {}` (which declares "no
    /// effects" and is distinct from no annotation at all).
    def_effects: HashMap<String, Vec<EffectExpr>>,
}

impl DesugarCtx {
    fn new(decls: &[Decl]) -> Self {
        let mut top_level_fn_params = HashMap::new();
        let mut top_level_fn_tensor_param_prec = HashMap::new();
        let mut explicit_sig_names = HashSet::new();
        let mut def_effects = HashMap::new();
        for decl in decls {
            for_each_decl(decl, &mut |d| {
                collect_top_level_fn_params(d, &mut top_level_fn_params);
                collect_top_level_fn_tensor_param_prec(d, &mut top_level_fn_tensor_param_prec);
                collect_explicit_sig_names(d, &mut explicit_sig_names);
                collect_def_effects(d, &mut def_effects);
            });
        }
        Self {
            top_level_fn_params,
            top_level_fn_tensor_param_prec,
            explicit_sig_names,
            def_effects,
        }
    }
}

#[cfg(test)]
fn desugar_decl(decl: &Decl) -> Vec<deep::Expr> {
    DesugarCtx::default().desugar_decl(decl)
}

#[cfg(test)]
fn desugar_expr(expr: &Expr) -> deep::Expr {
    DesugarCtx::default().desugar_expr(expr)
}

// ---------------------------------------------------------------------------
// Helpers for building Deep AST nodes (3-tuple format)
// ---------------------------------------------------------------------------

fn sp() -> Span {
    Span::new(0, 0)
}

fn sym(s: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(s.to_string()), sp())
}

fn int(n: i64) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Int(n), sp())
}

fn string(value: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Str(value.to_string()), sp())
}

fn meta_empty() -> deep::Expr {
    deep::Expr::Map(deep::MetaMap::default(), sp())
}

fn surf_span_id(span: Span) -> Option<String> {
    if span.len == 0 {
        None
    } else {
        Some(format!("surf:{}..{}", span.offset, span.end()))
    }
}

fn span_entry(span: Span) -> Option<(String, deep::Expr)> {
    surf_span_id(span).map(|id| {
        (
            "span".to_string(),
            deep::Expr::Atom(deep::Atom::Str(id), sp()),
        )
    })
}

fn meta_with_type(ty: deep::Expr) -> deep::Expr {
    meta_with_entries(vec![("type".to_string(), ty)])
}

fn meta_with_entries(entries: Vec<(String, deep::Expr)>) -> deep::Expr {
    deep::Expr::Map(deep::MetaMap { entries }, sp())
}

/// Build a 3-tuple Deep node: (tag {} children...)
fn node(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![sym(tag), meta_empty()];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, sp())
}

/// Build a 3-tuple Deep node with custom metadata: (tag {meta} children...)
fn node_meta(tag: &str, meta: deep::Expr, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![sym(tag), meta];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, sp())
}

/// Variable reference: (var {} name)
fn dvar(name: &str) -> deep::Expr {
    node("var", vec![sym(name)])
}

fn attach_span_metadata(expr: deep::Expr, span: Span) -> deep::Expr {
    let Some(entry) = span_entry(span) else {
        return expr;
    };

    match expr {
        deep::Expr::List(list, list_span) => {
            let mut elements = list.elements;
            if let Some(deep::Expr::Map(map, _)) = elements.get_mut(1) {
                map.entries.retain(|(key, _)| key != "span");
                map.entries.push(entry);
            }
            deep::Expr::List(deep::List { elements }, list_span)
        }
        other => other,
    }
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
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
    }
}

/// Build a bare list (no tag/meta) for structural helpers like params, bind
fn bare_list(elements: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::List(deep::List { elements }, sp())
}

fn lower_module_path(path: &str) -> String {
    path.to_ascii_lowercase()
}

fn desugar_param(param: &Param) -> deep::Expr {
    desugar_param_with_dims(param, &HashSet::new())
}

fn desugar_param_with_dims(param: &Param, dim_vars: &HashSet<String>) -> deep::Expr {
    desugar_param_with_scope(param, dim_vars, &HashSet::new())
}

/// Desugar a parameter with both a declared dim-vars scope and a
/// quantified type-variable scope. The tvar scope is non-empty for
/// `def f[..](...)` parameters where the def's quantifier list (`[..]`)
/// participates in the contextual precision rule of
/// `spec/02-surf-syntax.md` §P4b: a name in the precision slot of a
/// `tensor[..., <name>]` annotation that matches the def's quantifier
/// list desugars to `(t-var {} <name>)` rather than
/// `(t-prim {} <name>)`. This is the WS-A6 extension of the WS-A5 rule
/// from sigs to def parameter annotations.
fn desugar_param_with_scope(
    param: &Param,
    dim_vars: &HashSet<String>,
    tvar_set: &HashSet<String>,
) -> deep::Expr {
    match &param.ty {
        Some(ty) if typed_param_needs_meta_wrapper(&param.name) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                entries: vec![(
                    "type".to_string(),
                    desugar_type_with_scope(ty, dim_vars, tvar_set),
                )],
                expr: Box::new(sym(&param.name)),
            },
            sp(),
        ),
        Some(ty) => deep::Expr::List(
            deep::List {
                elements: vec![
                    sym(&param.name),
                    meta_with_type(desugar_type_with_scope(ty, dim_vars, tvar_set)),
                ],
            },
            sp(),
        ),
        None => sym(&param.name),
    }
}

fn typed_param_needs_meta_wrapper(name: &str) -> bool {
    matches!(
        name,
        "module"
            | "import"
            | "import-all"
            | "export"
            | "def"
            | "defsig"
            | "deftype"
            | "typealias"
            | "variant"
            | "field"
            | "defdim"
            | "fn"
            | "app"
            | "let"
            | "match"
            | "arm"
            | "if"
            | "var"
            | "lit"
            | "record"
            | "access"
            | "pipe"
            | "block"
            | "tuple"
            | "tuple-get"
            | "record-update"
            | "par"
            | "pat-var"
            | "pat-lit"
            | "pat-ctor"
            | "pat-tuple"
            | "pat-record"
            | "pat-wild"
            | "pat-as"
            | "t-prim"
            | "t-fn"
            | "t-tensor"
            | "t-adt"
            | "t-var"
            | "t-unit"
            | "t-tuple"
            | "d-name"
            | "d-var"
            | "d-lit"
            | "grad"
            | "vmap"
            | "jit"
            | "realize"
            | "cast"
            | "copy"
            | "quote"
            | "unquote"
            | "splice"
            | "params"
            | "bind"
            | "kv"
            | "defmacro"
            | "expand"
            | "effects"
            | "resource"
            | "handle-effect"
    )
}

/// Inject a type annotation into the metadata of a desugared expression.
fn inject_type_metadata(expr: deep::Expr, ty: deep::Expr) -> deep::Expr {
    match expr {
        deep::Expr::List(list, span) => {
            let mut elements = list.elements;
            if elements.len() >= 2 {
                let mut entries = match elements.remove(1) {
                    deep::Expr::Map(map, _) => map
                        .entries
                        .into_iter()
                        .filter(|(key, _)| key != "type")
                        .collect(),
                    _ => Vec::new(),
                };
                entries.push(("type".to_string(), ty));
                elements.insert(1, meta_with_entries(entries));
            }
            deep::Expr::List(deep::List { elements }, span)
        }
        // For atoms, wrap in an annotated var node
        other => node_meta("var", meta_with_type(ty), vec![other]),
    }
}

fn desugar_effect_set(effects: &[EffectExpr]) -> deep::Expr {
    let mut children = Vec::new();
    for effect in effects {
        children.push(match effect {
            EffectExpr::Diff(_) => sym("diff"),
            EffectExpr::Random(_) => sym("random"),
            EffectExpr::Accum(_) => sym("accum"),
            EffectExpr::Io(_) => sym("io"),
            EffectExpr::Test(_) => sym("test"),
            EffectExpr::Resource(device, _) => node(
                "resource",
                vec![deep::Expr::Atom(deep::Atom::Str(device.clone()), sp())],
            ),
        });
    }
    node("effects", children)
}

fn apply_effect_metadata(ty_expr: deep::Expr, effects: &Option<Vec<EffectExpr>>) -> deep::Expr {
    match (effects, ty_expr) {
        // An explicit `! { ... }` clause — even the empty `! {}` — must be preserved in
        // the Deep AST so the effect checker can distinguish "declared empty" from
        // "no annotation" when validating declared vs inferred effects.
        (Some(effects), deep::Expr::List(list, span)) => {
            let mut elements = list.elements;
            if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = elements.first()
                && tag == "t-fn"
            {
                elements[1] =
                    meta_with_entries(vec![("eff".to_string(), desugar_effect_set(effects))]);
            }
            deep::Expr::List(deep::List { elements }, span)
        }
        (_, other) => other,
    }
}

fn fresh_pipe_param_name(stage: &Expr) -> String {
    let base = "__chelis_pipe";
    let mut index = 0;
    loop {
        let candidate = if index == 0 {
            base.to_string()
        } else {
            format!("{base}{index}")
        };
        if !expr_mentions_name(stage, &candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn expr_mentions_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Lit(_, _) => false,
        Expr::Var(found, _) | Expr::Constructor(found, _) => found == name,
        Expr::List(items, _) => items.iter().any(|item| expr_mentions_name(item, name)),
        Expr::Apply(func, args, _) => {
            expr_mentions_name(func, name) || args.iter().any(|arg| expr_mentions_name(arg, name))
        }
        Expr::Record(_, fields, _) => fields
            .iter()
            .any(|(_, value)| expr_mentions_name(value, name)),
        Expr::Access(base, field, _) => expr_mentions_name(base, name) || field == name,
        Expr::TupleGet(base, _, _) => expr_mentions_name(base, name),
        Expr::Binary(_, lhs, rhs, _) => {
            expr_mentions_name(lhs, name) || expr_mentions_name(rhs, name)
        }
        Expr::Unary(_, operand, _) => expr_mentions_name(operand, name),
        Expr::Pipe(seed, stages, _) => {
            expr_mentions_name(seed, name)
                || stages.iter().any(|stage| expr_mentions_name(stage, name))
        }
        Expr::If(cond, then_e, else_e, _) => {
            expr_mentions_name(cond, name)
                || expr_mentions_name(then_e, name)
                || expr_mentions_name(else_e, name)
        }
        Expr::Match(scrutinee, arms, _) => {
            expr_mentions_name(scrutinee, name)
                || arms.iter().any(|arm| {
                    pattern_mentions_name(&arm.pattern, name)
                        || arm
                            .guard
                            .as_ref()
                            .is_some_and(|guard| expr_mentions_name(guard, name))
                        || expr_mentions_name(&arm.body, name)
                })
        }
        Expr::Block(bindings, body, _) => {
            bindings.iter().any(|binding| {
                let_pattern_mentions_name(&binding.pattern, name)
                    || binding
                        .ty
                        .as_ref()
                        .is_some_and(|ty| type_mentions_name(ty, name))
                    || expr_mentions_name(&binding.value, name)
            }) || expr_mentions_name(body, name)
        }
        Expr::Lambda(params, body, _) => {
            params.iter().any(|param| {
                param.name == name
                    || param
                        .ty
                        .as_ref()
                        .is_some_and(|ty| type_mentions_name(ty, name))
            }) || expr_mentions_name(body, name)
        }
        Expr::Tuple(items, _) | Expr::Par(items, _) => {
            items.iter().any(|item| expr_mentions_name(item, name))
        }
        Expr::Cast(expr, precision, _) => expr_mentions_name(expr, name) || precision == name,
        Expr::Grad(expr, wrt, _) => {
            expr_mentions_name(expr, name)
                || wrt
                    .as_ref()
                    .is_some_and(|names| names.iter().any(|wrt_name| wrt_name == name))
        }
        Expr::Vmap(expr, _, _)
        | Expr::Jit(expr, _)
        | Expr::Realize(expr, _)
        | Expr::Copy(expr, _)
        | Expr::Borrow(expr, _)
        | Expr::Annotate(expr, _, _) => expr_mentions_name(expr, name),
        Expr::WithSeed(seed, body, _) | Expr::WithDevice(seed, body, _) => {
            expr_mentions_name(seed, name) || expr_mentions_name(body, name)
        }
    }
}

fn let_pattern_mentions_name(pattern: &LetPattern, name: &str) -> bool {
    match pattern {
        LetPattern::Var(found, _) => found == name,
        LetPattern::Wildcard(_) => false,
        LetPattern::Tuple(items, _) => items
            .iter()
            .any(|item| let_pattern_mentions_name(item, name)),
    }
}

fn pattern_mentions_name(pattern: &Pattern, name: &str) -> bool {
    match pattern {
        Pattern::Wildcard(_) | Pattern::Lit(_, _) => false,
        Pattern::Var(found, _) => found == name,
        Pattern::Constructor(found, items, _) => {
            found == name || items.iter().any(|item| pattern_mentions_name(item, name))
        }
        Pattern::Tuple(items, _) => items.iter().any(|item| pattern_mentions_name(item, name)),
        Pattern::Record(found, fields, _) => {
            found == name
                || fields
                    .iter()
                    .any(|(field, value)| field == name || pattern_mentions_name(value, name))
        }
        Pattern::As(found, inner, _) => found == name || pattern_mentions_name(inner, name),
    }
}

fn type_mentions_name(ty: &TypeExpr, name: &str) -> bool {
    match ty {
        TypeExpr::Named(found, _) => found == name,
        TypeExpr::RankSpread(found, _) => found == name,
        TypeExpr::Tensor(items, precision, _) => {
            precision == name || items.iter().any(|item| type_mentions_name(item, name))
        }
        TypeExpr::Arrow(args, ret, _) => {
            args.iter().any(|arg| type_mentions_name(arg, name)) || type_mentions_name(ret, name)
        }
        TypeExpr::Ref(inner, _) => type_mentions_name(inner, name),
        TypeExpr::App(found, args, _) => {
            found == name || args.iter().any(|arg| type_mentions_name(arg, name))
        }
        TypeExpr::Tuple(items, _) => items.iter().any(|item| type_mentions_name(item, name)),
        TypeExpr::Infer(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Primitive type names
// ---------------------------------------------------------------------------

// `f8e4m3` is intentionally absent: spec/04-type-system.md §1.1.1
// rejects it as a deferred precision. Keeping the name out of the
// primitives list prevents the implicit-quantifier collector from
// treating `f8e4m3` as a fresh tvar candidate, so the type-checker's
// §1.1.1 rejection path fires with the correct diagnostic.
const PRIMITIVES: &[&str] = &[
    "f32", "f64", "f16", "bf16", "int8", "int16", "int32", "int64", "bool", "string", "unit",
];

/// Unsigned dtype names per `spec/04-type-system.md` §1.1.2. These are
/// not in the active numeric primitive set, but they are well-known
/// dtype identifiers that users (especially LLMs translating from
/// numpy/PyTorch) reach for. Treat them as "intended-precision"
/// identifiers in desugar so they reach the type-checker's §1.1.2
/// rejection path with a precise diagnostic, NOT as candidate
/// quantified type variables.
///
/// Mirrors `chelis_types::infer::is_unsigned_dtype_name`. Kept as a
/// parallel const here because chelis-surf does not depend on
/// chelis-types and pulling in the dependency just for this list
/// would invert the desugar / typecheck layering.
const UNSIGNED_DTYPE_NAMES: &[&str] = &[
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

/// Visit `decl` and every declaration nested inside a `Decl::Module`,
/// calling `visit` on each. Centralizes the module descent shared by the
/// `DesugarCtx::new` pre-pass collectors so a future nesting variant only
/// needs handling in one place. Every idiomatic Surf source wraps its
/// declarations in a single `module`, so without this descent the collectors
/// would see only the wrapper and miss everything inside.
fn for_each_decl(decl: &Decl, visit: &mut impl FnMut(&Decl)) {
    visit(decl);
    if let Decl::Module { decls, .. } = decl {
        for d in decls {
            for_each_decl(d, visit);
        }
    }
}

fn collect_top_level_fn_params(decl: &Decl, out: &mut HashMap<String, Vec<String>>) {
    if let Decl::FunDef { name, params, .. } | Decl::Property { name, params, .. } = decl {
        out.insert(
            name.clone(),
            params.iter().map(|param| param.name.clone()).collect(),
        );
    }
}

/// Collect names that carry an explicit standalone `sig` declaration, so
/// `desugar_fun_def` can suppress the redundant wildcard-filled `defsig` it
/// would otherwise synthesize for a same-name annotated `def` (chelis#285).
fn collect_explicit_sig_names(decl: &Decl, out: &mut HashSet<String>) {
    if let Decl::Sig { name, .. } = decl {
        out.insert(name.clone());
    }
}

/// Collect the explicit effect clause (`! { ... }`) declared on each `def`,
/// keyed by name, so `Decl::Sig` desugaring can inherit it when the explicit
/// sig declares no effects of its own (chelis#285 — see the `def_effects`
/// field doc). An empty `! {}` is stored too: it declares "no effects" and
/// must be distinguished from no annotation at all.
fn collect_def_effects(decl: &Decl, out: &mut HashMap<String, Vec<EffectExpr>>) {
    if let Decl::FunDef {
        name,
        effects: Some(effects),
        ..
    } = decl
    {
        out.insert(name.clone(), effects.clone());
    }
}

/// Collect per-position tensor element-prim names from each top-level
/// function's declared signature. Used by the contextual tensor-literal
/// inference rule (spec §P10b / §5.6) to narrow numeric literals in
/// argument positions whose declared parameter type is a tensor.
///
/// `Decl::Sig` (a separate signature declaration) is also collected so
/// `sig f: tensor[3, f64] -> ...` followed by an untyped `def f` participates.
fn collect_top_level_fn_tensor_param_prec(
    decl: &Decl,
    out: &mut HashMap<String, Vec<Option<String>>>,
) {
    match decl {
        Decl::FunDef { name, params, .. } => {
            let entry: Vec<Option<String>> = params
                .iter()
                .map(|p| p.ty.as_ref().and_then(tensor_element_prim_name))
                .collect();
            // Only insert if at least one parameter has a tensor element
            // type — otherwise an entry would still be returned but with
            // all-None which the lookup harmlessly ignores. Cheap
            // pre-filter to keep the map sparse.
            if entry.iter().any(Option::is_some) {
                out.insert(name.clone(), entry);
            }
        }
        Decl::Sig {
            name,
            ty: TypeExpr::Arrow(args, _ret, _),
            ..
        } => {
            // sig f: A -> B -> C is a flat Arrow; collect each non-return
            // arrow position.
            let entry: Vec<Option<String>> = args.iter().map(tensor_element_prim_name).collect();
            if entry.iter().any(Option::is_some) {
                out.insert(name.clone(), entry);
            }
        }
        _ => {}
    }
}

/// Return the precision name (e.g. `"f64"`, `"int32"`) for a tensor type
/// expression, or `None` for any other shape. Tensor type expressions in
/// Surf carry the precision as a `String` in `TypeExpr::Tensor`.
fn tensor_element_prim_name(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Tensor(_, prec, _) => Some(prec.clone()),
        _ => None,
    }
}

impl DesugarCtx {
    fn desugar_pipe_stage(&self, stage: &Expr, local_fn_params: &[String]) -> deep::Expr {
        match stage {
            Expr::Apply(func, args, span) if !args.is_empty() => {
                let pipe_param = fresh_pipe_param_name(stage);
                let mut applied_args = Vec::with_capacity(args.len() + 1);
                applied_args.push(Expr::Var(pipe_param.clone(), *span));
                applied_args.extend(args.iter().cloned());
                let lambda = Expr::Lambda(
                    vec![Param {
                        name: pipe_param,
                        ty: None,
                        span: *span,
                    }],
                    Box::new(Expr::Apply(func.clone(), applied_args, *span)),
                    *span,
                );
                self.desugar_expr_with_scope(&lambda, local_fn_params)
            }
            _ => self.desugar_expr_with_scope(stage, local_fn_params),
        }
    }

    fn desugar_decl(&self, decl: &Decl) -> Vec<deep::Expr> {
        match decl {
            Decl::FunDef {
                name,
                dim_params,
                params,
                ret_ty,
                effects,
                body,
                ..
            } => self.desugar_fun_def(name, dim_params, params, ret_ty, effects, body),

            Decl::Property {
                name,
                params,
                preconditions,
                body,
                options,
                ..
            } => self.desugar_property(name, params, preconditions, body, options),

            Decl::LetDef {
                name,
                ty: Some(t),
                value,
                ..
            } => {
                // Position 1 (spec §P10b / §5.6): RHS of a let-binding
                // whose declared type is a tensor type. Narrow numeric
                // literals in `value` to the tensor element type.
                let body = match (tensor_element_prim_name(t), value) {
                    (Some(prec), Expr::List(items, _)) => {
                        self.desugar_list_as_tensor_literal(items, &prec, &[])
                    }
                    _ => self.desugar_expr(value),
                };
                vec![
                    node("defsig", vec![sym(name), desugar_type(t)]),
                    node("def", vec![sym(name), body]),
                ]
            }

            Decl::LetDef {
                name,
                ty: None,
                value,
                ..
            } => {
                vec![node("def", vec![sym(name), self.desugar_expr(value)])]
            }

            Decl::MacroDef {
                name, params, body, ..
            } => {
                let params = node("params", params.iter().map(|param| sym(param)).collect());
                vec![node(
                    "defmacro",
                    vec![sym(name), params, self.desugar_expr(body)],
                )]
            }

            Decl::TypeDef {
                name,
                params,
                variants,
                ..
            } => vec![desugar_type_def(name, params, variants)],

            Decl::TypeAlias {
                name, params, ty, ..
            } => {
                let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
                vec![node(
                    "typealias",
                    vec![sym(name), param_list, desugar_type(ty)],
                )]
            }

            Decl::Sig {
                name, ty, effects, ..
            } => {
                // chelis#285: the synthesized `defsig` that used to carry a
                // same-named `def`'s `! { ... }` clause is suppressed when this
                // explicit sig exists (see `desugar_fun_def`). The effect
                // upper-bound check reads the declared effect set only from a
                // `defsig`'s `t-fn` `eff` metadata, so if this sig declares no
                // effects of its own, inherit the def's clause here. Otherwise
                // suppressing the synthesized `defsig` would silently drop the
                // def's effect contract and let its body leak effects unchecked.
                let effects = effects
                    .clone()
                    .or_else(|| self.def_effects.get(name).cloned());
                vec![node(
                    "defsig",
                    vec![
                        sym(name),
                        // WS-A5: standalone sigs use the contextual rule so a
                        // lowercase non-primitive name in the precision slot
                        // becomes a quantified type variable per
                        // spec/04-type-system.md §5.8.
                        apply_effect_metadata(desugar_sig_type(ty, &HashSet::new()), &effects),
                    ],
                )]
            }

            Decl::Dim { names, .. } => names
                .iter()
                .map(|name| node("defdim", vec![sym(name)]))
                .collect(),

            Decl::Module { name, decls, .. } => {
                let mut children = vec![sym(&lower_module_path(name))];
                for d in decls {
                    children.extend(self.desugar_decl(d));
                }
                vec![node("module", children)]
            }

            Decl::Import { module, kind, .. } => match kind {
                ImportKind::Names(ns) => {
                    let name_list = bare_list(ns.iter().map(|n| sym(n)).collect());
                    vec![node(
                        "import",
                        vec![sym(&lower_module_path(module)), name_list],
                    )]
                }
                ImportKind::Qualified => vec![node(
                    "import",
                    vec![sym(&lower_module_path(module)), bare_list(vec![])],
                )],
                ImportKind::All => vec![node("import-all", vec![sym(&lower_module_path(module))])],
            },

            Decl::Export { names, .. } => {
                let mut children = Vec::new();
                for n in names {
                    children.push(sym(n));
                }
                vec![node("export", children)]
            }
        }
    }

    fn desugar_fun_def(
        &self,
        name: &str,
        dim_params: &[String],
        params: &[Param],
        ret_ty: &Option<TypeExpr>,
        effects: &Option<Vec<EffectExpr>>,
        body: &Expr,
    ) -> Vec<deep::Expr> {
        // Function-level dim params are polymorphic d-vars, NOT module-level defdim.
        // Build a set so desugar_type_with_scope treats them as d-var.
        let dim_set: HashSet<String> = dim_params.iter().cloned().collect();

        // WS-A6 (spec/02-surf-syntax.md §P4b): when a def declares an
        // explicit quantifier list `def f[..](...)`, names in that list
        // that appear in the precision slot of a `tensor[..., <name>]`
        // parameter annotation desugar to `(t-var {} <name>)` rather
        // than `(t-prim {} <name>)`. The same identifier may also act
        // as a dim-var when it appears in a dim slot — the
        // dim/precision distinction is determined by position inside
        // the tensor type, not by per-name kind tracking. When
        // `dim_params` is empty, the WS-A5 implicit collection on the
        // synthesized sig is preserved and parameter annotations keep
        // their pre-WS-A6 behavior (an unbound precision name surfaces
        // a diagnostic via `validate_tensor_precisions_in_program`).
        let param_ann_tvar_set: HashSet<String> = dim_set.clone();

        let param_names: Vec<deep::Expr> = params
            .iter()
            .map(|param| desugar_param_with_scope(param, &dim_set, &param_ann_tvar_set))
            .collect();
        let params_node = node("params", param_names);
        let body_scope: Vec<String> = params.iter().map(|param| param.name.clone()).collect();
        // Position 3 (spec §P10b / §5.6): body expression of a function
        // whose declared return type is a tensor type and whose body is
        // itself a tensor literal. Narrow numeric literals in `body` to
        // the tensor element type.
        let desugared_body = match (ret_ty.as_ref().and_then(tensor_element_prim_name), body) {
            (Some(prec), Expr::List(items, _)) => {
                self.desugar_list_as_tensor_literal(items, &prec, &body_scope)
            }
            _ => self.desugar_expr_with_scope(body, &body_scope),
        };
        let fn_node = node("fn", vec![params_node, desugared_body]);
        let def_node = node("def", vec![sym(name), fn_node]);

        // chelis#285: when an explicit standalone `sig` already declares this
        // name, the signature synthesized below from inline annotations is
        // redundant and weaker — it fills every un-annotated position with a
        // wildcard `(t-var {} _)` and (being last-write-wins in the checker's
        // defsig binding) would overwrite the concrete explicit sig, dropping
        // the body-vs-signature contract on those positions. Suppress it and
        // let the explicit sig drive body validation.
        if (params.iter().any(|p| p.ty.is_some()) || ret_ty.is_some())
            && !self.explicit_sig_names.contains(name)
        {
            // Tvar set for the synthesized sig:
            //
            // - When the def declares an explicit quantifier list
            //   (`def f[..]`), use that list as the authoritative source
            //   of precision tvars. Names not in the list that appear
            //   in a precision slot stay as `t-prim` and the validator
            //   surfaces the unbound-name diagnostic; this matches the
            //   WS-A6 rule in spec/02-surf-syntax.md §P4b.
            // - When the def has no explicit quantifier list, fall back
            //   to the WS-A5 implicit collection over typed params and
            //   the return type (spec/04-type-system.md §5.8) so a
            //   bare `def f(x: tensor[3, p])` continues to work.
            let tvar_set: HashSet<String> = if !dim_params.is_empty() {
                dim_params.iter().cloned().collect()
            } else {
                let mut acc: HashSet<String> = HashSet::new();
                for p in params {
                    if let Some(ty) = &p.ty {
                        collect_sig_type_vars(ty, &mut acc);
                    }
                }
                if let Some(ty) = ret_ty {
                    collect_sig_type_vars(ty, &mut acc);
                }
                acc
            };
            let mut type_parts: Vec<deep::Expr> = params
                .iter()
                .map(|p| match &p.ty {
                    Some(ty) => desugar_type_with_scope(ty, &dim_set, &tvar_set),
                    None => node("t-var", vec![sym("_")]),
                })
                .collect();
            type_parts.push(match ret_ty {
                Some(ty) => desugar_type_with_scope(ty, &dim_set, &tvar_set),
                None => node("t-var", vec![sym("_")]),
            });
            let sig = node(
                "defsig",
                vec![
                    sym(name),
                    apply_effect_metadata(node("t-fn", type_parts), effects),
                ],
            );
            vec![sig, def_node]
        } else {
            vec![def_node]
        }
    }

    fn desugar_property(
        &self,
        name: &str,
        params: &[Param],
        preconditions: &[Expr],
        body: &Expr,
        options: &[PropertyOption],
    ) -> Vec<deep::Expr> {
        let param_scope = params
            .iter()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>();
        let param_nodes = params.iter().map(desugar_param).collect::<Vec<_>>();
        let params_node = node("params", param_nodes.clone());
        let precondition_node = node(
            "tuple",
            preconditions
                .iter()
                .map(|expr| self.desugar_expr_with_scope(expr, &param_scope))
                .collect(),
        );

        let mut meta_entries = vec![
            ("chelis_role".to_string(), string("property")),
            ("property_source_kind".to_string(), string("user")),
            ("property_quantifiers".to_string(), params_node.clone()),
            ("property_preconditions".to_string(), precondition_node),
        ];
        for option in options {
            match option {
                PropertyOption::Tolerance(value, _) => meta_entries.push((
                    "property_tolerance".to_string(),
                    self.desugar_expr_with_scope(value, &param_scope),
                )),
                PropertyOption::Seed(value, _) => meta_entries.push((
                    "property_seed".to_string(),
                    self.desugar_expr_with_scope(value, &param_scope),
                )),
                PropertyOption::Samples(value, _) => meta_entries.push((
                    "property_samples".to_string(),
                    self.desugar_expr_with_scope(value, &param_scope),
                )),
            }
        }

        let fn_node = node(
            "fn",
            vec![
                params_node,
                self.desugar_expr_with_scope(body, &param_scope),
            ],
        );
        let def_node = node_meta(
            "def",
            meta_with_entries(meta_entries),
            vec![sym(name), fn_node],
        );
        let mut type_parts = params
            .iter()
            .map(|param| desugar_type(param.ty.as_ref().expect("property params are typed")))
            .collect::<Vec<_>>();
        type_parts.push(node("t-prim", vec![sym("bool")]));
        let sig_node = node("defsig", vec![sym(name), node("t-fn", type_parts)]);
        vec![sig_node, def_node]
    }
}

fn desugar_type_def(name: &str, params: &[String], variants: &[Variant]) -> deep::Expr {
    let param_list = bare_list(params.iter().map(|p| sym(p)).collect());
    let mut children = vec![sym(name), param_list];
    for v in variants {
        children.push(desugar_variant(v));
    }
    node("deftype", children)
}

fn desugar_variant(variant: &Variant) -> deep::Expr {
    match &variant.fields {
        VariantFields::Positional(fields) => {
            let mut children = vec![sym(&variant.name)];
            for f in fields {
                children.push(desugar_type(f));
            }
            node("variant", children)
        }
        VariantFields::Record(fields) => {
            let mut children = vec![sym(&variant.name)];
            for (field_name, field_ty) in fields {
                children.push(node("field", vec![sym(field_name), desugar_type(field_ty)]));
            }
            node("variant", children)
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl DesugarCtx {
    fn desugar_expr(&self, expr: &Expr) -> deep::Expr {
        self.desugar_expr_with_scope(expr, &[])
    }

    fn resolve_grad_wrt_indices(
        &self,
        f: &Expr,
        wrt: &[String],
        local_fn_params: &[String],
    ) -> Option<Vec<i64>> {
        let params = match f {
            Expr::Var(name, _) => self.top_level_fn_params.get(name)?.clone(),
            Expr::Lambda(params, _, _) => params.iter().map(|param| param.name.clone()).collect(),
            _ if !local_fn_params.is_empty() => local_fn_params.to_vec(),
            _ => return None,
        };

        wrt.iter()
            .map(|name| {
                params
                    .iter()
                    .position(|param| param == name)
                    .map(|index| index as i64)
            })
            .collect()
    }

    fn desugar_grad(
        &self,
        f: &Expr,
        wrt: Option<&[String]>,
        local_fn_params: &[String],
    ) -> deep::Expr {
        let desugared_fn = self.desugar_expr_with_scope(f, local_fn_params);
        let Some(wrt) = wrt else {
            return node("grad", vec![desugared_fn]);
        };

        let indices = self
            .resolve_grad_wrt_indices(f, wrt, local_fn_params)
            .unwrap_or_else(|| (0..wrt.len()).map(|index| index as i64).collect());

        let wrt_meta = if wrt.len() == 1 {
            dvar(&wrt[0])
        } else {
            node("tuple", wrt.iter().map(|name| dvar(name)).collect())
        };
        let index_expr = if indices.len() == 1 {
            node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym("int32")])),
                vec![int(indices[0])],
            )
        } else {
            node(
                "tuple",
                indices
                    .into_iter()
                    .map(|index| {
                        node_meta(
                            "lit",
                            meta_with_type(node("t-prim", vec![sym("int32")])),
                            vec![int(index)],
                        )
                    })
                    .collect(),
            )
        };

        node_meta(
            "grad",
            meta_with_entries(vec![("wrt".to_string(), wrt_meta)]),
            vec![desugared_fn, index_expr],
        )
    }

    fn desugar_expr_with_scope(&self, expr: &Expr, local_fn_params: &[String]) -> deep::Expr {
        let desugared = match expr {
            Expr::Lit(lit, _) => desugar_literal(lit),
            Expr::Var(name, _) => dvar(name),
            Expr::Constructor(name, _) => dvar(name),
            Expr::List(items, _) => desugar_list_literal(
                &items
                    .iter()
                    .map(|item| self.desugar_expr_with_scope(item, local_fn_params))
                    .collect::<Vec<_>>(),
            ),
            Expr::Record(name, fields, _) => {
                let mut fields = fields.clone();
                fields.sort_by(|a, b| a.0.cmp(&b.0));
                let mut children = vec![sym(name)];
                for (field, value) in fields {
                    children.push(node(
                        "kv",
                        vec![
                            sym(&field),
                            self.desugar_expr_with_scope(&value, local_fn_params),
                        ],
                    ));
                }
                node("record", children)
            }
            Expr::Access(target, field, _) => node(
                "access",
                vec![
                    self.desugar_expr_with_scope(target, local_fn_params),
                    sym(field),
                ],
            ),
            Expr::TupleGet(target, index, _) => node(
                "tuple-get",
                vec![
                    self.desugar_expr_with_scope(target, local_fn_params),
                    node_meta(
                        "lit",
                        meta_with_type(node("t-prim", vec![sym("int32")])),
                        vec![deep::Expr::Atom(deep::Atom::Int(*index), sp())],
                    ),
                ],
            ),

            Expr::Apply(func, args, _) => self.desugar_apply(func, args, local_fn_params),

            Expr::Binary(op, lhs, rhs, _) => {
                let op_name = binop_name(*op);
                // a > b -> (app {} (var {} cmplt) b' a') -- swap operands
                match op {
                    BinOp::Gt => node(
                        "app",
                        vec![
                            dvar(op_name),
                            self.desugar_expr_with_scope(rhs, local_fn_params),
                            self.desugar_expr_with_scope(lhs, local_fn_params),
                        ],
                    ),
                    _ => node(
                        "app",
                        vec![
                            dvar(op_name),
                            self.desugar_expr_with_scope(lhs, local_fn_params),
                            self.desugar_expr_with_scope(rhs, local_fn_params),
                        ],
                    ),
                }
            }

            Expr::Unary(op, operand, _) => {
                let op_name = match op {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Not => "not",
                };
                node(
                    "app",
                    vec![
                        dvar(op_name),
                        self.desugar_expr_with_scope(operand, local_fn_params),
                    ],
                )
            }

            Expr::Pipe(head, stages, _) => {
                let mut children = vec![self.desugar_expr_with_scope(head, local_fn_params)];
                children.extend(
                    stages
                        .iter()
                        .map(|expr| self.desugar_pipe_stage(expr, local_fn_params)),
                );
                node("pipe", children)
            }

            Expr::If(cond, then_e, else_e, _) => node(
                "if",
                vec![
                    self.desugar_expr_with_scope(cond, local_fn_params),
                    self.desugar_expr_with_scope(then_e, local_fn_params),
                    self.desugar_expr_with_scope(else_e, local_fn_params),
                ],
            ),

            Expr::Match(scrutinee, arms, _) => {
                let mut children = vec![self.desugar_expr_with_scope(scrutinee, local_fn_params)];
                for arm in arms {
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                        .unwrap_or_else(|| bare_list(vec![]));
                    children.push(node(
                        "arm",
                        vec![
                            desugar_pattern(&arm.pattern),
                            guard,
                            self.desugar_expr_with_scope(&arm.body, local_fn_params),
                        ],
                    ));
                }
                node("match", children)
            }

            Expr::Lambda(params, body, _) => {
                let param_names: Vec<deep::Expr> = params.iter().map(desugar_param).collect();
                let params_node = node("params", param_names);
                let lambda_params = params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<Vec<_>>();
                node(
                    "fn",
                    vec![
                        params_node,
                        self.desugar_expr_with_scope(body, &lambda_params),
                    ],
                )
            }

            Expr::Tuple(elems, _) if elems.is_empty() => node_meta(
                "lit",
                meta_with_type(node("t-unit", vec![])),
                vec![bare_list(vec![])],
            ),
            Expr::Tuple(elems, _) => node(
                "tuple",
                elems
                    .iter()
                    .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                    .collect(),
            ),

            Expr::Cast(e, prec, _) => {
                // Position 4 (spec §P10b / §5.6): first argument of a
                // `cast(literal, p)` expression. When the inner is a
                // bare list literal, narrow numeric entries to `p` and
                // produce a tensor literal. The outer `cast` then
                // becomes a no-op precision-confirm at the type level
                // (tensor[N, p] cast to p), which `infer_cast` accepts
                // because the precision matches.
                let inner = match e.as_ref() {
                    Expr::List(items, _) => {
                        self.desugar_list_as_tensor_literal(items, prec, local_fn_params)
                    }
                    other => self.desugar_expr_with_scope(other, local_fn_params),
                };
                node("cast", vec![inner, node("t-prim", vec![sym(prec)])])
            }

            Expr::Grad(f, wrt, _) => self.desugar_grad(f, wrt.as_deref(), local_fn_params),

            Expr::Vmap(f, axis, _) => {
                let axis_node = node_meta(
                    "lit",
                    meta_with_type(node("t-prim", vec![sym("int32")])),
                    vec![deep::Expr::Atom(deep::Atom::Int(axis.unwrap_or(0)), sp())],
                );
                node(
                    "vmap",
                    vec![self.desugar_expr_with_scope(f, local_fn_params), axis_node],
                )
            }

            Expr::Jit(f, _) => node(
                "jit",
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Realize(f, _) => node(
                "realize",
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Copy(f, _) => node(
                "copy",
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::Borrow(f, _) => node(
                "borrow",
                vec![self.desugar_expr_with_scope(f, local_fn_params)],
            ),
            Expr::WithSeed(seed, body, _) => node_meta(
                "handle-effect",
                meta_with_entries(vec![("effect".to_string(), sym("random"))]),
                vec![
                    self.desugar_expr_with_scope(seed, local_fn_params),
                    self.desugar_expr_with_scope(body, local_fn_params),
                ],
            ),
            Expr::WithDevice(device, body, _) => node_meta(
                "handle-effect",
                meta_with_entries(vec![("effect".to_string(), sym("resource"))]),
                vec![
                    self.desugar_expr_with_scope(device, local_fn_params),
                    self.desugar_expr_with_scope(body, local_fn_params),
                ],
            ),
            Expr::Par(exprs, _) => node(
                "par",
                exprs
                    .iter()
                    .map(|expr| self.desugar_expr_with_scope(expr, local_fn_params))
                    .collect(),
            ),

            Expr::Annotate(e, ty, _) => {
                // Type annotation pushed into metadata of the desugared expression
                let desugared = self.desugar_expr_with_scope(e, local_fn_params);
                inject_type_metadata(desugared, desugar_type(ty))
            }

            Expr::Block(bindings, final_expr, _) => {
                if bindings.is_empty() {
                    self.desugar_expr_with_scope(final_expr, local_fn_params)
                } else {
                    self.desugar_let_bindings(
                        bindings,
                        self.desugar_expr_with_scope(final_expr, local_fn_params),
                    )
                }
            }
        };
        attach_span_metadata(desugared, expr_span(expr))
    }
}

fn tuple_index_expr(target: deep::Expr, index: i64) -> deep::Expr {
    node(
        "tuple-get",
        vec![
            target,
            node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym("int32")])),
                vec![deep::Expr::Atom(deep::Atom::Int(index), sp())],
            ),
        ],
    )
}

fn bind_name_value(name: &str, value: deep::Expr, body: deep::Expr) -> deep::Expr {
    let bind_node = node("bind", vec![sym(name), value]);
    node("let", vec![bind_node, body])
}

/// Synthesized destructure bind (Linearity-F2).  Marks the `bind`
/// node with `destructure: true` in its meta-map so the linearity
/// checker can distinguish synthesized-tmp scopes from regular
/// `let` scopes.  Linearity-F2 W2 cascade (this PR) treats
/// use-after-consume inside a destructure-marked scope as an
/// error: implicit Copy insertion does not apply to destructured
/// components because tuple-get produces a fresh owned value, not
/// an aliased borrow.
fn bind_destructure_value(name: &str, value: deep::Expr, body: deep::Expr) -> deep::Expr {
    let bind_node = node_meta(
        "bind",
        meta_with_entries(vec![(
            "destructure".to_string(),
            deep::Expr::Atom(deep::Atom::Bool(true), sp()),
        )]),
        vec![sym(name), value],
    );
    node("let", vec![bind_node, body])
}

fn destructure_pattern(
    pattern: &LetPattern,
    source_name: &str,
    body: deep::Expr,
    next_tmp: &mut usize,
) -> deep::Expr {
    match pattern {
        LetPattern::Var(name, _) => bind_destructure_value(name, dvar(source_name), body),
        LetPattern::Wildcard(_) => body,
        LetPattern::Tuple(parts, _) => {
            let mut out = body;
            for (index, part) in parts.iter().enumerate().rev() {
                let tuple_value = tuple_index_expr(dvar(source_name), index as i64);
                let tmp_name = format!("__chelis_tmp{}", *next_tmp);
                *next_tmp += 1;
                out = destructure_pattern(part, &tmp_name, out, next_tmp);
                out = bind_destructure_value(&tmp_name, tuple_value, out);
            }
            out
        }
    }
}

impl DesugarCtx {
    fn desugar_let_bindings(&self, bindings: &[LetBinding], body: deep::Expr) -> deep::Expr {
        let mut out = body;
        let mut next_tmp = 0usize;
        for binding in bindings.iter().rev() {
            match &binding.pattern {
                LetPattern::Var(name, _) => {
                    // Position 1 (spec §P10b / §5.6) at block scope:
                    // `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]` inside
                    // a block uses the same contextual rule as the
                    // top-level form. Module-level LetDef and
                    // block-level let bindings are both let-bindings
                    // per §5.6 enumerated position 1.
                    let value = match (
                        binding.ty.as_ref().and_then(tensor_element_prim_name),
                        &binding.value,
                    ) {
                        (Some(prec), Expr::List(items, _)) => {
                            self.desugar_list_as_tensor_literal(items, &prec, &[])
                        }
                        _ => self.desugar_expr(&binding.value),
                    };
                    if let Some(ty) = &binding.ty {
                        out = bind_name_value(
                            name,
                            inject_type_metadata(value, desugar_type(ty)),
                            out,
                        );
                    } else {
                        out = bind_name_value(name, value, out);
                    }
                }
                pattern => {
                    let temp_name = format!("__chelis_tmp{}", next_tmp);
                    next_tmp += 1;
                    let value = self.desugar_expr(&binding.value);
                    out = destructure_pattern(pattern, &temp_name, out, &mut next_tmp);
                    out = bind_destructure_value(&temp_name, value, out);
                }
            }
        }
        out
    }
}

fn desugar_literal(lit: &Literal) -> deep::Expr {
    match lit {
        Literal::Int(n) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("int32")])),
            vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
        ),
        Literal::Float(f) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("f32")])),
            vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
        ),
        // Typed-suffix literals (spec/02-surf-syntax.md §P10a /
        // spec/04-type-system.md §5.5): bind at exactly the suffix
        // precision with no inference, no widening, no narrowing. The
        // type metadata is the user-facing contract.
        Literal::TypedInt(n, suffix) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym(suffix.t_prim_name())])),
            vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
        ),
        Literal::TypedFloat(f, suffix) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym(suffix.t_prim_name())])),
            vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
        ),
        Literal::Bool(b) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("bool")])),
            vec![deep::Expr::Atom(deep::Atom::Bool(*b), sp())],
        ),
        Literal::Str(s) => node_meta(
            "lit",
            meta_with_type(node("t-prim", vec![sym("string")])),
            vec![deep::Expr::Atom(deep::Atom::Str(s.clone()), sp())],
        ),
    }
}

fn desugar_list_literal(items: &[deep::Expr]) -> deep::Expr {
    let mut out = dvar("Nil");
    for item in items.iter().rev() {
        out = node("app", vec![dvar("Cons"), item.clone(), out]);
    }
    out
}

// ---------------------------------------------------------------------------
// Contextual tensor-literal inference (spec §P10b / §5.6)
// ---------------------------------------------------------------------------
//
// When a tensor literal `[e1, e2, ...]` appears in a position with a known
// element type, the numeric literals in the body adopt that element type
// instead of the §5.3 / §P10 literal default (int32 for integer literals,
// f32 for float literals).
//
// The closed set of "known-element-type" positions is exactly four,
// per spec §5.6:
//
//   1. RHS of a `let`-binding whose declared type is a tensor type
//      `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
//   2. Argument position of a call whose callee has a declared signature
//      with a tensor parameter at that position
//      `f(xs)` where `f : tensor[3, f64] -> ...`
//   3. Body expression of a function with a declared return type that is
//      a tensor type, when the body is itself a tensor literal
//   4. First argument of an explicit `cast(literal, p)`
//
// Outside this closed set, numeric literals fall back to the §5.3 / §P10
// defaults; this is the WS-0 / D1 default and is implemented by
// `desugar_literal` above.
//
// The helpers here transform `Expr::List` into a `to_tensor` call wrapping
// a `Cons/Nil` chain whose numeric literal entries carry the contextual
// element type instead of the default. Non-literal entries (variables,
// function calls, ...) pass through unchanged; type unification at
// `Cons`/`to_tensor` will reject them if their inferred type does not
// match the contextual element type.
//
// Future agents reading this code: do NOT silently extend the closed
// set. Adding new positions (e.g. "any context where a tensor type might
// be inferred backward") is a spec change requiring an amendment to
// §5.6 / §P10b.

impl DesugarCtx {
    /// Desugar `items` as the body of a contextual tensor literal whose
    /// element type is `prec_name` (a precision name like `"f64"` or
    /// `"int32"`). Numeric literals in `items` are emitted with
    /// `(lit {type: (t-prim {} <prec_name>)} value)` instead of the
    /// default int32/f32. Non-literal entries are desugared normally.
    /// The chain is wrapped in `to_tensor` so type inference resolves
    /// the result as a tensor.
    fn desugar_list_as_tensor_literal(
        &self,
        items: &[Expr],
        prec_name: &str,
        local_fn_params: &[String],
    ) -> deep::Expr {
        let desugared_items: Vec<deep::Expr> = items
            .iter()
            .map(|item| self.desugar_tensor_literal_item(item, prec_name, local_fn_params))
            .collect();
        let list = desugar_list_literal(&desugared_items);
        node("app", vec![dvar("to_tensor"), list])
    }

    /// Desugar a single entry of a contextual tensor literal. Numeric
    /// literals are narrowed to the contextual element prim. Nested
    /// `Expr::List` entries (rank > 1) recurse with the same element
    /// type. Anything else falls back to the standard expression
    /// desugarer; the type checker will validate compatibility via
    /// the `Cons` element-type unification path.
    fn desugar_tensor_literal_item(
        &self,
        item: &Expr,
        prec_name: &str,
        local_fn_params: &[String],
    ) -> deep::Expr {
        match item {
            Expr::Lit(Literal::Int(n), _) => node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym(prec_name)])),
                vec![deep::Expr::Atom(deep::Atom::Int(*n), sp())],
            ),
            Expr::Lit(Literal::Float(f), _) => node_meta(
                "lit",
                meta_with_type(node("t-prim", vec![sym(prec_name)])),
                vec![deep::Expr::Atom(deep::Atom::Float(*f), sp())],
            ),
            // RT-2 fixup P2: the surface parser turns `-128` into
            // `Unary(Neg, Lit(Int(128)))`. In a contextual tensor
            // literal position we fold the sign into the literal so
            // the WS-A0 D1 / WS-A0 D1-extension range checks see the
            // user-facing value (`-128` for int8) rather than the
            // raw inner literal (`128`, which overflows int8 max).
            // The same applies to negative float literals.
            Expr::Unary(UnaryOp::Neg, inner, _) => match inner.as_ref() {
                Expr::Lit(Literal::Int(n), _) => node_meta(
                    "lit",
                    meta_with_type(node("t-prim", vec![sym(prec_name)])),
                    vec![deep::Expr::Atom(deep::Atom::Int(-*n), sp())],
                ),
                Expr::Lit(Literal::Float(f), _) => node_meta(
                    "lit",
                    meta_with_type(node("t-prim", vec![sym(prec_name)])),
                    vec![deep::Expr::Atom(deep::Atom::Float(-*f), sp())],
                ),
                // Non-literal `neg` operand falls through to the
                // standard desugar; the type checker will validate
                // the resulting expression's type against the
                // contextual element type via the Cons unification
                // path.
                _ => self.desugar_expr_with_scope(item, local_fn_params),
            },
            // Nested list — rank-N contextual tensor literal.
            Expr::List(nested_items, _) => {
                // The inner list is itself a contextual tensor literal
                // body: numeric literals at every depth adopt the same
                // element type. We do NOT wrap each inner level in
                // to_tensor (only the outermost wrap is needed).
                let inner_items: Vec<deep::Expr> = nested_items
                    .iter()
                    .map(|n| self.desugar_tensor_literal_item(n, prec_name, local_fn_params))
                    .collect();
                desugar_list_literal(&inner_items)
            }
            // Anything else: normal desugar. Type unification at Cons
            // will catch a mismatch.
            other => self.desugar_expr_with_scope(other, local_fn_params),
        }
    }
}

impl DesugarCtx {
    fn desugar_apply(&self, func: &Expr, args: &[Expr], local_fn_params: &[String]) -> deep::Expr {
        // Flatten nested Apply chains. We collect the original surface
        // arguments (not their pre-desugared forms) so the contextual
        // tensor-literal rule (position 2, spec §P10b / §5.6) can inspect
        // each argument against the callee's declared signature.
        let mut surf_args: Vec<&Expr> = Vec::new();
        let base_func = self.collect_apply_chain_surf(func, &mut surf_args);
        for arg in args {
            surf_args.push(arg);
        }

        // Position 2 (spec §P10b / §5.6): if the callee is a top-level
        // function with a declared signature whose i-th parameter is a
        // tensor type with element prim P, and the i-th argument is a
        // bare list literal, narrow the literal entries to P.
        //
        // The callee must be a `Var` for the lookup to apply — function
        // values from local bindings, partial applications, and lambda
        // returns do not carry a declared signature at desugar time.
        // The type checker still validates non-direct-call positions
        // through the standard `Cons`/`to_tensor` element-type
        // unification path; the contextual narrowing here is the
        // ergonomic affordance for the named-callee case.
        let callee_param_prec: Option<&Vec<Option<String>>> =
            if let Expr::Var(callee_name, _) = base_func {
                self.top_level_fn_tensor_param_prec.get(callee_name)
            } else {
                None
            };

        let desugared_args: Vec<deep::Expr> = surf_args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let prec = callee_param_prec
                    .and_then(|v| v.get(i))
                    .and_then(|opt| opt.as_deref());
                match (prec, arg) {
                    (Some(prec_name), Expr::List(items, _)) => {
                        self.desugar_list_as_tensor_literal(items, prec_name, local_fn_params)
                    }
                    _ => self.desugar_expr_with_scope(arg, local_fn_params),
                }
            })
            .collect();

        let mut children = vec![self.desugar_expr_with_scope(base_func, local_fn_params)];
        children.extend(desugared_args);
        node("app", children)
    }

    /// Like `collect_apply_chain` but returns surface-level argument
    /// references so contextual tensor-literal inference can inspect
    /// the arg AST against the callee's signature before desugar.
    fn collect_apply_chain_surf<'a>(&self, expr: &'a Expr, args: &mut Vec<&'a Expr>) -> &'a Expr {
        match expr {
            Expr::Apply(inner_func, inner_args, _) => {
                let base = self.collect_apply_chain_surf(inner_func, args);
                for arg in inner_args {
                    args.push(arg);
                }
                base
            }
            other => other,
        }
    }
}

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Mod => "mod",
        BinOp::Eq => "eq",
        BinOp::Ne => "neq",
        BinOp::Lt => "cmplt",
        BinOp::Gt => "cmplt", // handled specially with swap
        BinOp::Le => "lte",
        BinOp::Ge => "gte",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
}

// ---------------------------------------------------------------------------
// Type Expressions
// ---------------------------------------------------------------------------

/// Desugar a type with no declared dim params or quantified type vars
/// (module-level context).
fn desugar_type(ty: &TypeExpr) -> deep::Expr {
    desugar_type_with_scope(ty, &HashSet::new(), &HashSet::new())
}

/// True if `name` is a candidate quantified type variable per
/// `spec/04-type-system.md` §5.8: lowercase, not a known active
/// primitive, and not a §1.1.2 unsigned alias (those should reach the
/// type-checker's rejection path as `(t-prim {} <name>)`, not be
/// quietly absorbed as a quantifier).
fn is_candidate_tvar_name(name: &str) -> bool {
    name.starts_with(|c: char| c.is_lowercase())
        && !PRIMITIVES.contains(&name)
        && !UNSIGNED_DTYPE_NAMES.contains(&name)
}

/// Compute the set of free, lowercase, non-primitive identifiers used
/// as type names anywhere inside `ty` and its sub-types. These are the
/// candidate quantified type variables for a sig per `spec/04-type-system.md`
/// §5.8: an unbound lowercase name in a `sig` is treated as a `forall`-
/// quantified type variable.
///
/// The set INCLUDES the precision slot of `tensor[..., <ident>]`
/// because WS-A5 (`spec/04-type-system.md` §5.8 / `spec/02-surf-syntax.md`)
/// pins the contextual rule: a lowercase non-primitive name in the
/// precision slot of a tensor type, when it appears in a sig, becomes
/// a quantified type variable. That is the load-bearing change WS-A5
/// makes possible.
///
/// Names listed in `UNSIGNED_DTYPE_NAMES` are EXCLUDED so the type
/// checker still surfaces a `spec/04-type-system.md §1.1.2`-citing
/// diagnostic for them via the `(t-prim {} u8)` path.
fn collect_sig_type_vars(ty: &TypeExpr, out: &mut HashSet<String>) {
    match ty {
        TypeExpr::Named(name, _) => {
            if is_candidate_tvar_name(name) {
                out.insert(name.clone());
            }
        }
        // `..r` is a rank variable, not a type variable — it is collected
        // separately (the checker treats `(d-rank {} r)` as a bound rank var).
        TypeExpr::RankSpread(_, _) => {}
        TypeExpr::Tensor(dims, precision, _) => {
            for d in dims {
                collect_sig_type_vars(d, out);
            }
            // Precision slot: a lowercase non-primitive name here is a
            // candidate quantified type variable per WS-A5. The dim
            // names themselves are handled by the d-name / d-var
            // contextual rules elsewhere and are not type variables.
            if is_candidate_tvar_name(precision) {
                out.insert(precision.clone());
            }
        }
        TypeExpr::Arrow(params, ret, _) => {
            for p in params {
                collect_sig_type_vars(p, out);
            }
            collect_sig_type_vars(ret, out);
        }
        TypeExpr::Ref(inner, _) => collect_sig_type_vars(inner, out),
        TypeExpr::App(_, args, _) => {
            for a in args {
                collect_sig_type_vars(a, out);
            }
        }
        TypeExpr::Tuple(elems, _) => {
            for e in elems {
                collect_sig_type_vars(e, out);
            }
        }
        TypeExpr::Infer(_) => {}
    }
}

/// Desugar a sig's type, computing the implicit quantifier set for the
/// sig and applying the WS-A5 contextual precision rule.
///
/// Per `spec/04-type-system.md` §5.8 and `spec/02-surf-syntax.md`:
/// inside a sig, an identifier in the precision slot of a `tensor[...]`
/// type is desugared to `(t-var {} <name>)` when it is one of the sig's
/// implicitly quantified type variables, and to `(t-prim {} <name>)`
/// when it is a primitive. Outside a sig the same desugar is invoked
/// with an empty quantifier set, so only primitive names are accepted.
fn desugar_sig_type(ty: &TypeExpr, dim_vars: &HashSet<String>) -> deep::Expr {
    let mut tvars = HashSet::new();
    collect_sig_type_vars(ty, &mut tvars);
    desugar_type_with_scope(ty, dim_vars, &tvars)
}

/// Desugar a type with declared dimension parameters and a set of
/// in-scope quantified type variable names. `tvar_set` is non-empty
/// only when desugaring inside a sig that declared (or implicitly
/// introduced) quantified type variables. The contextual rule for the
/// tensor precision slot lives here:
///
/// - A primitive name (`f32`, `int32`, ...) becomes `(t-prim {} <name>)`.
/// - A name found in `tvar_set` becomes `(t-var {} <name>)`.
/// - Any other name in the precision slot is encoded as `(t-prim {} <name>)`
///   so the type checker can surface a precise diagnostic
///   (`unbound type variable in precision slot`) via the existing
///   `Prim::parse_name` rejection path.
fn desugar_type_with_scope(
    ty: &TypeExpr,
    dim_vars: &HashSet<String>,
    tvar_set: &HashSet<String>,
) -> deep::Expr {
    match ty {
        TypeExpr::Named(name, _) => {
            // The contextual rule for type-name positions:
            //
            // - A primitive name (`f32`, `int32`, ...) is a `t-prim`.
            // - A name that appears in the enclosing quantifier set
            //   (`tvar_set` — a def's explicit `[..]` clause or a sig's
            //   implicit quantifiers) is a quantified type variable and
            //   becomes `(t-var {} <name>)` REGARDLESS of case. The
            //   `[..]` clause is the authoritative, unkinded source per
            //   `spec/02-surf-syntax.md` §P4b, so a name the user
            //   explicitly bound there overrides the lexical
            //   case-split. This is what threads a general type
            //   variable (e.g. `def f[n, P](.., f: .. -> P -> ..)`)
            //   through arrow argument positions so it can unify at the
            //   call site (chelis#293). Without this, an uppercase
            //   quantifier name was misclassified as a rigid ADT and
            //   every call site failed with `type mismatch: P vs ..`.
            // - Otherwise the lexical case-split applies: a PascalCase
            //   name is an ADT; a lowercase name is a free `t-var`
            //   whose binding the type checker resolves downstream.
            if PRIMITIVES.contains(&name.as_str()) {
                node("t-prim", vec![sym(name)])
            } else if tvar_set.contains(name.as_str()) {
                node("t-var", vec![sym(name)])
            } else if name.starts_with(|c: char| c.is_uppercase()) {
                node("t-adt", vec![sym(name)])
            } else {
                node("t-var", vec![sym(name)])
            }
        }

        // A bare `..r` reaching here (outside a tensor dim list) is not a
        // valid standalone type, but desugar defensively to the rank node so
        // the match stays exhaustive; validation rejects the misuse upstream.
        TypeExpr::RankSpread(name, _) => node("d-rank", vec![sym(name)]),
        TypeExpr::Tensor(dims, precision, _) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|d| match d {
                    TypeExpr::Named(n, _) if n.parse::<i64>().is_ok() => {
                        node("d-lit", vec![int(n.parse::<i64>().unwrap())])
                    }
                    TypeExpr::Named(n, _) if n == "*" => node("d-name", vec![sym("*")]),
                    // Declared dim param → always d-var (polymorphic)
                    TypeExpr::Named(n, _) if dim_vars.contains(n.as_str()) => {
                        node("d-var", vec![sym(n)])
                    }
                    // Single lowercase letter → d-var (heuristic fallback)
                    TypeExpr::Named(n, _)
                        if n.len() == 1 && n.starts_with(|c: char| c.is_lowercase()) =>
                    {
                        node("d-var", vec![sym(n)])
                    }
                    // Everything else → d-name (concrete)
                    TypeExpr::Named(n, _) => node("d-name", vec![sym(n)]),
                    // `..r` rank-variable spread → (d-rank {} r). May be
                    // interleaved with concrete anchors (Tier-3); a multi-letter
                    // anchor such as `seq` desugars to `d-name` above, which is
                    // what the name-preserving split matches on.
                    TypeExpr::RankSpread(n, _) => node("d-rank", vec![sym(n)]),
                    _ => node(
                        "d-var",
                        vec![desugar_type_with_scope(d, dim_vars, tvar_set)],
                    ),
                })
                .collect();
            // WS-A5 contextual precision rule (spec/04-type-system.md §5.8,
            // spec/02-surf-syntax.md): the precision slot is a t-var when
            // its name is in `tvar_set` (a sig-quantified type variable),
            // otherwise it stays as t-prim and the type checker validates
            // it against the closed primitive set via Prim::parse_name.
            let prec_node = if tvar_set.contains(precision.as_str())
                && !PRIMITIVES.contains(&precision.as_str())
            {
                node("t-var", vec![sym(precision)])
            } else {
                node("t-prim", vec![sym(precision)])
            };
            children.push(prec_node);
            node("t-tensor", children)
        }

        TypeExpr::Arrow(params, ret, _) => {
            let mut children: Vec<deep::Expr> = params
                .iter()
                .map(|p| desugar_type_with_scope(p, dim_vars, tvar_set))
                .collect();
            children.push(desugar_type_with_scope(ret, dim_vars, tvar_set));
            node("t-fn", children)
        }

        TypeExpr::Ref(inner, _) => node(
            "t-ref",
            vec![desugar_type_with_scope(inner, dim_vars, tvar_set)],
        ),

        TypeExpr::App(name, args, _) => {
            let mut children = vec![sym(name)];
            children.extend(
                args.iter()
                    .map(|a| desugar_type_with_scope(a, dim_vars, tvar_set)),
            );
            node("t-adt", children)
        }

        TypeExpr::Tuple(elems, _) if elems.is_empty() => node("t-unit", vec![]),
        TypeExpr::Tuple(elems, _) => node(
            "t-tuple",
            elems
                .iter()
                .map(|e| desugar_type_with_scope(e, dim_vars, tvar_set))
                .collect(),
        ),

        TypeExpr::Infer(_) => node("t-var", vec![sym("_")]),
    }
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

fn desugar_pattern(pat: &Pattern) -> deep::Expr {
    match pat {
        Pattern::Wildcard(_) => node("pat-wild", vec![]),
        Pattern::Var(name, _) => node("pat-var", vec![sym(name)]),
        Pattern::Lit(lit, _) => {
            // pat-lit contains the raw literal value, NOT a typed (lit ...) node.
            // Typed-suffix patterns desugar to the bare value: pattern matching
            // does not enforce the suffix dtype at the pattern level (the
            // checker reconciles it via the surrounding scrutinee type).
            let val = match lit {
                Literal::Int(n) | Literal::TypedInt(n, _) => {
                    deep::Expr::Atom(deep::Atom::Int(*n), sp())
                }
                Literal::Float(f) | Literal::TypedFloat(f, _) => {
                    deep::Expr::Atom(deep::Atom::Float(*f), sp())
                }
                Literal::Bool(b) => deep::Expr::Atom(deep::Atom::Bool(*b), sp()),
                Literal::Str(s) => deep::Expr::Atom(deep::Atom::Str(s.clone()), sp()),
            };
            node("pat-lit", vec![val])
        }
        Pattern::Constructor(name, sub_pats, _) => {
            let mut children = vec![sym(name)];
            children.extend(sub_pats.iter().map(desugar_pattern));
            node("pat-ctor", children)
        }
        Pattern::Tuple(pats, _) => node("pat-tuple", pats.iter().map(desugar_pattern).collect()),
        Pattern::Record(name, fields, _) => {
            let mut fields = fields.clone();
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            let mut children = vec![sym(name)];
            for (field_name, field_pat) in fields {
                children.push(node(
                    "kv",
                    vec![sym(&field_name), desugar_pattern(&field_pat)],
                ));
            }
            node("pat-record", children)
        }
        Pattern::As(name, inner, _) => node("pat-as", vec![sym(name), desugar_pattern(inner)]),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::printer::print_expr;

    fn s() -> Span {
        Span::new(0, 0)
    }

    fn tvar(name: &str) -> Expr {
        Expr::Var(name.to_string(), s())
    }

    fn int_lit(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), s())
    }

    fn float_lit(f: f64) -> Expr {
        Expr::Lit(Literal::Float(f), s())
    }

    fn param(name: &str, ty: Option<TypeExpr>) -> Param {
        Param {
            name: name.to_string(),
            ty,
            span: s(),
        }
    }

    fn named_ty(name: &str) -> TypeExpr {
        TypeExpr::Named(name.to_string(), s())
    }

    /// Helper: desugar a decl and print all resulting nodes.
    fn desugar_decl_strs(decl: &Decl) -> Vec<String> {
        desugar_decl(decl).iter().map(print_expr).collect()
    }

    // --- Literals ---

    #[test]
    fn test_int_literal() {
        let result = print_expr(&desugar_expr(&int_lit(42)));
        assert_eq!(result, "(lit {type: (t-prim {} int32)} 42)");
    }

    #[test]
    fn test_float_literal() {
        let result = print_expr(&desugar_expr(&float_lit(3.125)));
        assert_eq!(result, "(lit {type: (t-prim {} f32)} 3.125)");
    }

    #[test]
    fn test_bool_literal() {
        let result = print_expr(&desugar_expr(&Expr::Lit(Literal::Bool(true), s())));
        assert_eq!(result, "(lit {type: (t-prim {} bool)} true)");
    }

    #[test]
    fn test_string_literal() {
        let result = print_expr(&desugar_expr(&Expr::Lit(
            Literal::Str("hello".to_string()),
            s(),
        )));
        assert_eq!(result, "(lit {type: (t-prim {} string)} \"hello\")");
    }

    // --- Variables ---

    #[test]
    fn test_var() {
        assert_eq!(print_expr(&desugar_expr(&tvar("x"))), "(var {} x)");
    }

    // --- Binary operators (all go through app) ---

    #[test]
    fn test_add() {
        let expr = Expr::Binary(BinOp::Add, Box::new(tvar("a")), Box::new(tvar("b")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} add) (var {} a) (var {} b))"
        );
    }

    #[test]
    fn test_gt_swaps_operands() {
        let expr = Expr::Binary(BinOp::Gt, Box::new(tvar("a")), Box::new(tvar("b")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} cmplt) (var {} b) (var {} a))"
        );
    }

    #[test]
    fn test_nested_binops() {
        // a + b * c
        let expr = Expr::Binary(
            BinOp::Add,
            Box::new(tvar("a")),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(tvar("b")),
                Box::new(tvar("c")),
                s(),
            )),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} add) (var {} a) (app {} (var {} mul) (var {} b) (var {} c)))"
        );
    }

    // --- Unary ---

    #[test]
    fn test_unary_neg() {
        let expr = Expr::Unary(UnaryOp::Neg, Box::new(tvar("a")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} neg) (var {} a))"
        );
    }

    // --- Function application ---

    #[test]
    fn test_apply() {
        let expr = Expr::Apply(Box::new(tvar("f")), vec![tvar("x"), tvar("y")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(app {} (var {} f) (var {} x) (var {} y))"
        );
    }

    #[test]
    fn test_apply_chain_flattening() {
        let inner = Expr::Apply(Box::new(tvar("f")), vec![tvar("x")], s());
        let outer = Expr::Apply(Box::new(inner), vec![tvar("y")], s());
        assert_eq!(
            print_expr(&desugar_expr(&outer)),
            "(app {} (var {} f) (var {} x) (var {} y))"
        );
    }

    // --- Pipe ---

    #[test]
    fn test_pipe() {
        let expr = Expr::Pipe(Box::new(tvar("x")), vec![tvar("f"), tvar("g")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(pipe {} (var {} x) (var {} f) (var {} g))"
        );
    }

    #[test]
    fn test_pipe_stage_call_desugars_to_unary_lambda() {
        let expr = Expr::Pipe(
            Box::new(tvar("x")),
            vec![Expr::Apply(Box::new(tvar("add")), vec![tvar("y")], s())],
            s(),
        );
        let rendered = print_expr(&desugar_expr(&expr));
        assert!(rendered.contains("(var {} x)"));
        assert!(rendered.contains("(params {} __chelis_pipe)"));
        assert!(rendered.contains("(app {} (var {} add) (var {} __chelis_pipe) (var {} y))"));
    }

    // --- If ---

    #[test]
    fn test_if() {
        let expr = Expr::If(
            Box::new(tvar("a")),
            Box::new(tvar("b")),
            Box::new(tvar("c")),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(if {} (var {} a) (var {} b) (var {} c))"
        );
    }

    // --- Match ---

    #[test]
    fn test_match() {
        let expr = Expr::Match(
            Box::new(tvar("x")),
            vec![
                MatchArm {
                    pattern: Pattern::Constructor(
                        "Some".to_string(),
                        vec![Pattern::Var("y".to_string(), s())],
                        s(),
                    ),
                    guard: None,
                    body: tvar("y"),
                    span: s(),
                },
                MatchArm {
                    pattern: Pattern::Constructor("None".to_string(), vec![], s()),
                    guard: None,
                    body: int_lit(0),
                    span: s(),
                },
            ],
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert_eq!(
            result,
            "(match {}\n  (var {} x)\n  (arm {} (pat-ctor {} Some (pat-var {} y)) () (var {} y))\n  (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int32)} 0)))"
        );
    }

    // --- Block bindings ---

    #[test]
    fn test_block_binding() {
        let expr = Expr::Block(
            vec![LetBinding {
                pattern: LetPattern::Var("x".to_string(), s()),
                ty: None,
                value: int_lit(1),
            }],
            Box::new(tvar("x")),
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert_eq!(
            result,
            "(let {}\n  (bind {}\n    x\n    (lit {type: (t-prim {} int32)} 1))\n  (var {} x))"
        );
    }

    #[test]
    fn test_block_tuple_destructuring() {
        let expr = Expr::Block(
            vec![LetBinding {
                pattern: LetPattern::Tuple(
                    vec![
                        LetPattern::Var("a".to_string(), s()),
                        LetPattern::Wildcard(s()),
                        LetPattern::Var("c".to_string(), s()),
                    ],
                    s(),
                ),
                ty: None,
                value: tvar("triple"),
            }],
            Box::new(tvar("c")),
            s(),
        );
        let result = print_expr(&desugar_expr(&expr));
        assert!(result.contains("__chelis_tmp0"));
        assert!(result.contains("(var {} triple)"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 0)"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 1)"));
        assert!(result.contains("(lit {type: (t-prim {} int32)} 2)"));
        assert_eq!(result.matches("(tuple-get {}").count(), 3);
        assert!(result.contains("(var {} __chelis_tmp0)"));
        assert!(result.contains("(var {} c)"));
        assert!(!result.contains("(bind {} _ "));
    }

    // --- Lambda ---

    #[test]
    fn test_lambda() {
        let expr = Expr::Lambda(vec![param("x", None)], Box::new(tvar("x")), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(fn {}\n  (params {} x)\n  (var {} x))"
        );
    }

    // --- Fun def (no types) ---

    #[test]
    fn test_fun_def() {
        let decl = Decl::FunDef {
            name: "f".to_string(),
            dim_params: vec![],
            params: vec![param("x", None)],
            ret_ty: None,
            effects: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(def {} f (fn {} (params {} x) (var {} x)))");
    }

    // --- Fun def (with types) produces defsig + def ---

    #[test]
    fn test_fun_def_typed() {
        // def f(x: f32): f32 = x
        let decl = Decl::FunDef {
            name: "f".to_string(),
            dim_params: vec![],
            params: vec![param("x", Some(named_ty("f32")))],
            ret_ty: Some(named_ty("f32")),
            effects: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0],
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))"
        );
        assert_eq!(
            nodes[1],
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))"
        );
    }

    // --- Fun def with dim params ---

    #[test]
    fn test_fun_def_with_dim_params() {
        // def transpose[batch, hidden](x: tensor[batch, hidden, f32]): tensor[hidden, batch, f32] = x
        // batch and hidden should be d-var (polymorphic), NOT d-name
        let decl = Decl::FunDef {
            name: "transpose".to_string(),
            dim_params: vec!["batch".to_string(), "hidden".to_string()],
            params: vec![param(
                "x",
                Some(TypeExpr::Tensor(
                    vec![named_ty("batch"), named_ty("hidden")],
                    "f32".to_string(),
                    s(),
                )),
            )],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("hidden"), named_ty("batch")],
                "f32".to_string(),
                s(),
            )),
            effects: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2); // defsig + def, NO defdim
        // defsig must use d-var for batch and hidden (declared dim params)
        assert!(
            nodes[0].contains("(d-var {} batch)"),
            "expected d-var for 'batch' (declared dim param), got:\n{}",
            nodes[0]
        );
        assert!(
            nodes[0].contains("(d-var {} hidden)"),
            "expected d-var for 'hidden' (declared dim param), got:\n{}",
            nodes[0]
        );
        // Must NOT contain defdim (those are module-level)
        for n in &nodes {
            assert!(
                !n.contains("defdim"),
                "function dim params should NOT emit defdim, got:\n{n}"
            );
        }
    }

    #[test]
    fn def_quantifier_general_tvar_through_arrow_is_t_var() {
        // chelis#293: a def generic over a general type variable declared in
        // the explicit `[..]` quantifier list, threaded through a
        // function-typed parameter, must desugar that name to `(t-var {} P)`
        // — NOT `(t-adt {} P)` — even though it is uppercase. The `[..]`
        // clause is the authoritative, unkinded quantifier source
        // (spec/02-surf-syntax.md §P4b), so it overrides the lexical
        // case-split.
        //
        // def apply_resid[n, P](
        //     x: tensor[n, f32],
        //     inner_p: P,
        //     f: tensor[n, f32] -> P -> tensor[n, f32],
        // ) -> tensor[n, f32] = add(x, f(x, inner_p))
        let decl = Decl::FunDef {
            name: "apply_resid".to_string(),
            dim_params: vec!["n".to_string(), "P".to_string()],
            params: vec![
                param(
                    "x",
                    Some(TypeExpr::Tensor(
                        vec![named_ty("n")],
                        "f32".to_string(),
                        s(),
                    )),
                ),
                param("inner_p", Some(named_ty("P"))),
                param(
                    "f",
                    Some(TypeExpr::Arrow(
                        vec![
                            TypeExpr::Tensor(vec![named_ty("n")], "f32".to_string(), s()),
                            named_ty("P"),
                        ],
                        Box::new(TypeExpr::Tensor(
                            vec![named_ty("n")],
                            "f32".to_string(),
                            s(),
                        )),
                        s(),
                    )),
                ),
            ],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("n")],
                "f32".to_string(),
                s(),
            )),
            effects: None,
            body: Expr::Apply(
                Box::new(tvar("add")),
                vec![
                    tvar("x"),
                    Expr::Apply(Box::new(tvar("f")), vec![tvar("x"), tvar("inner_p")], s()),
                ],
                s(),
            ),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2, "expected defsig + def, got: {nodes:?}");
        let sig = &nodes[0];
        // The general type variable `P` must be a t-var everywhere it
        // appears in the signature — the bare `inner_p` annotation AND the
        // arrow parameter — so it can unify at the call site.
        assert!(
            sig.contains("(t-var {} P)"),
            "expected `P` to desugar to (t-var {{}} P) in the signature, got:\n{sig}"
        );
        assert!(
            !sig.contains("(t-adt {} P)"),
            "general type var `P` from the `[..]` quantifier list must NOT be a \
             rigid ADT (chelis#293), got:\n{sig}"
        );
        // `n` is a declared dim param and must stay a d-var.
        assert!(
            sig.contains("(d-var {} n)"),
            "expected `n` to desugar to (d-var {{}} n), got:\n{sig}"
        );
    }

    #[test]
    fn def_uppercase_name_not_in_quantifier_stays_adt() {
        // Negative control for the chelis#293 fix: an uppercase type name
        // that is NOT in the `[..]` quantifier list keeps the lexical
        // case-split and stays a `(t-adt {} Activation)`. Only names the
        // user explicitly bound in `[..]` are promoted to type variables.
        //
        // def run[n](x: tensor[n, f32], a: Activation) -> tensor[n, f32] = x
        let decl = Decl::FunDef {
            name: "run".to_string(),
            dim_params: vec!["n".to_string()],
            params: vec![
                param(
                    "x",
                    Some(TypeExpr::Tensor(
                        vec![named_ty("n")],
                        "f32".to_string(),
                        s(),
                    )),
                ),
                param("a", Some(named_ty("Activation"))),
            ],
            ret_ty: Some(TypeExpr::Tensor(
                vec![named_ty("n")],
                "f32".to_string(),
                s(),
            )),
            effects: None,
            body: tvar("x"),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        let sig = &nodes[0];
        assert!(
            sig.contains("(t-adt {} Activation)"),
            "an uppercase name NOT in the quantifier list must stay an ADT, got:\n{sig}"
        );
        assert!(
            !sig.contains("(t-var {} Activation)"),
            "an unquantified ADT name must not be promoted to a type variable, got:\n{sig}"
        );
    }

    // --- Let def (with type) produces defsig + def ---

    #[test]
    fn test_let_def_typed() {
        // let x: f32 = 1.0
        let decl = Decl::LetDef {
            name: "x".to_string(),
            ty: Some(named_ty("f32")),
            value: float_lit(1.0),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0], "(defsig {} x (t-prim {} f32))");
        assert_eq!(nodes[1], "(def {} x (lit {type: (t-prim {} f32)} 1.0))");
    }

    // --- Let def (no type) produces just def ---

    #[test]
    fn test_let_def_untyped() {
        let decl = Decl::LetDef {
            name: "x".to_string(),
            ty: None,
            value: int_lit(42),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(def {} x (lit {type: (t-prim {} int32)} 42))");
    }

    // --- Annotate preserves type in metadata ---

    #[test]
    fn test_annotate_var() {
        // x : f32
        let expr = Expr::Annotate(Box::new(tvar("x")), named_ty("f32"), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(var {type: (t-prim {} f32)} x)"
        );
    }

    // --- Transforms (tags, not app) ---

    #[test]
    fn test_grad() {
        let expr = Expr::Grad(Box::new(tvar("f")), None, s());
        assert_eq!(
            print_expr(&DesugarCtx::default().desugar_expr(&expr)),
            "(grad {} (var {} f))"
        );
    }

    #[test]
    fn test_grad_with_wrt() {
        let expr = Expr::Grad(
            Box::new(tvar("loss")),
            Some(vec!["w".to_string(), "b".to_string()]),
            s(),
        );
        let ctx = DesugarCtx {
            top_level_fn_params: HashMap::from([(
                "loss".to_string(),
                vec!["x".to_string(), "w".to_string(), "b".to_string()],
            )]),
            top_level_fn_tensor_param_prec: HashMap::new(),
            explicit_sig_names: HashSet::new(),
            def_effects: HashMap::new(),
        };
        let actual = print_expr(&ctx.desugar_expr(&expr))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            actual,
            "(grad {wrt: (tuple {} (var {} w) (var {} b))} (var {} loss) (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 2)))"
        );
    }

    #[test]
    fn test_cast() {
        let expr = Expr::Cast(Box::new(tvar("x")), "bf16".to_string(), s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(cast {} (var {} x) (t-prim {} bf16))"
        );
    }

    #[test]
    fn test_jit() {
        let expr = Expr::Jit(Box::new(tvar("f")), s());
        assert_eq!(print_expr(&desugar_expr(&expr)), "(jit {} (var {} f))");
    }

    // --- Type expressions ---

    #[test]
    fn test_type_prim() {
        assert_eq!(
            print_expr(&desugar_type(&named_ty("f32"))),
            "(t-prim {} f32)"
        );
    }

    #[test]
    fn test_type_var() {
        // Lowercase non-primitive → t-var
        assert_eq!(print_expr(&desugar_type(&named_ty("a"))), "(t-var {} a)");
    }

    #[test]
    fn test_uppercase_named_is_adt() {
        // Uppercase non-primitive → t-adt (concrete ADT, zero args)
        assert_eq!(
            print_expr(&desugar_type(&named_ty("Activation"))),
            "(t-adt {} Activation)"
        );
        assert_eq!(
            print_expr(&desugar_type(&named_ty("MyType"))),
            "(t-adt {} MyType)"
        );
    }

    #[test]
    fn test_type_tensor() {
        let ty = TypeExpr::Tensor(
            vec![named_ty("batch"), named_ty("hidden")],
            "f32".to_string(),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_tensor_with_literal_dims() {
        let ty = TypeExpr::Tensor(
            vec![named_ty("32"), named_ty("784")],
            "f32".to_string(),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_tensor_rank_spread() {
        // `tensor[..r, f32]` desugars to a sole `(d-rank {} r)` dim node.
        let ty = TypeExpr::Tensor(
            vec![TypeExpr::RankSpread("r".to_string(), s())],
            "f32".to_string(),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-tensor {} (d-rank {} r) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_arrow() {
        let ty = TypeExpr::Arrow(
            vec![named_ty("f32"), named_ty("f32")],
            Box::new(named_ty("f32")),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32))"
        );
    }

    #[test]
    fn test_type_adt() {
        let ty = TypeExpr::App("Option".to_string(), vec![named_ty("f32")], s());
        assert_eq!(
            print_expr(&desugar_type(&ty)),
            "(t-adt {} Option (t-prim {} f32))"
        );
    }

    // --- Type def with type variable ---

    #[test]
    fn test_type_def() {
        let decl = Decl::TypeDef {
            name: "Option".to_string(),
            params: vec!["a".to_string()],
            variants: vec![
                Variant {
                    name: "Some".to_string(),
                    fields: VariantFields::Positional(vec![named_ty("a")]),
                    span: s(),
                },
                Variant {
                    name: "None".to_string(),
                    fields: VariantFields::Positional(vec![]),
                    span: s(),
                },
            ],
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0],
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))"
        );
    }

    #[test]
    fn test_record_variant() {
        let decl = Decl::TypeDef {
            name: "T".to_string(),
            params: vec![],
            variants: vec![Variant {
                name: "V".to_string(),
                fields: VariantFields::Record(vec![("x".to_string(), named_ty("f32"))]),
                span: s(),
            }],
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0],
            "(deftype {} T () (variant {} V (field {} x (t-prim {} f32))))"
        );
    }

    // --- Tuple ---

    #[test]
    fn test_tuple() {
        let expr = Expr::Tuple(vec![tvar("a"), tvar("b")], s());
        assert_eq!(
            print_expr(&desugar_expr(&expr)),
            "(tuple {} (var {} a) (var {} b))"
        );
    }

    // --- Patterns ---

    #[test]
    fn test_pat_lit_int() {
        let pat = Pattern::Lit(Literal::Int(42), s());
        assert_eq!(print_expr(&desugar_pattern(&pat)), "(pat-lit {} 42)");
    }

    #[test]
    fn test_pat_tuple() {
        let pat = Pattern::Tuple(
            vec![
                Pattern::Var("a".to_string(), s()),
                Pattern::Var("b".to_string(), s()),
            ],
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-tuple {} (pat-var {} a) (pat-var {} b))"
        );
    }

    // --- Import ---

    #[test]
    fn test_import_all() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            kind: ImportKind::All,
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(import-all {} foo)");
    }

    #[test]
    fn test_import_selective() {
        let decl = Decl::Import {
            module: "Foo".to_string(),
            kind: ImportKind::Names(vec!["a".to_string(), "b".to_string()]),
            span: s(),
        };
        let nodes = desugar_decl_strs(&decl);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "(import {} foo (a b))");
    }

    // --- Record pattern ---

    #[test]
    fn test_pat_record() {
        let pat = Pattern::Record(
            "Adam".to_string(),
            vec![
                ("lr".to_string(), Pattern::Var("lr".to_string(), s())),
                ("eps".to_string(), Pattern::Var("eps".to_string(), s())),
            ],
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-record {} Adam (kv {} eps (pat-var {} eps)) (kv {} lr (pat-var {} lr)))"
        );
    }

    // --- As pattern ---

    #[test]
    fn test_pat_as() {
        let pat = Pattern::As(
            "y".to_string(),
            Box::new(Pattern::Constructor(
                "Some".to_string(),
                vec![Pattern::Var("z".to_string(), s())],
                s(),
            )),
            s(),
        );
        assert_eq!(
            print_expr(&desugar_pattern(&pat)),
            "(pat-as {} y (pat-ctor {} Some (pat-var {} z)))"
        );
    }
}
