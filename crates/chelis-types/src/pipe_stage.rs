//! Shared pipe-stage callee resolution.
//!
//! `chelis_surf::desugar::desugar_pipe_stage` lowers a pipe stage with
//! explicit args (`x |> f(y)`) into a synthesized one-arg lambda
//!
//! ```text
//! (fn (params __chelis_pipe<N>) (app callee ... (var __chelis_pipe<N>) ...))
//! ```
//!
//! where the piped value lands at whatever position the desugarer placed
//! `__chelis_pipe<N>` in the inner app's args (today position 0, but the
//! scan stays robust against future stage shapes).
//!
//! Multiple passes need to classify whether the piped value is borrowed
//! or consumed by the stage. Without peering through the synthesized
//! lambda, every non-bare-var stage looks like a generic owned-arg call.
//! That mis-classification was the root cause of:
//!
//! - chelis#226 in `chelis_types::linearity::check_pipe` (fixed in PR
//!   #228) — explicit-arg pipe stages over borrow-arg builtins
//!   (`shape`, `add`, `mul`, …) were tagged with a structural-pipe
//!   consume, so later reads of the piped variable tripped
//!   `UseAfterConsume`.
//! - chelis#229 in `chelis_types::infer::pipe_consumes_param` — the
//!   same gap in the auto-borrow inferencer, so a function whose body
//!   only uses its param through an explicit-arg pipe stage stays
//!   owned-linear instead of being inferred read-only.
//!
//! [`resolve_pipe_stage_callee`] is the one helper both consumers
//! should reach for. The two callers (`linearity` and `infer`) consume
//! the result through their own borrow-arg classifiers, but the shape
//! recognition lives here so future sweeps stay aligned.

use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom, Expr, List};

/// Resolve the effective callee of a pipe stage and the arg position
/// the piped value occupies in that callee.
///
/// Two stage shapes are produced by `chelis_surf::desugar::desugar_pipe_stage`:
///
/// 1. Bare var stage (e.g. `x |> f`): `(var f)`. The piped value is the
///    only arg, at position 0. Returns `(stage, Some("f"), 0)`.
///
/// 2. Synthesized lambda stage (e.g. `x |> f(y)`): a one-arg lambda
///    `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe) ...))`
///    where the piped value lands at whatever position the desugarer
///    placed `__chelis_pipe`. Returns `(inner_callee_expr,
///    builtin_name_or_none, piped_arg_index)`.
///
/// Any other stage shape (parser-level oddity, macro-expanded result
/// not classified here) falls back to the historical contract of
/// `var_name(stage)` so the bare-var case never regresses.
pub(crate) fn resolve_pipe_stage_callee(stage: &Expr) -> (Option<&Expr>, Option<&str>, usize) {
    if let Some(name) = var_name(stage) {
        return (Some(stage), Some(name), 0);
    }
    if let Some((callee, idx)) = pipe_lambda_callee_and_pipe_arg_index(stage) {
        return (Some(callee), var_name(callee), idx);
    }
    (Some(stage), None, 0)
}

/// If `stage` is `(fn (params __chelis_pipe) (app callee args...))` with
/// exactly one of `args` being `(var __chelis_pipe)`, return the inner
/// callee and the index of the pipe arg within `args`.
fn pipe_lambda_callee_and_pipe_arg_index(stage: &Expr) -> Option<(&Expr, usize)> {
    let Expr::List(list, _) = stage else {
        return None;
    };
    if get_tag(list) != Some(DeepTag::Fn) {
        return None;
    }
    let kids = children(list);
    let params = kids.first()?;
    let body = kids.get(1)?;
    let pipe_param = sole_pipe_param_name(params)?;
    let Expr::List(body_list, _) = body else {
        return None;
    };
    if get_tag(body_list) != Some(DeepTag::App) {
        return None;
    }
    let app_kids = children(body_list);
    let callee = app_kids.first()?;
    for (offset, arg) in app_kids.iter().skip(1).enumerate() {
        if var_name(arg) == Some(pipe_param) {
            return Some((callee, offset));
        }
    }
    None
}

/// If `params` is `(params __chelis_pipe[N])` with exactly one
/// pipe-synthesized parameter, return its name. The desugarer prefixes
/// every synthetic pipe parameter with `__chelis_pipe`; a user-written
/// lambda that happens to take a single param does not match.
fn sole_pipe_param_name(params: &Expr) -> Option<&str> {
    let Expr::List(list, _) = params else {
        return None;
    };
    if get_tag(list) != Some(DeepTag::Params) {
        return None;
    }
    let kids = children(list);
    if kids.len() != 1 {
        return None;
    }
    let name = symbol_name(&kids[0])?;
    if name.starts_with("__chelis_pipe") {
        Some(name)
    } else {
        None
    }
}

fn get_tag(list: &List) -> Option<DeepTag> {
    list.tag()
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
