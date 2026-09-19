//! What a pipe denotes, stated once.
//!
//! `spec/02-surf-syntax.md` §0.1 fixes the meaning: "pipe stages use
//! first-argument insertion: `x |> f(y)` means `f(x, y)`". A pipe is
//! therefore not a separate semantic form. It is notation for an
//! application, and the application is the thing every later pass is
//! supposed to reason about.
//!
//! Before chelis#1923 each consumer that met a `Pipe` node re-derived that
//! sentence for itself, and they did not all derive it the same way. The
//! checker typed a bare-name stage from the callee's FUNCTION type rather
//! than as the application `f(acc)`, so every rule that reads an
//! application's arguments was lost in pipe position: `to_tensor`'s literal
//! shape, `sum`'s axis, `expand`'s size. The same program written applied
//! type-checked; written as a pipe stage it did not. The lowerer had the
//! matching defect on its own side (chelis#1791 half A).
//!
//! So the fold lives here, in the crate that owns the Deep vocabulary, and
//! the checker applies it before it infers anything. Every later consumer
//! then sees the application, and none of them needs a pipe arm at all.
//! That is the point of folding rather than teaching one more consumer to
//! handle pipes correctly: a second derivation of one sentence is what
//! produced the defect, and one that stayed would hide the day the first
//! stopped agreeing with it.

use crate::annotations::{Metadata, MetadataKey};
use crate::ast::{Atom, Expr, List};
use crate::tag::DeepTag;

/// The parts of a stamped node, read through either carrier.
///
/// chelis#1107: a node reaches here as `Expr::List` or as the typed
/// `Expr::Node`, and a reader that knows only one silently declines the
/// other. Both are read here so the fold cannot depend on which producer
/// built the tree.
fn stamped(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        crate::ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        crate::ExprCarrier::StructuralList(_)
        | crate::ExprCarrier::UndecodableHead(_, _, _)
        | crate::ExprCarrier::Atom(_)
        | crate::ExprCarrier::MetadataMap(_)
        | crate::ExprCarrier::MetadataExpression(_)
        | crate::ExprCarrier::MalformedLegacyList(_) => None,
    }
}

/// The single parameter name of a unary `fn` stage, if that is its shape.
fn unary_param(stage: &Expr) -> Option<&str> {
    let (tag, _, kids) = stamped(stage)?;
    if tag != DeepTag::Fn {
        return None;
    }
    let (params_tag, _, params) = stamped(kids.first()?)?;
    if params_tag != DeepTag::Params || params.len() != 1 {
        return None;
    }
    match &params[0] {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

/// Fold one stage over the accumulator, per §0.1's first-argument
/// insertion.
///
/// A stage that directly forwards its parameter as a named callee's first
/// argument is reduced to that call (or its dedicated cast/copy/realize form).
/// The desugarer uses this shape for
/// `expand(0, n)`, exposing the inserted operand to argument-sensitive rules.
/// Insertion stays outside every binder and before every other argument.
/// General lambdas retain ordinary application semantics: even one use of a
/// parameter can be conditional, deferred, shadowed, or preceded by effects.
fn fold_stage(stage: &Expr, acc: Expr) -> Expr {
    if let Some(param) = unary_param(stage)
        && let Some((_, _, stage_kids)) = stamped(stage)
        && let Some(body) = stage_kids.get(1)
        && let Some((tag, meta, args)) = stamped(body)
    {
        // Surf's call-first stages also use dedicated operand forms for
        // cast, copy, and realize. Each evaluates its operand first.
        let input = match tag {
            DeepTag::App
                if matches!(args.first().and_then(stamped), Some((DeepTag::Var, _, _))) =>
            {
                Some(1)
            }
            DeepTag::Cast | DeepTag::Copy | DeepTag::Realize => Some(0),
            _ => None,
        };
        if let Some(input) = input
            && args.get(input).is_some_and(|arg| is_var(arg, param))
            && !args
                .iter()
                .enumerate()
                .any(|(index, arg)| index != input && mentions_name(arg, param))
        {
            let mut children = args.to_vec();
            children[input] = acc;
            return rebuild(body, tag, meta, children);
        }
    }
    // This origin marker is valid only inside a pipe (spec/03 [03-META-1/2]).
    // A retained lambda is now an ordinary callee, so consume the marker.
    let stage = match stamped(stage) {
        Some((DeepTag::Fn, meta, kids)) if meta.surf_pipe_stage().is_some() => {
            let mut meta = meta.clone();
            meta.remove(MetadataKey::SurfPipeStage);
            rebuild(stage, DeepTag::Fn, &meta, kids.to_vec())
        }
        _ => stage.clone(),
    };
    // The synthesized application carries the STAGE's span and metadata, so
    // a diagnostic about it points at the stage the user wrote, and any key
    // a later pass mints from the span lands on the node it expects.
    //
    // It also carries the stage's CARRIER. chelis#1107 cuts both ways: a
    // reader that knows one carrier declines the other, so a fold that reads
    // both and always WRITES the typed one hands every list-carrier consumer
    // a node it cannot match. The host expression lowerer is one such
    // consumer, and a typed `app` reached its catch-all as "no
    // host-expression lowering rule exists for this checked form" across the
    // whole grad suite. Matching the stage keeps the synthesized node in the
    // carrier its siblings are already in.
    let (meta, span) = match stamped(&stage) {
        Some((_, meta, _)) => (meta.clone(), stage.span()),
        None => (Metadata::default(), stage.span()),
    };
    let children = vec![stage.clone(), acc];
    match &stage {
        Expr::Node(..) => Expr::node(DeepTag::App, meta, children, span),
        _ => Expr::List(
            List {
                elements: std::iter::once(Expr::Atom(Atom::Tag(DeepTag::App), span))
                    .chain(std::iter::once(Expr::Map(meta, span)))
                    .chain(children)
                    .collect(),
            },
            span,
        ),
    }
}

#[cfg(test)]
thread_local! {
    /// Expression nodes copied by the pipe fold on this thread since the last
    /// reset (chelis#2207). Counted at the copy, never estimated.
    static FOLD_COPIED_NODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Copy `expr` for the fold's result, recording the nodes copied.
fn copied(expr: &Expr) -> Expr {
    #[cfg(test)]
    FOLD_COPIED_NODES.with(|nodes| nodes.set(nodes.get() + expr_nodes(expr)));
    expr.clone()
}

/// Expression nodes in `expr`, every carrier included; metadata is not
/// counted.
#[cfg(test)]
fn expr_nodes(expr: &Expr) -> usize {
    1 + match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => 0,
        Expr::MetaExpr(meta, _) => expr_nodes(&meta.expr),
        Expr::List(list, _) => list.elements.iter().map(expr_nodes).sum(),
        Expr::Node(node, _) => node.children_slice().iter().map(expr_nodes).sum(),
        Expr::BareList(elements, _) => elements.iter().map(expr_nodes).sum(),
        Expr::UnknownForm(data) => data.children.iter().map(expr_nodes).sum(),
    }
}

/// Expression nodes copied by the pipe fold on this thread since the last
/// reset.
#[cfg(test)]
pub(crate) fn fold_copied_nodes() -> usize {
    FOLD_COPIED_NODES.with(std::cell::Cell::get)
}

/// Reset [`fold_copied_nodes`] for this thread.
#[cfg(test)]
pub(crate) fn reset_fold_copied_nodes() {
    FOLD_COPIED_NODES.with(|nodes| nodes.set(0));
}

/// Fold every `Pipe` in `expr`, innermost first.
///
/// Bottom-up matters: a stage body may itself contain a pipe, and
/// `fold_stage` copies the stage as written, so a pipe nested inside one
/// would survive a top-down pass and reach a consumer that no longer has an
/// arm for it.
///
/// A subtree with no pipe in it is returned unchanged, and a rebuilt one
/// keeps its original carrier (chelis#1107): this pass must not turn a
/// `List` tree into a `Node` tree as a side effect, because the carrier is
/// observable to printers, validators and the stamping rules.
pub fn fold_pipes(expr: &Expr) -> Expr {
    // The non-stamped carriers hold children too, and a pipe can sit inside
    // one: a typed pipe-stage parameter arrives as a `MetaExpr` wrapper, and
    // a stage body can reach the fold through a `BareList` or an
    // `UnknownForm`. Declining to descend leaves that pipe for the
    // fail-closed raise, which is a real program failing rather than a
    // malformed tree, so every carrier is walked.
    match expr {
        Expr::MetaExpr(meta_expr, span) => {
            return Expr::MetaExpr(
                crate::ast::MetaExpr {
                    metadata: meta_expr.metadata.clone(),
                    expr: Box::new(fold_pipes(&meta_expr.expr)),
                },
                *span,
            );
        }
        Expr::BareList(items, span) => {
            return Expr::BareList(items.iter().map(fold_pipes).collect(), *span);
        }
        Expr::UnknownForm(data) => {
            return Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
                head: data.head.clone(),
                meta: data.meta.clone(),
                children: data.children.iter().map(fold_pipes).collect(),
                span: data.span,
            }));
        }
        _ => {}
    }
    let Some((tag, meta, kids)) = stamped(expr) else {
        return copied(expr);
    };
    let folded: Vec<Expr> = kids.iter().map(fold_pipes).collect();
    let children_moved = folded.iter().zip(kids).any(|(new, old)| new != old);
    if tag != DeepTag::Pipe && !children_moved {
        return copied(expr);
    }
    let rebuilt = rebuild(expr, tag, meta, folded);
    if tag != DeepTag::Pipe {
        return rebuilt;
    }
    fold_one(&rebuilt).unwrap_or(rebuilt)
}

/// Rebuild `expr` with new children, in the carrier it arrived in.
fn rebuild(expr: &Expr, tag: DeepTag, meta: &Metadata, children: Vec<Expr>) -> Expr {
    match expr {
        Expr::List(_, span) => Expr::List(
            List {
                elements: std::iter::once(Expr::Atom(Atom::Tag(tag), *span))
                    .chain(std::iter::once(Expr::Map(meta.clone(), *span)))
                    .chain(children)
                    .collect(),
            },
            *span,
        ),
        _ => Expr::node(tag, meta.clone(), children, expr.span()),
    }
}

/// Fold one already-child-folded `Pipe` expression, in either carrier.
fn fold_one(pipe: &Expr) -> Option<Expr> {
    let (tag, _, kids) = stamped(pipe)?;
    if tag != DeepTag::Pipe {
        return None;
    }
    let (seed, stages) = kids.split_first()?;
    let mut acc = seed.clone();
    for stage in stages {
        acc = fold_stage(stage, acc);
    }
    Some(acc)
}

/// Fold every `Pipe` in a whole program.
///
/// The type checker calls this on its input, so inference AND the
/// annotation pass that follows it see the same tree: `annotate` walks the
/// expression it was given, so folding only inside inference would leave
/// the annotated tree, which is the checker's output and what every later
/// pass reads, still carrying the `Pipe` node.
pub fn fold_program_pipes(exprs: &[Expr]) -> Vec<Expr> {
    exprs.iter().map(fold_pipes).collect()
}

fn is_var(expr: &Expr, name: &str) -> bool {
    matches!(stamped(expr), Some((DeepTag::Var, _, [Expr::Atom(Atom::Name(found), _)])) if found == name)
}

/// Conservatively decline forwarding when another argument mentions the
/// parameter. No binder analysis is needed: declining leaves a normal call.
fn mentions_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Atom(Atom::Name(found), _) => found == name,
        Expr::List(list, _) => list.elements.iter().any(|child| mentions_name(child, name)),
        Expr::Node(node, _) => node
            .children_slice()
            .iter()
            .any(|child| mentions_name(child, name)),
        Expr::MetaExpr(wrapper, _) => mentions_name(&wrapper.expr, name),
        Expr::BareList(items, _) => items.iter().any(|child| mentions_name(child, name)),
        Expr::UnknownForm(form) => form.children.iter().any(|child| mentions_name(child, name)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    fn fold_source(src: &str) -> String {
        let exprs = parse_str(src).expect("parse");
        let folded = fold_program_pipes(&exprs);
        crate::printer::print_canonical_flat(&folded)
    }

    /// The shape chelis#1923 is about: a bare-name stage becomes the
    /// application it denotes, so the next stage's arm sees the real operand.
    #[test]
    fn a_bare_name_stage_becomes_an_application() {
        let folded = fold_source("(pipe {} (var {} xs) (var {} to_tensor))");
        assert!(
            folded.contains("to_tensor") && !folded.contains("pipe"),
            "{folded}"
        );
    }

    /// A direct first-argument forwarding stage exposes the literal operand.
    #[test]
    fn a_direct_first_argument_stage_is_folded() {
        let folded = fold_source(
            "(pipe {} (var {} xs) \
             (fn {} (params {} __chelis_pipe) \
             (app {} (var {} expand) (var {} __chelis_pipe) (lit {} 0))))",
        );
        assert!(
            folded.contains("expand") && folded.contains("xs") && !folded.contains("__chelis_pipe"),
            "{folded}"
        );
    }

    #[test]
    fn dedicated_operand_stages_forward_only_the_direct_parameter() {
        for (tag, tail) in [("cast", " (t-prim {} i64)"), ("copy", ""), ("realize", "")] {
            let folded = fold_source(&format!(
                "(pipe {{}} (var {{}} xs) (fn {{surf_pipe_stage: \"call-first\"}} (params {{}} p) ({tag} {{}} (var {{}} p){tail})))"
            ));
            assert!(
                !folded.contains("params") && !folded.contains("surf_pipe_stage"),
                "{folded}"
            );
            assert_eq!(folded.matches("xs").count(), 1, "{folded}");
            let deferred = fold_source(&format!(
                "(pipe {{}} (var {{}} xs) (fn {{}} (params {{}} p) ({tag} {{}} (if {{}} (lit {{}} false) (var {{}} p) (lit {{}} 7)){tail})))"
            ));
            assert!(deferred.contains("params"), "{deferred}");
        }
    }

    /// A parameter used TWICE is not substituted: doing so would duplicate
    /// the accumulator expression and any effect or cost the source wrote
    /// once. The stage is applied instead, which denotes the same call.
    #[test]
    fn a_twice_used_stage_parameter_is_applied_not_duplicated() {
        let folded = fold_source(
            "(pipe {} (var {} xs) \
             (fn {surf_pipe_stage: \"call-first\"} (params {} p) (app {} (var {} add) (var {} p) (var {} p))))",
        );
        assert_eq!(folded.matches("xs").count(), 1, "{folded}");
        assert!(
            folded.contains("params") && !folded.contains("surf_pipe_stage"),
            "the stage stays a lambda: {folded}"
        );
    }

    /// A pipe nested inside a stage body is folded too, which a top-down
    /// pass would miss and leave for a consumer that no longer has an arm.
    #[test]
    fn a_pipe_nested_in_a_stage_body_is_folded() {
        let folded = fold_source(
            "(pipe {} (var {} xs) \
             (fn {} (params {} p) \
             (app {} (var {} f) (var {} p) (pipe {} (var {} ys) (var {} g)))))",
        );
        assert!(!folded.contains("pipe"), "{folded}");
    }

    /// chelis#2207: folding a pipe-free tree must cost work linear in its
    /// size. The fold used to clone every child subtree, deep-compare the
    /// clones against the originals to detect movement, and then clone the
    /// unchanged subtree again at every stamped node, so a nested expression
    /// cost O(size x depth) and a 2000-element literal took seconds per
    /// fold.
    ///
    /// Counted receipt in the shape of chelis#2181's `substitution_nodes`:
    /// `fold_copied_nodes` counts expression nodes the fold copies. The
    /// assertion is a ratio, so it states the asymptotic promise with no
    /// machine budget: doubling the depth of a pipe-free chain may at most
    /// triple the nodes copied.
    ///
    /// Evidentiary status: REGRESSION TEST, proven failing first. With the
    /// counter on the unchanged fold, a 64-deep chain copied 6,563 nodes and a
    /// 128-deep chain 25,411, a ratio of 3.9 (quadratic), and this assertion
    /// failed.
    /// After the fix the same chains copy PASS_SMALL and PASS_LARGE: the one
    /// owned result, and nothing per level.
    #[test]
    fn folding_a_pipe_free_chain_copies_linear_work() {
        fn chain(depth: usize) -> Expr {
            let mut source = "(var {} x)".to_string();
            for _ in 0..depth {
                source = format!("(app {{}} (var {{}} f) {source})");
            }
            parse_str(&source).expect("parse").remove(0)
        }
        fn copied_nodes(depth: usize) -> (usize, usize) {
            let expr = chain(depth);
            let size = expr_nodes(&expr);
            reset_fold_copied_nodes();
            let folded = fold_pipes(&expr);
            assert_eq!(folded, expr, "a pipe-free chain folds to itself");
            (fold_copied_nodes(), size)
        }
        let (small, small_size) = copied_nodes(64);
        let (large, _) = copied_nodes(128);
        eprintln!("#2207 receipt: depth 64 copied {small} nodes, depth 128 copied {large}");
        assert!(
            small >= small_size,
            "#2207: the fold returns an owned copy, so it must record at least the \
             {small_size} nodes of the result; {small} means the counter is not measuring"
        );
        assert!(
            large <= small.saturating_mul(3),
            "#2207: doubling the chain depth must at most triple the nodes the fold copies; \
             depth 64 copied {small} and depth 128 copied {large}, a factor of {}",
            large / small.max(1)
        );
    }

    /// A subtree with no pipe is returned unchanged, carrier included.
    #[test]
    fn a_program_without_a_pipe_is_untouched() {
        let src = "(def {} f (fn {} (params {} x) (app {} (var {} g) (var {} x))))";
        let exprs = parse_str(src).expect("parse");
        assert_eq!(fold_program_pipes(&exprs), exprs);
    }

    /// Transitional carriers retain their contents when there is no pipe.
    #[test]
    fn a_transitional_variant_is_passed_through() {
        let span = crate::span::Span::new(0, 0);
        let bare = Expr::BareList(
            vec![Expr::Atom(Atom::Name("other_name".to_string()), span)],
            span,
        );
        assert_eq!(fold_pipes(&bare), bare);
    }

    /// A parameter outside the first argument is not moved into the body.
    /// Assert exact structure through both stamped carriers; execution rows
    /// check the lexical and eager-evaluation consequences at the CLI.
    #[test]
    fn general_lambda_stages_remain_applications_in_both_carriers() {
        fn typed(expr: &Expr) -> Expr {
            match stamped(expr) {
                Some((tag, meta, kids)) => Expr::node(
                    tag,
                    meta.clone(),
                    kids.iter().map(typed).collect(),
                    expr.span(),
                ),
                None => expr.clone(),
            }
        }
        for body in [
            "(fn {} (params {} y) (var {} p))",
            "(if {} (lit {} false) (var {} p) (lit {} 7))",
            "(app {} (var {} add) (lit {} 1) (var {} p))",
            "(app {} (var {} p) (var {} p))",
            "(app {} (var {} add) (var {} p) (fn {} (params {} y) (var {} p)))",
        ] {
            let source = format!("(pipe {{}} (var {{}} xs) (fn {{}} (params {{}} p) {body}))");
            let parsed = parse_str(&source).expect("parse").remove(0);
            for pipe in [parsed.clone(), typed(&parsed)] {
                let (_, _, kids) = stamped(&pipe).expect("pipe");
                let actual = fold_pipes(&pipe);
                let (tag, _, args) = stamped(&actual).expect("application");
                assert_eq!(tag, DeepTag::App);
                assert_eq!(args, &[kids[1].clone(), kids[0].clone()], "{source}");
            }
        }
    }

    /// A stage that shadows the pipe parameter does not capture it.
    #[test]
    fn a_shadowing_binder_keeps_its_lambda() {
        let folded = fold_source(
            "(pipe {} (var {} xs) \
             (fn {} (params {} p) (fn {} (params {} p) (var {} p))))",
        );
        assert!(
            folded.matches("xs").count() == 1 && folded.matches("params").count() == 2,
            "the inner binder owns `p`: {folded}"
        );
    }
}
