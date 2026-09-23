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
use crate::ast::{Atom, Expr};
use crate::tag::DeepTag;

/// The parts of a stamped node.
fn stamped(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        crate::ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        crate::ExprCarrier::StructuralList(_)
        | crate::ExprCarrier::UndecodableHead(_, _, _)
        | crate::ExprCarrier::Atom(_)
        | crate::ExprCarrier::MetadataMap(_)
        | crate::ExprCarrier::MetadataExpression(_) => None,
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
///
/// `None` when no `app` node admits the synthesized application. A pipe stage
/// is an inference-bypass child, so a hand-built `Pipe` node can carry a bare
/// name there. A stage whose own metadata is bound to its tag cannot lend it
/// to an `app` either, and valid Surf reaches that case:
/// `x |> grad(f, wrt=v)` copies `grad`'s `wrt` (chelis#2430). The pipe then
/// stays unfolded for the consumers' fail-closed pipe rejection.
fn fold_stage(stage: &Expr, acc: Expr) -> Option<Expr> {
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
            return Some(rebuild(body, tag, meta, children));
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
    let (meta, span) = match stamped(&stage) {
        Some((_, meta, _)) => (meta.clone(), stage.span()),
        None => (Metadata::default(), stage.span()),
    };
    crate::node::Node::try_new(DeepTag::App, meta, vec![stage, acc])
        .ok()
        .map(|node| Expr::Node(Box::new(node), span))
}

#[cfg(test)]
thread_local! {
    /// Expression nodes copied by the pipe fold on this thread since the last
    /// reset (chelis#2207). Counted at the copy, never estimated.
    static FOLD_COPIED_NODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The subset of those nodes rebuilt through the validating node
    /// constructor rather than copied (chelis#2434).
    static FOLD_REBUILT_NODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Record one expression node of the fold's result that was rebuilt.
fn count_rebuilt_node() {
    #[cfg(test)]
    {
        FOLD_COPIED_NODES.with(|nodes| nodes.set(nodes.get() + 1));
        FOLD_REBUILT_NODES.with(|nodes| nodes.set(nodes.get() + 1));
    }
}

/// Copy the unchanged subtree `expr` into the fold's result, recording the
/// nodes copied.
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

/// Expression nodes the pipe fold rebuilt on this thread since the last
/// reset.
#[cfg(test)]
pub(crate) fn fold_rebuilt_nodes() -> usize {
    FOLD_REBUILT_NODES.with(std::cell::Cell::get)
}

/// Reset [`fold_copied_nodes`] and [`fold_rebuilt_nodes`] for this thread.
#[cfg(test)]
pub(crate) fn reset_fold_copied_nodes() {
    FOLD_COPIED_NODES.with(|nodes| nodes.set(0));
    FOLD_REBUILT_NODES.with(|nodes| nodes.set(0));
}

/// Fold every `Pipe` in `expr`, innermost first.
///
/// Bottom-up matters: a stage body may itself contain a pipe, and
/// `fold_stage` copies the stage as written, so a pipe nested inside one
/// would survive a top-down pass and reach a consumer that no longer has an
/// arm for it.
///
/// The walk is an explicit heap worklist, not native recursion. The checker
/// folds its input before any of its stack-guarded walkers runs, and the fold
/// has no diagnostic channel of its own, so a recursive fold would abort the
/// process on the same deep input those walkers reject with a typed
/// diagnostic.
///
/// Only a node with a `Pipe` at or below it is rebuilt, through the
/// validating node constructor, from its already-folded children. A subtree
/// with no pipe in it is not rebuilt: it is copied once, where a rebuilt
/// ancestor needs it or at the root, so the work is linear in the size of the
/// tree (chelis#2207) and a pipe-free tree costs one copy, not one validated
/// rebuild per node (chelis#2434). That copy is the derived `Clone`, which
/// does recurse on the native stack, at roughly four times a `Drop`'s stack per
/// level; [`fold_pipes_if_changed`] makes no copy of a pipe-free tree at all.
///
/// Every carrier is walked, because a pipe can sit inside any of them: a
/// typed pipe-stage parameter arrives as a `MetaExpr` wrapper, and a stage
/// body can reach the fold through a `BareList` or an `UnknownForm`.
/// Declining to descend would leave that pipe for the fail-closed raise, a
/// real program failing rather than a malformed tree. Metadata values are
/// copied as written.
pub fn fold_pipes(expr: &Expr) -> Expr {
    fold_pipes_if_changed(expr).unwrap_or_else(|| copied(expr))
}

/// [`fold_pipes`], reporting whether the tree held a pipe: `None` when `expr`
/// contains no `Pipe` node and would fold to itself, `Some(folded)` otherwise.
/// A pipe whose stage cannot be applied stays unfolded, so `Some` can hold a
/// tree equal to the input. A caller that can keep its input pays nothing for
/// a pipe-free tree.
pub fn fold_pipes_if_changed(expr: &Expr) -> Option<Expr> {
    enum Step<'a> {
        Enter(&'a Expr),
        Exit(&'a Expr, usize),
    }
    let mut steps = vec![Step::Enter(expr)];
    // `None` stands for a subtree with no pipe in it, still unchanged.
    let mut values: Vec<Option<Expr>> = Vec::new();
    while let Some(step) = steps.pop() {
        match step {
            Step::Enter(expr) => match fold_children_of(expr) {
                None => values.push(None),
                Some(children) => {
                    steps.push(Step::Exit(expr, children.len()));
                    steps.extend(children.iter().rev().map(Step::Enter));
                }
            },
            Step::Exit(expr, child_count) => {
                let start = values
                    .len()
                    .checked_sub(child_count)
                    .expect("the fold's step and value stacks stay balanced");
                let folded = values.split_off(start);
                let is_pipe = matches!(expr, Expr::Node(node, _) if node.tag() == DeepTag::Pipe);
                if !is_pipe && folded.iter().all(Option::is_none) {
                    values.push(None);
                    continue;
                }
                let children = folded
                    .into_iter()
                    .zip(fold_children_of(expr).unwrap_or_default())
                    .map(|(folded, original)| folded.unwrap_or_else(|| copied(original)))
                    .collect();
                count_rebuilt_node();
                values.push(Some(rebuild_folded(expr, children)));
            }
        }
    }
    let folded = values.pop().expect("the fold produces its root");
    debug_assert!(values.is_empty(), "the fold produces exactly one root");
    folded
}

/// The children the fold descends into, or `None` for a leaf.
fn fold_children_of(expr: &Expr) -> Option<&[Expr]> {
    match expr {
        Expr::Atom(..) | Expr::Map(..) => None,
        Expr::MetaExpr(meta_expr, _) => Some(std::slice::from_ref(meta_expr.expr.as_ref())),
        Expr::Node(node, _) => Some(node.children_slice()),
        Expr::BareList(items, _) => Some(items),
        Expr::UnknownForm(data) => Some(&data.children),
    }
}

/// Rebuild `expr` over its folded `children`, folding it too when it is a
/// `Pipe`.
fn rebuild_folded(expr: &Expr, mut children: Vec<Expr>) -> Expr {
    match expr {
        Expr::MetaExpr(meta_expr, span) => Expr::MetaExpr(
            crate::ast::MetaExpr {
                metadata: meta_expr.metadata.clone(),
                expr: Box::new(
                    children
                        .pop()
                        .expect("a metadata wrapper has one expression"),
                ),
            },
            *span,
        ),
        Expr::BareList(_, span) => Expr::BareList(children, *span),
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
            head: data.head.clone(),
            meta: data.meta.clone(),
            children,
            span: data.span,
        })),
        Expr::Node(node, _) => {
            let rebuilt = rebuild(expr, node.tag(), node.meta(), children);
            if node.tag() == DeepTag::Pipe {
                fold_one(&rebuilt).unwrap_or(rebuilt)
            } else {
                rebuilt
            }
        }
        Expr::Atom(..) | Expr::Map(..) => unreachable!("leaves are copied on entry"),
    }
}

/// Rebuild the stamped node `expr` with new children.
fn rebuild(expr: &Expr, tag: DeepTag, meta: &Metadata, children: Vec<Expr>) -> Expr {
    Expr::node(tag, meta.clone(), children, expr.span())
}

/// Fold one already-child-folded `Pipe` expression.
fn fold_one(pipe: &Expr) -> Option<Expr> {
    let (tag, _, kids) = stamped(pipe)?;
    if tag != DeepTag::Pipe {
        return None;
    }
    let (seed, stages) = kids.split_first()?;
    let mut acc = seed.clone();
    for stage in stages {
        acc = fold_stage(stage, acc)?;
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
    /// triple the nodes copied. The exact-count assertion beside it locks the
    /// constant too: a pipe-free tree is copied once, at the root.
    ///
    /// Evidentiary status: REGRESSION TEST, proven failing first. With the
    /// counter on the unchanged fold, a 64-deep chain copied 6,563 nodes and a
    /// 128-deep chain 25,411, a ratio of 3.9 (quadratic), and this assertion
    /// failed.
    /// After the fix the same chains copy 194 and 386 nodes, a ratio of 2.0:
    /// the one owned result, and nothing per level.
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
        assert_eq!(
            small, small_size,
            "#2207: a pipe-free tree is copied exactly once, at the root: the fold must \
             record the {small_size} nodes of the result and nothing per level"
        );
        assert!(
            large <= small.saturating_mul(3),
            "#2207: doubling the chain depth must at most triple the nodes the fold copies; \
             depth 64 copied {small} and depth 128 copied {large}, a factor of {}",
            large / small.max(1)
        );
    }

    /// chelis#2434: a node with no pipe at or below it must not be rebuilt
    /// through the validating node constructor. The host runtime prepares a
    /// lowering context per `grad` or `vmap` application and folds every
    /// program definition there, so a fold that rebuilt every node doubled the
    /// cost of each application in a package that includes the standard
    /// library. Only the ancestors of a folded pipe are rebuilt; everything
    /// else is copied, and a pipe-free tree reports no change at all.
    ///
    /// Evidentiary status: REGRESSION TEST. With the exit step rebuilding
    /// every node, as the chelis#2427 worklist did, a pipe-free chain reported
    /// a change and this test failed at its first assertion. The copy-count
    /// test above passed under that mutation: it counts nodes copied, and a
    /// rebuilt node is copied too.
    #[test]
    fn only_the_ancestors_of_a_folded_pipe_are_rebuilt() {
        const DEPTH: usize = 64;
        let mut pipe_free = "(var {} x)".to_string();
        let mut with_pipe = "(pipe {} (var {} x) (var {} f))".to_string();
        for _ in 0..DEPTH {
            pipe_free = format!("(app {{}} (var {{}} g) {pipe_free})");
            with_pipe = format!("(app {{}} (var {{}} g) {with_pipe})");
        }
        let pipe_free = parse_str(&pipe_free).expect("parse").remove(0);
        let with_pipe = parse_str(&with_pipe).expect("parse").remove(0);

        reset_fold_copied_nodes();
        assert_eq!(fold_pipes_if_changed(&pipe_free), None);
        assert_eq!(fold_pipes(&pipe_free), pipe_free);
        assert_eq!(
            fold_rebuilt_nodes(),
            0,
            "#2434: a pipe-free tree is copied, never rebuilt"
        );
        assert_eq!(fold_copied_nodes(), expr_nodes(&pipe_free));

        reset_fold_copied_nodes();
        let folded = fold_pipes_if_changed(&with_pipe).expect("the pipe folds");
        assert!(
            !crate::printer::print_canonical_flat(std::slice::from_ref(&folded)).contains("pipe")
        );
        assert_eq!(
            fold_rebuilt_nodes(),
            DEPTH + 1,
            "#2434: only the pipe and its {DEPTH} ancestors are rebuilt; each sibling \
             `(var {{}} g)` is copied"
        );
    }

    /// The fold runs before the checker's stack-guarded walkers and has no
    /// diagnostic channel, so it must not recurse on the native stack: a
    /// 5000-deep `app` chain with a pipe at its bottom folds on a 256 KiB
    /// thread. Building, comparing and dropping the trees recurse, so they run
    /// on a large-stack thread; only the fold runs on the small stack.
    #[test]
    fn a_deep_chain_folds_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(fold_a_deep_chain_on_a_small_stack)
            .expect("spawn the large-stack harness")
            .join()
            .expect("the large-stack harness must not abort");
    }

    fn fold_a_deep_chain_on_a_small_stack() {
        const DEPTH: usize = 5000;
        let span = crate::span::Span::new(0, 0);
        let var = |name: &str| {
            Expr::node(
                DeepTag::Var,
                Metadata::default(),
                vec![Expr::Atom(Atom::Name(name.to_string()), span)],
                span,
            )
        };
        let app = |callee: Expr, argument: Expr| {
            Expr::node(
                DeepTag::App,
                Metadata::default(),
                vec![callee, argument],
                span,
            )
        };
        let mut input = Expr::node(
            DeepTag::Pipe,
            Metadata::default(),
            vec![var("x"), var("f")],
            span,
        );
        let mut expected = app(var("f"), var("x"));
        for _ in 0..DEPTH {
            input = app(var("g"), input);
            expected = app(var("g"), expected);
        }
        let input = std::sync::Arc::new(input);
        let worker_input = std::sync::Arc::clone(&input);
        let folded = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || fold_pipes(&worker_input))
            .expect("spawn the small-stack fold")
            .join()
            .expect("the fold must not exhaust a small native stack");
        assert!(
            folded == expected,
            "the bottom pipe folds and the chain is kept"
        );
    }

    /// A subtree with no pipe is returned unchanged, carrier included.
    #[test]
    fn a_program_without_a_pipe_is_untouched() {
        let src = "(def {} f (fn {} (params {} x) (app {} (var {} g) (var {} x))))";
        let exprs = parse_str(src).expect("parse");
        assert_eq!(fold_program_pipes(&exprs), exprs);
    }

    /// Structural carriers retain their contents when there is no pipe.
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
    /// Assert exact structure; execution rows check the lexical and
    /// eager-evaluation consequences at the CLI.
    #[test]
    fn general_lambda_stages_remain_applications() {
        for body in [
            "(fn {} (params {} y) (var {} p))",
            "(if {} (lit {} false) (var {} p) (lit {} 7))",
            "(app {} (var {} add) (lit {} 1) (var {} p))",
            "(app {} (var {} p) (var {} p))",
            "(app {} (var {} add) (var {} p) (fn {} (params {} y) (var {} p)))",
        ] {
            let source = format!("(pipe {{}} (var {{}} xs) (fn {{}} (params {{}} p) {body}))");
            let pipe = parse_str(&source).expect("parse").remove(0);
            let (_, _, kids) = stamped(&pipe).expect("pipe");
            let actual = fold_pipes(&pipe);
            let (tag, _, args) = stamped(&actual).expect("application");
            assert_eq!(tag, DeepTag::App);
            assert_eq!(args, &[kids[1].clone(), kids[0].clone()], "{source}");
        }
    }

    /// A pipe stage is an inference-bypass child, so a hand-built `Pipe`
    /// node admits a bare name there, which no `app` node admits as its
    /// callee. The fold leaves that pipe in place for the consumers'
    /// fail-closed rejection instead of building an invalid application.
    /// Negative control: the same pipe with a `var` stage folds.
    #[test]
    fn a_bare_name_stage_is_left_unfolded() {
        let span = crate::span::Span::new(0, 0);
        let seed = parse_str("(var {} xs)").expect("parse").remove(0);
        let bare_stage = Expr::Atom(Atom::Name("f".to_string()), span);
        let pipe = Expr::node(
            DeepTag::Pipe,
            Metadata::default(),
            vec![seed.clone(), bare_stage],
            span,
        );
        assert_eq!(fold_pipes(&pipe), pipe);

        let var_stage = parse_str("(var {} f)").expect("parse").remove(0);
        let pipe = Expr::node(
            DeepTag::Pipe,
            Metadata::default(),
            vec![seed, var_stage],
            span,
        );
        let (tag, _, _) = stamped(&fold_pipes(&pipe)).expect("application");
        assert_eq!(tag, DeepTag::App);
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
