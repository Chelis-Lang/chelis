//! A closure's body and the variable reads in it that are their binding's
//! last use (chelis#3720).
//!
//! Applying a closure binds its parameters in a fresh innermost scope, and its
//! body binds more names there. Reading such a binding shares its value, so a
//! container the body passes to a writing builtin (`append`, `concat`,
//! `dict_insert`, `dict_merge`) still has the binding as a second owner, and
//! the write copies every element. A fold that grows a list one element per
//! step was quadratic for that reason alone.
//!
//! A read that no other read of its name can follow, on any path through the
//! body, is its binding's last use; the evaluator takes the value there
//! instead of sharing it. The decision is made once per closure, from the
//! body's syntax, by counting the reads of each name on the path that reads
//! it most:
//!
//! - an `if` evaluates one branch and a `match` one arm body, so each counts
//!   its larger branch, after the condition, the scrutinee and every guard
//!   (a guard that fails passes control to the next arm's guard);
//! - a construct the evaluator runs each child of at most once per evaluation
//!   adds its children's counts;
//! - every other construct, a nested `fn` above all, may evaluate a child
//!   more than once or read a binding without a variable read of its own, so
//!   it counts every name read inside it as read more than once.
//!
//! A read is a last use when its name counts exactly once. Counting by name
//! ignores shadowing, which can only overcount. The evaluator releases only a
//! container binding of the running activation's innermost scope, never a
//! captured one (`Frame::release_local`).

use std::collections::BTreeMap;
use std::sync::Arc;

use chelis_deep::DeepTag;
use chelis_deep::ast::{Expr, ExprCarrier};
use chelis_unord::UnordSet;

use super::frame::{Frame, ResultProducer};
use super::{RuntimeValue, symbol_name};

/// The count for a name read more than once, or inside a construct that may
/// read it more than once. Counts saturate here.
const MANY: u8 = 2;

type Reads = BTreeMap<String, u8>;

#[derive(Debug)]
pub struct ClosureBody {
    expr: Expr,
    /// The address of every variable node of `expr` that is its binding's
    /// last use. The body is shared behind an `Arc` and never mutated, so its
    /// nodes keep these addresses for as long as any clone of the closure
    /// lives, and no node of another expression can share one.
    last_uses: UnordSet<usize>,
}

impl ClosureBody {
    pub(crate) fn new(expr: Expr) -> Arc<Self> {
        let mut body = Arc::new(Self {
            expr,
            last_uses: UnordSet::new(),
        });
        // Addresses are taken where the body now lives.
        let mut reads = Reads::new();
        count_reads(&body.expr, &mut reads);
        let mut last_uses = UnordSet::new();
        collect_last_uses(&body.expr, &reads, &mut last_uses);
        Arc::get_mut(&mut body)
            .expect("a body just built has one owner")
            .last_uses = last_uses;
        body
    }

    pub(crate) fn expr(&self) -> &Expr {
        &self.expr
    }

    /// Whether `read`, a variable node this evaluator is visiting, is a node
    /// of this body that is its binding's last use.
    pub(crate) fn is_last_use(&self, read: &Expr) -> bool {
        self.last_uses.contains(&node_address(read))
    }
}

fn node_address(expr: &Expr) -> usize {
    std::ptr::from_ref(expr) as usize
}

fn add_read(reads: &mut Reads, name: &str, count: u8) {
    let entry = reads.entry(name.to_owned()).or_default();
    *entry = entry.saturating_add(count).min(MANY);
}

fn add_reads(reads: &mut Reads, more: Reads) {
    for (name, count) in more {
        add_read(reads, &name, count);
    }
}

/// The reads of whichever branch runs: the larger count per name.
fn larger_reads(branches: impl IntoIterator<Item = Reads>) -> Reads {
    let mut larger = Reads::new();
    for branch in branches {
        for (name, count) in branch {
            let entry = larger.entry(name).or_default();
            *entry = (*entry).max(count);
        }
    }
    larger
}

fn branch_reads(expr: &Expr) -> Reads {
    let mut reads = Reads::new();
    count_reads(expr, &mut reads);
    reads
}

/// Constructs whose children the evaluator evaluates at most once each per
/// evaluation of the node, reading bindings only through variable nodes.
fn evaluates_each_child_once(tag: DeepTag) -> bool {
    matches!(
        tag,
        DeepTag::App
            | DeepTag::Lit
            | DeepTag::Block
            | DeepTag::Tuple
            | DeepTag::TupleGet
            | DeepTag::Record
            | DeepTag::Kv
            | DeepTag::Access
            | DeepTag::Cast
            | DeepTag::Copy
            | DeepTag::Borrow
    )
}

fn count_reads(expr: &Expr, reads: &mut Reads) {
    // A body is as deep as the checked program allows; grow rather than
    // overflow on a deep one.
    stacker::maybe_grow(64 * 1024, 1024 * 1024, || match expr.carrier() {
        ExprCarrier::MetadataExpression(meta) => count_reads(&meta.expr, reads),
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            if let Some(name) = children.first().and_then(symbol_name) {
                add_read(reads, name, 1);
            }
        }
        ExprCarrier::DecodedNode(DeepTag::If, _, [condition, branches @ ..]) => {
            count_reads(condition, reads);
            add_reads(reads, larger_reads(branches.iter().map(branch_reads)));
        }
        ExprCarrier::DecodedNode(DeepTag::Match, _, [scrutinee, arms @ ..]) => {
            count_reads(scrutinee, reads);
            let mut bodies = Vec::with_capacity(arms.len());
            for arm in arms {
                match arm.carrier() {
                    // The pattern binds names and reads none.
                    ExprCarrier::DecodedNode(DeepTag::Arm, _, [_pattern, guard, body]) => {
                        count_reads(guard, reads);
                        bodies.push(branch_reads(body));
                    }
                    _ => read_many(arm, reads),
                }
            }
            add_reads(reads, larger_reads(bodies));
        }
        ExprCarrier::DecodedNode(DeepTag::Let, _, [bindings, body]) => {
            match bindings.carrier() {
                // Each pair is a binder, which reads nothing, and its
                // initializer.
                ExprCarrier::DecodedNode(DeepTag::Bind, _, pairs) if pairs.len() % 2 == 0 => {
                    for pair in pairs.as_chunks::<2>().0 {
                        count_reads(&pair[1], reads);
                    }
                }
                _ => read_many(bindings, reads),
            }
            count_reads(body, reads);
        }
        ExprCarrier::DecodedNode(tag, _, children) if evaluates_each_child_once(tag) => {
            for child in children {
                count_reads(child, reads);
            }
        }
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => {}
        _ => read_many(expr, reads),
    });
}

/// Count every name read anywhere in `expr` as read more than once.
fn read_many(expr: &Expr, reads: &mut Reads) {
    stacker::maybe_grow(64 * 1024, 1024 * 1024, || match expr.carrier() {
        ExprCarrier::MetadataExpression(meta) => read_many(&meta.expr, reads),
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            if let Some(name) = children.first().and_then(symbol_name) {
                add_read(reads, name, MANY);
            }
        }
        ExprCarrier::DecodedNode(_, _, children)
        | ExprCarrier::UndecodableHead(_, _, children)
        | ExprCarrier::StructuralList(children) => {
            for child in children {
                read_many(child, reads);
            }
        }
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => {}
    });
}

/// Every variable node `count_reads` counted whose name counts once. The walk
/// visits exactly the positions `count_reads` counts once, so a node inside a
/// construct counted as reading more than once is never collected; its name
/// counts `MANY` regardless.
fn collect_last_uses(expr: &Expr, reads: &Reads, last_uses: &mut UnordSet<usize>) {
    stacker::maybe_grow(64 * 1024, 1024 * 1024, || match expr.carrier() {
        ExprCarrier::MetadataExpression(meta) => collect_last_uses(&meta.expr, reads, last_uses),
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            if children
                .first()
                .and_then(symbol_name)
                .is_some_and(|name| reads.get(name) == Some(&1))
            {
                last_uses.insert(node_address(expr));
            }
        }
        ExprCarrier::DecodedNode(DeepTag::If, _, children) => {
            for child in children {
                collect_last_uses(child, reads, last_uses);
            }
        }
        ExprCarrier::DecodedNode(DeepTag::Match, _, [scrutinee, arms @ ..]) => {
            collect_last_uses(scrutinee, reads, last_uses);
            for arm in arms {
                if let ExprCarrier::DecodedNode(DeepTag::Arm, _, [_pattern, guard, body]) =
                    arm.carrier()
                {
                    collect_last_uses(guard, reads, last_uses);
                    collect_last_uses(body, reads, last_uses);
                }
            }
        }
        ExprCarrier::DecodedNode(DeepTag::Let, _, [bindings, body]) => {
            if let ExprCarrier::DecodedNode(DeepTag::Bind, _, pairs) = bindings.carrier()
                && pairs.len() % 2 == 0
            {
                for pair in pairs.as_chunks::<2>().0 {
                    collect_last_uses(&pair[1], reads, last_uses);
                }
            }
            collect_last_uses(body, reads, last_uses);
        }
        ExprCarrier::DecodedNode(tag, _, children) if evaluates_each_child_once(tag) => {
            for child in children {
                collect_last_uses(child, reads, last_uses);
            }
        }
        _ => {}
    });
}

/// The executing closure activation's last-use reads, and the frame
/// snapshots its `let` and `match` forms take on entry and restore on exit.
///
/// A snapshot is a second owner of every value its frame binds, so releasing
/// a binding at its last use also releases that name in every snapshot the
/// activation took. The name is read nowhere else in the activation, so a
/// restored snapshot never needs it. Snapshots below `base` belong to callers'
/// activations, whose same-named bindings are other variables.
#[derive(Debug, Default)]
pub(crate) struct LastUses {
    body: Option<Arc<ClosureBody>>,
    snapshots: Vec<Frame>,
    base: usize,
}

/// A caller's last-use state while one of its callees executes.
pub(crate) struct CallerLastUses {
    body: Option<Arc<ClosureBody>>,
    base: usize,
}

impl LastUses {
    /// Enter an activation of `body`, returning the caller's state.
    pub(crate) fn enter(&mut self, body: Arc<ClosureBody>) -> CallerLastUses {
        CallerLastUses {
            body: self.body.replace(body),
            base: std::mem::replace(&mut self.base, self.snapshots.len()),
        }
    }

    /// Return to the caller's activation. Every snapshot the callee took has
    /// already been restored.
    pub(crate) fn exit(&mut self, caller: CallerLastUses) {
        debug_assert_eq!(self.snapshots.len(), self.base);
        self.body = caller.body;
        self.base = caller.base;
    }

    pub(crate) fn push_snapshot(&mut self, frame: Frame) {
        self.snapshots.push(frame);
    }

    pub(crate) fn pop_snapshot(&mut self) -> Frame {
        self.snapshots
            .pop()
            .expect("every restored frame snapshot was taken")
    }

    pub(crate) fn last_snapshot(&self) -> &Frame {
        self.snapshots
            .last()
            .expect("a taken frame snapshot is still held")
    }

    /// Whether `read` is a last-use read of the executing activation's body.
    pub(crate) fn is_last_use(&self, read: &Expr) -> bool {
        self.body
            .as_ref()
            .is_some_and(|body| body.is_last_use(read))
    }

    /// Release `name` from the innermost scope of `bindings` and from every
    /// snapshot this activation took, returning its value and producer. `None`
    /// when the innermost scope does not own a live binding of `name`.
    pub(crate) fn release(
        &mut self,
        bindings: &mut Frame,
        name: &str,
    ) -> Option<(RuntimeValue, Option<ResultProducer>)> {
        let released = bindings.release_local(name)?;
        for snapshot in &mut self.snapshots[self.base..] {
            snapshot.release_local(name);
        }
        // Every snapshot has dropped its share, so the value moves out
        // solely owned.
        Some(released.into_parts())
    }
}
