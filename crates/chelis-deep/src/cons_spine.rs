//! One iterative reader for canonical `Cons(head, tail)` spines.

use crate::{Atom, DeepTag, Expr, ExprCarrier};

/// A carrier that exposes its canonical Cons cell and Nil terminator.
///
/// Compiler value carriers can implement this trait without another traversal
/// loop. A noncanonical node remains the actual terminal for its reader to
/// reject or evaluate according to that reader's contract.
pub trait ConsSpineNode: Sized {
    type Head;

    fn unwrapped_for_spine(&self) -> &Self {
        self
    }
    fn cons_parts(&self) -> Option<(&Self::Head, &Self)>;
    fn cons_parts_with_terminal_name(&self) -> Option<(&Self::Head, &Self)> {
        self.cons_parts()
    }
    fn is_nil(&self) -> bool;
}

/// One canonical cell. `node` retains metadata needed by individual readers.
pub struct ConsCell<'a, N: ConsSpineNode> {
    pub node: &'a N,
    pub head: &'a N::Head,
}

/// The exact expression or value after the last canonical Cons cell.
pub enum ConsSpineTail<'a, N> {
    Nil(&'a N),
    Other(&'a N),
}

impl<'a, N> Copy for ConsSpineTail<'a, N> {}

impl<'a, N> Clone for ConsSpineTail<'a, N> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A typed rejection with the actual offending tail available for location.
pub struct ImproperConsTail<'a, N> {
    pub tail: &'a N,
}

impl<N> std::fmt::Debug for ImproperConsTail<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ImproperConsTail")
    }
}

impl<N> std::fmt::Display for ImproperConsTail<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("canonical Cons spine has an improper tail")
    }
}

impl<N> std::error::Error for ImproperConsTail<'_, N> {}

/// Iterates over heads without using one native call frame per list element.
pub struct ConsSpine<'a, N: ConsSpineNode> {
    cursor: &'a N,
    terminal: Option<ConsSpineTail<'a, N>>,
    terminal_names: bool,
    metadata_wrappers: bool,
}

impl<'a, N: ConsSpineNode> ConsSpine<'a, N> {
    pub fn new(root: &'a N) -> Self {
        Self {
            cursor: root,
            terminal: None,
            terminal_names: false,
            metadata_wrappers: false,
        }
    }

    /// Use the terminal component of a constructor name, for readers whose
    /// established input also accepts a qualified `Cons` spelling.
    pub fn with_terminal_names(root: &'a N) -> Self {
        Self {
            cursor: root,
            terminal: None,
            terminal_names: true,
            metadata_wrappers: false,
        }
    }

    /// Accept legacy metadata wrappers around any cell or terminator.
    pub fn with_metadata_wrappers(root: &'a N) -> Self {
        Self {
            cursor: root,
            terminal: None,
            terminal_names: false,
            metadata_wrappers: true,
        }
    }

    /// Available after iteration has reached the terminal node.
    pub fn tail(&self) -> Option<ConsSpineTail<'a, N>> {
        self.terminal
    }

    /// Drain any unread heads and require a canonical Nil terminal.
    pub fn require_nil(&mut self) -> Result<(), ImproperConsTail<'a, N>> {
        for _ in self.by_ref() {}
        match self
            .terminal
            .expect("draining a finite spine reaches its terminal")
        {
            ConsSpineTail::Nil(_) => Ok(()),
            ConsSpineTail::Other(tail) => Err(ImproperConsTail { tail }),
        }
    }
}

impl<'a, N: ConsSpineNode> Iterator for ConsSpine<'a, N> {
    type Item = ConsCell<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.terminal.is_some() {
            return None;
        }
        let node = if self.metadata_wrappers {
            self.cursor.unwrapped_for_spine()
        } else {
            self.cursor
        };
        let parts = if self.terminal_names {
            node.cons_parts_with_terminal_name()
        } else {
            node.cons_parts()
        };
        if let Some((head, tail)) = parts {
            self.cursor = tail;
            Some(ConsCell { node, head })
        } else {
            self.terminal = Some(if node.is_nil() {
                ConsSpineTail::Nil(node)
            } else {
                ConsSpineTail::Other(node)
            });
            None
        }
    }
}

fn is_var(expr: &Expr, name: &str, terminal_name: bool) -> bool {
    matches!(
        expr.carrier(),
        ExprCarrier::DecodedNode(
            DeepTag::Var,
            _,
            [Expr::Atom(Atom::Name(found), _)]
        ) if found == name || terminal_name && found
            .rsplit_once("__")
            .map(|(_, tail)| tail)
            .or_else(|| found.rsplit_once('.').map(|(_, tail)| tail)) == Some(name)
    )
}

impl ConsSpineNode for Expr {
    type Head = Expr;

    fn unwrapped_for_spine(&self) -> &Self {
        let mut expr = self;
        while let Expr::MetaExpr(meta, _) = expr {
            expr = &meta.expr;
        }
        expr
    }

    fn cons_parts(&self) -> Option<(&Self::Head, &Self)> {
        match self.carrier() {
            ExprCarrier::DecodedNode(DeepTag::App, _, [constructor, head, tail])
                if is_var(constructor, "Cons", false) =>
            {
                Some((head, tail))
            }
            _ => None,
        }
    }

    fn cons_parts_with_terminal_name(&self) -> Option<(&Self::Head, &Self)> {
        match self.carrier() {
            ExprCarrier::DecodedNode(DeepTag::App, _, [constructor, head, tail])
                if is_var(constructor.unwrapped_for_spine(), "Cons", true) =>
            {
                Some((head, tail))
            }
            _ => None,
        }
    }

    fn is_nil(&self) -> bool {
        is_var(self, "Nil", false)
    }
}
