//! The gated `Node` type — a stamped vocabulary node with private fields.
//!
//! `Node` is constructible only through `try_new` (boundary — returns
//! Result) or `new` (internal — panics on violation, validates always).
//! Children are accessible only through role-typed accessors and the
//! `ChildRef` total-traversal iterator.
//!
//! Custom `Deserialize` routes through `try_new` and surfaces failures
//! as `serde::de::Error` rather than panicking.

use serde::{Deserialize, Serialize};

use crate::ast::{Atom, Expr, MetaMap};
use crate::role::{AritySpec, ChildStampRole, arity_contract, child_stamp_role};
use crate::tag::DeepTag;

/// Error from `Node::try_new` — a child violated its role constraint or
/// arity was wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeError {
    /// A Name atom appeared at a RuntimeExpr child position.
    NameAtExprSlot {
        tag: DeepTag,
        index: usize,
        name: String,
    },
    /// Child count violates `arity_contract`.
    ArityViolation {
        tag: DeepTag,
        expected: AritySpec,
        actual: usize,
    },
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::NameAtExprSlot { tag, index, name } => {
                write!(
                    f,
                    "structural name `{name}` at RuntimeExpr child position \
                     (tag={}, index={index}); use `(var {{}} {name})` to reference a binding",
                    tag.as_str()
                )
            }
            NodeError::ArityViolation {
                tag,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "wrong child count for `{}`: expected {expected:?}, got {actual}",
                    tag.as_str()
                )
            }
        }
    }
}

impl std::error::Error for NodeError {}

/// A role-tagged reference to a child, for total traversal by
/// printers and serializers.
#[derive(Debug, Clone, Copy)]
pub enum ChildRef<'a> {
    Expr(&'a Expr),
    Binder(&'a str),
    Selector(&'a str),
    Syntax(&'a Expr),
    Type(&'a Expr),
    EffectHandler(&'a Expr),
    Bypass(&'a Expr),
}

/// A stamped vocabulary node with private fields.
///
/// Invariant: no `Atom::Symbol` or `Atom::Tag` at a `RuntimeExpr` child
/// position. Enforced at construction (both `try_new` and `new`
/// validate in all build modes).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Node {
    tag: DeepTag,
    meta: MetaMap,
    children: Vec<Expr>,
}

impl Node {
    /// Boundary constructor — returns `Err` when the user wrote
    /// something wrong (Name at RuntimeExpr slot, wrong arity).
    pub fn try_new(tag: DeepTag, meta: MetaMap, children: Vec<Expr>) -> Result<Self, NodeError> {
        Self::validate(tag, &children)?;
        Ok(Node {
            tag,
            meta,
            children,
        })
    }

    /// Internal constructor — panics on violation. Use only when a
    /// violation would indicate a compiler bug (rewriting
    /// already-validated trees).
    pub fn new(tag: DeepTag, meta: MetaMap, children: Vec<Expr>) -> Self {
        if let Err(e) = Self::validate(tag, &children) {
            panic!("Node::new invariant violation (compiler bug): {e}");
        }
        Node {
            tag,
            meta,
            children,
        }
    }

    fn validate(tag: DeepTag, children: &[Expr]) -> Result<(), NodeError> {
        // Arity check.
        let spec = arity_contract(tag);
        let n = children.len();
        let arity_ok = match spec {
            AritySpec::Fixed(expected) => n == expected,
            AritySpec::AtLeast(min) => n >= min,
            AritySpec::Range(lo, hi) => n >= lo && n <= hi,
        };
        if !arity_ok {
            return Err(NodeError::ArityViolation {
                tag,
                expected: spec,
                actual: n,
            });
        }

        // Per-child role check: reject Name at RuntimeExpr positions.
        for (index, child) in children.iter().enumerate() {
            let role = child_stamp_role(tag, index, n);
            if role == ChildStampRole::RuntimeExpr {
                if let Expr::Atom(Atom::Symbol(s), _) = child {
                    return Err(NodeError::NameAtExprSlot {
                        tag,
                        index,
                        name: s.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    // === Observation ===

    /// The node's vocabulary tag (total — no Option).
    pub fn tag(&self) -> DeepTag {
        self.tag
    }

    /// The node's metadata map.
    pub fn meta(&self) -> &MetaMap {
        &self.meta
    }

    /// Mutable access to metadata (for annotation passes that add
    /// type/span info without rebuilding the node).
    pub fn meta_mut(&mut self) -> &mut MetaMap {
        &mut self.meta
    }

    /// Number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    // === Role-typed accessors ===

    /// Iterator over children at RuntimeExpr positions only.
    pub fn expr_children(&self) -> impl Iterator<Item = &Expr> {
        let tag = self.tag;
        let arity = self.children.len();
        self.children.iter().enumerate().filter_map(move |(i, child)| {
            if child_stamp_role(tag, i, arity) == ChildStampRole::RuntimeExpr {
                Some(child)
            } else {
                None
            }
        })
    }

    /// Indexed expression child. Panics if the role at `index` is not
    /// RuntimeExpr (consumer bug).
    pub fn expr_child(&self, index: usize) -> &Expr {
        let role = child_stamp_role(self.tag, index, self.children.len());
        assert_eq!(
            role,
            ChildStampRole::RuntimeExpr,
            "expr_child({index}) on `{}`: role is {role:?}, not RuntimeExpr",
            self.tag.as_str()
        );
        &self.children[index]
    }

    /// Iterator over binder-position children, yielding their Name string.
    pub fn binder_names(&self) -> impl Iterator<Item = &str> {
        let tag = self.tag;
        let arity = self.children.len();
        self.children.iter().enumerate().filter_map(move |(i, child)| {
            if child_stamp_role(tag, i, arity) == ChildStampRole::Binder {
                match child {
                    Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
                    _ => None,
                }
            } else {
                None
            }
        })
    }

    /// Indexed binder name. Panics on role mismatch.
    pub fn binder_name(&self, index: usize) -> &str {
        let role = child_stamp_role(self.tag, index, self.children.len());
        assert_eq!(
            role,
            ChildStampRole::Binder,
            "binder_name({index}) on `{}`: role is {role:?}, not Binder",
            self.tag.as_str()
        );
        match &self.children[index] {
            Expr::Atom(Atom::Symbol(s), _) => s.as_str(),
            other => panic!(
                "binder_name({index}) on `{}`: child is {other:?}, not Name",
                self.tag.as_str()
            ),
        }
    }

    /// Iterator over children at Type positions.
    pub fn type_children(&self) -> impl Iterator<Item = &Expr> {
        let tag = self.tag;
        let arity = self.children.len();
        self.children.iter().enumerate().filter_map(move |(i, child)| {
            if child_stamp_role(tag, i, arity) == ChildStampRole::Type {
                Some(child)
            } else {
                None
            }
        })
    }

    /// Total traversal yielding role-tagged references.
    pub fn children_iter(&self) -> impl Iterator<Item = ChildRef<'_>> {
        let tag = self.tag;
        let arity = self.children.len();
        self.children.iter().enumerate().map(move |(i, child)| {
            match child_stamp_role(tag, i, arity) {
                ChildStampRole::RuntimeExpr => ChildRef::Expr(child),
                ChildStampRole::Binder => match child {
                    Expr::Atom(Atom::Symbol(s), _) => ChildRef::Binder(s.as_str()),
                    _ => ChildRef::Bypass(child),
                },
                ChildStampRole::Selector => match child {
                    Expr::Atom(Atom::Symbol(s), _) => ChildRef::Selector(s.as_str()),
                    _ => ChildRef::Bypass(child),
                },
                ChildStampRole::Syntax => ChildRef::Syntax(child),
                ChildStampRole::Type => ChildRef::Type(child),
                ChildStampRole::EffectHandler => ChildRef::EffectHandler(child),
                ChildStampRole::ExplicitInferenceBypass => ChildRef::Bypass(child),
            }
        })
    }
}

/// Custom Deserialize: routes through `try_new`, surfaces failure as
/// `D::Error` rather than panicking.
impl<'de> Deserialize<'de> for Node {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NodeShadow {
            tag: DeepTag,
            meta: MetaMap,
            children: Vec<Expr>,
        }

        let shadow = NodeShadow::deserialize(deserializer)?;
        Node::try_new(shadow.tag, shadow.meta, shadow.children)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn name(s: &str) -> Expr {
        Expr::Atom(Atom::Symbol(s.to_string()), sp())
    }

    fn int(n: i64) -> Expr {
        Expr::Atom(Atom::Int(n), sp())
    }

    #[test]
    fn try_new_rejects_name_at_runtime_expr_slot() {
        // App children are all RuntimeExpr. A bare Name there is invalid.
        let result = Node::try_new(DeepTag::App, MetaMap::default(), vec![name("x")]);
        assert!(matches!(result, Err(NodeError::NameAtExprSlot { .. })));
    }

    #[test]
    fn try_new_permits_name_at_binder_slot() {
        // Var child 0 is Syntax (name is content). Def child 0 is Binder.
        let result = Node::try_new(DeepTag::Var, MetaMap::default(), vec![name("x")]);
        assert!(result.is_ok());
    }

    #[test]
    fn try_new_permits_name_at_type_slot() {
        // TVar child 0 is Type. A Name there is a type variable.
        let result = Node::try_new(DeepTag::TVar, MetaMap::default(), vec![name("a")]);
        assert!(result.is_ok());
    }

    #[test]
    fn try_new_rejects_wrong_arity() {
        // Var requires exactly 1 child.
        let result = Node::try_new(DeepTag::Var, MetaMap::default(), vec![name("x"), name("y")]);
        assert!(matches!(result, Err(NodeError::ArityViolation { .. })));
    }

    #[test]
    fn try_new_accepts_correct_arity() {
        let result = Node::try_new(
            DeepTag::If,
            MetaMap::default(),
            vec![int(1), int(2), int(3)],
        );
        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "compiler bug")]
    fn new_panics_on_violation() {
        Node::new(DeepTag::App, MetaMap::default(), vec![name("x")]);
    }

    #[test]
    fn expr_child_returns_runtime_expr_children() {
        let node = Node::new(
            DeepTag::App,
            MetaMap::default(),
            vec![int(1), int(2), int(3)],
        );
        assert_eq!(node.expr_children().count(), 3);
    }

    #[test]
    fn binder_name_works() {
        let node = Node::new(DeepTag::Def, MetaMap::default(), vec![name("f"), int(42)]);
        assert_eq!(node.binder_name(0), "f");
    }

    #[test]
    #[should_panic(expected = "not RuntimeExpr")]
    fn expr_child_panics_on_role_mismatch() {
        let node = Node::new(DeepTag::Def, MetaMap::default(), vec![name("f"), int(42)]);
        // Index 0 of Def is Binder, not RuntimeExpr.
        let _ = node.expr_child(0);
    }

    #[test]
    fn children_iter_yields_correct_roles() {
        let node = Node::new(DeepTag::Def, MetaMap::default(), vec![name("f"), int(42)]);
        let refs: Vec<_> = node.children_iter().collect();
        assert!(matches!(refs[0], ChildRef::Binder("f")));
        assert!(matches!(refs[1], ChildRef::Expr(_)));
    }
}
