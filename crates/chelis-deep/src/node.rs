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

use crate::Metadata;
use crate::ast::{Atom, Expr, List};
use crate::role::{AritySpec, ChildStampRole, arity_contract, child_stamp_role};
use crate::span::Span;
use crate::tag::DeepTag;

/// Error from `Node::try_new` — a child violated its role constraint or
/// arity was wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeError {
    /// Metadata violates its spec-owned shape or locally decidable placement.
    Metadata(crate::metadata::MetadataError),
    /// A Name atom appeared at a RuntimeExpr child position.
    NameAtExprSlot {
        tag: DeepTag,
        index: usize,
        name: String,
    },
    /// A transitional decoded tag atom appeared where a runtime expression
    /// node is required.
    TagAtExprSlot {
        tag: DeepTag,
        index: usize,
        child_tag: DeepTag,
    },
    /// Child count violates `arity_contract`.
    ArityViolation {
        tag: DeepTag,
        expected: AritySpec,
        actual: usize,
    },
    /// A transactional replacement addressed a child that does not exist.
    ChildOutOfBounds {
        tag: DeepTag,
        index: usize,
        child_count: usize,
    },
    /// A closed-vocabulary tag survived as a raw string below the gate.
    RawVocabularyTag { container: DeepTag, raw_tag: String },
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::Metadata(error) => error.fmt(f),
            NodeError::NameAtExprSlot { tag, index, name } => {
                write!(
                    f,
                    "structural name `{name}` at RuntimeExpr child position \
                     (tag={}, index={index}); use `(var {{}} {name})` to reference a binding",
                    tag.as_str()
                )
            }
            NodeError::TagAtExprSlot {
                tag,
                index,
                child_tag,
            } => write!(
                f,
                "transitional tag atom `{}` at RuntimeExpr child position \
                 (tag={}, index={index}); construct a validated Node expression",
                child_tag.as_str(),
                tag.as_str()
            ),
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
            NodeError::ChildOutOfBounds {
                tag,
                index,
                child_count,
            } => write!(
                f,
                "child index {index} is out of bounds for `{}` with {child_count} children",
                tag.as_str()
            ),
            NodeError::RawVocabularyTag { container, raw_tag } => write!(
                f,
                "raw closed-vocabulary tag `{raw_tag}` below stamped `{}` node",
                container.as_str()
            ),
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
/// Invariant: no `Atom::Name` or `Atom::Tag` at a `RuntimeExpr` child
/// position. Enforced at construction (both `try_new` and `new`
/// validate in all build modes).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Node {
    tag: DeepTag,
    meta: Metadata,
    children: Vec<Expr>,
}

impl Node {
    /// Consume a node for atomic rebuilding of coupled metadata and children.
    pub fn into_parts(self) -> (DeepTag, Metadata, Vec<Expr>) {
        (self.tag, self.meta, self.children)
    }

    /// Boundary constructor — returns `Err` when the user wrote
    /// something wrong (Name at RuntimeExpr slot, wrong arity).
    pub fn try_new(tag: DeepTag, meta: Metadata, children: Vec<Expr>) -> Result<Self, NodeError> {
        Self::validate(tag, &meta, &children)?;
        Ok(Node {
            tag,
            meta,
            children,
        })
    }

    /// Internal constructor — panics on violation. Use only when a
    /// violation would indicate a compiler bug (rewriting
    /// already-validated trees).
    pub fn new(tag: DeepTag, meta: Metadata, children: Vec<Expr>) -> Self {
        if let Err(e) = Self::validate(tag, &meta, &children) {
            panic!("Node::new invariant violation (compiler bug): {e}");
        }
        Node {
            tag,
            meta,
            children,
        }
    }

    fn validate(tag: DeepTag, meta: &Metadata, children: &[Expr]) -> Result<(), NodeError> {
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

        // Per-child role check: RuntimeExpr positions contain expression
        // nodes or literal atoms, never structural names or the transitional
        // Atom::Tag carrier.
        for (index, child) in children.iter().enumerate() {
            let role = child_stamp_role(tag, index, n);
            if role == ChildStampRole::RuntimeExpr {
                match child {
                    Expr::Atom(Atom::Name(s), _) => {
                        return Err(NodeError::NameAtExprSlot {
                            tag,
                            index,
                            name: s.clone(),
                        });
                    }
                    Expr::Atom(Atom::Tag(child_tag), _) => {
                        return Err(NodeError::TagAtExprSlot {
                            tag,
                            index,
                            child_tag: *child_tag,
                        });
                    }
                    _ => {}
                }
            }
        }

        // Decode-once is recursive: a public caller cannot smuggle an old
        // raw-string vocabulary node through a metadata value or child of the
        // new gated carrier (chelis#731 Phase 3 successor acceptance).
        //
        // The scan stops at an already-stamped `Expr::Node` boundary
        // (chelis#1109): that child cleared this same scan when IT was
        // constructed, so re-walking its subtree here only repeats work, once
        // per ancestor, which made bottom-up stamping quadratic in nesting
        // depth. Every non-Node carrier is still walked to any depth. See
        // `find_raw_vocabulary_tag_below_gate` for the induction.
        let mut raw_tag = None;
        meta.visit_expressions(&mut |value, _| {
            if raw_tag.is_none() {
                raw_tag = crate::validate::find_raw_vocabulary_tag_below_gate(
                    std::slice::from_ref(value),
                );
            }
        });
        let raw_tag =
            raw_tag.or_else(|| crate::validate::find_raw_vocabulary_tag_below_gate(children));
        if let Some(raw_tag) = raw_tag {
            return Err(NodeError::RawVocabularyTag {
                container: tag,
                raw_tag,
            });
        }

        crate::metadata::validate_node(tag, meta, children).map_err(NodeError::Metadata)?;
        Ok(())
    }

    // === Observation ===

    /// The node's vocabulary tag (total — no Option).
    pub fn tag(&self) -> DeepTag {
        self.tag
    }

    /// The node's metadata map.
    pub fn meta(&self) -> &Metadata {
        &self.meta
    }

    /// Replace metadata only after revalidating the complete node.
    ///
    /// The replacement is transactional: validation failure leaves the
    /// original metadata untouched. Public callers never receive a mutable
    /// reference that could reopen the raw-vocabulary domain after
    /// construction (chelis#731 Phase 3).
    pub fn try_replace_meta(&mut self, meta: Metadata) -> Result<(), NodeError> {
        Self::validate(self.tag, &meta, &self.children)?;
        self.meta = meta;
        Ok(())
    }

    /// Replace one child only after validating the complete candidate node.
    ///
    /// The replacement is transactional: every failure leaves the original
    /// child vector untouched.
    pub fn try_replace_child(&mut self, index: usize, child: Expr) -> Result<(), NodeError> {
        if index >= self.children.len() {
            return Err(NodeError::ChildOutOfBounds {
                tag: self.tag,
                index,
                child_count: self.children.len(),
            });
        }
        let mut children = self.children.clone();
        children[index] = child;
        self.try_replace_children(children)
    }

    /// Replace all children only after validating the complete candidate
    /// node. Validation failure is transactional.
    pub fn try_replace_children(&mut self, children: Vec<Expr>) -> Result<(), NodeError> {
        Self::validate(self.tag, &self.meta, &children)?;
        self.children = children;
        Ok(())
    }

    /// Number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Read-only bridge for remaining positional consumers. This cannot
    /// reopen the validated domain; mutable access is intentionally absent.
    pub fn children_slice(&self) -> &[Expr] {
        &self.children
    }

    // === Bridge: reconstruct List for transition-period consumers ===

    /// Reconstruct the canonical `List` representation that existing
    /// consumer dispatch functions (`infer_var`, `infer_app`, etc.)
    /// expect. This is a **transitional bridge**: once all consumers are
    /// migrated to use the Node API directly, this method becomes dead
    /// code and should be removed.
    ///
    /// The returned List has the same shape as what `Expr::node()` would
    /// produce: `elements[0]` = `Atom::Tag(self.tag)`,
    /// `elements[1]` = `Map(self.meta)`, `elements[2..]` = children.
    pub fn to_list(&self, span: Span) -> List {
        let mut elements = Vec::with_capacity(self.children.len() + 2);
        elements.push(Expr::Atom(Atom::Tag(self.tag), span));
        elements.push(Expr::Map(self.meta.clone(), span));
        elements.extend(self.children.clone());
        List { elements }
    }

    // === Role-typed accessors ===

    /// Iterator over children at RuntimeExpr positions only.
    pub fn expr_children(&self) -> impl Iterator<Item = &Expr> {
        let tag = self.tag;
        let arity = self.children.len();
        self.children
            .iter()
            .enumerate()
            .filter_map(move |(i, child)| {
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
        self.children
            .iter()
            .enumerate()
            .filter_map(move |(i, child)| {
                if child_stamp_role(tag, i, arity) == ChildStampRole::Binder {
                    match child {
                        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
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
            Expr::Atom(Atom::Name(s), _) => s.as_str(),
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
        self.children
            .iter()
            .enumerate()
            .filter_map(move |(i, child)| {
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
                    Expr::Atom(Atom::Name(s), _) => ChildRef::Binder(s.as_str()),
                    _ => ChildRef::Bypass(child),
                },
                ChildStampRole::Selector => match child {
                    Expr::Atom(Atom::Name(s), _) => ChildRef::Selector(s.as_str()),
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
            meta: Metadata,
            children: Vec<Expr>,
        }

        let shadow = NodeShadow::deserialize(deserializer)?;
        Node::try_new(shadow.tag, shadow.meta, shadow.children).map_err(serde::de::Error::custom)
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
        Expr::Atom(Atom::Name(s.to_string()), sp())
    }

    fn int(n: i64) -> Expr {
        Expr::Atom(Atom::Int(n), sp())
    }

    #[test]
    fn try_new_rejects_name_at_runtime_expr_slot() {
        // App children are all RuntimeExpr. A bare Name there is invalid.
        let result = Node::try_new(DeepTag::App, Metadata::default(), vec![name("x")]);
        assert!(matches!(result, Err(NodeError::NameAtExprSlot { .. })));
    }

    #[test]
    fn try_new_permits_name_at_binder_slot() {
        // Var child 0 is Syntax (name is content). Def child 0 is Binder.
        let result = Node::try_new(DeepTag::Var, Metadata::default(), vec![name("x")]);
        assert!(result.is_ok());
    }

    #[test]
    fn try_new_permits_name_at_type_slot() {
        // TVar child 0 is Type. A Name there is a type variable.
        let result = Node::try_new(DeepTag::TVar, Metadata::default(), vec![name("a")]);
        assert!(result.is_ok());
    }

    #[test]
    fn try_new_rejects_wrong_arity() {
        // Var requires exactly 1 child.
        let result = Node::try_new(
            DeepTag::Var,
            Metadata::default(),
            vec![name("x"), name("y")],
        );
        assert!(matches!(result, Err(NodeError::ArityViolation { .. })));
    }

    #[test]
    fn try_new_accepts_correct_arity() {
        let result = Node::try_new(
            DeepTag::If,
            Metadata::default(),
            vec![int(1), int(2), int(3)],
        );
        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "compiler bug")]
    fn new_panics_on_violation() {
        Node::new(DeepTag::App, Metadata::default(), vec![name("x")]);
    }

    #[test]
    fn expr_child_returns_runtime_expr_children() {
        let node = Node::new(
            DeepTag::App,
            Metadata::default(),
            vec![int(1), int(2), int(3)],
        );
        assert_eq!(node.expr_children().count(), 3);
    }

    #[test]
    fn binder_name_works() {
        let node = Node::new(DeepTag::Def, Metadata::default(), vec![name("f"), int(42)]);
        assert_eq!(node.binder_name(0), "f");
    }

    #[test]
    #[should_panic(expected = "not RuntimeExpr")]
    fn expr_child_panics_on_role_mismatch() {
        let node = Node::new(DeepTag::Def, Metadata::default(), vec![name("f"), int(42)]);
        // Index 0 of Def is Binder, not RuntimeExpr.
        let _ = node.expr_child(0);
    }

    /// chelis#1109: the construction scan stops at an already-stamped
    /// `Expr::Node`. This pins both halves of that induction.
    ///
    /// The smuggling node below is fabricated through the private fields,
    /// which only this module can reach — `try_new`, `new`, `Deserialize`,
    /// and the three `try_replace_*` mutators all route through `validate`,
    /// so no caller can produce a real `Node` carrying a raw vocabulary tag.
    /// The permanent boundary oracle `find_raw_vocabulary_tag` still
    /// descends into Node subtrees, so if a future edit ever did produce
    /// one, the lowering-boundary assertion still catches it.
    #[test]
    fn construction_scan_stops_at_a_stamped_node_boundary() {
        let raw = Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Name("lit".to_string()), sp()),
                    Expr::Map(Metadata::default(), sp()),
                    int(0),
                ],
            },
            sp(),
        );

        // Unreachable outside this module: the fields are private.
        let smuggled = Expr::Node(
            Box::new(Node {
                tag: DeepTag::Var,
                meta: Metadata::default(),
                children: vec![raw.clone()],
            }),
            sp(),
        );

        assert_eq!(
            crate::validate::find_raw_vocabulary_tag(std::slice::from_ref(&smuggled)).as_deref(),
            Some("lit"),
            "the permanent boundary oracle must still descend into Node subtrees"
        );
        assert!(
            Node::try_new(DeepTag::App, Metadata::default(), vec![smuggled]).is_ok(),
            "construction must trust an already-validated Node child instead of \
             re-walking it once per ancestor"
        );

        // The same raw form under an unvalidated carrier is still rejected.
        assert!(matches!(
            Node::try_new(DeepTag::App, Metadata::default(), vec![raw]),
            Err(NodeError::RawVocabularyTag { .. })
        ));
    }

    #[test]
    fn children_iter_yields_correct_roles() {
        let node = Node::new(DeepTag::Def, Metadata::default(), vec![name("f"), int(42)]);
        let refs: Vec<_> = node.children_iter().collect();
        assert!(matches!(refs[0], ChildRef::Binder("f")));
        assert!(matches!(refs[1], ChildRef::Expr(_)));
    }
}
