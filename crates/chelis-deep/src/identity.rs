//! Explicit identity of decoded Deep nodes.
//!
//! The type checker keys facts about individual nodes: the owner stamps that
//! inference writes and annotation reads, and the deferred application
//! ledger. Two structurally equal subtrees at different positions are
//! different nodes for those facts, so the key cannot be the node's content.
//! It is a value the checker assigns to its own copy of the program before it
//! infers anything, carried in the node's [`Metadata`]:
//!
//! - every decoded node in a structural position gets a distinct identity;
//! - a clone keeps the identity of the node it copies, so a transient view of
//!   a node (the `Node::to_list` bridge, a suspended call the ledger replays)
//!   names the node it was taken from;
//! - a rebuilt node starts without one, like any node the checker synthesizes;
//! - equality, serialization and `Debug` ignore it, so no observable content
//!   of a tree depends on whether it was numbered.
//!
//! An identity means something only to the checker entry that assigned it.
//! Every entry numbers its own copy from zero, and reassigning overwrites
//! whatever an earlier entry left on the nodes it copied.

use crate::annotations::Metadata;
use crate::ast::{Atom, Expr, List};

/// The checker-assigned identity of one decoded Deep node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIdentity(usize);

impl NodeIdentity {
    /// Preorder position of the node in the numbered program.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl Expr {
    /// The identity of this expression when it is a decoded node.
    ///
    /// Only a decoded node (either carrier) has one. Atoms, maps, structural
    /// lists, legacy metadata wrappers and undecodable heads answer `None`.
    pub fn node_identity(&self) -> Option<NodeIdentity> {
        match self.carrier() {
            crate::ExprCarrier::DecodedNode(_, metadata, _) => metadata.node_identity(),
            crate::ExprCarrier::StructuralList(_)
            | crate::ExprCarrier::UndecodableHead(_, _, _)
            | crate::ExprCarrier::Atom(_)
            | crate::ExprCarrier::MetadataMap(_)
            | crate::ExprCarrier::MetadataExpression(_)
            | crate::ExprCarrier::MalformedLegacyList(_) => None,
        }
    }
}

impl List {
    /// The identity of the decoded node this list carries, read exactly as
    /// [`Expr::node_identity`] reads an `Expr::List`.
    pub fn node_identity(&self) -> Option<NodeIdentity> {
        match self.elements.as_slice() {
            [Expr::Atom(Atom::Tag(_), _), Expr::Map(metadata, _), ..] => metadata.node_identity(),
            _ => None,
        }
    }
}

/// Give every decoded node in a structural position of `exprs` a distinct
/// identity, numbered in preorder from zero, and return how many were
/// assigned.
///
/// Structural positions are the children of nodes, lists, structural lists
/// and undecodable heads, and the expression a legacy metadata wrapper
/// annotates. Expressions inside metadata payloads are not program nodes the
/// checker keys, and keep whatever they carry.
pub fn assign_node_identities(exprs: &mut [Expr]) -> usize {
    let mut next = 0;
    for expr in exprs {
        assign(expr, &mut next);
    }
    next
}

fn assign(expr: &mut Expr, next: &mut usize) {
    match expr {
        Expr::Node(node, _) => {
            let (metadata, children) = node.identity_parts_mut();
            stamp(metadata, next);
            for child in children {
                assign(child, next);
            }
        }
        // The shape test reads a shared slice and the writes go through
        // indices: the extraction does not support enum patterns, or several
        // bindings, inside a slice pattern on a mutable slice.
        Expr::List(list, _) => {
            if matches!(
                list.elements.as_slice(),
                [Expr::Atom(Atom::Tag(_), _), Expr::Map(_, _), ..]
            ) {
                if let Expr::Map(metadata, _) = &mut list.elements[1] {
                    stamp(metadata, next);
                }
                for child in &mut list.elements[2..] {
                    assign(child, next);
                }
            } else {
                for element in &mut list.elements {
                    assign(element, next);
                }
            }
        }
        Expr::BareList(children, _) => {
            for child in children {
                assign(child, next);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &mut data.children {
                assign(child, next);
            }
        }
        Expr::MetaExpr(wrapper, _) => assign(&mut wrapper.expr, next),
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
    }
}

fn stamp(metadata: &mut Metadata, next: &mut usize) {
    metadata.set_node_identity(NodeIdentity(*next));
    *next += 1;
}
