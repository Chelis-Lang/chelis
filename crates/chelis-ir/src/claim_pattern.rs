//! Extent claims on tensors reached through aggregate and nominal types
//! (spec/design/runtime_extents.md C6.5).
//!
//! A claim source (a formal, a declared result, a local ascription) states
//! its obligations through its authored type. A tensor nested in a tuple, a
//! `List`, an `Option` or a nominal application keeps its literal and named
//! axes as obligations, and a nominal dimension argument becomes one by
//! substitution into the declared field types: under `Box[n] = Box { v:
//! tensor[n, f32] }`, the authored `Box[3]` claims axis 0 of `v`.
//!
//! [`ClaimPattern::derive`] walks the authored type once. Nominal
//! applications are memoized by name and substituted argument tuple, so a
//! recursive declaration whose argument tuples close becomes a finite graph.
//! A declaration whose argument tuples never close (polymorphic recursion
//! through an argument that grows) has no finite pattern; deriving through
//! it is the typed refusal [`ClaimPatternError::NonRegularRecursion`], never
//! a skipped claim.

use chelis_deep::ast::{Atom, Expr, Metadata};
use chelis_deep::{DeepTag, ExprCarrier};
use chelis_types::adt::{AdtDef, AdtRegistry};
use chelis_types::infer::type_to_deep_expr;
use chelis_types::types::{Dim, NominalArg, TensorPrec, Type};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One node of a [`ClaimPattern`] graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimNodeId(usize);

impl ClaimNodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Where a claimed axis sits in a tensor whose rank may contain a spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimAxisPosition {
    /// The axis counted from the first.
    Front(usize),
    /// The axis counted from the last, after a rank spread.
    Back(usize),
}

impl ClaimAxisPosition {
    /// The concrete axis of a value of rank `rank`, if it has one.
    pub fn resolve(self, rank: usize) -> Option<usize> {
        match self {
            Self::Front(axis) => (axis < rank).then_some(axis),
            Self::Back(axis) => rank.checked_sub(axis + 1),
        }
    }
}

/// What one claimed axis requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimDim {
    Literal(i64),
    /// A dimension binder of the claim source's signature, by its authored
    /// spelling. The spelling is scoped to that one signature.
    Binder(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimAxis {
    pub position: ClaimAxisPosition,
    pub claim: ClaimDim,
}

/// A tensor position with at least one obligation.
#[derive(Debug, Clone)]
pub struct ClaimTensor {
    /// The authored tensor type after nominal substitution.
    pub ty: Expr,
    /// The declared rank, or `None` when a rank spread makes it a minimum.
    pub rank: Option<usize>,
    /// The fixed axes on either side of a spread.
    pub min_rank: usize,
    pub axes: Vec<ClaimAxis>,
}

#[derive(Debug, Clone)]
pub struct ClaimField {
    pub name: Option<String>,
    pub node: Option<ClaimNodeId>,
}

#[derive(Debug, Clone)]
pub struct ClaimConstructor {
    /// The constructor's identity in the checked program.
    pub name: String,
    /// The spelling a compiled value stores as its tag.
    pub stored_name: String,
    /// Every declared field, in declared order; `node` is `None` for a field
    /// that owes nothing.
    pub fields: Vec<ClaimField>,
}

#[derive(Debug, Clone)]
pub enum ClaimNode {
    Tensor(ClaimTensor),
    /// Every component by position; `None` owes nothing.
    Tuple(Vec<Option<ClaimNodeId>>),
    /// Every element.
    List(ClaimNodeId),
    /// The `Some` payload.
    Option(ClaimNodeId),
    /// Every constructor of a nominal application, in declared order. Only
    /// the constructor a value carries is walked.
    Nominal {
        name: String,
        constructors: Vec<ClaimConstructor>,
    },
}

/// The obligations of one authored type, as a finite graph.
#[derive(Debug, Clone, Default)]
pub struct ClaimPattern {
    nodes: Vec<ClaimNode>,
    root: Option<ClaimNodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimPatternError {
    /// The claim passes through a nominal declaration whose argument tuples
    /// never close.
    NonRegularRecursion {
        nominal: String,
    },
    /// A nominal application that may carry a claimed tensor has no
    /// declaration the derivation can substitute into.
    UnresolvedNominal {
        nominal: String,
    },
    Malformed(String),
}

impl fmt::Display for ClaimPatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonRegularRecursion { nominal } => write!(
                f,
                "an extent claim passes through `{nominal}`, whose recursive type arguments \
                 never repeat, so the claim has no finite pattern to check"
            ),
            Self::UnresolvedNominal { nominal } => write!(
                f,
                "an extent claim passes through `{nominal}`, which has no declaration to \
                 instantiate"
            ),
            Self::Malformed(detail) => write!(f, "malformed claimed type: {detail}"),
        }
    }
}

impl std::error::Error for ClaimPatternError {}

impl ClaimPattern {
    /// The pattern of `authored`, an alias-normalized authored type.
    pub fn derive(authored: &Expr, registry: &AdtRegistry) -> Result<Self, ClaimPatternError> {
        let mut builder = Builder {
            registry,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            analysis: std::cell::OnceCell::new(),
        };
        let root = builder.node(authored)?;
        Ok(builder.finish(root))
    }

    /// Whether the claim source owes nothing.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn root(&self) -> Option<ClaimNodeId> {
        self.root
    }

    pub fn node(&self, id: ClaimNodeId) -> &ClaimNode {
        &self.nodes[id.0]
    }

    pub fn nodes(&self) -> &[ClaimNode] {
        &self.nodes
    }

    /// The root obligation when the claimed type is not itself a tensor: the
    /// part of the claim that existing tensor-level carriers cannot hold.
    pub fn nested_root(&self) -> Option<ClaimNodeId> {
        self.root
            .filter(|root| !matches!(self.node(*root), ClaimNode::Tensor(_)))
    }

    /// Every binder spelling the pattern names, in first-reach order.
    pub fn binders(&self) -> Vec<String> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if let ClaimNode::Tensor(tensor) = node {
                for axis in &tensor.axes {
                    if let ClaimDim::Binder(name) = &axis.claim
                        && !out.contains(name)
                    {
                        out.push(name.clone());
                    }
                }
            }
        }
        out
    }

    /// The node a value's component reaches: a tuple component, a list
    /// element, an option payload or a constructor field.
    pub fn child(&self, id: ClaimNodeId, step: ClaimStep<'_>) -> Option<ClaimNodeId> {
        match (self.node(id), step) {
            (ClaimNode::Tuple(items), ClaimStep::Component(index)) => {
                items.get(index).copied().flatten()
            }
            (ClaimNode::List(item), ClaimStep::Element) => Some(*item),
            (ClaimNode::Option(item), ClaimStep::Some) => Some(*item),
            (ClaimNode::Nominal { constructors, .. }, ClaimStep::Field { ctor, field }) => {
                constructors
                    .iter()
                    .find(|constructor| constructor.name == ctor || constructor.stored_name == ctor)
                    .and_then(|constructor| constructor.fields.get(field))
                    .and_then(|field| field.node)
            }
            (ClaimNode::Nominal { constructors, .. }, ClaimStep::NamedField { ctor, field }) => {
                constructors
                    .iter()
                    .find(|constructor| constructor.name == ctor || constructor.stored_name == ctor)
                    .and_then(|constructor| {
                        constructor
                            .fields
                            .iter()
                            .find(|candidate| candidate.name.as_deref() == Some(field))
                    })
                    .and_then(|field| field.node)
            }
            _ => None,
        }
    }
}

/// One structural step from a value to a component it holds.
#[derive(Debug, Clone, Copy)]
pub enum ClaimStep<'a> {
    Component(usize),
    Element,
    Some,
    Field { ctor: &'a str, field: usize },
    NamedField { ctor: &'a str, field: &'a str },
}

struct Builder<'a> {
    registry: &'a AdtRegistry,
    nodes: Vec<Option<ClaimNode>>,
    memo: BTreeMap<String, ClaimNodeId>,
    /// The non-regular and tensor-carrying declarations, computed only when
    /// the walk first meets a nominal application.
    analysis: std::cell::OnceCell<(BTreeSet<String>, BTreeSet<String>)>,
}

impl Builder<'_> {
    fn non_regular(&self) -> &BTreeSet<String> {
        &self.analysis().0
    }

    fn carrying(&self) -> &BTreeSet<String> {
        &self.analysis().1
    }

    fn analysis(&self) -> &(BTreeSet<String>, BTreeSet<String>) {
        self.analysis.get_or_init(|| {
            (
                non_regular_declarations(self.registry),
                carrying_declarations(self.registry),
            )
        })
    }

    fn push(&mut self, node: ClaimNode) -> ClaimNodeId {
        self.nodes.push(Some(node));
        ClaimNodeId(self.nodes.len() - 1)
    }

    fn node(&mut self, ty: &Expr) -> Result<Option<ClaimNodeId>, ClaimPatternError> {
        let Some((tag, children)) = type_parts(ty) else {
            return Ok(None);
        };
        match tag {
            DeepTag::TRef => match children.first() {
                Some(inner) => self.node(inner),
                None => Ok(None),
            },
            DeepTag::TTensor => {
                Ok(tensor_claim(ty, children)?.map(|t| self.push(ClaimNode::Tensor(t))))
            }
            DeepTag::TTuple => {
                let items = children
                    .iter()
                    .map(|item| self.node(item))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(items
                    .iter()
                    .any(Option::is_some)
                    .then(|| self.push(ClaimNode::Tuple(items))))
            }
            DeepTag::TAdt => {
                let (name, args) = children
                    .split_first()
                    .ok_or_else(|| ClaimPatternError::Malformed("t-adt has no name".into()))?;
                let name = symbol(name)
                    .ok_or_else(|| ClaimPatternError::Malformed("t-adt name".into()))?;
                match (name, args) {
                    ("List", [item]) => Ok(self
                        .node(item)?
                        .map(|item| self.push(ClaimNode::List(item)))),
                    ("Option", [item]) => Ok(self
                        .node(item)?
                        .map(|item| self.push(ClaimNode::Option(item)))),
                    // A dictionary's values are not a constructor field; the
                    // walk keeps its established owner.
                    ("Dict" | "MappedFile", _) => Ok(None),
                    _ => self.nominal(name, args),
                }
            }
            _ => Ok(None),
        }
    }

    fn nominal(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Option<ClaimNodeId>, ClaimPatternError> {
        let key = format!(
            "{name}[{}]",
            args.iter().map(canonical).collect::<Vec<_>>().join(",")
        );
        if let Some(id) = self.memo.get(&key) {
            return Ok(Some(*id));
        }
        let Some(definition) = self.registry.lookup(name) else {
            return if args.iter().any(|arg| self.may_carry(arg)) {
                Err(ClaimPatternError::UnresolvedNominal {
                    nominal: name.to_string(),
                })
            } else {
                Ok(None)
            };
        };
        if !self.carrying().contains(name) && !args.iter().any(|arg| self.may_carry(arg)) {
            return Ok(None);
        }
        if self.non_regular().contains(name) {
            return Err(ClaimPatternError::NonRegularRecursion {
                nominal: name.to_string(),
            });
        }
        let substitution = parameter_substitution(definition, args)?;
        self.nodes.push(None);
        let id = ClaimNodeId(self.nodes.len() - 1);
        self.memo.insert(key, id);
        let mut constructors = Vec::with_capacity(definition.variants.len());
        for variant in &definition.variants {
            let mut fields = Vec::with_capacity(variant.fields.len());
            for (field_name, field_ty) in &variant.fields {
                let field_ty = substitute(&type_to_deep_expr(field_ty), &substitution);
                fields.push(ClaimField {
                    name: field_name.clone(),
                    node: self.node(&field_ty)?,
                });
            }
            constructors.push(ClaimConstructor {
                stored_name: chelis_types::linked_constructor_source_name(name, &variant.name)
                    .unwrap_or(&variant.name)
                    .to_string(),
                name: variant.name.clone(),
                fields,
            });
        }
        self.nodes[id.0] = Some(ClaimNode::Nominal {
            name: name.to_string(),
            constructors,
        });
        Ok(Some(id))
    }

    /// Whether a type argument may hold a tensor.
    fn may_carry(&self, ty: &Expr) -> bool {
        let Some((tag, children)) = type_parts(ty) else {
            return false;
        };
        match tag {
            DeepTag::TTensor => true,
            DeepTag::TAdt => {
                let own = children
                    .first()
                    .and_then(symbol)
                    .is_some_and(|name| self.carrying().contains(name));
                own || children.iter().skip(1).any(|arg| self.may_carry(arg))
            }
            _ => children.iter().any(|child| self.may_carry(child)),
        }
    }

    /// Drop every node that owes nothing, including a cycle that reaches no
    /// tensor, and renumber the rest.
    fn finish(self, root: Option<ClaimNodeId>) -> ClaimPattern {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| node.expect("every reserved claim node is filled"))
            .collect::<Vec<_>>();
        let children = |node: &ClaimNode| -> Vec<ClaimNodeId> {
            match node {
                ClaimNode::Tensor(_) => Vec::new(),
                ClaimNode::Tuple(items) => items.iter().flatten().copied().collect(),
                ClaimNode::List(item) | ClaimNode::Option(item) => vec![*item],
                ClaimNode::Nominal { constructors, .. } => constructors
                    .iter()
                    .flat_map(|constructor| constructor.fields.iter().filter_map(|f| f.node))
                    .collect(),
            }
        };
        let mut live = nodes
            .iter()
            .map(|node| matches!(node, ClaimNode::Tensor(_)))
            .collect::<Vec<_>>();
        loop {
            let mut changed = false;
            for (index, node) in nodes.iter().enumerate() {
                if !live[index] && children(node).iter().any(|child| live[child.0]) {
                    live[index] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut renumber = vec![None; nodes.len()];
        let mut next = 0;
        for (index, alive) in live.iter().enumerate() {
            if *alive {
                renumber[index] = Some(ClaimNodeId(next));
                next += 1;
            }
        }
        let map = |id: ClaimNodeId| renumber[id.0];
        let kept = nodes
            .into_iter()
            .zip(&live)
            .filter(|(_, alive)| **alive)
            .map(|(node, _)| match node {
                ClaimNode::Tensor(tensor) => ClaimNode::Tensor(tensor),
                ClaimNode::Tuple(items) => {
                    ClaimNode::Tuple(items.into_iter().map(|item| item.and_then(map)).collect())
                }
                ClaimNode::List(item) => ClaimNode::List(map(item).expect("live child")),
                ClaimNode::Option(item) => ClaimNode::Option(map(item).expect("live child")),
                ClaimNode::Nominal { name, constructors } => ClaimNode::Nominal {
                    name,
                    constructors: constructors
                        .into_iter()
                        .map(|constructor| ClaimConstructor {
                            fields: constructor
                                .fields
                                .into_iter()
                                .map(|field| ClaimField {
                                    name: field.name,
                                    node: field.node.and_then(map),
                                })
                                .collect(),
                            ..constructor
                        })
                        .collect(),
                },
            })
            .collect();
        ClaimPattern {
            nodes: kept,
            root: root.and_then(map),
        }
    }
}

fn type_parts(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => Some((tag, children)),
        ExprCarrier::MetadataExpression(meta) => type_parts(&meta.expr),
        _ => None,
    }
}

fn symbol(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name),
        Expr::MetaExpr(meta, _) => symbol(&meta.expr),
        _ => None,
    }
}

/// The obligations of one tensor type, or `None` when it claims nothing.
fn tensor_claim(ty: &Expr, children: &[Expr]) -> Result<Option<ClaimTensor>, ClaimPatternError> {
    let Some((_, dims)) = children.split_last() else {
        return Err(ClaimPatternError::Malformed(
            "t-tensor has no precision".into(),
        ));
    };
    let spread = dims
        .iter()
        .position(|dim| type_parts(dim).is_some_and(|(tag, _)| tag == DeepTag::DRank));
    let fixed = dims
        .iter()
        .filter(|dim| type_parts(dim).is_none_or(|(tag, _)| tag != DeepTag::DRank))
        .count();
    let mut axes = Vec::new();
    for (index, dim) in dims.iter().enumerate() {
        let Some((tag, kids)) = type_parts(dim) else {
            continue;
        };
        let claim = match tag {
            DeepTag::DLit => match kids.first() {
                Some(Expr::Atom(Atom::Int(value), _)) => ClaimDim::Literal(*value),
                _ => return Err(ClaimPatternError::Malformed("d-lit has no integer".into())),
            },
            DeepTag::DName | DeepTag::DVar => match kids.first().and_then(symbol) {
                Some("*") | Some("_") | None => continue,
                Some(name) => ClaimDim::Binder(name.to_string()),
            },
            _ => continue,
        };
        let position = match spread {
            Some(at) if index > at => ClaimAxisPosition::Back(dims.len() - 1 - index),
            _ => ClaimAxisPosition::Front(index),
        };
        axes.push(ClaimAxis { position, claim });
    }
    Ok((!axes.is_empty()).then(|| ClaimTensor {
        ty: ty.clone(),
        rank: spread.is_none().then_some(fixed),
        min_rank: fixed,
        axes,
    }))
}

/// The deterministic spelling of a substituted type argument, for the memo.
fn canonical(expr: &Expr) -> String {
    match type_parts(expr) {
        Some((tag, children)) => format!(
            "({} {})",
            tag.as_str(),
            children.iter().map(canonical).collect::<Vec<_>>().join(" ")
        ),
        None => match expr {
            Expr::Atom(Atom::Name(name), _) => name.clone(),
            Expr::Atom(Atom::Int(value), _) => value.to_string(),
            Expr::Atom(atom, _) => format!("{atom:?}"),
            other => format!("{other:?}"),
        },
    }
}

/// The declared field types name a nominal's parameters by the checker's
/// variables (`t7`, `d3`); map each to the application's argument.
fn parameter_substitution(
    definition: &AdtDef,
    args: &[Expr],
) -> Result<BTreeMap<String, Expr>, ClaimPatternError> {
    if definition.param_args.len() != args.len() {
        return Err(ClaimPatternError::Malformed(format!(
            "`{}` applied to {} arguments, declared with {}",
            definition.name,
            args.len(),
            definition.param_args.len()
        )));
    }
    let mut out = BTreeMap::new();
    for (parameter, arg) in definition.param_args.iter().zip(args) {
        let (key, arg) = match parameter {
            NominalArg::Type(Type::Var(var)) => (format!("t{}", var.0), arg.clone()),
            // Surf spells a binder argument `Box[k]` as a type variable; in
            // a dimension slot it names the dimension binder `k`.
            NominalArg::Dimension(Dim::Var(var)) => {
                (format!("d{}", var.0), dimension_argument(arg))
            }
            _ => {
                return Err(ClaimPatternError::Malformed(format!(
                    "`{}` has a parameter without a declaration variable",
                    definition.name
                )));
            }
        };
        out.insert(key, arg);
    }
    Ok(out)
}

fn dimension_argument(arg: &Expr) -> Expr {
    match type_parts(arg) {
        Some((DeepTag::TVar, children)) => Expr::node(
            DeepTag::DVar,
            Metadata::default(),
            children.to_vec(),
            arg.span(),
        ),
        _ => arg.clone(),
    }
}

fn substitute(ty: &Expr, substitution: &BTreeMap<String, Expr>) -> Expr {
    match ty.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => {
            if matches!(tag, DeepTag::TVar | DeepTag::DVar)
                && let Some(replacement) = children
                    .first()
                    .and_then(symbol)
                    .and_then(|name| substitution.get(name))
            {
                return replacement.clone();
            }
            Expr::node(
                tag,
                Metadata::default(),
                children
                    .iter()
                    .map(|child| substitute(child, substitution))
                    .collect(),
                ty.span(),
            )
        }
        ExprCarrier::MetadataExpression(meta) => substitute(&meta.expr, substitution),
        _ => ty.clone(),
    }
}

/// The declaration variable of each parameter, aligned with `param_args`.
enum Param {
    Type(chelis_types::types::TypeVar),
    Dim(chelis_types::types::DimVar),
}

fn params(definition: &AdtDef) -> Vec<Option<Param>> {
    definition
        .param_args
        .iter()
        .map(|argument| match argument {
            NominalArg::Type(Type::Var(var)) => Some(Param::Type(*var)),
            NominalArg::Dimension(Dim::Var(var)) => Some(Param::Dim(*var)),
            _ => None,
        })
        .collect()
}

fn mentions(ty: &Type, param: &Param) -> bool {
    match ty {
        Type::Var(var) => matches!(param, Param::Type(p) if p == var),
        Type::Tensor(dims, prec) => {
            matches!((param, prec), (Param::Type(p), TensorPrec::Var(v)) if p == v)
                || dims
                    .iter()
                    .any(|dim| matches!((param, dim), (Param::Dim(p), Dim::Var(v)) if p == v))
        }
        Type::Fn(args, ret) => args.iter().any(|arg| mentions(arg, param)) || mentions(ret, param),
        Type::Ref(inner) => mentions(inner, param),
        Type::Tuple(items) | Type::Adt(_, items) => items.iter().any(|item| mentions(item, param)),
        Type::KindedAdt(_, args) => args.iter().any(|arg| match arg {
            NominalArg::Type(ty) => mentions(ty, param),
            NominalArg::Dimension(Dim::Var(v)) => matches!(param, Param::Dim(p) if p == v),
            NominalArg::Dimension(_) => false,
        }),
        Type::Prim(_) | Type::Unit | Type::Error(_) => false,
    }
}

fn is_exactly(arg: &NominalArg, param: &Param) -> bool {
    match (arg, param) {
        (NominalArg::Type(Type::Var(v)), Param::Type(p)) => v == p,
        (NominalArg::Dimension(Dim::Var(v)), Param::Dim(p)) => v == p,
        _ => false,
    }
}

/// Every nominal application inside `ty`, with its arguments.
fn applications<'t>(ty: &'t Type, out: &mut Vec<(&'t str, Vec<NominalArg>)>) {
    match ty {
        Type::Adt(name, args) => {
            out.push((name, args.iter().cloned().map(NominalArg::Type).collect()));
            for arg in args {
                applications(arg, out);
            }
        }
        Type::KindedAdt(name, args) => {
            out.push((name, args.clone()));
            for arg in args {
                if let NominalArg::Type(arg) = arg {
                    applications(arg, out);
                }
            }
        }
        Type::Fn(args, ret) => {
            for arg in args {
                applications(arg, out);
            }
            applications(ret, out);
        }
        Type::Ref(inner) => applications(inner, out),
        Type::Tuple(items) => {
            for item in items {
                applications(item, out);
            }
        }
        Type::Tensor(_, _) | Type::Var(_) | Type::Prim(_) | Type::Unit | Type::Error(_) => {}
    }
}

/// Declarations whose recursive instantiation never closes.
///
/// The parameter-flow graph has a node per (declaration, parameter) and an
/// edge for every occurrence of a parameter inside a field's nominal
/// argument; the edge is expansive when the parameter sits strictly inside
/// that argument. The set of instantiations reachable from any application
/// is finite exactly when no cycle contains an expansive edge (Kennedy and
/// Pierce, "On Decidability of Nominal Subtyping with Variance").
fn non_regular_declarations(registry: &AdtRegistry) -> BTreeSet<String> {
    type Vertex = (String, usize);
    let mut edges: Vec<(Vertex, Vertex, bool)> = Vec::new();
    for definition in registry.defs.values() {
        let own = params(definition);
        for variant in &definition.variants {
            for (_, field) in &variant.fields {
                let mut found = Vec::new();
                applications(field, &mut found);
                for (target, args) in found {
                    for (to, arg) in args.iter().enumerate() {
                        for (from, param) in own.iter().enumerate() {
                            let Some(param) = param else { continue };
                            let occurs = match arg {
                                NominalArg::Type(ty) => mentions(ty, param),
                                NominalArg::Dimension(Dim::Var(v)) => {
                                    matches!(param, Param::Dim(p) if p == v)
                                }
                                NominalArg::Dimension(_) => false,
                            };
                            if occurs {
                                edges.push((
                                    (definition.name.clone(), from),
                                    (target.to_string(), to),
                                    !is_exactly(arg, param),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    let reaches = |start: &Vertex, goal: &Vertex| {
        let mut seen = BTreeSet::new();
        let mut stack = vec![start.clone()];
        while let Some(vertex) = stack.pop() {
            if &vertex == goal {
                return true;
            }
            if seen.insert(vertex.clone()) {
                stack.extend(
                    edges
                        .iter()
                        .filter(|(from, _, _)| *from == vertex)
                        .map(|(_, to, _)| to.clone()),
                );
            }
        }
        false
    };
    let mut out = BTreeSet::new();
    for (from, to, expansive) in &edges {
        if *expansive && reaches(to, from) {
            out.insert(from.0.clone());
            out.insert(to.0.clone());
        }
    }
    out
}

/// Declarations some field of which holds a tensor whatever their
/// arguments, as the least fixed point over field references. A tensor an
/// argument supplies is found in the argument itself.
fn carrying_declarations(registry: &AdtRegistry) -> BTreeSet<String> {
    fn direct(ty: &Type, carrying: &BTreeSet<String>) -> bool {
        match ty {
            Type::Tensor(_, _) => true,
            Type::Adt(name, args) => {
                carrying.contains(name) || args.iter().any(|arg| direct(arg, carrying))
            }
            Type::KindedAdt(name, args) => {
                carrying.contains(name)
                    || args
                        .iter()
                        .any(|arg| arg.as_type().is_some_and(|arg| direct(arg, carrying)))
            }
            Type::Ref(inner) => direct(inner, carrying),
            Type::Tuple(items) => items.iter().any(|item| direct(item, carrying)),
            Type::Fn(_, _) | Type::Var(_) | Type::Prim(_) | Type::Unit | Type::Error(_) => false,
        }
    }
    let mut carrying = BTreeSet::new();
    loop {
        let before = carrying.len();
        for definition in registry.defs.values() {
            if !carrying.contains(&definition.name)
                && definition.variants.iter().any(|variant| {
                    variant
                        .fields
                        .iter()
                        .any(|(_, field)| direct(field, &carrying))
                })
            {
                carrying.insert(definition.name.clone());
            }
        }
        if carrying.len() == before {
            return carrying;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(source: &str) -> AdtRegistry {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        let deep = chelis_surf::desugar::desugar_program(&decls).expect("desugar");
        chelis_types::check_ir_program(&deep)
            .unwrap_or_else(|result| panic!("check failed: {:?}", result.errors))
            .adt_registry()
            .clone()
    }

    fn ty(text: &str) -> Expr {
        chelis_deep::parser::parse_str(text)
            .expect("type syntax")
            .remove(0)
    }

    fn tensor(pattern: &ClaimPattern, id: ClaimNodeId) -> &ClaimTensor {
        match pattern.node(id) {
            ClaimNode::Tensor(tensor) => tensor,
            other => panic!("expected a tensor node, got {other:?}"),
        }
    }

    fn field(pattern: &ClaimPattern, id: ClaimNodeId, ctor: &str, index: usize) -> ClaimNodeId {
        pattern
            .child(id, ClaimStep::Field { ctor, field: index })
            .unwrap_or_else(|| panic!("{ctor}.{index} owes nothing"))
    }

    const BOX: &str = "type Box[n] =\n  | Box { v: tensor[n, f32], w: i64 }\n";

    #[test]
    fn nominal_dimension_argument_becomes_the_field_claim() {
        let registry = registry(BOX);
        for (arg, claim) in [
            ("(d-lit {} 3)", ClaimDim::Literal(3)),
            ("(d-var {} k)", ClaimDim::Binder("k".into())),
            ("(t-var {} k)", ClaimDim::Binder("k".into())),
        ] {
            let pattern =
                ClaimPattern::derive(&ty(&format!("(t-adt {{}} Box {arg})")), &registry).unwrap();
            let root = pattern.nested_root().expect("Box owes its field claim");
            let v = field(&pattern, root, "Box", 0);
            assert_eq!(
                tensor(&pattern, v).axes,
                vec![ClaimAxis {
                    position: ClaimAxisPosition::Front(0),
                    claim,
                }]
            );
            assert!(
                pattern
                    .child(
                        root,
                        ClaimStep::Field {
                            ctor: "Box",
                            field: 1
                        }
                    )
                    .is_none()
            );
        }
        let wildcard =
            ClaimPattern::derive(&ty("(t-adt {} Box (d-name {} *))"), &registry).unwrap();
        assert!(wildcard.is_empty(), "a wildcard argument claims nothing");
    }

    #[test]
    fn tuple_option_and_list_positions_keep_their_claims() {
        let registry = registry(BOX);
        let pattern = ClaimPattern::derive(
            &ty("(t-tuple {} (t-prim {} i64) (t-adt {} Option (t-adt {} List (t-tensor {} (d-lit {} 2) (d-name {} *) (t-prim {} f32)))))"),
            &registry,
        )
        .unwrap();
        let root = pattern.root().unwrap();
        assert!(pattern.child(root, ClaimStep::Component(0)).is_none());
        let option = pattern.child(root, ClaimStep::Component(1)).unwrap();
        let list = pattern.child(option, ClaimStep::Some).unwrap();
        let element = pattern.child(list, ClaimStep::Element).unwrap();
        assert_eq!(tensor(&pattern, element).rank, Some(2));
        assert_eq!(tensor(&pattern, element).axes.len(), 1);
    }

    #[test]
    fn literal_field_of_a_nonparametric_record_is_a_claim() {
        let registry = registry("type Holder =\n  | Holder { v: tensor[3, f32] }\n");
        let pattern = ClaimPattern::derive(&ty("(t-adt {} Holder)"), &registry).unwrap();
        let v = field(&pattern, pattern.root().unwrap(), "Holder", 0);
        assert_eq!(tensor(&pattern, v).axes[0].claim, ClaimDim::Literal(3));
    }

    #[test]
    fn closing_recursion_is_a_finite_graph() {
        let registry = registry(
            "type Pair[n] =\n  | Leaf { v: tensor[n, f32] }\n  | Node { l: Pair[n], r: Pair[n] }\n\
             type T[n] =\n  | Tip { v: tensor[n, f32] }\n  | Fix { t: T[3] }\n",
        );
        let pattern = ClaimPattern::derive(&ty("(t-adt {} Pair (d-lit {} 3))"), &registry).unwrap();
        let root = pattern.root().unwrap();
        assert_eq!(field(&pattern, root, "Node", 0), root);
        assert_eq!(field(&pattern, root, "Node", 1), root);
        assert_eq!(pattern.nodes().len(), 2);

        let pattern = ClaimPattern::derive(&ty("(t-adt {} T (d-var {} k))"), &registry).unwrap();
        let root = pattern.root().unwrap();
        let fixed = field(&pattern, root, "Fix", 0);
        assert_ne!(fixed, root);
        assert_eq!(field(&pattern, fixed, "Fix", 0), fixed);
        let tip = field(&pattern, fixed, "Tip", 0);
        assert_eq!(tensor(&pattern, tip).axes[0].claim, ClaimDim::Literal(3));
        let own = field(&pattern, root, "Tip", 0);
        assert_eq!(
            tensor(&pattern, own).axes[0].claim,
            ClaimDim::Binder("k".into())
        );
    }

    #[test]
    fn non_closing_recursion_is_a_typed_refusal_only_when_it_can_carry_a_claim() {
        let registry =
            registry("type Nest[a] =\n  | Done { h: a }\n  | More { t: Nest[List[a]] }\n");
        let claimed = ty("(t-adt {} Nest (t-tensor {} (d-lit {} 3) (t-prim {} f32)))");
        assert_eq!(
            ClaimPattern::derive(&claimed, &registry).unwrap_err(),
            ClaimPatternError::NonRegularRecursion {
                nominal: "Nest".into()
            }
        );
        let plain =
            ClaimPattern::derive(&ty("(t-adt {} Nest (t-prim {} i64))"), &registry).unwrap();
        assert!(plain.is_empty());
    }

    #[test]
    fn rank_spread_claims_count_from_the_end() {
        let pattern = ClaimPattern::derive(
            &ty("(t-tensor {} (d-lit {} 2) (d-rank {} r) (d-lit {} 4) (t-prim {} f32))"),
            &AdtRegistry::new(),
        )
        .unwrap();
        let root = tensor(&pattern, pattern.root().unwrap());
        assert_eq!(root.rank, None);
        assert_eq!(root.axes[0].position, ClaimAxisPosition::Front(0));
        assert_eq!(root.axes[1].position, ClaimAxisPosition::Back(0));
        assert_eq!(root.axes[1].position.resolve(5), Some(4));
    }
}
