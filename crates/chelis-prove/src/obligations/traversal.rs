//! One finite graph for proof-type projection and reachability (LU2, #872).
//! Alias and record recursion are edges, not Rust recursion or depth fuses.

use super::{ProducedPosition, children, symbol_text, tag};
use chelis_deep::{DeepTag, Expr};
use chelis_types::types::Type;
use std::collections::{BTreeMap, BTreeSet};

pub(super) const DEFAULT_BUDGET: usize = 100_000;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TraversalError {
    #[error("proof-type traversal exhausted its {limit}-step resource budget (chelis#872)")]
    Exhausted { limit: usize },
    #[error("malformed proof type: {0}")]
    Malformed(&'static str),
}

pub(super) struct Budget {
    remaining: usize,
    limit: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET)
    }
}

impl Budget {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            remaining: limit,
            limit,
        }
    }

    fn step(&mut self) -> Result<(), TraversalError> {
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or(TraversalError::Exhausted { limit: self.limit })?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) enum Source<'a> {
    Deep(&'a Expr),
    Checked(&'a Type),
}

impl Source<'_> {
    fn identity(self) -> (bool, usize) {
        match self {
            Self::Deep(value) => (false, std::ptr::from_ref(value) as usize),
            Self::Checked(value) => (true, std::ptr::from_ref(value) as usize),
        }
    }
}

type Id = usize;

enum Node {
    Pending,
    Leaf,
    Link(Id),
    Adt {
        name: String,
        args: Vec<Id>,
        fields: Vec<Id>,
    },
    Tuple(Vec<Id>),
    Function(Vec<Id>, Id),
    Ref(Id),
}

pub(super) struct ProofTypes<'a> {
    nodes: Vec<Node>,
    identities: BTreeMap<(bool, usize), Id>,
    pending: Vec<(Id, Source<'a>)>,
    aliases: BTreeMap<&'a str, &'a Expr>,
    records: BTreeMap<&'a str, Vec<Source<'a>>>,
    pub(super) budget: Budget,
}

impl<'a> ProofTypes<'a> {
    pub(super) fn new(
        aliases: BTreeMap<&'a str, &'a Expr>,
        records: BTreeMap<&'a str, Vec<Source<'a>>>,
        budget: Budget,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            identities: BTreeMap::new(),
            pending: Vec::new(),
            aliases,
            records,
            budget,
        }
    }

    fn intern(&mut self, source: Source<'a>) -> Result<Id, TraversalError> {
        if let Some(id) = self.identities.get(&source.identity()) {
            return Ok(*id);
        }
        self.budget.step()?;
        let id = self.nodes.len();
        self.identities.insert(source.identity(), id);
        self.nodes.push(Node::Pending);
        self.pending.push((id, source));
        Ok(id)
    }

    fn intern_all(
        &mut self,
        sources: impl IntoIterator<Item = Source<'a>>,
    ) -> Result<Vec<Id>, TraversalError> {
        sources
            .into_iter()
            .map(|source| self.intern(source))
            .collect()
    }

    fn adt(&mut self, name: &str, args: Vec<Id>) -> Result<Node, TraversalError> {
        if args.is_empty()
            && let Some(target) = self.aliases.get(name)
        {
            return Ok(Node::Link(self.intern(Source::Deep(target))?));
        }
        let sources = self.records.get(name).cloned().unwrap_or_default();
        Ok(Node::Adt {
            name: name.to_string(),
            args,
            fields: self.intern_all(sources)?,
        })
    }

    pub(super) fn project(&mut self, source: Source<'a>) -> Result<Id, TraversalError> {
        let root = self.intern(source)?;
        while let Some((id, source)) = self.pending.pop() {
            let node = match source {
                Source::Checked(ty) => match ty {
                    Type::Adt(name, args) => {
                        let args = self.intern_all(args.iter().map(Source::Checked))?;
                        self.adt(name, args)?
                    }
                    Type::KindedAdt(name, args) => {
                        let args = self.intern_all(args.iter().filter_map(|arg| match arg {
                            chelis_types::types::NominalArg::Type(ty) => Some(Source::Checked(ty)),
                            chelis_types::types::NominalArg::Dimension(_) => None,
                        }))?;
                        self.adt(name, args)?
                    }
                    Type::Tuple(items) => {
                        Node::Tuple(self.intern_all(items.iter().map(Source::Checked))?)
                    }
                    Type::Fn(args, ret) => Node::Function(
                        self.intern_all(args.iter().map(Source::Checked))?,
                        self.intern(Source::Checked(ret))?,
                    ),
                    Type::Ref(inner) => Node::Ref(self.intern(Source::Checked(inner))?),
                    // Numeric tensors, scalar primitives and type variables cannot
                    // contain nominal opaque values. This is semantic, not a fuse.
                    Type::Prim(_)
                    | Type::Tensor(_, _)
                    | Type::Var(_)
                    | Type::Unit
                    | Type::Error(_) => Node::Leaf,
                },
                Source::Deep(ty) => {
                    let kids = children(ty);
                    match tag(ty) {
                        Some(DeepTag::TAdt) => {
                            let name = kids
                                .first()
                                .and_then(symbol_text)
                                .ok_or(TraversalError::Malformed("ADT without a name"))?;
                            // Explicit nominal dimension arguments have no opaque
                            // values, just like NominalArg::Dimension above.
                            let args = self.intern_all(kids[1..].iter().filter_map(|arg| {
                                (!matches!(
                                    tag(arg),
                                    Some(DeepTag::DName | DeepTag::DVar | DeepTag::DLit)
                                ))
                                .then_some(Source::Deep(arg))
                            }))?;
                            self.adt(name, args)?
                        }
                        Some(DeepTag::TTuple) => {
                            Node::Tuple(self.intern_all(kids.iter().map(Source::Deep))?)
                        }
                        Some(DeepTag::TFn) => {
                            let (ret, args) = kids
                                .split_last()
                                .ok_or(TraversalError::Malformed("function without a result"))?;
                            Node::Function(
                                self.intern_all(args.iter().map(Source::Deep))?,
                                self.intern(Source::Deep(ret))?,
                            )
                        }
                        Some(DeepTag::TRef) => {
                            Node::Ref(self.intern(Source::Deep(kids.first().ok_or(
                                TraversalError::Malformed("reference without an inner type"),
                            )?))?)
                        }
                        Some(
                            DeepTag::TPrim | DeepTag::TTensor | DeepTag::TVar | DeepTag::TUnit,
                        ) => Node::Leaf,
                        _ => return Err(TraversalError::Malformed("expected a Deep type node")),
                    }
                }
            };
            self.nodes[id] = node;
        }
        Ok(root)
    }

    fn edges(&self, id: Id) -> Vec<Id> {
        match &self.nodes[id] {
            Node::Adt { args, fields, .. } => args.iter().chain(fields).copied().collect(),
            Node::Tuple(items) => items.clone(),
            Node::Function(args, ret) => args.iter().copied().chain([*ret]).collect(),
            Node::Link(inner) | Node::Ref(inner) => vec![*inner],
            Node::Leaf => Vec::new(),
            Node::Pending => unreachable!("projection drains all pending nodes before querying"),
        }
    }

    pub(super) fn contains(&mut self, root: Id, name: &str) -> Result<bool, TraversalError> {
        let mut work = vec![root];
        let mut visited = BTreeSet::new();
        while let Some(id) = work.pop() {
            if !visited.insert(id) {
                continue;
            }
            self.budget.step()?;
            if matches!(&self.nodes[id], Node::Adt { name: found, .. } if found == name) {
                return Ok(true);
            }
            work.extend(self.edges(id));
        }
        Ok(false)
    }

    pub(super) fn caller_receives(&mut self, root: Id, name: &str) -> Result<bool, TraversalError> {
        let mut work = vec![root];
        let mut visited = BTreeSet::new();
        while let Some(id) = work.pop() {
            if !visited.insert(id) {
                continue;
            }
            self.budget.step()?;
            match &self.nodes[id] {
                Node::Function(args, ret) => {
                    let args = args.clone();
                    work.push(*ret);
                    for arg in args {
                        if self.contains(arg, name)? {
                            return Ok(true);
                        }
                        work.push(arg);
                    }
                }
                Node::Adt { args, .. } | Node::Tuple(args) => work.extend(args),
                Node::Link(inner) | Node::Ref(inner) => work.push(*inner),
                Node::Leaf => {}
                Node::Pending => unreachable!("projection completed"),
            }
        }
        Ok(false)
    }

    pub(super) fn function(&self, id: Id) -> Option<(Vec<Id>, Id)> {
        match &self.nodes[id] {
            Node::Function(args, ret) => Some((args.clone(), *ret)),
            _ => None,
        }
    }

    pub(super) fn decompose(
        &mut self,
        root: Id,
        name: &str,
    ) -> Result<Option<ProducedPosition>, DecomposeError> {
        enum Task {
            Visit(Id, usize),
            Leave(Id),
            Option(usize, usize),
            Tuple(usize, Vec<usize>),
        }
        let mut work = vec![Task::Visit(root, 0)];
        let mut positions = vec![None];
        let mut active = BTreeSet::new();
        while let Some(task) = work.pop() {
            self.budget.step()?;
            match task {
                Task::Leave(id) => {
                    active.remove(&id);
                }
                Task::Option(out, inner) => {
                    positions[out] = positions[inner]
                        .take()
                        .map(|position| ProducedPosition::InsideOption(Box::new(position)));
                }
                Task::Tuple(out, slots) => {
                    let components: Vec<_> = slots
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, slot)| {
                            positions[slot].take().map(|position| (index, position))
                        })
                        .collect();
                    if !components.is_empty() {
                        positions[out] = Some(ProducedPosition::TupleComponents(components));
                    }
                }
                Task::Visit(id, out) => {
                    if !active.insert(id) {
                        if self.contains(id, name)? {
                            return Err(DecomposeError::Container("recursive alias".to_string()));
                        }
                        continue;
                    }
                    work.push(Task::Leave(id));
                    match &self.nodes[id] {
                        Node::Adt { name: found, .. } if found == name => {
                            positions[out] = Some(ProducedPosition::Direct)
                        }
                        Node::Link(inner) => work.push(Task::Visit(*inner, out)),
                        Node::Adt {
                            name: found, args, ..
                        } if found == "Option" => {
                            let inner = *args
                                .first()
                                .ok_or_else(|| DecomposeError::Container("Option".to_string()))?;
                            let slot = positions.len();
                            positions.push(None);
                            work.push(Task::Option(out, slot));
                            work.push(Task::Visit(inner, slot));
                        }
                        Node::Tuple(items) => {
                            let slots: Vec<_> =
                                (positions.len()..positions.len() + items.len()).collect();
                            positions.resize(positions.len() + items.len(), None);
                            work.push(Task::Tuple(out, slots.clone()));
                            for (id, slot) in items.iter().zip(slots).rev() {
                                work.push(Task::Visit(*id, slot));
                            }
                        }
                        other => {
                            let container = match other {
                                Node::Adt { name, .. }
                                    if self.records.contains_key(name.as_str()) =>
                                {
                                    format!("record `{name}`")
                                }
                                Node::Adt { name, .. } => format!("generic `{name}`"),
                                Node::Ref(_) => "borrow".to_string(),
                                Node::Function(_, _) => "function type".to_string(),
                                Node::Leaf => continue,
                                _ => unreachable!("handled above"),
                            };
                            if self.contains(id, name)? {
                                return Err(DecomposeError::Container(container));
                            }
                        }
                    }
                }
            }
        }
        Ok(positions[0].take())
    }
}

pub(super) enum DecomposeError {
    Container(String),
    Traversal(TraversalError),
}

impl From<TraversalError> for DecomposeError {
    fn from(error: TraversalError) -> Self {
        Self::Traversal(error)
    }
}
