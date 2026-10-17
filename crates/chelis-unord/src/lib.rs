#![forbid(unsafe_code)]

//! Order-free collections for compiler internals.
//!
//! `spec/design/hash_order_determinism.md` bans `std::collections::HashMap`
//! and `HashSet` across the workspace because their per-process hash seeds let
//! an iteration order reach a checker verdict, a dispatch choice, emitted
//! artifact bytes, or a persisted cache payload. These two types are the
//! replacement for the hashed-lookup uses.
//!
//! The storage is a `BTreeMap`/`BTreeSet`, so there is no randomized order to
//! leak in the first place: cloning, comparison, serialization, `clear`, and
//! destruction all traverse key order because that is the only order the
//! structure has. Every key already carried `Ord`, since the ordered exits
//! below require it.
//!
//! What the type still buys over using `BTreeMap` directly is the *boundary*:
//! there is no `iter`, `keys`, `values`, `drain`, `IntoIterator`, or `Deref`.
//! A consumer that needs an order must spell [`UnordMap::to_sorted`] or
//! [`UnordMap::into_sorted`], which reads as a claim that its owning authority
//! makes key order canonical here, rather than picking up whatever order the
//! container happened to have.
//!
//! Two scope types serve walkers that extend an environment at each binder.
//! [`ScopedMap`] undoes a scope's changes when the scope ends, for an
//! environment no copy outlives. [`PersistentMap`] and [`PersistentSet`] keep
//! the same ordered boundary over shared storage, so a copy that outlives its
//! scope costs O(1) to take and O(log n) per later change.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::Arc;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A map whose public API cannot expose an incidental iteration order.
#[derive(Clone, PartialEq, Eq)]
pub struct UnordMap<K, V> {
    inner: BTreeMap<K, V>,
}

impl<K, V> fmt::Debug for UnordMap<K, V> {
    /// Report the kind and length only.
    ///
    /// A content-bearing `Debug` is an ordered rendering of the contents, and
    /// a diagnostic or snapshot that embeds one turns a container's order into
    /// observable output. Callers that want the contents ask for them.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "UnordMap {{ len: {} }}", self.inner.len())
    }
}

/// An empty map needs nothing of `K` or `V`, so this carries no bound.
impl<K, V> Default for UnordMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> UnordMap<K, V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<K: Ord, V> UnordMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(key)
    }

    #[must_use]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_key_value(key)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_mut(key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(key)
    }

    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains_key(key)
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.inner.entry(key) {
            btree_map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry(entry)),
            btree_map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry(entry)),
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.inner.extend(other.inner);
    }

    /// Return the contents in the key's canonical `Ord` order.
    ///
    /// Available only where the caller's owning authority makes key order
    /// canonical; see `spec/design/hash_order_determinism.md` C3.1.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<(&K, &V)> {
        self.inner.iter().collect()
    }

    /// Consume the map and return its contents in the key's canonical `Ord`
    /// order. See [`Self::to_sorted`] for the authority requirement.
    #[must_use]
    pub fn into_sorted(self) -> Vec<(K, V)> {
        self.inner.into_iter().collect()
    }
}

impl<K: Ord, V> Extend<(K, V)> for UnordMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        self.inner.extend(iter);
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for UnordMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for UnordMap<K, V> {
    fn from(entries: [(K, V); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<K, Q, V> Index<&Q> for UnordMap<K, V>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        self.get(index).expect("no entry found for key")
    }
}

pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    pub fn and_modify<F: FnOnce(&mut V)>(self, update: F) -> Self {
        match self {
            Self::Occupied(mut entry) => {
                update(entry.get_mut());
                Self::Occupied(entry)
            }
            Self::Vacant(entry) => Self::Vacant(entry),
        }
    }

    #[must_use]
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(entry) => entry.key(),
            Self::Vacant(entry) => entry.key(),
        }
    }
}

pub struct OccupiedEntry<'a, K, V>(btree_map::OccupiedEntry<'a, K, V>);

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    #[must_use]
    pub fn key(&self) -> &K {
        self.0.key()
    }

    #[must_use]
    pub fn get(&self) -> &V {
        self.0.get()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.0.get_mut()
    }

    pub fn into_mut(self) -> &'a mut V {
        self.0.into_mut()
    }

    pub fn insert(&mut self, value: V) -> V {
        self.0.insert(value)
    }

    pub fn remove(self) -> V {
        self.0.remove()
    }

    pub fn remove_entry(self) -> (K, V) {
        self.0.remove_entry()
    }
}

pub struct VacantEntry<'a, K, V>(btree_map::VacantEntry<'a, K, V>);

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    #[must_use]
    pub fn key(&self) -> &K {
        self.0.key()
    }

    pub fn into_key(self) -> K {
        self.0.into_key()
    }

    pub fn insert(self, value: V) -> &'a mut V {
        self.0.insert(value)
    }
}

/// A set whose public API cannot expose an incidental iteration order.
#[derive(Clone, PartialEq, Eq)]
pub struct UnordSet<T> {
    inner: BTreeSet<T>,
}

impl<T> fmt::Debug for UnordSet<T> {
    /// Report the kind and length only; see [`UnordMap`]'s `Debug`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "UnordSet {{ len: {} }}", self.inner.len())
    }
}

/// An empty set needs nothing of `T`, so this carries no bound.
impl<T> Default for UnordSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> UnordSet<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<T: Ord> UnordSet<T> {
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value)
    }

    #[must_use]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains(value)
    }

    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(value)
    }

    pub fn merge(&mut self, other: Self) {
        self.inner.extend(other.inner);
    }

    /// Return the contents in the element's canonical `Ord` order.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<&T> {
        self.inner.iter().collect()
    }

    /// Consume the set and return its contents in the element's canonical
    /// `Ord` order.
    #[must_use]
    pub fn into_sorted(self) -> Vec<T> {
        self.inner.into_iter().collect()
    }
}

impl<T: Ord> Extend<T> for UnordSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
    }
}

impl<T: Ord> FromIterator<T> for UnordSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}

impl<T: Ord, const N: usize> From<[T; N]> for UnordSet<T> {
    fn from(entries: [T; N]) -> Self {
        entries.into_iter().collect()
    }
}

/// A map with nested lexical scopes, for a walker whose environment gains
/// bindings at each binder and loses them when the binder's body ends.
///
/// Entering a scope with [`ScopedMap::scope`] copies nothing. Each
/// [`ScopedMap::insert`] or [`ScopedMap::remove`] inside a scope records the
/// entry it displaced, and leaving the scope restores those entries in
/// reverse order, so the map after a scope equals the map before it. A
/// walker that copied its whole environment at every binder pays instead for
/// the bindings each scope makes, which keeps a chain of `k` nested binders
/// linear rather than quadratic in `k`.
///
/// Like [`UnordMap`], it has no iteration exit. It is deliberately not
/// `Clone`: a copy of the environment per scope is the cost it removes.
pub struct ScopedMap<K, V> {
    map: UnordMap<K, V>,
    /// The entries the open scopes displaced, oldest first: a key and the
    /// value it held before, `None` when it was absent.
    undo: Vec<(K, Option<V>)>,
    /// How many scopes are open. A change outside every scope is permanent
    /// and records nothing.
    depth: usize,
}

impl<K, V> fmt::Debug for ScopedMap<K, V> {
    /// Report the kind, length and scope depth only, as [`UnordMap`] does.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ScopedMap {{ len: {}, depth: {} }}",
            self.map.len(),
            self.depth
        )
    }
}

impl<K, V> Default for ScopedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> ScopedMap<K, V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: UnordMap::new(),
            undo: Vec::new(),
            depth: 0,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<K: Ord + Clone, V> ScopedMap<K, V> {
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.get(key)
    }

    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.contains_key(key)
    }

    /// Bind `key` to `value` until the innermost open scope ends.
    pub fn insert(&mut self, key: K, value: V) {
        let displaced = self.map.insert(key.clone(), value);
        if self.depth > 0 {
            self.undo.push((key, displaced));
        }
    }

    /// Unbind `key` until the innermost open scope ends.
    pub fn remove<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if let Some((key, displaced)) = self.map.inner.remove_entry(key)
            && self.depth > 0
        {
            self.undo.push((key, Some(displaced)));
        }
    }

    /// Run `body` in a new scope, then restore every entry its changes
    /// displaced.
    pub fn scope<R>(&mut self, body: impl FnOnce(&mut Self) -> R) -> R {
        let mark = self.undo.len();
        self.depth += 1;
        let result = body(self);
        while self.undo.len() > mark {
            let (key, displaced) = self.undo.pop().expect("the undo log is above its mark");
            match displaced {
                Some(value) => {
                    self.map.insert(key, value);
                }
                None => {
                    self.map.remove(&key);
                }
            }
        }
        self.depth -= 1;
        result
    }
}

impl<K: Ord + Clone, V> FromIterator<(K, V)> for ScopedMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            map: iter.into_iter().collect(),
            undo: Vec::new(),
            depth: 0,
        }
    }
}

/// A map whose copies share their storage, for a scope that is snapshotted
/// and then extended: a lexical environment that every nested binder copies
/// before adding its own names, and whose copies can outlive the binder.
///
/// The storage is a balanced binary search tree whose nodes are reference
/// counted. [`Clone`] copies one pointer, so taking a snapshot costs O(1)
/// whatever the map holds. A change through one copy duplicates only the
/// nodes on the path to the key it touches, O(log n) of them, and every
/// other copy keeps seeing the map as it was. A node that only one copy
/// holds is changed in place. A walker that once copied its whole
/// environment at every binder therefore pays for the bindings each binder
/// makes, which keeps a chain of `k` nested binders near linear in `k`
/// rather than quadratic, in time and in the memory its live snapshots hold.
///
/// Like [`UnordMap`], it has no iteration exit beyond [`Self::to_sorted`],
/// and its `Debug` reports the length only. It serializes exactly as an
/// [`UnordMap`] with the same contents does. The tree's shape, and so the
/// order in which a dropped map releases its entries, is a function of the
/// sequence of changes that built it; no seed or address enters it.
pub struct PersistentMap<K, V> {
    root: Link<K, V>,
    len: usize,
}

type Link<K, V> = Option<Arc<Node<K, V>>>;

/// One tree node. `height` is the AVL height: a leaf is 1, and the heights of
/// a node's two subtrees differ by at most one.
struct Node<K, V> {
    key: K,
    value: V,
    left: Link<K, V>,
    right: Link<K, V>,
    height: u8,
}

impl<K: Clone, V: Clone> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            value: self.value.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
            height: self.height,
        }
    }
}

fn height<K, V>(link: &Link<K, V>) -> u8 {
    link.as_ref().map_or(0, |node| node.height)
}

impl<K, V> Node<K, V> {
    fn update_height(&mut self) {
        self.height = 1 + height(&self.left).max(height(&self.right));
    }

    fn balance(&self) -> i16 {
        i16::from(height(&self.left)) - i16::from(height(&self.right))
    }
}

/// The node at `link`, copied first if another map shares it.
fn unique<K: Clone, V: Clone>(link: &mut Link<K, V>) -> &mut Node<K, V> {
    Arc::make_mut(
        link.as_mut()
            .expect("a rotation or rebalance names a present node"),
    )
}

/// Turn `link`'s subtree so its left child becomes its root.
fn rotate_right<K: Clone, V: Clone>(link: &mut Link<K, V>) {
    let mut root = link.take().expect("a rotation names a present node");
    let mut pivot = {
        let node = Arc::make_mut(&mut root);
        let mut pivot = node.left.take().expect("a right rotation has a left child");
        node.left = Arc::make_mut(&mut pivot).right.take();
        node.update_height();
        pivot
    };
    let node = Arc::make_mut(&mut pivot);
    node.right = Some(root);
    node.update_height();
    *link = Some(pivot);
}

/// Turn `link`'s subtree so its right child becomes its root.
fn rotate_left<K: Clone, V: Clone>(link: &mut Link<K, V>) {
    let mut root = link.take().expect("a rotation names a present node");
    let mut pivot = {
        let node = Arc::make_mut(&mut root);
        let mut pivot = node
            .right
            .take()
            .expect("a left rotation has a right child");
        node.right = Arc::make_mut(&mut pivot).left.take();
        node.update_height();
        pivot
    };
    let node = Arc::make_mut(&mut pivot);
    node.left = Some(root);
    node.update_height();
    *link = Some(pivot);
}

/// Restore the AVL invariant at `link` after one of its subtrees changed
/// height by at most one.
fn rebalance<K: Clone, V: Clone>(link: &mut Link<K, V>) {
    let node = unique(link);
    node.update_height();
    let balance = node.balance();
    if balance > 1 {
        if unique(&mut node.left).balance() < 0 {
            rotate_left(&mut node.left);
        }
        rotate_right(link);
    } else if balance < -1 {
        if unique(&mut node.right).balance() > 0 {
            rotate_right(&mut node.right);
        }
        rotate_left(link);
    }
}

fn insert_at<K: Ord + Clone, V: Clone>(link: &mut Link<K, V>, key: K, value: V) -> Option<V> {
    let Some(shared) = link.as_mut() else {
        *link = Some(Arc::new(Node {
            key,
            value,
            left: None,
            right: None,
            height: 1,
        }));
        return None;
    };
    let node = Arc::make_mut(shared);
    let displaced = match key.cmp(&node.key) {
        Ordering::Less => insert_at(&mut node.left, key, value),
        Ordering::Greater => insert_at(&mut node.right, key, value),
        Ordering::Equal => return Some(std::mem::replace(&mut node.value, value)),
    };
    if displaced.is_none() {
        rebalance(link);
    }
    displaced
}

/// Remove and return the least entry of the nonempty subtree at `link`.
fn remove_least<K: Clone, V: Clone>(link: &mut Link<K, V>) -> (K, V) {
    let node = unique(link);
    if node.left.is_some() {
        let least = remove_least(&mut node.left);
        rebalance(link);
        return least;
    }
    let right = node.right.take();
    let Node { key, value, .. } =
        Arc::unwrap_or_clone(std::mem::replace(link, right).expect("the subtree is nonempty"));
    (key, value)
}

/// Remove `key` from the subtree at `link`, which holds it.
fn remove_at<K, V, Q>(link: &mut Link<K, V>, key: &Q) -> V
where
    K: Ord + Clone + Borrow<Q>,
    V: Clone,
    Q: Ord + ?Sized,
{
    let node = unique(link);
    let removed = match key.cmp(node.key.borrow()) {
        Ordering::Less => remove_at(&mut node.left, key),
        Ordering::Greater => remove_at(&mut node.right, key),
        Ordering::Equal => match (node.left.take(), node.right.take()) {
            (None, child) | (child, None) => {
                let removed = std::mem::replace(link, child).expect("the subtree holds the key");
                return Arc::unwrap_or_clone(removed).value;
            }
            (Some(left), Some(right)) => {
                node.left = Some(left);
                node.right = Some(right);
                let (least_key, least_value) = remove_least(&mut node.right);
                node.key = least_key;
                std::mem::replace(&mut node.value, least_value)
            }
        },
    };
    rebalance(link);
    removed
}

fn get_at<'a, K, V, Q>(mut link: &'a Link<K, V>, key: &Q) -> Option<&'a V>
where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
{
    while let Some(node) = link {
        match key.cmp(node.key.borrow()) {
            Ordering::Less => link = &node.left,
            Ordering::Greater => link = &node.right,
            Ordering::Equal => return Some(&node.value),
        }
    }
    None
}

fn get_mut_at<'a, K, V, Q>(mut link: &'a mut Link<K, V>, key: &Q) -> Option<&'a mut V>
where
    K: Clone + Borrow<Q>,
    V: Clone,
    Q: Ord + ?Sized,
{
    loop {
        let node = Arc::make_mut(link.as_mut()?);
        match key.cmp(node.key.borrow()) {
            Ordering::Less => link = &mut node.left,
            Ordering::Greater => link = &mut node.right,
            Ordering::Equal => return Some(&mut node.value),
        }
    }
}

fn collect_sorted<'a, K, V>(link: &'a Link<K, V>, out: &mut Vec<(&'a K, &'a V)>) {
    if let Some(node) = link {
        collect_sorted(&node.left, out);
        out.push((&node.key, &node.value));
        collect_sorted(&node.right, out);
    }
}

fn drain_sorted<K: Clone, V: Clone>(link: Link<K, V>, out: &mut Vec<(K, V)>) {
    if let Some(node) = link {
        let Node {
            key,
            value,
            left,
            right,
            ..
        } = Arc::unwrap_or_clone(node);
        drain_sorted(left, out);
        out.push((key, value));
        drain_sorted(right, out);
    }
}

impl<K, V> Clone for PersistentMap<K, V> {
    /// Share the storage: O(1), whatever the map holds.
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<K, V> fmt::Debug for PersistentMap<K, V> {
    /// Report the kind and length only, as [`UnordMap`] does.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "PersistentMap {{ len: {} }}", self.len)
    }
}

impl<K, V> Default for PersistentMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> PersistentMap<K, V> {
    #[must_use]
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.len = 0;
    }
}

impl<K: Ord + Clone, V: Clone> PersistentMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let displaced = insert_at(&mut self.root, key, value);
        if displaced.is_none() {
            self.len += 1;
        }
        displaced
    }

    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        get_at(&self.root, key)
    }

    /// The value at `key`, unshared from every other copy of the map first.
    /// An absent key copies nothing.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if !self.contains_key(key) {
            return None;
        }
        get_mut_at(&mut self.root, key)
    }

    /// The value at `key`, inserting `default()` first when it is absent.
    pub fn get_or_insert_with(&mut self, key: K, default: impl FnOnce() -> V) -> &mut V {
        if !self.contains_key(&key) {
            self.insert(key.clone(), default());
        }
        get_mut_at(&mut self.root, &key).expect("the key was just made present")
    }

    /// Remove `key`. An absent key copies nothing.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if !self.contains_key(key) {
            return None;
        }
        self.len -= 1;
        Some(remove_at(&mut self.root, key))
    }

    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Return the contents in the key's canonical `Ord` order. See
    /// [`UnordMap::to_sorted`] for the authority requirement.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<(&K, &V)> {
        let mut out = Vec::with_capacity(self.len);
        collect_sorted(&self.root, &mut out);
        out
    }

    /// Consume the map and return its contents in the key's canonical `Ord`
    /// order, moving each entry no other copy shares and cloning the rest.
    #[must_use]
    pub fn into_sorted(self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.len);
        drain_sorted(self.root, &mut out);
        out
    }
}

impl<K: Ord + Clone, V: Clone> Extend<(K, V)> for PersistentMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Ord + Clone, V: Clone> FromIterator<(K, V)> for PersistentMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

/// A set whose copies share their storage; see [`PersistentMap`].
pub struct PersistentSet<T> {
    map: PersistentMap<T, ()>,
}

impl<T> Clone for PersistentSet<T> {
    /// Share the storage: O(1), whatever the set holds.
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<T> fmt::Debug for PersistentSet<T> {
    /// Report the kind and length only, as [`UnordSet`] does.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "PersistentSet {{ len: {} }}", self.map.len())
    }
}

impl<T> Default for PersistentSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PersistentSet<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: PersistentMap::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

impl<T: Ord + Clone> PersistentSet<T> {
    /// Whether `value` was absent.
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    #[must_use]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.contains_key(value)
    }

    /// Whether `value` was present. An absent value copies nothing.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Return the contents in the element's canonical `Ord` order.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<&T> {
        self.map
            .to_sorted()
            .into_iter()
            .map(|(value, ())| value)
            .collect()
    }

    /// Consume the set and return its contents in the element's canonical
    /// `Ord` order.
    #[must_use]
    pub fn into_sorted(self) -> Vec<T> {
        self.map
            .into_sorted()
            .into_iter()
            .map(|(value, ())| value)
            .collect()
    }
}

impl<T: Ord + Clone> Extend<T> for PersistentSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<T: Ord + Clone> FromIterator<T> for PersistentSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<K, V> Serialize for UnordMap<K, V>
where
    K: Ord + Serialize,
    V: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.inner.len()))?;
        for entry in &self.inner {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
}

struct MapVisitor<K, V>(PhantomData<fn() -> (K, V)>);

impl<'de, K, V> Visitor<'de> for MapVisitor<K, V>
where
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = UnordMap<K, V>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence of unique key-value pairs")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut result = UnordMap::new();
        while let Some((key, value)) = sequence.next_element()? {
            if result.insert(key, value).is_some() {
                return Err(A::Error::custom("duplicate key in UnordMap"));
            }
        }
        Ok(result)
    }
}

impl<'de, K, V> Deserialize<'de> for UnordMap<K, V>
where
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(MapVisitor(PhantomData))
    }
}

impl<T> Serialize for UnordSet<T>
where
    T: Ord + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.inner.len()))?;
        for value in &self.inner {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

struct SetVisitor<T>(PhantomData<fn() -> T>);

impl<'de, T> Visitor<'de> for SetVisitor<T>
where
    T: Ord + Deserialize<'de>,
{
    type Value = UnordSet<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence of unique set elements")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut result = UnordSet::new();
        while let Some(value) = sequence.next_element()? {
            if !result.insert(value) {
                return Err(A::Error::custom("duplicate element in UnordSet"));
            }
        }
        Ok(result)
    }
}

impl<'de, T> Deserialize<'de> for UnordSet<T>
where
    T: Ord + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SetVisitor(PhantomData))
    }
}

impl<K, V> Serialize for PersistentMap<K, V>
where
    K: Ord + Serialize,
    V: Serialize,
{
    /// The same sequence of key-value pairs an [`UnordMap`] writes.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = Vec::with_capacity(self.len);
        collect_sorted(&self.root, &mut entries);
        let mut sequence = serializer.serialize_seq(Some(entries.len()))?;
        for entry in entries {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
}

impl<'de, K, V> Deserialize<'de> for PersistentMap<K, V>
where
    K: Ord + Clone + Deserialize<'de>,
    V: Clone + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries = UnordMap::<K, V>::deserialize(deserializer)?;
        Ok(entries.into_sorted().into_iter().collect())
    }
}
