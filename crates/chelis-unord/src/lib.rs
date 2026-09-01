#![forbid(unsafe_code)]
#![deny(clippy::iter_over_hash_type)]

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

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Index;

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
