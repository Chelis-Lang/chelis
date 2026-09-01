#![forbid(unsafe_code)]
#![deny(clippy::iter_over_hash_type)]

//! Order-free hash collections.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, RandomState};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Index;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[allow(clippy::disallowed_types)]
mod raw {
    pub(crate) type Map<K, V> = std::collections::HashMap<K, V>;
}

/// A hash map whose public API cannot expose the table's iteration order.
pub struct UnordMap<K, V> {
    storage: Box<Storage<K, V>>,
}

struct Storage<K, V> {
    /// The raw hash table contains numeric digests and numeric entry indices
    /// only. User-owned keys and values never live in, or are walked through,
    /// randomized table order.
    buckets: raw::Map<u64, Vec<usize>>,
    hash_builder: RandomState,
    entries: Vec<Option<(K, V)>>,
    /// Entry indices in canonical key order, maintained at mutation time.
    order: Vec<usize>,
    free: Vec<usize>,
    len: usize,
}

impl<K, V> Drop for UnordMap<K, V> {
    fn drop(&mut self) {
        // `HashMap`'s own drop walks randomized table order. User-owned
        // entries live outside that table, so retire them through the
        // maintained canonical index before the numeric buckets are dropped.
        for index in std::mem::take(&mut self.storage.order) {
            drop(self.storage.entries[index].take());
        }
    }
}

impl<K, V> Clone for UnordMap<K, V>
where
    K: Clone + Eq + Hash + Ord,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut cloned = Self::new();
        for index in &self.storage.order {
            let (key, value) = self.entry_at(*index);
            cloned.insert(key.clone(), value.clone());
        }
        cloned
    }
}

impl<K, V> fmt::Debug for UnordMap<K, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnordMap")
            .field("len", &self.storage.len)
            .finish()
    }
}

impl<K, V> PartialEq for UnordMap<K, V>
where
    K: Eq + Hash + Ord,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.storage.len == other.storage.len
            && self
                .storage
                .order
                .iter()
                .zip(&other.storage.order)
                .all(|(left, right)| {
                    let left = self.entry_at(*left);
                    let right = other.entry_at(*right);
                    left.0 == right.0 && left.1 == right.1
                })
    }
}

impl<K, V> Eq for UnordMap<K, V>
where
    K: Eq + Hash + Ord,
    V: Eq,
{
}

impl<K, V> Default for UnordMap<K, V> {
    fn default() -> Self {
        Self {
            storage: Box::new(Storage {
                buckets: raw::Map::new(),
                hash_builder: RandomState::new(),
                entries: Vec::new(),
                order: Vec::new(),
                free: Vec::new(),
                len: 0,
            }),
        }
    }
}

impl<K, V> UnordMap<K, V> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.len == 0
    }

    pub fn clear(&mut self) {
        for index in std::mem::take(&mut self.storage.order) {
            drop(self.storage.entries[index].take());
        }
        self.storage.buckets.clear();
        self.storage.entries.clear();
        self.storage.free.clear();
        self.storage.len = 0;
    }

    fn entry_at(&self, index: usize) -> (&K, &V) {
        let (key, value) = self.storage.entries[index]
            .as_ref()
            .expect("UnordMap index must name a live entry");
        (key, value)
    }

    fn entry_at_mut(&mut self, index: usize) -> (&K, &mut V) {
        let (key, value) = self.storage.entries[index]
            .as_mut()
            .expect("UnordMap index must name a live entry");
        (key, value)
    }
}

impl<K: Eq + Hash + Ord, V> UnordMap<K, V> {
    fn hash<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        self.storage.hash_builder.hash_one(key)
    }

    fn find_index<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.storage
            .buckets
            .get(&self.hash(key))
            .and_then(|bucket| {
                bucket.iter().copied().find(|index| {
                    self.storage.entries[*index]
                        .as_ref()
                        .is_some_and(|(stored, _)| stored.borrow() == key)
                })
            })
    }

    fn insert_new(&mut self, key: K, value: V) -> usize {
        let digest = self.hash(&key);
        let index = match self.storage.free.pop() {
            Some(index) => {
                self.storage.entries[index] = Some((key, value));
                index
            }
            None => {
                let index = self.storage.entries.len();
                self.storage.entries.push(Some((key, value)));
                index
            }
        };
        let ordered_position = self
            .storage
            .order
            .binary_search_by(|other| self.entry_at(*other).0.cmp(self.entry_at(index).0))
            .expect_err("a vacant UnordMap key must have a unique canonical position");
        self.storage.order.insert(ordered_position, index);
        self.storage.buckets.entry(digest).or_default().push(index);
        self.storage.len += 1;
        index
    }

    fn remove_index(&mut self, index: usize) -> (K, V) {
        let digest = self.hash(self.entry_at(index).0);
        let ordered_position = self
            .storage
            .order
            .iter()
            .position(|candidate| *candidate == index)
            .expect("live UnordMap entry must appear in canonical order");
        self.storage.order.remove(ordered_position);
        let empty_bucket = {
            let bucket = self
                .storage
                .buckets
                .get_mut(&digest)
                .expect("live UnordMap entry must appear in its hash bucket");
            let bucket_position = bucket
                .iter()
                .position(|candidate| *candidate == index)
                .expect("live UnordMap entry index must appear in its hash bucket");
            bucket.swap_remove(bucket_position);
            bucket.is_empty()
        };
        if empty_bucket {
            self.storage.buckets.remove(&digest);
        }
        let entry = self.storage.entries[index]
            .take()
            .expect("removed UnordMap index must name a live entry");
        self.storage.free.push(index);
        self.storage.len -= 1;
        entry
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(index) = self.find_index(&key) {
            return Some(std::mem::replace(self.entry_at_mut(index).1, value));
        }
        self.insert_new(key, value);
        None
    }

    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_index(key).map(|index| self.entry_at(index).1)
    }

    #[must_use]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_index(key).map(|index| self.entry_at(index))
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        Some(self.entry_at_mut(index).1)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.find_index(key)?;
        Some(self.remove_index(index).1)
    }

    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_index(key).is_some()
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.find_index(&key) {
            Some(index) => Entry::Occupied(OccupiedEntry { map: self, index }),
            None => Entry::Vacant(VacantEntry { map: self, key }),
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.extend(other.into_sorted());
    }

    /// Return the contents in the key's canonical `Ord` order.
    ///
    /// This exit is available only when the caller's owning authority makes
    /// the key order canonical. Unlike a projection-based sort, no user code
    /// runs while the raw table order is still observable.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<(&K, &V)> {
        self.storage
            .order
            .iter()
            .map(|index| self.entry_at(*index))
            .collect()
    }

    /// Consume the map and return its contents in the key's canonical `Ord`
    /// order. See [`Self::to_sorted`] for the authority requirement.
    #[must_use]
    pub fn into_sorted(mut self) -> Vec<(K, V)> {
        let order = std::mem::take(&mut self.storage.order);
        order
            .into_iter()
            .map(|index| {
                self.storage.entries[index]
                    .take()
                    .expect("canonical UnordMap index must name a live entry")
            })
            .collect()
    }
}

impl<K: Eq + Hash + Ord, V> Extend<(K, V)> for UnordMap<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Eq + Hash + Ord, V> FromIterator<(K, V)> for UnordMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

impl<K: Eq + Hash + Ord, V, const N: usize> From<[(K, V); N]> for UnordMap<K, V> {
    fn from(entries: [(K, V); N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<K, Q, V> Index<&Q> for UnordMap<K, V>
where
    K: Eq + Hash + Ord + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
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

impl<'a, K: Eq + Hash + Ord, V> Entry<'a, K, V> {
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

pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut UnordMap<K, V>,
    index: usize,
}

impl<'a, K: Eq + Hash + Ord, V> OccupiedEntry<'a, K, V> {
    #[must_use]
    pub fn key(&self) -> &K {
        self.map.entry_at(self.index).0
    }

    #[must_use]
    pub fn get(&self) -> &V {
        self.map.entry_at(self.index).1
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.map.entry_at_mut(self.index).1
    }

    pub fn into_mut(self) -> &'a mut V {
        self.map.entry_at_mut(self.index).1
    }

    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.map.entry_at_mut(self.index).1, value)
    }

    pub fn remove(self) -> V {
        self.map.remove_index(self.index).1
    }

    pub fn remove_entry(self) -> (K, V) {
        self.map.remove_index(self.index)
    }
}

pub struct VacantEntry<'a, K, V> {
    map: &'a mut UnordMap<K, V>,
    key: K,
}

impl<'a, K: Eq + Hash + Ord, V> VacantEntry<'a, K, V> {
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn into_key(self) -> K {
        self.key
    }

    pub fn insert(self, value: V) -> &'a mut V {
        let index = self.map.insert_new(self.key, value);
        self.map.entry_at_mut(index).1
    }
}

/// A hash set whose public API cannot expose the table's iteration order.
pub struct UnordSet<T> {
    inner: UnordMap<T, ()>,
}

impl<T> Clone for UnordSet<T>
where
    T: Clone + Eq + Hash + Ord,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> fmt::Debug for UnordSet<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnordSet")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl<T: Eq + Hash + Ord> PartialEq for UnordSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Eq + Hash + Ord> Eq for UnordSet<T> {}

impl<T> Default for UnordSet<T> {
    fn default() -> Self {
        Self {
            inner: UnordMap::new(),
        }
    }
}

impl<T> UnordSet<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

impl<T: Eq + Hash + Ord> UnordSet<T> {
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    #[must_use]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.contains_key(value)
    }

    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    pub fn merge(&mut self, other: Self) {
        self.inner.merge(other.inner);
    }

    /// Return the contents in the element's canonical `Ord` order.
    #[must_use]
    pub fn to_sorted(&self) -> Vec<&T>
    where
        T: Ord,
    {
        self.inner
            .to_sorted()
            .into_iter()
            .map(|(value, ())| value)
            .collect()
    }

    /// Consume the set and return its contents in the element's canonical
    /// `Ord` order.
    #[must_use]
    pub fn into_sorted(self) -> Vec<T>
    where
        T: Ord,
    {
        self.inner
            .into_sorted()
            .into_iter()
            .map(|(value, ())| value)
            .collect()
    }
}

impl<T: Eq + Hash + Ord> Extend<T> for UnordSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter.into_iter().map(|value| (value, ())));
    }
}

impl<T: Eq + Hash + Ord> FromIterator<T> for UnordSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().map(|value| (value, ())).collect(),
        }
    }
}

impl<T: Eq + Hash + Ord, const N: usize> From<[T; N]> for UnordSet<T> {
    fn from(entries: [T; N]) -> Self {
        entries.into_iter().collect()
    }
}

impl<K, V> Serialize for UnordMap<K, V>
where
    K: Eq + Hash + Ord + Serialize,
    V: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let entries = self.to_sorted();
        let mut sequence = serializer.serialize_seq(Some(entries.len()))?;
        for entry in entries {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
}

struct MapVisitor<K, V>(PhantomData<fn() -> (K, V)>);

impl<'de, K, V> Visitor<'de> for MapVisitor<K, V>
where
    K: Eq + Hash + Ord + Deserialize<'de>,
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
    K: Eq + Hash + Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(MapVisitor(PhantomData))
    }
}

impl<T> Serialize for UnordSet<T>
where
    T: Eq + Hash + Ord + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let entries = self.to_sorted();
        let mut sequence = serializer.serialize_seq(Some(entries.len()))?;
        for value in entries {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

struct SetVisitor<T>(PhantomData<fn() -> T>);

impl<'de, T> Visitor<'de> for SetVisitor<T>
where
    T: Eq + Hash + Ord + Deserialize<'de>,
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
    T: Eq + Hash + Ord + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SetVisitor(PhantomData))
    }
}
