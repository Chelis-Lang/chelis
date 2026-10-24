//! Fixed-hash tables for private graph lookup.
//!
//! Public map contracts retain `chelis_unord`. Ordered exits sort keys explicitly.

use alloc::vec::Vec;
use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasherDefault, Hash};
use core::ops::Index;

pub type FixedBuildHasher = BuildHasherDefault<rustc_hash::FxHasher>;

#[derive(Clone)]
pub struct HashMap<K, V>(hashbrown::HashMap<K, V, FixedBuildHasher>);

impl<K, V> Default for HashMap<K, V> {
    fn default() -> Self {
        Self(hashbrown::HashMap::with_hasher(FixedBuildHasher::default()))
    }
}

impl<K, V> HashMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(hashbrown::HashMap::with_capacity_and_hasher(
            capacity,
            FixedBuildHasher::default(),
        ))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear()
    }
}

impl<K: Eq + Hash, V> HashMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.0.insert(key, value)
    }
    pub fn get<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.0.get(key)
    }
    pub fn get_mut<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.0.get_mut(key)
    }
    pub fn contains_key<Q: Eq + Hash + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.contains_key(key)
    }
    pub fn remove<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.0.remove(key)
    }
    pub fn entry(&mut self, key: K) -> hashbrown::hash_map::Entry<'_, K, V, FixedBuildHasher> {
        self.0.entry(key)
    }
    pub fn merge(&mut self, other: Self) {
        self.0.extend(other.0)
    }
}

impl<K: Ord, V> HashMap<K, V> {
    pub fn to_sorted(&self) -> Vec<(&K, &V)> {
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        entries
    }
    pub fn into_sorted(self) -> Vec<(K, V)> {
        let mut entries: Vec<_> = self.0.into_iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        entries
    }
}

impl<K, V> fmt::Debug for HashMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashMap {{ len: {} }}", self.len())
    }
}

impl<K: Eq + Hash, V: PartialEq> PartialEq for HashMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K: Eq + Hash, V: Eq> Eq for HashMap<K, V> {}
impl<K: Eq + Hash, V> Extend<(K, V)> for HashMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, values: I) {
        self.0.extend(values)
    }
}
impl<K: Eq + Hash, V> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(values: I) -> Self {
        let mut map = Self::new();
        map.extend(values);
        map
    }
}
impl<K: Eq + Hash, V, const N: usize> From<[(K, V); N]> for HashMap<K, V> {
    fn from(values: [(K, V); N]) -> Self {
        values.into_iter().collect()
    }
}
impl<K, Q, V> Index<&Q> for HashMap<K, V>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    type Output = V;
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

#[derive(Clone)]
pub struct HashSet<T>(hashbrown::HashSet<T, FixedBuildHasher>);

impl<T> Default for HashSet<T> {
    fn default() -> Self {
        Self(hashbrown::HashSet::with_hasher(FixedBuildHasher::default()))
    }
}
impl<T> HashSet<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear()
    }
}
impl<T: Eq + Hash> HashSet<T> {
    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }
    pub fn contains<Q: Eq + Hash + ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.0.contains(value)
    }
    pub fn remove<Q: Eq + Hash + ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.0.remove(value)
    }
    pub fn merge(&mut self, other: Self) {
        self.0.extend(other.0)
    }
}
impl<T: Ord> HashSet<T> {
    pub fn to_sorted(&self) -> Vec<&T> {
        let mut values: Vec<_> = self.0.iter().collect();
        values.sort_unstable();
        values
    }
    pub fn into_sorted(self) -> Vec<T> {
        let mut values: Vec<_> = self.0.into_iter().collect();
        values.sort_unstable();
        values
    }
}
impl<T> fmt::Debug for HashSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashSet {{ len: {} }}", self.len())
    }
}
impl<T: Eq + Hash> PartialEq for HashSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: Eq + Hash> Eq for HashSet<T> {}
impl<T: Eq + Hash> Extend<T> for HashSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        self.0.extend(values)
    }
}
impl<T: Eq + Hash> FromIterator<T> for HashSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let mut set = Self::new();
        set.extend(values);
        set
    }
}
impl<T: Eq + Hash, const N: usize> From<[T; N]> for HashSet<T> {
    fn from(values: [T; N]) -> Self {
        values.into_iter().collect()
    }
}
