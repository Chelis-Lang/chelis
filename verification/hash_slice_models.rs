//! `Hash for [T]` (Aeneas.Std Core/Hash.lean): `write_length_prefix(len)`, then the element
//! type's `Hash::hash_slice`, which the integer types override with one `write` of the slice's
//! native-order bytes, while the other types hash each element in turn (the provided method). As
//! the derived `Hash` of chelis-ir `capacity_key::CanonicalCapacity` uses it: an enum recursive
//! through `Box<[Self]>` and `Box<Self>`.
use std::hash::{Hash, Hasher};

/// A hasher that records every byte written and counts the `write` calls.
pub struct Log {
    pub bytes: Vec<u8>,
    pub writes: u32,
}

impl Hasher for Log {
    fn finish(&self) -> u64 {
        self.bytes.len() as u64 * 1000 + self.writes as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
        self.writes += 1;
    }
}

/// As `capacity_key::CapacitySource`: a derived `Hash` of a field-carrying enum.
#[derive(Clone, Copy, Hash)]
pub enum Source {
    Axis { load: u32, axis: usize },
    Scalar { node: u32 },
}

/// As `capacity_key::CanonicalCapacity`: recursive through `Box<[Self]>` and `Box<Self>`.
#[derive(Clone, Hash)]
pub enum Canonical {
    Literal(u64),
    Source(Source),
    Product(Box<[Canonical]>),
    ExactQuotient { dividend: Box<Canonical>, divisor: Box<Canonical> },
}

/// `Product` of the factors (`canonical_factors.into_boxed_slice()`).
pub fn product(factors: Vec<Canonical>) -> Canonical {
    Canonical::Product(factors.into_boxed_slice())
}

pub fn quotient(dividend: Canonical, divisor: Canonical) -> Canonical {
    Canonical::ExactQuotient { dividend: Box::new(dividend), divisor: Box::new(divisor) }
}

/// The derived `Hash` of a canonical capacity, into a caller's hasher (given back).
pub fn hash_canonical(value: &Canonical, log: &mut Log) {
    value.hash(log);
}

/// Integer slices: the length prefix, then one `write` of all the elements' bytes.
pub fn hash_ints(a: &Vec<u32>, b: &Vec<u8>, c: &Vec<i16>, d: &Vec<usize>, log: &mut Log) {
    a[..].hash(log);
    b[..].hash(log);
    c[..].hash(log);
    d[..].hash(log);
}

/// Other element types: the length prefix, then each element's own `hash`.
pub fn hash_others(a: &Vec<bool>, b: &Vec<String>, c: &Vec<Option<u32>>, log: &mut Log) {
    a[..].hash(log);
    b[..].hash(log);
    c[..].hash(log);
}

/// A slice of slices: each inner slice is hashed with its own length prefix.
pub fn hash_nested(rows: &Vec<Box<[u64]>>, log: &mut Log) {
    rows[..].hash(log);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> Log {
        Log { bytes: vec![9], writes: 1 }
    }

    fn le(n: u64, width: usize) -> Vec<u8> {
        n.to_le_bytes()[..width].to_vec()
    }

    #[test]
    fn ints() {
        let mut l = log();
        hash_ints(&vec![0x01020304, 5], &vec![7, 8, 9], &vec![-2], &vec![], &mut l);
        let mut expected = vec![9];
        expected.extend(le(2, 8));
        expected.extend([4, 3, 2, 1, 5, 0, 0, 0]);
        expected.extend(le(3, 8));
        expected.extend([7, 8, 9]);
        expected.extend(le(1, 8));
        expected.extend([0xfe, 0xff]);
        expected.extend(le(0, 8));
        assert_eq!(l.bytes, expected);
        // One write per length prefix and one per slice (also for the empty one).
        assert_eq!(l.writes, 1 + 8);
    }

    #[test]
    fn others() {
        let mut l = log();
        hash_others(
            &vec![true, false],
            &vec!["ab".to_string(), String::new()],
            &vec![None, Some(3)],
            &mut l,
        );
        let mut expected = vec![9];
        expected.extend(le(2, 8));
        expected.extend([1, 0]);
        expected.extend(le(2, 8));
        expected.extend([b'a', b'b', 0xff, 0xff]);
        expected.extend(le(2, 8));
        expected.extend(le(0, 8));
        expected.extend(le(1, 8));
        expected.extend([3, 0, 0, 0]);
        assert_eq!(l.bytes, expected);
        // bools: 2 writes; strings: 2 each (bytes, 0xff); options: discriminant (+ payload).
        assert_eq!(l.writes, 1 + 1 + 2 + 1 + 4 + 1 + 3);
    }

    #[test]
    fn nested() {
        let mut l = log();
        hash_nested(&vec![vec![1u64, 2].into_boxed_slice(), vec![].into_boxed_slice()], &mut l);
        let mut expected = vec![9];
        expected.extend(le(2, 8));
        expected.extend(le(2, 8));
        expected.extend(le(1, 8));
        expected.extend(le(2, 8));
        expected.extend(le(0, 8));
        assert_eq!(l.bytes, expected);
        assert_eq!(l.writes, 1 + 1 + 2 + 2);
    }

    #[test]
    fn canonical() {
        let axis = Canonical::Source(Source::Axis { load: 3, axis: 1 });
        let value = quotient(
            product(vec![Canonical::Literal(6), axis.clone()]),
            product(vec![Canonical::Source(Source::Scalar { node: 2 })]),
        );
        let mut l = log();
        hash_canonical(&value, &mut l);
        let mut expected = vec![9];
        expected.extend(le(3, 8)); // ExactQuotient
        expected.extend(le(2, 8)); // Product
        expected.extend(le(2, 8)); // 2 factors
        expected.extend(le(0, 8)); // Literal
        expected.extend(le(6, 8));
        expected.extend(le(1, 8)); // Source
        expected.extend(le(0, 8)); // Axis
        expected.extend(le(3, 4));
        expected.extend(le(1, 8));
        expected.extend(le(2, 8)); // Product
        expected.extend(le(1, 8)); // 1 factor
        expected.extend(le(1, 8)); // Source
        expected.extend(le(1, 8)); // Scalar
        expected.extend(le(2, 4));
        assert_eq!(l.bytes, expected);
        assert_eq!(l.writes, 1 + 14);
        let mut empty = log();
        hash_canonical(&product(vec![]), &mut empty);
        let mut expected = vec![9];
        expected.extend(le(2, 8));
        expected.extend(le(0, 8));
        assert_eq!((empty.bytes, empty.writes), (expected, 3));
    }
}
