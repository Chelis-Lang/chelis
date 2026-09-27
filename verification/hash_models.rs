//! The `Hash` impls that the production crates' `#[derive(Hash)]` expansions
//! call (Aeneas.Std Core/Hash.lean): integers, `bool`, `char`, `str`, `&T`,
//! `String` and `Option<T>`, and the provided `Hasher::write_*` methods they
//! reach, including an implementation that overrides one of them. The integer
//! methods write `to_ne_bytes`, which the model defines as little-endian: the
//! platform assumption `Platform.littleEndian` of verification/trusted_base.json,
//! checked here against the build host (aarch64).
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

/// A hasher that overrides the provided `write_u32` (as `FxHasher` overrides
/// the integer writes): the derived impls must dispatch to the override.
pub struct Tagged {
    pub bytes: Vec<u8>,
}

impl Hasher for Tagged {
    fn finish(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn write_u32(&mut self, i: u32) {
        self.bytes.push(0xAA);
        self.bytes.push(i as u8);
    }
}

/// A fieldless enum, as chelis-deep `dtype_bounds::DtypeFamily`: the derive
/// hashes the discriminant as an `isize`.
#[derive(Hash)]
pub enum Family {
    Float,
    Int,
    Bool,
}

/// An enum with payloads of every modeled `Hash` impl.
#[derive(Hash)]
pub enum Key {
    Named(String),
    Index(u32),
    Pair(i64, usize),
    Maybe(Option<u32>),
    Flag(bool),
    Letter(char),
    Small(u8, i8, u16, i16),
    Wide(u128, i128, isize),
}

/// A struct with a borrowed `str` field (`Hash for &T`).
#[derive(Hash)]
pub struct Site<'a> {
    pub name: &'a str,
    pub line: u64,
    pub delta: i32,
}

pub fn log_of<T: Hash>(x: &T) -> (Vec<u8>, u32, u64) {
    let mut state = Log { bytes: Vec::new(), writes: 0 };
    x.hash(&mut state);
    let finish = state.finish();
    (state.bytes, state.writes, finish)
}

pub fn tagged_of<T: Hash>(x: &T) -> Vec<u8> {
    let mut state = Tagged { bytes: Vec::new() };
    x.hash(&mut state);
    state.bytes
}

pub fn family_log(family: Family) -> (Vec<u8>, u32, u64) {
    log_of(&family)
}

pub fn key_log(key: &Key) -> (Vec<u8>, u32, u64) {
    log_of(key)
}

pub fn key_tagged(key: &Key) -> Vec<u8> {
    tagged_of(key)
}

pub fn site_log(name: &str, line: u64, delta: i32) -> (Vec<u8>, u32, u64) {
    log_of(&Site { name, line, delta })
}

/// `Hash for &T` and `Hash for str`, called directly.
pub fn str_log(text: &str) -> (Vec<u8>, u32, u64) {
    log_of(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_host() {
        // The platform assumption: `to_ne_bytes` is `to_le_bytes`.
        assert!(cfg!(target_endian = "little"));
        assert_eq!(0x0102_0304u32.to_ne_bytes(), [4, 3, 2, 1]);
    }

    #[test]
    fn fieldless_enum() {
        assert_eq!(family_log(Family::Float), (vec![0, 0, 0, 0, 0, 0, 0, 0], 1, 8001));
        assert_eq!(family_log(Family::Bool), (vec![2, 0, 0, 0, 0, 0, 0, 0], 1, 8001));
    }

    #[test]
    fn payloads() {
        let named = key_log(&Key::Named("ab".to_string()));
        assert_eq!(named, (vec![0, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', 0xff], 3, 11003));
        assert_eq!(
            key_log(&Key::Index(0x0102_0304)),
            (vec![1, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1], 2, 12002)
        );
        assert_eq!(
            key_log(&Key::Pair(-2, 258)).0,
            vec![2, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255, 2, 1, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            key_log(&Key::Maybe(None)).0,
            vec![3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            key_log(&Key::Maybe(Some(7))).0,
            vec![3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0]
        );
        assert_eq!(key_log(&Key::Flag(true)).0, vec![4, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(key_log(&Key::Letter('é')).0, vec![5, 0, 0, 0, 0, 0, 0, 0, 0xe9, 0, 0, 0]);
        assert_eq!(
            key_log(&Key::Small(0x81, -1, 0x0102, -2)).0,
            vec![6, 0, 0, 0, 0, 0, 0, 0, 0x81, 0xff, 2, 1, 0xfe, 0xff]
        );
        let wide = key_log(&Key::Wide(1 << 120, -1, -3)).0;
        let mut expected = vec![7, 0, 0, 0, 0, 0, 0, 0];
        expected.extend_from_slice(&(1u128 << 120).to_le_bytes());
        expected.extend_from_slice(&[0xff; 16]);
        expected.extend_from_slice(&[0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(wide, expected);
    }

    #[test]
    fn borrowed_str_field() {
        assert_eq!(
            site_log("f", 3, -1),
            (vec![b'f', 0xff, 3, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff], 4, 14004)
        );
        assert_eq!(str_log(""), (vec![0xff], 2, 1002));
        assert_eq!(str_log("é"), (vec![0xc3, 0xa9, 0xff], 2, 3002));
    }

    #[test]
    fn overridden_write_u32() {
        assert_eq!(key_tagged(&Key::Index(0x0102_0304)), vec![1, 0, 0, 0, 0, 0, 0, 0, 0xAA, 4]);
        assert_eq!(key_tagged(&Key::Letter('A')), vec![5, 0, 0, 0, 0, 0, 0, 0, 0xAA, b'A']);
        assert_eq!(key_tagged(&Key::Maybe(Some(9))), vec![3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0xAA, 9]);
    }
}
