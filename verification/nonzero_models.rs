//! The `NonZero` trait impls that the derives of chelis-types `unsupported::IssueRef(NonZeroU32)`
//! call (Aeneas.Std Core/NonZeroTraits.lean): `PartialEq` (`eq` and `ne`: `self.get() ==
//! other.get()`), `Hash` (`self.get().hash(state)`) and `Debug` (`self.get().fmt(f)`, the
//! primitive's formatting options included).
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;

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

/// As chelis-types `unsupported::IssueRef`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IssueRef(NonZeroU32);

/// As `IssueRef::new`: `None` for zero.
pub fn issue(number: u32) -> Option<IssueRef> {
    NonZeroU32::new(number).map(IssueRef)
}

/// The derived `eq` of two issues, and `NonZero`'s own `ne`.
pub fn compare(a: u32, b: u32) -> Option<(bool, bool)> {
    let (a, b) = (issue(a)?, issue(b)?);
    Some((a == b, a.0 != b.0))
}

/// The derived `Hash` of an issue, into a caller's hasher (given back).
pub fn hash_issue(number: u32, log: &mut Log) -> bool {
    match issue(number) {
        Some(issue) => {
            issue.hash(log);
            true
        }
        None => false,
    }
}

/// The derived `Debug` of an issue, and a padded `Debug` of the `NonZero` itself.
pub fn show(number: u32) -> Option<String> {
    let issue = issue(number)?;
    Some(format!("{issue:?}|{:>5?}|{:#?}", issue.0, issue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equalities() {
        assert_eq!(compare(7, 7), Some((true, false)));
        assert_eq!(compare(7, 8), Some((false, true)));
        assert_eq!(compare(0, 8), None);
    }

    #[test]
    fn hashes() {
        let mut log = Log { bytes: vec![9], writes: 1 };
        assert!(hash_issue(0x0102_0304, &mut log));
        assert_eq!(log.bytes, vec![9, 4, 3, 2, 1]);
        assert_eq!(log.writes, 2);
        assert!(!hash_issue(0, &mut log));
        assert_eq!(log.writes, 2);
    }

    #[test]
    fn debug() {
        assert_eq!(show(42), Some("IssueRef(42)|   42|IssueRef(\n    42,\n)".to_string()));
        assert_eq!(show(0), None);
    }
}
