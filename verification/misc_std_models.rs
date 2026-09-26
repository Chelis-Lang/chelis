//! Models of misc std items reached by the production core: `core::mem::take`,
//! `Default for Option<T>`, `{:X}` of `u8` (`UpperHex` and
//! `Argument::new_upper_hex`), `Debug for [T]`, `mem::discriminant` with the
//! `PartialEq` and `Debug` of `Discriminant<T>`, `Clone` and `len` of
//! `BTreeMap`, and `FnMut`/`FnOnce` for `&mut F`.
use std::collections::BTreeMap;
use std::mem;

/// `mem::take` of a `Vec`: the old value, leaving `Vec::default()`.
pub fn take_vec(mut values: Vec<u32>) -> (Vec<u32>, usize) {
    let old = mem::take(&mut values);
    (old, values.len())
}

/// `mem::take` of an `Option`, leaving `None`.
pub fn take_option(mut value: Option<u32>) -> (Option<u32>, Option<u32>) {
    let old = mem::take(&mut value);
    (old, value)
}

/// `Option::<u32>::default()`.
pub fn default_option() -> Option<u32> {
    Default::default()
}

/// `{:02X}`, `{:X}` and `{:#X}` of a byte.
pub fn upper_hex(byte: u8) -> String {
    format!("{byte:02X}|{byte:X}|{byte:#X}")
}

/// `Debug for [T]`, compact and pretty.
pub fn debug_slice(values: &[u32], pretty: bool) -> String {
    if pretty {
        format!("{values:#?}")
    } else {
        format!("{values:?}")
    }
}

pub enum Token {
    Word(u32),
    Comma,
    End,
}

/// `mem::discriminant` compared with `==`: the same variant, whatever the payload.
pub fn same_kind(a: &Token, b: &Token) -> bool {
    mem::discriminant(a) == mem::discriminant(b)
}

/// `Debug for Discriminant<Token>`.
pub fn discriminant_debug(token: &Token) -> String {
    format!("{:?}", mem::discriminant(token))
}

/// `Clone for BTreeMap`: the clone has the same entries and leaves the original intact.
pub fn cloned_map(entries: &[(u32, u32)]) -> (Vec<(u32, u32)>, usize) {
    let map: BTreeMap<u32, u32> = entries.iter().copied().collect();
    let copy = map.clone();
    (copy.into_iter().collect(), map.len())
}

fn apply<F: FnMut(u32) -> u32>(mut f: F, x: u32) -> u32 {
    f(x)
}

fn apply_once<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

/// `FnMut` and `FnOnce` for `&mut F`: calls through the reference update the
/// closure's own state.
pub fn through_mut_ref(x: u32) -> (u32, u32, u32) {
    let mut total = 0u32;
    let mut add = move |y: u32| {
        total += y;
        total
    };
    let first = apply(&mut add, x);
    let second = apply_once(&mut add, 4);
    (first, second, add(0))
}

/// `Option::<&mut u32>::expect`: monomorphized over the mutable borrow, so core's
/// body is translated; `None` reaches `core::option::expect_failed`, which panics.
pub fn expect_slot(slot: Option<&mut u32>) -> u32 {
    let value = slot.expect("a slot");
    *value += 1;
    *value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_and_default() {
        assert_eq!(take_vec(vec![1, 2]), (vec![1, 2], 0));
        assert_eq!(take_option(Some(3)), (Some(3), None));
        assert_eq!(take_option(None), (None, None));
        assert_eq!(default_option(), None);
    }

    #[test]
    fn formatting() {
        assert_eq!(upper_hex(0x0a), "0A|A|0xA");
        assert_eq!(upper_hex(0xff), "FF|FF|0xFF");
        assert_eq!(upper_hex(0), "00|0|0x0");
        assert_eq!(debug_slice(&[1, 2, 3], false), "[1, 2, 3]");
        assert_eq!(debug_slice(&[], false), "[]");
        assert_eq!(debug_slice(&[7, 8], true), "[\n    7,\n    8,\n]");
    }

    #[test]
    fn discriminants() {
        assert!(same_kind(&Token::Word(1), &Token::Word(2)));
        assert!(!same_kind(&Token::Word(1), &Token::Comma));
        assert!(same_kind(&Token::End, &Token::End));
        assert_eq!(discriminant_debug(&Token::Word(5)), "Discriminant(0)");
        assert_eq!(discriminant_debug(&Token::End), "Discriminant(2)");
    }

    #[test]
    fn closures_through_references() {
        assert_eq!(through_mut_ref(3), (3, 7, 7));
    }

    #[test]
    fn map_clone() {
        assert_eq!(cloned_map(&[(3, 30), (1, 10), (3, 31)]), (vec![(1, 10), (3, 31)], 2));
        assert_eq!(cloned_map(&[]), (vec![], 0));
    }

    #[test]
    fn expect_failed() {
        let mut slot = 3;
        assert_eq!(expect_slot(Some(&mut slot)), 4);
        assert_eq!(slot, 4);
        assert!(std::panic::catch_unwind(|| expect_slot(None)).is_err());
    }
}
