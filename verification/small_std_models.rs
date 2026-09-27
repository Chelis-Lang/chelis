//! Small std impls used once each by chelis-ir (Aeneas.Std Scalar/CoreConvertNum.lean,
//! Scalar/Ops/Add.lean, Scalar/Fmt.lean, Core/FmtWithSlice.lean, IteratorSources.lean):
//! `u32::try_from(usize)` (`StorageId::from_slot`), `usize += &usize`, `{:x}` of `u8`, `Debug`
//! of a triple, and `iter::empty()` consumed through a generic `&mut impl Iterator`.

/// As `StorageId::from_slot`: the slot index as an opaque `u32` key, or the error.
pub fn from_slot(index: usize) -> Result<u32, String> {
    let key = u32::try_from(index).map_err(|e| format!("slot {index} does not fit: {e:?}"))?;
    Ok(key)
}

/// A running total over borrowed counts.
pub fn total(counts: &Vec<usize>) -> usize {
    let mut sum = 0usize;
    for count in counts {
        sum += count;
    }
    sum
}

pub fn hex_bytes(bytes: &Vec<u8>) -> String {
    let mut out = String::new();
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out.push_str(&format!("|{:x}|{:#x}|{:>4x}|{:#06x}", 10u8, 255u8, 7u8, 171u8));
    out
}

pub fn show_triple(a: u32, b: bool, c: &str) -> String {
    let t = (a, b, c.to_string());
    format!("{t:?}|{t:#?}")
}

/// As `pack_grad_roots`: consumes up to `n` items of a caller's iterator.
fn take_roots<I: Iterator<Item = u32>>(n: usize, roots: &mut I) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n {
        match roots.next() {
            Some(root) => out.push(root),
            None => break,
        }
    }
    out
}

pub fn empty_roots(n: usize) -> (Vec<u32>, (usize, Option<usize>)) {
    let mut roots = std::iter::empty();
    let taken = take_roots(n, &mut roots);
    (taken, roots.size_hint())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots() {
        assert_eq!(from_slot(7), Ok(7));
        assert_eq!(from_slot(u32::MAX as usize), Ok(u32::MAX));
        assert_eq!(
            from_slot(u32::MAX as usize + 1),
            Err("slot 4294967296 does not fit: TryFromIntError(())".to_string())
        );
    }

    #[test]
    fn totals() {
        assert_eq!(total(&vec![1, 2, 3]), 6);
        assert_eq!(total(&vec![]), 0);
    }

    #[test]
    #[should_panic]
    fn total_overflows() {
        total(&vec![usize::MAX, 1]);
    }

    #[test]
    fn hexes() {
        assert_eq!(hex_bytes(&vec![0, 15, 16, 255]), "000f10ff|a|0xff|   7|0x00ab");
    }

    #[test]
    fn triples() {
        assert_eq!(
            show_triple(3, true, "x\"y"),
            "(3, true, \"x\\\"y\")|(\n    3,\n    true,\n    \"x\\\"y\",\n)"
        );
    }

    #[test]
    fn empties() {
        assert_eq!(empty_roots(3), (vec![], (0, Some(0))));
        assert_eq!(empty_roots(0), (vec![], (0, Some(0))));
    }
}
