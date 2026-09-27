//! Consumers of the production iterators and vectors (Aeneas.Std Core/Iter.lean,
//! IteratorAdapters.lean, Vec.lean): `Vec::dedup_by_key`, `Iterator::max`
//! through `FilterMap` and `FlatMap`, `Iterator::min_by_key` through `Copied`
//! and `Filter`, `Copied::product`, and the unsigned `abs_diff`.

pub struct Definition {
    pub eager: bool,
    pub item_index: u32,
    pub name: String,
}

/// As chelis-types `env` (`vars.dedup_by_key(|v| v.0)` after a sort).
pub fn dedup_names(vars: &mut Vec<(u32, u32)>) {
    vars.sort();
    vars.dedup_by_key(|v| v.0);
}

/// A key closure changing the elements it is called on (`key(current)` then
/// `key(previous kept)`), and a captured count of its calls.
pub fn dedup_marking(values: &mut Vec<(u32, u32)>) -> u32 {
    let mut calls = 0u32;
    values.dedup_by_key(|v| {
        calls += 1;
        v.1 += 1;
        v.0 / 10
    });
    calls
}

/// As chelis-deep `path::resolve_decl_index`: the last matching index.
pub fn last_match(names: &[String], wanted: &str) -> Option<usize> {
    names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| if name.as_str() == wanted { Some(index) } else { None })
        .max()
}

/// As chelis-ir `last_use`: the largest id of the blocks' operations.
pub fn highest_id(blocks: &[Vec<u32>]) -> Option<u32> {
    blocks.iter().flat_map(|block| block.iter().map(|id| *id + 1)).max()
}

/// `Iterator::max` keeps the last of equal maxima.
pub fn max_pair(pairs: &[(u32, u32)]) -> Option<(u32, u32)> {
    pairs.iter().filter_map(|p| if p.0 > 0 { Some((p.0, p.1)) } else { None }).max()
}

/// As chelis-types `deep_type::nearest_active_dtype` (`ACTIVE_DTYPE_NAMES.iter()
/// .copied().min_by_key(..)`), over names of a named lifetime (a slice of
/// `&'static str` stops Aeneas with "Symbolic borrows have no lender", reported
/// to the nested-borrows lane): the name nearest by length, the first of equal
/// minima.
pub fn nearest<'a>(names: &[&'a str], target: usize) -> Option<&'a str> {
    names.iter().copied().min_by_key(|candidate| candidate.len().abs_diff(target))
}

/// As chelis-types `agreement` (`ordered_f32(a).abs_diff(ordered_f32(b))`, and on
/// `u64`): the unsigned `abs_diff`.
pub fn distances(a: u32, b: u32, c: u64, d: u64) -> (u32, u64) {
    (a.abs_diff(b), c.abs_diff(d))
}

/// As chelis-types `declarations::eager_cycle_start`: the first eager vertex
/// by `(item_index, name)`.
pub fn eager_start(definitions: &[Definition], component: &[usize]) -> Option<usize> {
    component
        .iter()
        .copied()
        .filter(|vertex| definitions[*vertex].eager)
        .min_by_key(|vertex| {
            let definition = &definitions[*vertex];
            (definition.item_index, definition.name.clone())
        })
}

/// As chelis-types `static_value`: `shape.iter().copied().product::<usize>().max(1)`.
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(eager: bool, item_index: u32, name: &str) -> Definition {
        Definition { eager, item_index, name: name.to_string() }
    }

    #[test]
    fn dedup_by_key_keeps_the_first_of_each_run() {
        let mut vars = vec![(3, 1), (1, 2), (3, 0), (1, 1), (2, 5)];
        dedup_names(&mut vars);
        assert_eq!(vars, vec![(1, 1), (2, 5), (3, 0)]);
        let mut single = vec![(7, 7)];
        dedup_names(&mut single);
        assert_eq!(single, vec![(7, 7)]);
    }

    #[test]
    fn dedup_by_key_changes_what_the_key_sees() {
        let mut values = vec![(11, 0), (12, 0), (25, 0), (21, 0), (30, 0)];
        assert_eq!(dedup_marking(&mut values), 8);
        assert_eq!(values, vec![(11, 2), (25, 3), (30, 1)]);
        let mut one = vec![(1, 0)];
        assert_eq!(dedup_marking(&mut one), 0);
        assert_eq!(one, vec![(1, 0)]);
    }

    #[test]
    fn max_through_adapters() {
        let names: Vec<String> = ["a", "b", "a", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(last_match(&names, "a"), Some(2));
        assert_eq!(last_match(&names, "z"), None);
        assert_eq!(highest_id(&[vec![3, 9], vec![], vec![4]]), Some(10));
        assert_eq!(highest_id(&[vec![], vec![]]), None);
        assert_eq!(max_pair(&[(2, 1), (0, 9), (2, 0), (1, 5)]), Some((2, 1)));
    }

    #[test]
    fn min_by_key_through_adapters() {
        assert_eq!(nearest(&["f32", "f64", "bf16", "i8"], 3), Some("f32"));
        assert_eq!(nearest(&["float64", "i8", "u8"], 1), Some("i8"));
        assert_eq!(nearest(&[], 1), None);
        let defs = vec![def(true, 4, "b"), def(false, 0, "a"), def(true, 4, "a"), def(true, 9, "c")];
        assert_eq!(eager_start(&defs, &[0, 1, 2, 3]), Some(2));
        assert_eq!(eager_start(&defs, &[1]), None);
    }

    #[test]
    fn unsigned_abs_diff() {
        assert_eq!(distances(3, 10, u64::MAX, 0), (7, u64::MAX));
        assert_eq!(distances(10, 3, 5, 5), (7, 0));
        assert_eq!(distances(0, u32::MAX, 0, 1), (u32::MAX, 1));
    }

    #[test]
    fn copied_product() {
        assert_eq!(element_count(&[2, 3, 4]), 24);
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[5, 0]), 1);
    }
}
