//! Slice methods of the production crates (Aeneas.Std Slice.lean): `as_chunks::<2>` (the
//! binding pairs of chelis-ir `host`/`lower` and chelis-types `infer`), `binary_search` (the
//! sorted adjacency lists of chelis-types `infer::declarations`), `partition_point` (chelis-types
//! `unify` level transitions) and `fill` (chelis-types `infer::app_collection`, chelis-ir `host`).

/// A value whose `Clone` is not a copy: `fill` clones the value into every element but the
/// last, which receives the value itself.
#[derive(Debug, PartialEq)]
pub struct Tagged {
    pub id: u32,
    pub clones: u32,
}

impl Clone for Tagged {
    fn clone(&self) -> Self {
        Tagged { id: self.id, clones: self.clones + 1 }
    }
}

/// As chelis-types `infer::vmap_extent`: the binding pairs and the odd remainder.
pub fn pairs(kids: &[u32]) -> (Vec<(u32, u32)>, Vec<u32>) {
    let (pairs, rest) = kids.as_chunks::<2>();
    (pairs.iter().map(|[a, b]| (*a, *b)).collect(), rest.to_vec())
}

/// As chelis-ir `host`: `for pair in kids.as_chunks::<2>().0`, summing the second of each pair.
pub fn pair_seconds(kids: &[u64]) -> u64 {
    let mut total = 0;
    for pair in kids.as_chunks::<2>().0 {
        total += pair[1];
    }
    total
}

/// Chunks of three (the chunk length is a const generic).
pub fn triples(values: &[u8]) -> (usize, usize) {
    let (chunks, rest) = values.as_chunks::<3>();
    (chunks.len(), rest.len())
}

/// As chelis-types `infer::declarations`: a vertex has a self edge.
pub fn has_self_edge(adjacency: &[Vec<usize>], vertex: usize) -> bool {
    adjacency[vertex].binary_search(&vertex).is_ok()
}

/// `binary_search`, whose index among equal elements is std's algorithm's.
pub fn search(sorted: &[i32], x: i32) -> Result<usize, usize> {
    sorted.binary_search(&x)
}

/// As chelis-types `unify`: the first level whose transition is past `level`.
pub fn transition_index(levels: &[u32], level: u32) -> usize {
    levels.partition_point(|transition| *transition <= level)
}

/// `partition_point` with a predicate that counts its calls through a captured borrow.
pub fn counted_partition(values: &[u32], bound: u32, calls: &mut u32) -> usize {
    values.partition_point(|value| {
        *calls += 1;
        *value < bound
    })
}

/// As chelis-types `infer::app_collection`: `out_dims.fill(Dim::Wildcard)` on a caller's
/// buffer.
pub fn wildcard(dims: &mut [i64]) {
    dims.fill(-1);
}

/// `fill` with a non-trivial `Clone`.
pub fn fill_tagged(items: &mut Vec<Tagged>, id: u32) {
    items.fill(Tagged { id, clones: 0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks() {
        assert_eq!(pairs(&[1, 2, 3, 4, 5]), (vec![(1, 2), (3, 4)], vec![5]));
        assert_eq!(pairs(&[1, 2]), (vec![(1, 2)], vec![]));
        assert_eq!(pairs(&[]), (vec![], vec![]));
        assert_eq!(pairs(&[7]), (vec![], vec![7]));
        assert_eq!(pair_seconds(&[1, 10, 2, 20, 3]), 30);
        assert_eq!(triples(&[0; 7]), (2, 1));
        assert_eq!(triples(&[0; 6]), (2, 0));
        assert_eq!(triples(&[]), (0, 0));
    }

    #[test]
    fn searches() {
        let adjacency = vec![vec![0, 2], vec![0, 2], vec![]];
        assert!(has_self_edge(&adjacency, 0));
        assert!(!has_self_edge(&adjacency, 1));
        assert!(!has_self_edge(&adjacency, 2));
        let s = [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        assert_eq!(search(&s, 13), Ok(9));
        assert_eq!(search(&s, 4), Err(7));
        assert_eq!(search(&s, 100), Err(13));
        assert_eq!(search(&s, -1), Err(0));
        assert_eq!(search(&s, 1), Ok(4));
        assert_eq!(search(&[], 1), Err(0));
        assert_eq!(search(&[1, 1, 1, 1], 1), Ok(3));
        assert_eq!(search(&[1, 1, 1, 1, 1], 1), Ok(4));
        assert_eq!(search(&[1, 2], 1), Ok(0));
    }

    #[test]
    fn partitions() {
        assert_eq!(transition_index(&[1, 3, 3, 7], 3), 3);
        assert_eq!(transition_index(&[1, 3, 3, 7], 0), 0);
        assert_eq!(transition_index(&[1, 3, 3, 7], 9), 4);
        assert_eq!(transition_index(&[], 9), 0);
        let mut calls = 5;
        assert_eq!(counted_partition(&[1, 2, 3, 4, 5, 6, 7, 8], 6, &mut calls), 5);
        assert_eq!(calls, 9);
        let mut calls = 0;
        assert_eq!(counted_partition(&[], 6, &mut calls), 0);
        assert_eq!(calls, 0);
    }

    #[test]
    fn fills() {
        let mut dims = [3, 4, 5];
        wildcard(&mut dims);
        assert_eq!(dims, [-1, -1, -1]);
        let mut empty: [i64; 0] = [];
        wildcard(&mut empty);
        let mut items = vec![Tagged { id: 1, clones: 9 }, Tagged { id: 2, clones: 9 }, Tagged { id: 3, clones: 9 }];
        fill_tagged(&mut items, 7);
        assert_eq!(items, vec![Tagged { id: 7, clones: 1 }, Tagged { id: 7, clones: 1 }, Tagged { id: 7, clones: 0 }]);
        let mut none = vec![];
        fill_tagged(&mut none, 7);
        assert_eq!(none, vec![]);
    }
}
