//! The post-dominator tree of one unit's expanded scheduling CFG.
//!
//! A single-placement death frontier must post-dominate its owner's
//! definition. A search from the definition that stops at the frontier
//! answers that, but it walks the owner's whole live region, once per owner,
//! so a body whose owners all stay live to its end is quadratic
//! (chelis#3755). [`PostDominators`] is built once per CFG and answers each
//! query by an ancestor test in the tree.
//!
//! A point post-dominates another when every path from the other to `Exit`
//! passes through it. A point with no path to `Exit` (inside a nonterminating
//! component) is post-dominated by every point of the CFG, which is what the
//! search answers for it.

use std::collections::{BTreeMap, BTreeSet};

use super::{SchedulePoint, lookup_visited};

#[derive(Debug)]
pub(super) struct PostDominators {
    index: BTreeMap<SchedulePoint, usize>,
    /// For each point with a path to `Exit`, its pre-order entry and exit
    /// times in the post-dominator tree rooted at `Exit`.
    span: Vec<Option<(usize, usize)>>,
}

impl PostDominators {
    /// The tree over `successors` and its inverse `predecessors`, which name
    /// the same points; `Exit` is one of them.
    pub(super) fn new(
        successors: &BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
        predecessors: &BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
    ) -> Self {
        let index = successors
            .keys()
            .enumerate()
            .map(|(position, point)| (*point, position))
            .collect::<BTreeMap<_, _>>();
        let neighbours = |edges: &BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>| {
            successors
                .keys()
                .map(|point| {
                    edges
                        .get(point)
                        .into_iter()
                        .flatten()
                        .filter_map(|next| index.get(next).copied())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let forward = neighbours(successors);
        let backward = neighbours(predecessors);
        let count = index.len();
        lookup_visited(count + forward.iter().map(Vec::len).sum::<usize>());
        let mut span = vec![None; count];
        let Some(&exit) = index.get(&SchedulePoint::Exit) else {
            return Self { index, span };
        };

        // Post-order of the reverse CFG from `Exit`: the points that reach it.
        let mut postorder = vec![usize::MAX; count];
        let mut order = Vec::with_capacity(count);
        let mut seen = vec![false; count];
        seen[exit] = true;
        let mut stack = vec![(exit, 0)];
        while let Some((point, next)) = stack.last_mut() {
            if let Some(&previous) = backward[*point].get(*next) {
                *next += 1;
                if !seen[previous] {
                    seen[previous] = true;
                    stack.push((previous, 0));
                }
            } else {
                postorder[*point] = order.len();
                order.push(*point);
                stack.pop();
            }
        }

        // The iterative dominator algorithm of Cooper, Harvey and Kennedy on
        // the reverse CFG: a point's immediate post-dominator is the common
        // tree ancestor of its successors that reach `Exit`.
        let mut idom = vec![None; count];
        idom[exit] = Some(exit);
        let intersect = |idom: &[Option<usize>], mut left: usize, mut right: usize| {
            while left != right {
                while postorder[left] < postorder[right] {
                    left = idom[left].expect("a processed point has a post-dominator");
                }
                while postorder[right] < postorder[left] {
                    right = idom[right].expect("a processed point has a post-dominator");
                }
            }
            left
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &point in order.iter().rev() {
                if point == exit {
                    continue;
                }
                lookup_visited(forward[point].len());
                let mut candidate = None;
                for &next in &forward[point] {
                    if idom[next].is_none() {
                        continue;
                    }
                    candidate = Some(match candidate {
                        None => next,
                        Some(current) => intersect(&idom, next, current),
                    });
                }
                if candidate.is_some() && idom[point] != candidate {
                    idom[point] = candidate;
                    changed = true;
                }
            }
        }

        // Pre-order entry and exit times, so ancestry is interval nesting.
        let mut children = vec![Vec::new(); count];
        for &point in &order {
            if point != exit
                && let Some(parent) = idom[point]
            {
                children[parent].push(point);
            }
        }
        let mut clock = 0;
        let mut entered = vec![0; count];
        let mut stack = vec![(exit, 0)];
        entered[exit] = clock;
        clock += 1;
        while let Some((point, next)) = stack.last_mut() {
            if let Some(&child) = children[*point].get(*next) {
                *next += 1;
                entered[child] = clock;
                clock += 1;
                stack.push((child, 0));
            } else {
                span[*point] = Some((entered[*point], clock));
                stack.pop();
            }
        }
        Self { index, span }
    }

    /// Every path from `point` to `Exit` passes through `candidate`; false
    /// when either is not a point of the CFG.
    pub(super) fn postdominates(&self, candidate: SchedulePoint, point: SchedulePoint) -> bool {
        lookup_visited(1);
        let (Some(&candidate), Some(&point)) = (self.index.get(&candidate), self.index.get(&point))
        else {
            return false;
        };
        if candidate == point {
            return true;
        }
        match (self.span[candidate], self.span[point]) {
            (_, None) => true,
            (None, Some(_)) => false,
            (Some((enter, exit)), Some((point_enter, point_exit))) => {
                enter <= point_enter && point_exit <= exit
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::super::super::ir::BlockId;
    use super::super::SchedulePoint;
    use super::super::reference::point_postdominates;
    use super::PostDominators;

    /// chelis#3755: the tree answers every query the path search answers, on
    /// random graphs with branches, back edges, irreducible loops, points with
    /// no path to `Exit`, and points no other point reaches.
    #[test]
    fn post_dominator_tree_agrees_with_the_path_search() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u32::try_from(state % u64::from(bound)).unwrap()
        };
        for _ in 0..256 {
            let count = 1 + next(24);
            let point = |index: u32| {
                if index == count {
                    SchedulePoint::Exit
                } else {
                    SchedulePoint::BlockEntry(BlockId(index))
                }
            };
            let mut successors = (0..=count)
                .map(|index| (point(index), BTreeSet::new()))
                .collect::<BTreeMap<_, _>>();
            for index in 0..count {
                for _ in 0..next(4) {
                    successors
                        .get_mut(&point(index))
                        .unwrap()
                        .insert(point(next(count + 1)));
                }
            }
            let mut predecessors = successors
                .keys()
                .map(|point| (*point, BTreeSet::new()))
                .collect::<BTreeMap<_, _>>();
            for (&from, targets) in &successors {
                for target in targets {
                    predecessors.get_mut(target).unwrap().insert(from);
                }
            }
            let tree = PostDominators::new(&successors, &predecessors);
            let outside = SchedulePoint::BlockEntry(BlockId(count + 1));
            let queried = (0..=count).map(point).chain([outside]).collect::<Vec<_>>();
            for &candidate in &queried {
                for &target in &queried {
                    assert_eq!(
                        tree.postdominates(candidate, target),
                        point_postdominates(&successors, candidate, target),
                        "{candidate:?} over {target:?} in {successors:?}"
                    );
                }
            }
        }
    }
}
