//! Kernel launch configuration (grid size, block size).

/// Default block size for elementwise operations.
pub const BLOCK_SIZE: usize = 256;
pub const SMALL_SEGMENT_MAX: usize = 64;
pub const TINY_SEGMENT_MAX: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentedStrategy {
    Tiny,
    Small,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentedLaunch {
    pub strategy: SegmentedStrategy,
    pub grid: usize,
    pub block: usize,
    pub threads_per_segment: usize,
    pub segments_per_block: usize,
}

/// Compute grid dimensions for a 1D launch over `total_elements` threads.
#[must_use]
pub fn grid_1d(total_elements: usize) -> (usize, usize) {
    let block = BLOCK_SIZE;
    let grid = total_elements.div_ceil(block);
    (grid, block)
}

#[must_use]
pub fn floor_pow2(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        1usize << (usize::BITS as usize - 1 - value.leading_zeros() as usize)
    }
}

#[must_use]
pub fn reduction_block_size(axis_size: usize) -> usize {
    floor_pow2(axis_size.clamp(1, BLOCK_SIZE))
}

#[must_use]
pub fn segmented_strategy(axis_size: usize) -> SegmentedStrategy {
    if axis_size <= TINY_SEGMENT_MAX {
        SegmentedStrategy::Tiny
    } else if axis_size <= SMALL_SEGMENT_MAX {
        SegmentedStrategy::Small
    } else {
        SegmentedStrategy::Large
    }
}

#[must_use]
pub fn segmented_launch(out_size: usize, axis_size: usize) -> SegmentedLaunch {
    match segmented_strategy(axis_size) {
        SegmentedStrategy::Tiny => SegmentedLaunch {
            strategy: SegmentedStrategy::Tiny,
            grid: out_size.div_ceil(BLOCK_SIZE),
            block: BLOCK_SIZE,
            threads_per_segment: 1,
            segments_per_block: BLOCK_SIZE,
        },
        SegmentedStrategy::Small => {
            let threads_per_segment =
                floor_pow2(axis_size.next_power_of_two().clamp(1, SMALL_SEGMENT_MAX));
            let segments_per_block = (BLOCK_SIZE / threads_per_segment).max(1);
            let grid = out_size.div_ceil(segments_per_block);
            SegmentedLaunch {
                strategy: SegmentedStrategy::Small,
                grid,
                block: BLOCK_SIZE,
                threads_per_segment,
                segments_per_block,
            }
        }
        SegmentedStrategy::Large => SegmentedLaunch {
            strategy: SegmentedStrategy::Large,
            grid: out_size,
            block: reduction_block_size(axis_size),
            threads_per_segment: reduction_block_size(axis_size),
            segments_per_block: 1,
        },
    }
}

#[must_use]
pub fn staged_partial_count(total_elements: usize, block_size: usize) -> usize {
    total_elements.div_ceil(block_size.max(1))
}

#[must_use]
pub fn staged_chain_partial_counts(total_elements: usize, block_size: usize) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut current = total_elements;
    let block = block_size.max(1);
    loop {
        let partials = staged_partial_count(current, block);
        counts.push(partials);
        if partials <= 1 {
            break;
        }
        current = partials;
    }
    counts
}

#[must_use]
pub fn staged_chain_scratch_elements(total_elements: usize, block_size: usize) -> usize {
    let counts = staged_chain_partial_counts(total_elements, block_size);
    counts
        .into_iter()
        .take_while(|&partials| partials > 1)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_exact_multiple() {
        let (grid, block) = grid_1d(512);
        assert_eq!(block, 256);
        assert_eq!(grid, 2);
    }

    #[test]
    fn grid_not_exact() {
        let (grid, _) = grid_1d(257);
        assert_eq!(grid, 2);
    }

    #[test]
    fn grid_single_element() {
        let (grid, _) = grid_1d(1);
        assert_eq!(grid, 1);
    }

    #[test]
    fn reduction_block_is_power_of_two() {
        assert_eq!(reduction_block_size(300), 256);
        assert_eq!(reduction_block_size(63), 32);
        assert_eq!(reduction_block_size(1), 1);
    }

    #[test]
    fn segmented_thresholds_match_plan() {
        assert_eq!(segmented_strategy(8), SegmentedStrategy::Tiny);
        assert_eq!(segmented_strategy(9), SegmentedStrategy::Small);
        assert_eq!(segmented_strategy(64), SegmentedStrategy::Small);
        assert_eq!(segmented_strategy(65), SegmentedStrategy::Large);
    }

    #[test]
    fn segmented_small_launch_batches_segments() {
        let launch = segmented_launch(17, 16);
        assert_eq!(launch.strategy, SegmentedStrategy::Small);
        assert_eq!(launch.threads_per_segment, 16);
        assert_eq!(launch.segments_per_block, 16);
        assert_eq!(launch.grid, 2);
    }

    #[test]
    fn staged_chain_sums_intermediate_partials_only() {
        assert_eq!(staged_chain_partial_counts(1024, 256), vec![4, 1]);
        assert_eq!(staged_chain_scratch_elements(1024, 256), 4);
    }
}
