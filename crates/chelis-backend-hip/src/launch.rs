//! Kernel launch configuration (grid size, block size).

/// Default block size for elementwise operations.
pub const BLOCK_SIZE: usize = 256;

/// Compute grid dimensions for a 1D launch over `total_elements` threads.
pub fn grid_1d(total_elements: usize) -> (usize, usize) {
    let block = BLOCK_SIZE;
    let grid = total_elements.div_ceil(block);
    (grid, block)
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
}
