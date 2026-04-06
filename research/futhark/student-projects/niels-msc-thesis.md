# Memory Block Merging in Futhark

## Metadata
- **Authors:** Niels G. W. Serup
- **Venue/Year:** Master's Thesis, University of Copenhagen / 2017

## Summary
This thesis presents two memory optimization techniques for the Futhark programming language, which is designed for efficient GPU code generation. The optimizations aim to reduce memory footprint and data copying overhead by merging memory blocks used by arrays. The author implements and evaluates these optimizations on a suite of more than 30 Futhark benchmark programs, showing average memory footprint reductions ranging from 0% to 70%, but also observes runtime regressions ranging from -28% to +16%.

## Key Contributions
- Introduces memory block coalescing and memory block reuse optimizations for Futhark
- Implements enabling analyses (memory block aliases, last use analysis, interference analysis)
- Develops enabling transformations (allocation hoisting, allocation size hoisting)
- Provides formal and informal descriptions of the core transformations
- Evaluates optimizations on 30+ benchmark programs with quantitative results

## Technical Approach
The optimizations operate on Futhark's EXPLICITMEMORY intermediate representation where arrays are explicitly associated with memory blocks. The core transformations are:

**Memory Block Coalescing**: Allows an array to use the memory block of another array when certain safety conditions are met (last use, allocation order, no interference, etc.). This handles copy, concat, and in-place update patterns.

**Memory Block Reuse**: Allows an array to reuse a previously allocated memory block if the blocks don't interfere and have compatible sizes (using a "max trick" to handle size differences).

The enabling analyses determine liveness intervals, interferences, and aliases between memory blocks. The enabling transformations hoist allocations and allocation sizes to enable more merging opportunities.

## Results
The optimizations were evaluated on 30+ benchmark programs with these key findings:
- **Memory footprint**: Average reductions from 0% to 70% across benchmarks
- **Runtime**: Mixed results with average reductions from -28% to +16%
- **Validation**: All 95 memory-specific tests and 765 general tests passed
- **Best case**: Canny benchmark achieved 70% memory reduction
- **Worst case**: Some benchmarks showed runtime regressions up to 28%

## Relevance
This work is highly relevant to language design, compilers, and ML systems because it addresses fundamental challenges in memory management for GPU programming. The techniques bridge the gap between functional programming's abstraction benefits and the low-level memory control needed for efficient GPU execution. The approach of lifting register allocation concepts to array memory management provides a novel perspective that could influence future compiler designs for array programming languages and ML systems targeting heterogeneous hardware.
