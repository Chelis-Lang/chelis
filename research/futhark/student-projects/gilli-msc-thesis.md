# Extending automatic differentiation for an array language with nested parallelism

## Metadata
- **Authors:** Gilli Reynstind Fjallstein
- **Venue/Year:** Master's thesis, May 30, 2022

## Summary
This thesis explores automatic differentiation (AD) for functional languages, specifically focusing on the data-parallel language Futhark. The main contribution is deriving and implementing a rewrite rule for the `reduce_by_index` (RBI) construct in reverse mode AD, along with more efficient special cases for addition, multiplication, and min/max operations. The implementation is validated and benchmarked against the forward mode AD, showing that while the general reverse mode case is less efficient, the special cases significantly outperform both the general reverse mode and forward mode implementations.

## Key Contributions
- Derivation of a rewrite rule for reverse mode AD of the `reduce_by_index` construct
- Implementation of the rewrite rule in the Futhark compiler
- Development of special case optimizations for addition, multiplication, and min/max operations
- Benchmarking and validation of the implementations

## Technical Approach
The thesis begins by introducing the work-span model for reasoning about parallel algorithms, functional programming concepts, and the Futhark language. It then provides a thorough introduction to automatic differentiation, covering both forward and reverse modes. The core technical approach involves:

1. Deriving a rewrite rule for the `reduce` construct in reverse mode AD
2. Extending this rule to handle the `reduce_by_index` construct
3. Implementing the rewrite rule in the Futhark compiler using its intermediate representation (IR)
4. Developing special case optimizations for common operations
5. Validating the implementation through testing and benchmarking

## Results
The benchmarks show that while the general reverse mode AD implementation for `reduce_by_index` is less efficient than the forward mode, the special case optimizations for addition, multiplication, and min/max operations significantly outperform both. The multiplication special case, in particular, shows up to 17% speedup when the input contains zeros. The thesis also provides a detailed analysis of the AD overhead for different cases.

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. It extends the capabilities of the Futhark language by adding support for reverse mode AD of the `reduce_by_index` construct, which is not supported by other AD libraries.
2. The special case optimizations demonstrate how significant performance improvements can be achieved by deriving more efficient rewrite rules for common operations.
3. The implementation provides a proof of concept for supporting reverse mode AD in functional languages with nested parallelism, which could be applied to other languages and systems.
4. The benchmarks and analysis provide valuable insights into the performance characteristics of different AD approaches, which can inform future language and compiler design decisions.
