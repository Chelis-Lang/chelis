# Futhark Autotuners for Incremental Flattening

## Metadata
- **Authors:** Svend Lund Breddam
- **Venue/Year:** Master's Project, University of Copenhagen / 2019

## Summary
This thesis explores automatic tuning strategies for threshold values in Futhark's incremental flattening compilation scheme. Incremental flattening is a compiler technique that generates multiple semantically equivalent code versions, each representing different optimization strategies for exploiting parallelism on hardware. These versions are combined into a single program with predicates that select the best version at runtime based on dataset characteristics. The thesis investigates two main cases: a simple case where each threshold is compared against a single value of parallelism, and a more complex loop-based case where thresholds are compared against multiple values due to sequential loops with varying parallelism. Four different optimization strategies are implemented and evaluated: exhaustive search, binary search, evolutionary strategies (CMA-ES), active learning, and an experimental program instrumentation tuner. The best tuner achieves a 3.07x speedup over standard Futhark compilation into OpenCL code.

## Key Contributions
- Development of multiple autotuning strategies for Futhark's incremental flattening
- Implementation of a program instrumentation tuner that provides optimal results with provable guarantees
- Comprehensive evaluation across multiple hardware platforms (GTX 780 Ti, RTX 2080 Ti, Tesla K40c)
- Analysis of the trade-offs between different tuning approaches in terms of speed and result quality
- Demonstration that tuned incremental flattening provides substantial performance improvements (3.07x average speedup)

## Technical Approach
The thesis presents several tuning strategies:

1. **Simple Case Algorithm**: A recursive depth-first traversal algorithm that benchmarks each threshold against two choices (1 and ∞) to determine optimal ranges, then intersects these ranges across all datasets.

2. **Loop-Based Strategies**:
   - **Exhaustive Search**: Tests every possible value of parallelism for each threshold
   - **Binary Search**: Uses a logarithmic search strategy assuming polynomial behavior of the objective function
   - **Evolutionary Strategies (CMA-ES)**: Applies covariance matrix adaptation evolution strategies with both naive and index-based encodings
   - **Active Learning**: Uses polynomial regression to predict optimal thresholds and queries the most informative points
   - **Program Instrumentation**: Modifies the compiler to collect per-iteration runtime data, enabling complete information about the system

The instrumentation tuner is particularly notable as it provides optimal results with one benchmark per code version by collecting detailed timing information during execution.

## Results
The thesis evaluates the tuners on 11 real-world benchmarks (SRAD, LocVolCalib, BFast, Backprop, LavaMD, NN, NW, Pathfinder, Heston, OptionPricing) across three hardware platforms. Key findings include:

- The simple case tuner achieves an average 3.07x speedup over moderate flattening
- The instrumentation tuner provides the best results in the loop-based case with a 1.2x average speedup
- Different hardware platforms show varying preferences for code versions, demonstrating the value of runtime selection
- The binary search and active learning tuners provide good trade-offs between speed and accuracy
- CMA-ES approaches show inconsistent results due to the step-function nature of the objective function

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work because it addresses the critical challenge of automatically optimizing parallel code for different hardware and dataset characteristics. The techniques developed for Futhark's incremental flattening could be applied to other data-parallel languages and compilers. The instrumentation approach in particular demonstrates how compiler-level modifications can enable more effective autotuning by providing richer runtime information. The work also contributes to the broader field of program optimization by showing how machine learning techniques can be applied to compiler optimization problems, and how different optimization strategies compare in practice.
