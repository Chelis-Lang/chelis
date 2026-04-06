# Empirically comparing high level and low level AD

## Metadata
- **Authors:** Shatin Nguyen, Theis Baasch, Oliver Raaschou
- **Venue/Year:** Bachelor Thesis, University of Copenhagen, 2025

## Summary
This thesis presents an empirical comparison of high-level and low-level automatic differentiation (AD) systems using Futhark's native AD and Enzyme AD applied to Futhark-generated LLVM IR. The authors systematically benchmark eight different programs across various computational patterns to understand when each approach performs better. Their results reveal that the performance trade-off is dictated by workload characteristics: high-level AD excels on memory-bound computations with large matrix operations (up to 10x faster), while low-level AD performs better on computationally simple programs. The study demonstrates that high-level AD's ability to perform semantic transformations provides significant performance advantages for a wide range of programs, particularly those involving large-scale array operations.

## Key Contributions
- First systematic empirical comparison of high-level vs low-level AD performance
- Identification of three distinct performance categories based on computational patterns
- Demonstration that high-level AD's semantic transformations provide up to 10x performance advantage for memory-bound workloads
- Analysis of why low-level AD suffers from performance cliffs at specific problem scales
- Practical insights into the challenges of applying low-level AD to GPU code

## Technical Approach
The authors compared Futhark's built-in AD system (high-level) against Enzyme AD (low-level) applied to Futhark-generated LLVM IR. They selected eight benchmarks from gradBench and Futhark's AdBench covering diverse computational patterns: dot product, matrix multiplication, GMM, K-means, hand tracking, bundle adjustment, LSTM, and linear least squares. For each benchmark, they applied both AD approaches and measured runtime using Futhark's benchmark tool, which provides mean runtime with 95% confidence intervals. They also analyzed LLVM IR and used perf to understand performance characteristics. The Enzyme integration required manual modification of Futhark-generated C code to interface with Enzyme's C API, including handling allocation functions and specifying differentiation attributes.

## Results
The results show a clear performance trade-off based on computational patterns:
- **Memory-bound workloads** (GMM, LSTM): High-level AD dominates, with speedup ratios of 0.83x (1.20x faster) for GMM and 0.79x (1.27x faster) for LSTM
- **Many-small-tasks workloads** (Bundle Adjustment): High-level AD consistently faster by 1.5-1.7x
- **Computationally simple programs** (dot product): High-level AD faster by 4-16x
- **Forward-mode workloads** (hand tracking): Complex pattern with performance cliffs for low-level AD
- **Scalar-dominated workloads** (linear least squares): Nearly identical performance (0.98x speedup)

The analysis reveals that high-level AD's advantage comes from its ability to perform semantic transformations that improve memory locality and reduce overhead, while low-level AD's fixed code structure makes it vulnerable to performance bottlenecks.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work because it provides empirical evidence for the performance benefits of high-level AD systems. The findings suggest that language-integrated AD systems that can perform semantic transformations offer significant advantages over generic low-level approaches, particularly for memory-intensive workloads common in machine learning. The study also highlights practical challenges in applying low-level AD to GPU code, suggesting that high-level, language-integrated approaches may be more practical for real-world deployment. The identification of distinct performance categories based on computational patterns provides valuable guidance for choosing appropriate AD strategies in different contexts.
