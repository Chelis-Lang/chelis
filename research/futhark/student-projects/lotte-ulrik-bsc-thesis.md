# A Language for Parallel Generation of L-Systems

## Metadata
- **Authors:** Lotte Maria Bruun, Ulrik Stuhr Larsen
- **Venue/Year:** Bachelor thesis at University of Copenhagen, June 2020
- **Supervisor:** Troels Henriksen

## Summary
This thesis presents a domain-specific language (DSL) for defining Lindenmayer systems (L-systems) and a compiler that generates highly parallel programs in the Futhark programming language. The work focuses on parallel derivation and visualization of L-systems, leveraging GPU parallelism to achieve high performance. The authors implement four parallel derivation strategies and two parallel visualization algorithms, benchmarking them against various L-systems. Results show that their fastest derivation strategy can generate up to 17,695 symbols per microsecond (17.7 billion symbols per second), and the visualization algorithms can interpret up to 113 symbols per microsecond (113 million symbols per second) for their test cases.

## Key Contributions
- Design and implementation of a DSL for defining deterministic, context-free L-systems (bracketed or non-bracketed).
- Development of a compiler that translates the DSL into highly parallel Futhark programs.
- Implementation of four parallel derivation strategies with work-depth asymptotic analysis.
- Implementation of two parallel visualization algorithms for bracketed and non-bracketed L-systems.
- Comprehensive benchmarking and comparison of the strategies against various L-systems.

## Technical Approach
The thesis describes the theory of L-systems and their parallel generation and visualization. The authors implement a DSL for L-systems and a compiler that generates parallel Futhark programs. They implement four parallel derivation strategies and two parallel visualization algorithms. The strategies are benchmarked with multiple L-systems and different iteration numbers with 10 runs per test setting.

## Results
The benchmarks show that the fastest derivation strategy can generate up to 17,695 symbols per microsecond (17.7 billion symbols per second) and the visualization algorithms can interpret up to 113 symbols per microsecond (113 million symbols per second) for their test cases. The results also show that strategy 4 performs the best in a large majority of the tested L-systems.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it presents a DSL for defining L-systems and a compiler that generates highly parallel programs. The work also explores the use of GPU parallelism for L-system generation and visualization, which is relevant to ML systems that require high-performance computing.
