# Multi-GPU Futhark Using Parallel Streams

## Metadata
- **Authors:** Steffen Holst Larsen
- **Venue/Year:** Master's Thesis, September 19, 2019

## Summary
This thesis presents the design and implementation of a new internal streaming operator called the "husk operator" in the Futhark compiler. The husk operator enables the distribution of operations across multiple GPUs, allowing compiled Futhark programs to utilize multiple graphics processing units (GPUs) in the system they are executed on. The thesis describes the type and semantics of the husk operator, its transformations for map and reduce second-order array combinators (SOACs), and the implementation strategy for the Futhark compiler. It also discusses the changes made to the CUDA C backend of Futhark to support multiple GPUs and introduces a worker-thread runtime environment for isolated execution of operations contained in husk operators. The thesis evaluates the performance of the husk operator in both single-GPU and multi-GPU scenarios, analyzing cases where it introduces good scaling with large data sets and cases where inter-GPU data transfer overhead becomes too expensive.

## Key Contributions
- Design and implementation of the internal husk operator in the Futhark compiler.
- Transformations for map and reduce SOACs to utilize the husk operator.
- Changes to the CUDA C backend of Futhark to support multiple GPUs.
- Introduction of a worker-thread runtime environment for isolated execution of husk operations.
- Evaluation of the husk operator's performance in single-GPU and multi-GPU scenarios.

## Technical Approach
The thesis introduces the husk operator as an internal streaming operator in the Futhark compiler. It describes the type and semantics of the husk operator, which allows for the distribution of operations across multiple GPUs. The thesis then presents transformations for map and reduce SOACs to utilize the husk operator, relating their semantics to that of the husk operator. It discusses the implementation strategy for the Futhark compiler, including changes to the CUDA C backend to support multiple GPUs. The thesis also introduces a worker-thread runtime environment for isolated execution of husk operations. Finally, it evaluates the performance of the husk operator in both single-GPU and multi-GPU scenarios.

## Results
The thesis evaluates the performance of the husk operator in two parts. First, it compares the single-GPU performance with and without the husk operator, analyzing examples on both ends of the spectrum of relative performance. Secondly, it compares the single-GPU performance with the multi-GPU performance of select programs, analyzing cases where the husk operator introduces good scaling with large data sets and cases where the inter-GPU data transfer overhead becomes too expensive relative to the computations, resulting in undesirable performance.

## Relevance
This paper is relevant to language design, compilers, and ML systems work as it presents a new internal streaming operator in the Futhark compiler that enables the distribution of operations across multiple GPUs. The husk operator allows for the utilization of multiple GPUs in the system, improving performance for data-heavy computations. The thesis also discusses the implementation strategy for the Futhark compiler, including changes to the CUDA C backend and the introduction of a worker-thread runtime environment. The evaluation of the husk operator's performance in both single-GPU and multi-GPU scenarios provides insights into its effectiveness and limitations.
