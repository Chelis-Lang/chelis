# Benchmarking Futhark-AD using MINPACK-2

## Metadata
- **Authors:** Peter Kanstrup Larsen
- **Venue/Year:** Project Report, 7.5 ECTS, January 2023

## Summary
This project explores the capabilities of Futhark-AD, an automatic differentiation library for the Futhark programming language, by translating and benchmarking a problem from the MINPACK-2 test collection. The focus is on evaluating the performance and correctness of computing Jacobian matrices for a "Flow in a Channel" problem. The implementation leverages Futhark's parallel computing features to optimize performance. While the runtime results show promising speedups, particularly for large input sizes, the computed Jacobians fail validation when compared to the original FORTRAN implementation. This discrepancy highlights the need for further investigation into the correctness of the AD implementation.

## Key Contributions
- Translation of the MINPACK-2 "Flow in a Channel" problem into Futhark with a focus on parallelism.
- Implementation of the objective function and helper functions to compute collocation and continuity equations.
- Benchmarking of Futhark-AD's forward and reverse modes for Jacobian computation.
- Identification of validation issues in the computed Jacobians, suggesting potential bugs or implementation errors.

## Technical Approach
The project uses Futhark-AD to compute Jacobians of the objective function `dficfj`, which models fluid flow in a channel. The implementation includes:
- **task_XS**: Generates input vectors for the problem.
- **mk_eq**: Computes collocation and continuity equations.
- **dficfj**: The main objective function that evaluates the problem at a given point.
The Jacobian is computed using both forward mode (`jvp`) and reverse mode (`vjp`) AD, with results compared to the original FORTRAN implementation.

## Results
- **Runtime Performance**: The Futhark implementation achieves significant speedups on GPU for large input sizes, outperforming the FORTRAN implementation in some cases.
- **Validation Issues**: The computed Jacobians fail validation, with discrepancies in the results. The forward mode (`jvp`) shows small mismatches, while the reverse mode (`vjp`) has columns of zeros, indicating potential issues in the AD implementation.

## Relevance
This work is relevant to the development of automatic differentiation tools for parallel computing environments. It highlights the challenges of ensuring correctness in AD implementations while leveraging parallelism for performance. The findings suggest that further research is needed to address validation issues and improve the robustness of Futhark-AD for real-world applications.
