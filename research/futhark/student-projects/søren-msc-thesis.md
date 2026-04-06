# Reverse mode automatic differentiation of histograms in Futhark

## Metadata
- **Authors:** Søren Brix
- **Venue/Year:** MSc Thesis, University of Copenhagen, 2022

## Summary
This thesis presents a method for differentiating the parallel Futhark construct `reduce_by_index` using reverse mode automatic differentiation (AD). The work integrates a new rewrite rule into the Futhark compiler pipeline, transforming the original code into a program that computes the adjoint of its input. The thesis provides background on AD, including forward and reverse modes, and the Futhark language itself. It covers the main rewrite rule for reverse mode and its application to `reduce_by_index`. The implementation includes three special cases for popular operators (+, *, min/max) and a general approach for arbitrary associative and commutative operators. The correctness and performance of the implementation are evaluated.

## Key Contributions
- A fully developed rewrite rule for reverse mode AD of `reduce_by_index` in Futhark.
- Implementation of the rewrite rule as a compiler transformation, including three special cases and a general approach.
- Extension of the Futhark compiler to support reverse mode AD for `reduce_by_index`.
- Validation of the implementation by comparing results with forward mode AD.
- Performance evaluation showing the overhead of applying reverse mode AD to `reduce_by_index`.

## Technical Approach
The thesis presents the intermediate language (IR) of the Futhark compiler and the implementation of the rewrite rule. The core rule for rewriting statements in reverse mode is applied to `reduce_by_index`, which is a generalization of the `reduce` construct. The implementation involves sorting elements by their keys, computing partial reductions, and updating the adjoints of the input values and the destination array. The special cases for addition, multiplication, and min/max are optimized for their respective operators, while the general approach works for any associative and commutative operator.

## Results
The implementation is validated by comparing the results of reverse mode AD with forward mode AD on randomly generated data sets. The performance of the four different cases is evaluated, showing that the work-depth asymptotic of `reduce_by_index` is preserved when reverse mode AD is applied. The overhead of applying reverse mode AD to each operator is measured, with the special cases showing lower overhead compared to the general approach.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it extends the Futhark compiler to support reverse mode AD for a key parallel construct (`reduce_by_index`). This enables the expression of machine learning algorithms in Futhark and improves the accessibility of ML models by providing GPU support for reverse mode AD. The work also contributes to the understanding of AD and its application to parallel programming languages.
