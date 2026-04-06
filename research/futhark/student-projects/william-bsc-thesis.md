# University of Copenhagen Computer Science Department Parallel Parsing using Futhark The Implementation of a Parallel LL Parser Generator

## Metadata
- **Authors:** William Henrich Due
- **Venue/Year:** University of Copenhagen, 2023

## Summary
This paper presents an implementation of a deterministic parallel LL parser generator for the LLP(q, k) grammar class. The LLP grammar class allows for parallel parsing by imposing restrictions on LL grammars that enable substring validation and context checking. The implementation uses Haskell for the parser generator and Futhark for the generated parallel parsers. The work identifies and corrects errors in the original LLP papers, particularly in the PSLS definition and Algorithm 8, which could lead to infinite loops. The implementation includes memoization optimizations for FIRST and LAST set computations and uses string packing techniques to represent data in Futhark. The parser generator is thoroughly tested with both unit tests and property-based testing on random grammars.

## Key Contributions
- First implementation of a LLP(q, k) parser generator
- Identification and correction of errors in the original LLP papers
- Implementation of memoization for FIRST and LAST set computations
- String packing techniques for Futhark representation
- Comprehensive testing framework including property-based testing
- Parallel bracket matching algorithm for LLP parsing

## Technical Approach
The implementation follows a structured approach: parsing context-free grammars, checking for common left factors and left recursion, constructing PSLS tables using modified Algorithm 8, building LLP(q, k) tables using Algorithm 13, and generating Futhark source files. The FIRST and FOLLOW set computations use fixed-point iteration with memoization to improve performance. The LLP item collection uses a modified version of Algorithm 8 that addresses the infinite loop issue by checking if the lookahead string is in the FIRSTk set of prefixes. String packing represents terminals, nonterminals, and productions as integers with padding to handle Futhark's limitations with dynamic arrays. The parallel bracket matching algorithm checks for balanced brackets using a scan operation.

## Results
The implementation successfully generates LLP(q, k) parsers that can parse leftmost derivable strings and reject non-derivable strings. Testing shows that approximately 7-10% of randomly generated grammars with 3 terminals, 3 nonterminals, and 6 productions are accepted as LLP(q, k) grammars. The memoization optimization significantly improves performance, reducing test execution time from over 4 hours to 21 seconds. The parser generator correctly handles various grammar classes and produces valid Futhark code for parallel execution.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it presents a novel approach to parallel parsing that could enable more efficient compilation of programming languages on parallel architectures. The implementation demonstrates how functional programming languages like Haskell can be used for compiler construction, and how domain-specific languages like Futhark can be leveraged for parallel execution. The identification of errors in the original LLP papers and their correction contributes to the theoretical foundations of parallel parsing. The string packing techniques and memoization optimizations provide practical insights for implementing efficient parser generators. The work also highlights the challenges of implementing complex parsing algorithms and the importance of thorough testing, particularly when dealing with parallel and GPU-based execution.
