# Sum Types in Futhark

## Metadata
- **Authors:** Robert Schenck
- **Venue/Year:** Master's Thesis, University of Copenhagen / 2019

## Summary
This thesis presents the design and implementation of sum types in Futhark, a purely functional data-parallel array programming language. The work includes a formal type system for sum types, their implementation in the Futhark compiler, and an analysis of the implementation with suggestions for performance improvements. The author argues that sum types provide increased safety and better abstraction in Futhark, despite being uncommon in compute-oriented languages.

## Key Contributions
- Formal development of sum types in a model language based on Futhark, including a proof of soundness
- Extension of Futhark's syntax to support sum types, constructor expressions, and match expressions
- Augmentation of Futhark's type system to support sum types with constraint-based typing
- Implementation of pattern exhaustivity checking for match expressions
- Transformation logic to convert sum types into Futhark's intermediate representation
- Analysis of implementation performance and suggestions for improvements
- Case studies demonstrating the utility of sum types in real Futhark programs

## Technical Approach
The thesis develops a formal type system for sum types based on a λ-calculus model of Futhark. The system includes typing rules for constructors, match expressions, and patterns, with a focus on ensuring progress and preservation properties. The implementation extends Futhark's syntax and type checker to support sum types, using a constraint-based Damas-Milner type system. Constructors are transformed into tuples with numeric tags, and match expressions are converted into nested if-else expressions. The implementation includes pattern exhaustivity checking to ensure all cases are covered.

## Results
The implementation successfully adds sum type support to Futhark, enabling more expressive and safer programming. The thesis identifies several performance issues, including inefficient array representations of sum types and redundant tests in match expressions. Proposed improvements include constructor deduplication to reduce tuple sizes and decision tree compilation for more efficient pattern matching. Case studies show that sum types simplify code in real Futhark programs, such as combining multiple event handlers into a single type and improving the representation of particle simulation elements.

## Relevance
This work is relevant to language design and compiler development for functional and data-parallel programming languages. It demonstrates how to add sum types to a specialized compute-oriented language while maintaining performance characteristics. The techniques for transforming sum types into efficient representations and the analysis of pattern matching compilation are valuable for other language implementers. The work also contributes to the theoretical understanding of sum types in structurally typed languages and their practical implementation challenges.
