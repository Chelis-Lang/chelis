# Refinement Types in Futhark

## Metadata
- **Authors:** Henriks Urms (mgs837) and Anna Soﬁe Kiehn (mvq686)
- **Venue/Year:** Master's Thesis, September 2, 2019
- **Supervisor:** Martin Elsman

## Summary
This thesis explores the integration of refinement types into the Futhark programming language to enable compile-time verification of array bounds safety, particularly for GPU execution where runtime checks are problematic. The authors design a refinement type system, implement a type checker in Haskell, and demonstrate its effectiveness through examples. The work shows that refinement types can express conditions necessary for safe array indexing and can be used with higher-order functions and polymorphism.

## Key Contributions
- Design of a refinement type system for Futhark
- Implementation of a type checker based on bidirectional type rules
- Transformation functions between external (Futhark-like) and internal languages
- Demonstration of compile-time array bounds verification
- Evaluation of usability and performance benefits

## Technical Approach
The authors adopt a two-language approach: an external language (a subset of Futhark with refinement type annotations) and an internal language (fully indexed types for type checking). They implement bidirectional type rules inspired by previous work on refinement types, particularly Dunfield and Krishnaswami's approach. The type checker generates constraints that must be solved to verify program safety.

## Results
The implementation successfully type checks example programs demonstrating safe array indexing, including cases where indices are drawn from arrays. The system supports polymorphism and higher-order functions. The authors show that refinement types can express necessary conditions for safe array access and that the type annotations serve as useful documentation.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates how refinement types can be integrated into a practical functional language for GPU programming. The approach of using refinement types to eliminate runtime array bounds checks is particularly valuable for performance-critical applications on parallel architectures. The bidirectional type checking approach and the two-language design pattern are also valuable contributions to the broader field of type system design.
