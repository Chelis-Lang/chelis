# Translating Concepts of the Futhark Programming Language into an Extended fi-Calculus

## Metadata
- **Authors:** Lars Jensen, Chris Oliver Paulsen, Julian Jørgensen Teule
- **Venue/Year:** Master's Thesis, Aalborg University, 2023

## Summary
This thesis presents a translation of a subset of the Futhark programming language into an extended fi-calculus (Efi). Futhark is a data-parallel functional language designed for efficient GPU programming, while the fi-calculus is a process calculus that naturally expresses concurrency. The authors create a simplified version of Futhark called Basic Un-Typed Futhark (ButF) and extend the fi-calculus to better represent Futhark's array structures and second-order array combinators (SOACs). They then prove the correctness of this translation using operational correspondence and weak administrative barbed bisimulation.

## Key Contributions
- Design of ButF, a simplified subset of Futhark focusing on arrays and SOACs
- Extension of the fi-calculus with broadcasting and composite names (Efi)
- Translation of ButF constructs into Efi
- Analysis of primitive candidates to reduce translation complexity
- Proof of correctness using operational correspondence and weak administrative barbed bisimulation

## Technical Approach
The authors take a multi-step approach:
1. Create ButF by isolating Futhark's core features (arrays, SOACs) while omitting complex aspects like in-place updates
2. Extend the fi-calculus with broadcasting and composite names to naturally represent arrays
3. Develop a translation from ButF to Efi, categorizing it into expressions, arrays, and SOACs
4. Analyze primitive candidates (map, reduce, scan, size, iota, concat) to find an optimal subset
5. Prove correctness using operational correspondence with administrative reductions and weak administrative barbed bisimulation

## Results
The thesis successfully demonstrates:
- A correct translation from ButF to Efi
- That map, size, and iota can serve as primitives while other constructs can be derived
- The translation preserves both work and span complexity compared to Futhark
- Operational correspondence holds between ButF and its Efi translation
- The extended fi-calculus with broadcasting and composite names provides a natural representation for Futhark's array structures

## Relevance
This work is significant for language design and compiler construction as it:
- Provides a formal foundation for analyzing data-parallel languages
- Demonstrates how process calculi can model functional array languages
- Offers insights into the relationship between functional programming and concurrent computation
- Creates a framework for analyzing parallel properties of array operations
- Shows how to prove correctness of translations between different computational models

The translation enables formal analysis of Futhark programs without hardware constraints, potentially leading to better optimization strategies and deeper understanding of data-parallel computation patterns.
