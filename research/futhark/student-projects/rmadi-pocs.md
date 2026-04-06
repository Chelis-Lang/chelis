# Unifying Paths for Updates and Sections in Futhark

## Metadata
- **Authors:** Aziz Rmadi
- **Venue/Year:** 2026 (Project Report / Coursework, DIKU)

## Summary
Futhark is a purely functional, data-parallel programming language that relies on a uniqueness type system to enable efficient, in-place array updates while preserving functional semantics. Historically, the language treated array updates, record updates, and partial projection sections as distinct syntactic and AST constructs. This separation prevented programmers from naturally expressing mixed nested paths (e.g., `[i].f[j]` or `(.a[1:3])`) and often led to verbose workarounds or unintended consumption errors when updating nested structures like arrays inside records.

This project introduces a unified, path-based representation that treats array indexing, slicing, and record field access as sequential steps within a single abstraction. By refactoring the parser, type checker, normalization, and internalization phases to operate on this shared `UpdateStep` list, the compiler now supports mixed paths uniformly across `with` expressions, `let-with` shorthand, and section syntax. Crucially, the implementation is confined entirely to the frontend, leaving the core intermediate representation (IR) untouched.

The work demonstrates that syntactic unification, when paired with careful handling of evaluation order and consumption semantics, can significantly improve language ergonomics without compromising Futhark’s strict uniqueness guarantees. The result is a cleaner, more maintainable frontend that eliminates previous limitations around nested updates while providing a more intuitive surface syntax for complex data manipulations.

## Key Contributions
- Introduction of a unified `UpdateStep` AST datatype that represents array slices and record field accesses as a single, ordered path.
- Refactoring of the parser and frontend to support mixed paths in `with` expressions, `let-with` shorthand, and section syntax.
- A strictly frontend-only implementation that requires zero modifications to Futhark’s core compiler IR.
- Preservation of uniqueness and consumption semantics through carefully ordered normalization and consumption checking.
- Elimination of previous usability bottlenecks, such as consumption errors when updating arrays nested inside records.

## Technical Approach
The core technical innovation is the replacement of separate array and record update constructs with a single path abstraction. A path is represented as a list of `UpdateStep` values, where each step is either an `UpdateStepSlice` (for array indexing/slicing) or an `UpdateStepField` (for record projection). This structure is used uniformly across three frontend constructs: standard `Update` expressions, `LetWith` shorthand, and `UpdateSection` projections.

To maintain semantic correctness under Futhark’s uniqueness system, the implementation enforces a strict evaluation order during normalization and consumption checking: path index expressions are evaluated first, followed by the replacement value, and finally the source expression. This ordering prevents use-after-consume errors and avoids duplicating consuming source expressions during path traversal. During internalization, the unified path is recursively lowered to existing core operations: field steps decompose records structurally, while slice steps compute indexed subvalues, apply the update, and write the result back. Sections are desugared into projection lambdas using the same path-traversal logic, ensuring consistency across updates and projections.

## Results
The implementation was validated through a comprehensive suite of regression tests rather than performance benchmarks, as the changes are primarily syntactic and frontend-focused. Tests confirmed correct handling of mixed update paths, consecutive indexing, mixed field-and-slice updates, generalized sections, and edge cases involving uniqueness and consumption. Notably, the new design successfully resolves previous consumption errors that occurred when updating nested structures (e.g., arrays within records), eliminating the need for manual record decomposition. The project concludes that the unified path representation improves expressiveness and compiler maintainability without introducing runtime overhead or altering the core IR.

## Relevance
This paper is highly relevant to language design and compiler engineering, particularly for functional and data-parallel languages that employ linear, affine, or uniqueness type systems. It illustrates how frontend unification can resolve ergonomic limitations while strictly preserving consumption semantics—a common challenge in languages like Rust, Koka, or Futhark. The careful treatment of evaluation order and desugaring provides a practical blueprint for implementing path-based updates without introducing subtle aliasing or duplication bugs. Additionally, the approach demonstrates how to extend surface syntax expressively without bloating the core IR, a valuable pattern for compiler architects designing modular, maintainable language frontends. For ML systems work, where data-parallel array manipulations and nested data structures are common, such ergonomic and semantically sound update mechanisms can streamline high-performance numerical and tensor programming.
