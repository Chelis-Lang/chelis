# Shape-Constrained Array Programming with Size-Dependent Types

## Metadata
- **Authors:** Lubin Bailly, Troels Henriksen, Martin Elsman
- **Venue/Year:** FHPNC ’23 (11th ACM SIGPLAN International Workshop on Functional High-Performance and Numerical Computing), 2023

## Summary
Array programming frequently relies on shape and size constraints (e.g., matrix multiplication requires compatible dimensions, `filter` produces arrays of unknown length). In most languages, these constraints are enforced at runtime, leading to potential crashes or silent errors. This paper introduces a simplified dependent type system for an ML-style functional array language that statically enforces array-size consistency while avoiding the steep complexity and ergonomic overhead of full dependently typed languages like Agda or Idris.

The core innovation is a type system where sizes can be arbitrary integer expressions, but size equality is checked purely syntactically. The system introduces implicit size polymorphism and automatically manages existential sizes (e.g., the unknown output length of `filter`), eliminating the need for programmers to manually unpack dependent pairs. When static syntactic equality is insufficient, the system provides dynamically checked size coercions as a safe, explicit escape hatch.

The authors formalize a substantial subset of the language, prove type soundness, and detail how the system is adapted for practical use in the Futhark high-performance array language. By combining compile-time erasure, Hindley-Milner-style inference, and careful handling of size causality, the work demonstrates that strong shape guarantees can be integrated into real-world array programming without sacrificing performance or developer ergonomics.

## Key Contributions
- A lightweight, size-dependent type system tailored for functional array programming that avoids the complexity of full dependent type theories.
- Implicit size polymorphism and automated bookkeeping for existential sizes, removing the need for manual dependent-pair unpacking.
- Purely syntactic size equality with dynamic size coercions as a principled fallback for non-syntactic equivalences.
- Formalization of the core calculus with a complete type soundness proof and substitution properties.
- Practical implementation strategy in Futhark, including type inference, causality enforcement, and zero-runtime-overhead size tracking.

## Technical Approach
The paper defines a core calculus, **𝐹**, featuring size-polymorphic types, existential return types (`∃x.μ`), and implicit size arguments. Key technical mechanisms include:
- **Structural Equivalence & Subtyping:** Types are considered equivalent modulo size expressions (`∼s`). Subtyping allows replacing arbitrary size expressions with fresh existential variables, enabling flexible typing while discarding only size information.
- **Witnessed Existentials:** Existential sizes in return types must be "witnessed" by the underlying array structure, guaranteeing that their values can be extracted dynamically from the runtime value without requiring inverse computations.
- **Type Inference & Causality:** Adapts Hindley-Milner inference by treating sizes as a distinct kind of type variable. Unification is strictly syntactic to prevent arithmetic normalization from altering program semantics. Programs must satisfy a *causal coherence* property: sizes must be in scope before they are used, enforced via an implicit right-to-left evaluation order and ANF transformation.
- **Dynamic Coercions & Erasure:** The `e ⊲ τ` construct performs runtime shape verification when static checks fail. All size proofs and polymorphic instantiations are erased at compile time; at runtime, each array dimension is represented by a single 64-bit integer, ensuring zero storage or computational overhead.
- **Empty Array Handling:** The implementation extends the formal model by explicitly storing full shape metadata at runtime, allowing safe extraction of sizes even when outer dimensions are zero.

## Results
The paper's evaluation is primarily theoretical and implementation-focused rather than empirical:
- **Formal Guarantees:** Provides a rigorous metatheory, including preservation of well-formedness under substitution, value equivalence properties, and a full soundness proof (well-typed programs that terminate evaluate to well-typed values).
- **Futhark Prototype Integration:** Successfully adapts the calculus into a prototype of the Futhark compiler. The implementation automatically handles inference, ANF transformation, and t-relax rule application, demonstrating that the system scales to real-world array programming patterns.
- **Limitations & Trade-offs:** Acknowledges that strict syntactic equality misses commutative/associative arithmetic equivalences (e.g., `n+m ≠ m+n`), and that negative size instantiation can bypass safety without refinement types. Proposes future integration with SMT solvers or liquid types to address these gaps.

## Relevance
- **Language Design:** Demonstrates a pragmatic middle ground between simple ML type systems and full dependent types, showing how implicit polymorphism, existential automation, and dynamic escape hatches can make shape-dependent types usable by mainstream programmers.
- **Compilers:** Offers concrete techniques for erasing dependent type information, enforcing causal evaluation order for inference, and statically interpreting higher-order constructs to maintain high-performance compilation targets.
- **ML Systems & Tensor Computing:** Shape mismatches are a leading cause of runtime errors in ML frameworks (e.g., PyTorch, JAX, TensorFlow). This work provides a blueprint for statically verified, zero-overhead tensor shape checking, which could significantly improve the reliability, debuggability, and performance of domain-specific languages and compilers for machine learning workloads.
