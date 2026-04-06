# Static Interpretation of Higher-Order Modules in Futhark: Functional GPU Programming in the Large

## Metadata
- **Authors:** Martin Elsman, Troels Henriksen, Danil Annenkov, Cosmin E. Oancea
- **Venue/Year:** Proceedings of the ACM on Programming Languages (ICFP), 2018

## Summary
This paper presents a higher-order module system for the purely functional data-parallel array language Futhark. The module language is designed to be completely eliminated at compile time, providing a powerful tool for organizing libraries and complete programs without runtime overhead. The authors introduce the concept of "static interpretation" to achieve this, which involves elaborating module expressions and declarations into target language code. The paper provides a formal semantics for the module language, including a static type system and a provably terminating elaboration process. The development is formalized in Coq using a novel encoding of semantic objects based on products, sets, and finite maps. The module language features a unified treatment of module type abstraction and core language polymorphism, enabling practical forms of module composition. The authors demonstrate the usefulness of the module system through examples and argue that it introduces no performance overhead in the generated code.

## Key Contributions
- Introduction of higher-order static interpretation for eliminating modules at compile time
- Formal semantics for a higher-order module language with static type system and terminating elaboration
- Coq formalization of the module language and its properties
- Novel encoding of semantic objects using products, sets, and finite maps
- Practical examples demonstrating the usefulness of the module system for GPU programming

## Technical Approach
The paper presents a higher-order module language that is elaborated into a target language (Futhark) at compile time. The module language includes constructs for specifying module types (mty) and declaring modules (mdec). The elaboration process involves:
1. Elaborating module types and specifications into semantic objects (Σ and ∃T.E)
2. Elaborating module expressions and declarations into interpretation environments and target code
3. Using a logical relation argument to prove termination of static interpretation
4. Ensuring type consistency between source and target programs

The key technical ideas include:
- Enrichment and instantiation relations for handling parameterised modules
- Filtering interpretation environments to match elaboration environments
- Use of universal and existential quantification for type abstraction
- Novel encoding of semantic objects in Coq using products, sets, and finite maps

## Results
The paper presents several key results:
1. Static interpretation is terminating for all well-typed programs
2. Generated target programs are type-consistent with source programs
3. The module language can express practical forms of module composition
4. No performance overhead is introduced in the generated code
5. The Coq formalization provides a mechanized proof of the key properties

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work for several reasons:
1. It presents a novel approach to eliminating modules at compile time, which can be applied to other languages
2. The formal semantics and Coq formalization provide a solid foundation for reasoning about module systems
3. The technique of static interpretation can be used to implement efficient module systems for domain-specific languages
4. The unified treatment of module type abstraction and core language polymorphism offers insights into language design
5. The work demonstrates how to achieve modularity and code reuse without runtime overhead, which is crucial for high-performance computing

The paper's approach to static interpretation and its application to GPU programming make it particularly relevant for researchers and practitioners working on compilers for data-parallel languages and high-performance computing systems.
