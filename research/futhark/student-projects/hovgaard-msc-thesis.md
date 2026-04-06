# Higher-order functions for a high-performance programming language for GPUs

## Metadata
- **Authors:** Anders Kiel Hovgaard, Troels Henriksen, Martin Elsman
- **Venue/Year:** TFP 2018

## Summary
This thesis presents a defunctionalization transformation for a data-parallel functional language, Futhark, that enables efficient implementation of higher-order functions on GPUs. The key insight is that by restricting the use of functions in certain contexts (e.g., disallowing conditionals and loops from returning functions, and arrays from containing functions), the transformation can completely eliminate higher-order functions without introducing branching. This is crucial for GPU performance, as branching can cause significant inefficiencies due to branch divergence. The thesis proves the correctness of the transformation and discusses its implementation in Futhark, along with various extensions and optimizations. Empirical evaluation shows that the use of higher-order functions has no impact on runtime performance and that the restricted higher-order functions are useful in practice.

## Key Contributions
- A defunctionalization transformation for a data-parallel functional language that eliminates higher-order functions without introducing branching.
- A correctness proof of the transformation, showing that well-typed programs translate to well-typed programs and preserve meaning.
- An implementation of defunctionalization in the Futhark compiler, along with extensions and optimizations.
- An empirical evaluation demonstrating the performance and usefulness of the restricted higher-order functions.

## Technical Approach
The defunctionalization transformation works by representing each function abstraction as a record that captures the free variables in the function and replacing each application with a call to a specialized function that interprets the record. The key to avoiding branching is the type system restriction that ensures the form of the applied function can be statically determined at every application site. The transformation is defined on a simple functional language with features like records, arrays, and loops, and is proven correct using logical relations.

## Results
The empirical evaluation shows that the use of higher-order functions in Futhark has no impact on runtime performance. The restricted higher-order functions are useful in practice, enabling more modular and expressive code without sacrificing performance. The implementation has been used to rewrite the Futhark standard library and benchmark programs, demonstrating the practical benefits of the approach.

## Relevance
This work is relevant to language design, compilers, and ML systems because it provides a practical solution for implementing higher-order functions in high-performance functional languages for GPUs. The approach of using type-based restrictions to enable efficient defunctionalization is a novel contribution that could be applied to other languages and compilation targets. The empirical evaluation demonstrates the effectiveness of the approach in practice, making it a valuable contribution to the field.
