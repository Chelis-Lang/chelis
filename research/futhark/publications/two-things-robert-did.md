# Two Things I Did: Parallel Differentiation and Rank Polymorphism

## Metadata
- **Authors:** Robert Schenck
- **Venue/Year:** PhD Thesis, University of Copenhagen, 2024

## Summary
This thesis presents two distinct contributions to programming language research: (1) an efficient technique for reverse-mode automatic differentiation (AD) in a nested-parallel functional language, and (2) a mechanism for rank polymorphism in statically typed languages with parametric polymorphism and type inference. The AD work introduces a recomputation-based approach that eliminates the need for a tape in reverse-mode AD by redundantly re-executing forward computations when entering new scopes, combined with high-level rewrite rules for differentiating parallel combinators. The rank polymorphism work, called Automap, enables implicit map and replication operations to be inferred by a polymorphic type system using integer linear programming, allowing functions to operate on arguments of different ranks without explicit lifting or replication.

## Key Contributions
- A recomputation-based technique for efficient reverse-mode AD in nested-parallel contexts that eliminates tape overhead
- High-level rewrite rules for differentiating parallel combinators (map, reduce, scan, hist, scatter) in a nested-parallel language
- A rank polymorphism mechanism (Automap) that infers implicit map and replication operations using integer linear programming
- Formalization of the Automap mechanism with three languages (source, internal, target) and proofs of key properties
- Implementation of both techniques in the Futhark compiler with competitive performance evaluations

## Technical Approach

### Parallel Automatic Differentiation
The AD approach exploits the fact that applying reverse mode AD to a straight line of side-effect-free code doesn't require a tape because intermediate values remain available. The transformation expands this idea across lexical scopes by requiring that whenever the return sweep enters a new scope, it first redundantly re-executes the forward sweep of that scope to bring needed variables into scope. This technique preserves work-span asymptotics because the recomputation overhead is at worst proportional to the depth of the deepest nest of scopes, which is constant for non-recursive programs. Perfectly nested scopes (other than loops) are guaranteed not to introduce re-execution.

The approach uses high-level rewrite rules for differentiating parallel combinators. For example, the reduce combinator is differentiated by computing exclusive scans to determine the contributions of each element to the result, then mapping a function over these contributions to update the adjoints. Specialized rules exist for common operators like addition, multiplication, min, and max to improve efficiency.

### Rank Polymorphism (Automap)
Automap introduces a mechanism where functions on their own don't have rank-polymorphic types, but function applications can have map and rep operations inserted implicitly by the compiler during type inference elaboration. The system uses a constraint-based type system where applications are annotated with rank variables representing the number of implicit maps and reps. These constraints are relaxed into rank constraints and solved using integer linear programming to find minimal solutions that insert the fewest operations.

The elaboration process transforms source programs (with implicit maps/reps) into internal programs (with rank annotations) and then into target programs (with explicit maps/reps). The system satisfies key properties including well-typedness, determinism, disambiguation, forwards consistency, and backwards consistency.

## Results
The AD implementation in Futhark demonstrates competitive performance against established tools like Tapenade, Enzyme, PyTorch, and JAX on benchmarks including k-means clustering, GMM, and LSTM. The approach achieves memory overheads close to optimal (roughly twice the primal program) and handles nested parallelism effectively. The Automap implementation successfully removes 54% of map operations from real-world Futhark programs while maintaining type safety and introducing only modest type checking overhead (2.5× on average).

## Relevance
This work is highly relevant to language design, compilers, and ML systems because it addresses two fundamental challenges in scientific computing: efficient automatic differentiation for parallel programs and expressive array programming with static type safety. The AD technique demonstrates how high-level abstractions can enable simple, verifiable transformations that generate efficient code, while the rank polymorphism work shows how to bring the flexibility of dynamically typed array languages into a statically typed setting without sacrificing type safety or expressiveness. Both contributions are implemented in Futhark, a real-world functional array language, demonstrating their practical applicability to high-performance computing and machine learning workloads.
