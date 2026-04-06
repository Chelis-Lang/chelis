# AUTOMAP: Inferring Rank-Polymorphic Function Applications with Integer Linear Programming

## Metadata
- **Authors:** Robert Schenck, Nikolaj Hey Hinnerskov, Troels Henriksen, Magnus Madsen, Martin Elsman
- **Venue/Year:** OOPSLA 2024 (Proc. ACM Program. Lang. 8, OOPSLA2, Article 334)

## Summary
Dynamically typed array languages like Python/NumPy, APL, and MATLAB support rank polymorphism, allowing scalar operations to be implicitly lifted to arrays of any dimension through broadcasting and replication. While highly ergonomic, this flexibility often obscures type errors and complicates static analysis. Statically typed functional array languages typically require programmers to explicitly write `map` and `replicate` operations, which clutters code and hinders readability, particularly in mathematical and linear algebra domains.

This paper introduces AUTOMAP, a type inference mechanism that automatically inserts the minimal number of `map` and `rep` operations at function application sites to make statically typed programs well-typed. By treating rank polymorphism as a purely syntactic elaboration problem, AUTOMAP preserves the static guarantees of a Hindley-Milner-style type system while delivering the conciseness of dynamic array languages. The system uses Integer Linear Programming (ILP) to solve rank constraints, ensuring that inferred elaborations are unambiguous and minimize implicit operations.

Implemented as an extension to the Futhark compiler, AUTOMAP is formally proven to satisfy well-typedness, determinism, and consistency properties. Empirical evaluation on real-world benchmarks demonstrates that it eliminates 54% of explicit `map` operations, significantly improving code readability with only a modest 2.5× average type-checking overhead and zero impact on runtime performance.

## Key Contributions
- A novel constraint-based type inference system that automatically infers implicit `map` and `rep` (broadcasting) operations at function application sites in a polymorphic, higher-order array language.
- A formal elaboration framework with proven meta-theoretic properties: Well-Typedness, Determinism, Disambiguation, Forwards Consistency, and Backwards Consistency.
- An ILP-based rank analysis algorithm that minimizes the number of inserted operations, detects ambiguity, and integrates seamlessly with standard type unification.
- A production-ready implementation in the Futhark compiler, accompanied by an empirical evaluation showing substantial reductions in boilerplate code and practical type-checking performance.

## Technical Approach
AUTOMAP operates by annotating every function application in an intermediate representation with rank variables `△(M, R)`, where `M` and `R` represent the number of implicit `map` and `rep` operations, respectively. During type checking, the system generates constraints that relate the ranks of function parameters and arguments. A critical constraint is the zero-rank disjunction `M ∨· R`, which enforces that an application can have implicit maps *or* implicit reps, but never both.

These constraints are relaxed into a rank-only system and translated into an Integer Linear Program. The ILP's objective function minimizes `Σ(M + R)` to find the smallest number of implicit operations required for type correctness. To handle the disjunction constraint, binary variables and a large constant upper bound are used. After finding a minimal solution, the solver checks for ambiguity by adding constraints to ban the first solution and searching for a second solution of identical size; if one exists, the program is rejected.

Once a unique minimal rank substitution is found, standard structural type unification resolves remaining type variables. The `AM` transformation then replaces the `△(n_M, n_R)` annotations with explicit nested `map` and `rep` constructs, yielding a fully explicit target program. The implementation also employs "rep fusion" to ignore induced replicates (those necessitated by prior explicit maps) in the ILP objective, preventing false ambiguity and aligning better with programmer intent.

## Results
- **Code Reduction:** Evaluated on 67 real-world Futhark programs (8,600 SLOC) from benchmark suites like Parboil, PBBS, and Rodinia. AUTOMAP reduced explicit `map` operations by 54% (from 467 to 213), with the most significant readability gains in linear algebra and mathematical kernels.
- **Performance Overhead:** The ILP-based type checker introduces an average 2.5× slowdown compared to the baseline. Most ILP instances are small (median 18 constraints), though a worst-case dense function saw a ~13× slowdown due to a 28,104-constraint ILP. The authors note optimization opportunities like local constraint elision.
- **Runtime & Ambiguity:** Zero impact on compiled program performance. Ambiguity was rare in practice, and the system proved transparent: programmers can always validate elaborations by viewing the fully explicit target code or resolve ambiguities by adding explicit `map`/`rep` or type annotations.

## Relevance
- **Language Design:** Provides a principled, formally verified bridge between dynamic array languages (NumPy, APL) and static functional languages. It demonstrates how rank polymorphism can be safely integrated into ML-style type systems without sacrificing parametric polymorphism or higher-order functions.
- **Compilers:** Showcases a practical, production-grade application of Integer Linear Programming in type inference. The stratified solving approach (rank analysis via ILP followed by standard type unification) and ambiguity detection techniques offer a blueprint for other constraint-based elaboration systems.
- **ML Systems & Tensor Compilers:** Highly relevant for ML frameworks (JAX, PyTorch, TVM) where broadcasting and implicit dimension lifting are ubiquitous. AUTOMAP's deterministic, minimal-operation inference could inform static shape analysis, automatic differentiation pipelines, and compiler passes that need to resolve rank mismatches predictably and efficiently without runtime overhead.
