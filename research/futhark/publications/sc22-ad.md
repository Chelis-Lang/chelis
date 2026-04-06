# AD for an Array Language with Nested Parallelism

## Metadata
- **Authors:** Robert Schenck, Ola Rønning, Troels Henriksen, Cosmin E. Oancea
- **Venue/Year:** SC22, 2022

## Summary
This paper presents a technique for applying reverse mode automatic differentiation (AD) to a non-recursive second-order functional array language that supports nested parallelism and is primarily aimed at efficient GPU execution. The key innovation is eliminating the need for a tape by relying on redundant execution to bring into each new scope all program variables that may be needed by the differentiated code. The approach preserves work-span asymptotics and differentiates loops and bulk-parallel operators like map, reduce, scan, and scatter using specific rewrite rules. The authors report competitive performance on ten common benchmarks from recent applied AD literature.

## Key Contributions
- A redundant execution technique for reverse AD that eliminates the need for tape and does not introduce re-execution for perfectly nested scopes other than loops.
- A set of rewrite rules for differentiating higher-order parallel combinators, including in the presence of free variables.
- A collection of optimizations that rewrite common cases of accumulators to reductions, which benefit from specialized code generation.
- An experimental evaluation that demonstrates sequential and GPU performance competitive with Tapenade, Enzyme, PyTorch, and JAX.

## Technical Approach
The core technical idea is to eliminate the need for a tape by relying on redundant execution to bring into each new scope all program variables that may be needed by the differentiated code. The approach preserves work-span asymptotics because the recomputation overhead is at worst proportional to the depth of the deepest nest of scopes, which is constant for a given non-recursive program. Perfectly nested scopes (other than loops) are guaranteed to not introduce re-execution, hence the overhead can be minimized by classic compiler transformations such as flattening nested parallelism and polyhedral-like optimizations. The technique differentiates loops and bulk-parallel operators by specific rewrite rules and aggressively optimizes the resulting nested-parallel code.

## Results
The paper reports an evaluation that compares with established AD solutions and demonstrates competitive performance on ten common benchmarks from recent applied AD literature. The results show that the approach is effective in practice and competitive with both well-established frameworks that encompass more specialized languages such as PyTorch and JAX and with newer research efforts aimed at a lower-level language, such as Enzyme.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work. It presents a novel approach to automatic differentiation that is specifically designed for array languages with nested parallelism, which is a key feature of many modern machine learning frameworks. The approach is implemented as a compiler pass for the Futhark programming language, which is a functional array language that supports nested parallelism and is primarily aimed at efficient GPU execution. The paper's results demonstrate that the approach is effective in practice and competitive with established AD solutions, which suggests that it could be a valuable addition to the toolkit of language designers, compiler writers, and ML practitioners.
