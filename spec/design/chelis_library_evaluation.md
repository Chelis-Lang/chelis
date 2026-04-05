# Chelis Compiler: Rust Ecosystem Library Evaluation

## Executive Takeaway

The strongest near-term recommendation is **not** to replace working custom infrastructure just because a well-known crate exists. `logos`, `chumsky`, and `petgraph` all solve real problems, but none is a clear win against Chelis's current code shape before Phase 0 is complete. The strongest "later" recommendation is `salsa`, because incremental queries line up directly with Tide, REPL, LSP, and agent-loop use cases, but the adoption cost is architectural and should wait until the batch compiler path is complete. The strongest prototype candidate is `egg`, not for today's optimizer passes, but for Phase 1 fusion planning where rewrite search and cost modeling actually matter. `cranelift` is a plausible Phase 2 execution path for interactive latency, but it should coexist with, not displace, the C backend.

## Repo-Grounded Baseline

This evaluation is based on the current repo, not just the brief:

- `chelis-surf/src/lexer.rs` is about 1,000 lines and already has focused tests around the exact bug classes that would motivate a lexer generator: float exponents, token disambiguation, nested comments, and punctuation edge cases.
- `chelis-surf/src/parser.rs` is about 2,400 lines, combines recursive descent with Pratt parsing, and is covered by both unit tests and round-trip integration tests through desugaring and Deep reparse.
- `chelis-ir/src/dag.rs` is much simpler than a general graph engine: append-only, topologically ordered by construction, and backed by `verify.rs` for structural invariants.
- The optimizer in `chelis-ir/src/optimize.rs` is still intentionally simple: constant folding mutates in place; DCE and CSE rebuild fresh DAGs.
- The CLI and C backend are still early. Interactive mode, incrementality, GPU work, and fusion are planned but not yet driving the architecture.

That matters because the question is not "would these crates be good in the abstract?" The question is whether they beat today's working design at the current phase boundary.

## 1. `logos`

**Recommendation:** Do not adopt now. Revisit only if the Surf lexer becomes substantially more complex than it is today, or if the team wants a broad lexer simplification effort after Phase 0.  
**Confidence:** High

### Benefit Magnitude

- **Correctness:** Low-to-medium. `logos` would likely reduce some hand-written tokenization mistakes for fixed tokens and simple literal classes.
- **Maintainability:** Low. Chelis would still need manual handling for nested block comments, newline-sensitive behavior, and likely some numeric or context-specific token logic.
- **Performance:** Low. The current lexer is unlikely to be the compiler bottleneck in any planned phase.
- **Capability:** Low. `logos` does not unlock a new compiler feature.

### Migration Cost

- Medium. This is not a "swap the scanner and move on" change.
- `chelis-surf/src/token.rs`, `chelis-surf/src/lexer.rs`, parser token consumption, and lexer tests would all need coordinated changes.
- The hard part is not enumerating keywords and punctuation. The hard part is ending up with a split lexer where easy tokens live in `logos` and hard tokens still live in custom code.
- That hybrid design is harder to reason about than either a fully declarative lexer or a fully manual one.

### Timing

- **Now:** No.
- **Between Phase 0 and 1:** Still probably no.
- **Phase 2:** Only if front-end maintenance becomes a recurring cost center.
- **Never:** Plausible. This is a reasonable final answer if the current lexer remains stable.

### Reasoning

The current motivation for `logos` is mainly "hand-written lexers can be error-prone." That is true in general, but Chelis already paid most of the risk down with targeted tests around exponent parsing, symbol disambiguation, comments, and token boundaries. The remaining problem is that the language deliberately has one feature `logos` does not model cleanly: nested block comments. Once a manual escape hatch is required anyway, the main simplification benefit shrinks.

The current lexer also preserves explicit newline tokens for block and `par` separator semantics. That is a small but important hint that Chelis does not just want a generic regex tokenizer. It wants a language-specific token stream with layout-relevant behavior.

### Alternatives

- **Stay custom** and keep hardening tests. This is the best alternative today.
- Refactor the current lexer internally into smaller helper routines without changing the external model.
- Add more property-style or fuzz-style lexer tests before replacing the implementation.
- `winnow` or `nom` are possible parser-style scanning alternatives, but they do not obviously improve the nested-comment/layout case over the current code.

### Interactions

- `logos` becomes a little more attractive if the parser is later rewritten with a combinator stack that naturally consumes a token iterator, but it is still constrained by nested comments.
- `salsa` does not materially improve the case for `logos`; incrementality benefits come from query structure, not from how tokens are generated.
- `chumsky` examples often pair well with `logos`, but that coupling is not enough to justify either rewrite on its own.

## 2. `chumsky`

**Recommendation:** Do not rewrite the parser now. Consider a bounded prototype during Phase 2 planning if interactive diagnostics and recovery become central to Tide.  
**Confidence:** High

### Benefit Magnitude

- **Correctness:** Medium. A well-structured `chumsky` parser could reduce some handwritten parser bugs over time.
- **Maintainability:** Medium at best, but only after the rewrite stabilizes. In the short term it is worse.
- **Performance:** Low-to-medium. This is not the main value proposition.
- **Capability:** Medium-to-high if Chelis truly needs multi-error recovery and partial parsing for interactive tooling.

### Migration Cost

- High. `chelis-surf/src/parser.rs` is already a large, working parser with extensive direct tests plus end-to-end round-trips.
- This is a full parser rewrite, not an incremental refactor.
- The current parser mixes expression precedence, declaration parsing, block separator rules, and context-sensitive cases such as `let` inside braces. Those behaviors all need to be re-expressed and re-debugged in a different style.
- Stabilization risk is real because parser regressions can be semantically subtle while still passing many happy-path tests.

### Timing

- **Now:** No.
- **Between Phase 0 and 1:** No.
- **Phase 2:** Maybe, if Tide and editor tooling need robust recovery and richer syntax diagnostics.
- **Never:** Also plausible if the current parser remains maintainable and diagnostics are improved another way.

### Reasoning

`chumsky`'s best argument is not "parser combinators are nicer." It is "Chelis wants good recovery, multiple syntax diagnostics, and partial parse results for interactive agent loops." Today, that downstream consumer does not exist yet. The current fitness scoring pipeline is type-checker-driven after successful parse; it does not currently consume partial Surf parse artifacts. That means adopting `chumsky` now pays the rewrite cost before the product has a place to cash in the recovery benefits.

There is also a versioning concern. `chumsky` is active and valuable, but it has continued to evolve materially. That is not disqualifying, but it does raise the maintenance bar for a foundational parser dependency.

### Alternatives

- **Stay custom** and improve diagnostics around the existing parser.
- Add richer span-driven reporting with `ariadne`, `codespan-reporting`, or more structured `miette` integration before replacing the parser.
- Use a smaller, targeted refactor inside the current parser to reduce duplication without changing parsing strategy.
- `lalrpop` is a credible alternative for grammar-driven parsing, but it is a worse fit for the current Pratt-heavy expression design and would still be a rewrite.

### Interactions

- `logos` is slightly more appealing if `chumsky` is adopted later, but not enough to change the recommendation by itself.
- `salsa` raises the importance of stable parser outputs and file/module query boundaries. That argues for avoiding a parser rewrite until incremental architecture requirements are clearer.
- If Phase 2 chooses "interactive diagnostics first" as a product priority, `chumsky` moves from "no" to "prototype."

## 3. `petgraph`

**Recommendation:** Do not adopt as a replacement for the current DAG. Keep the custom DAG and only revisit if Chelis outgrows append-only DAG semantics.  
**Confidence:** High

### Benefit Magnitude

- **Correctness:** Low-to-medium. `petgraph` provides mature generic graph machinery, but Chelis already gets topological order for free and checks structural invariants explicitly.
- **Maintainability:** Low. The current DAG is small and intentionally specialized.
- **Performance:** Unclear. There is no evidence today that a generic graph library would improve optimizer or lowering performance.
- **Capability:** Low for current needs. Most of `petgraph`'s algorithm surface is unnecessary for a compiler IR that is deliberately acyclic and mostly rebuilt by passes.

### Migration Cost

- High relative to the benefit.
- `chelis-ir/src/dag.rs`, lowering, optimization, future AD rewrites, verification, and backend traversal would all change shape.
- The highest risk is not the initial port. It is making day-to-day optimization code less ergonomic because every rewrite now has to fight a general graph API rather than a compiler-specific DAG model.

### Timing

- **Now:** No.
- **Between Phase 0 and 1:** No.
- **Phase 1:** Only if fusion and memory planning force the IR toward a much richer mutable graph model.
- **Never:** Plausible.

### Reasoning

Chelis does not currently have a "graph problem." It has a "compact compiler IR" with these properties:

- nodes are append-only
- edges always point backward
- topological order is structural
- passes often rebuild new DAGs instead of performing arbitrary in-place surgery

That is exactly the kind of situation where a custom arena-plus-index design beats a general graph crate. `petgraph` would be more compelling if Chelis needed arbitrary graph algorithms, stable deletions in a long-lived mutable graph, or many alternate graph representations. Today it does not.

The current verifier is also doing compiler-specific work `petgraph` would not replace: arity checks, precision checks, dimension compatibility, reduction-axis validation, and dangling-node rules.

### Alternatives

- **Stay custom** and evolve the current DAG only when specific needs appear.
- If identity and deletion pressure increases, consider lighter building blocks first:
  - `slotmap`
  - `la-arena`
  - `id-arena`
- If the IR becomes truly mutable, consider adding explicit use-lists or consumer tables before replacing the whole storage layer.
- `daggy` exists, but it is not obviously a better fit than today's specialized DAG.

### Interactions

- Adopting `petgraph` does **not** materially improve the case for `egg`. `egg` wants an expression language and rewrite system, not a generic host graph.
- `petgraph` and `salsa` are mostly orthogonal; incremental queries care more about compiler boundaries than graph storage.
- A stronger AD/fusion pipeline could increase pressure on the DAG representation, but the first likely step is richer custom IR metadata, not a graph-library swap.

## 4. `egg`

**Recommendation:** Do not replace current optimizer passes with `egg`. Plan a Phase 1 prototype specifically for fusion and rewrite-search-heavy optimizations.  
**Confidence:** Medium

### Benefit Magnitude

- **Correctness:** Medium. Rewrite systems can reduce ad hoc optimizer bugs once the rewrite set is stable.
- **Maintainability:** Low for today's optimizer, potentially high for future fusion logic.
- **Performance:** Potentially high for optimization quality, but not necessarily for compile time or memory use.
- **Capability:** High for exploring non-greedy fusion strategies and cost-based extraction.

### Migration Cost

- Medium-to-high, depending on scope.
- For today's optimizer, cost is unjustified: DCE, CSE, and constant folding are simpler as direct passes over the current DAG.
- For Phase 1 fusion, cost becomes more defensible because the problem changes from "apply a few obvious local optimizations" to "search among many equivalent rewrites."
- Integrating `egg` means defining a language, rewrite rules, analyses, and a cost model, then translating between Chelis IR and e-graph terms.

### Timing

- **Now:** No for current optimization passes.
- **Between Phase 0 and 1:** A small research spike is reasonable if fusion design work starts early.
- **Phase 1:** Yes, as a prototype candidate for fusion.
- **Phase 2:** Still useful, but the best proving ground is likely Phase 1 kernel fusion.

### Reasoning

The current optimizer is simple enough that `egg` would be machinery in search of a problem. DCE and CSE are already natural graph operations. Constant folding is a direct local simplification. Replacing those with equality saturation would likely make the optimizer harder to explain, debug, and tune.

Fusion is different. Once Chelis starts asking questions like:

- which elementwise chains should fuse?
- when does fusion hurt reuse?
- how should costs trade off memory traffic, code size, launch count, and backend constraints?

then equality saturation starts to fit the shape of the problem. That is where `egg` can plausibly outperform a bespoke greedy rewrite engine.

### Alternatives

- **Stay custom** for Phase 0 optimizations.
- Build a lightweight custom rewrite engine for fusion if the rewrite space stays small and backend-specific.
- Consider `egglog` if Chelis later wants a more incremental, declarative rule system, though that is a bigger conceptual jump than `egg`.
- Use backend-specific greedy or DP-based fusion heuristics before committing to equality saturation.

### Interactions

- `petgraph` does not meaningfully de-risk `egg`; the two operate at different abstraction levels.
- `salsa` can make repeated optimization runs cheaper in interactive workflows, which slightly improves the long-term case for heavier optimizer infrastructure.
- `egg` is the one listed choice whose value rises sharply with Phase 1 fusion work. That makes sequencing important: do not adopt it for Phase 0 to solve a Phase 1 problem prematurely, but do not forget it once fusion planning becomes concrete.

## 5. `salsa`

**Recommendation:** Adopt later as an architectural direction, but not before Phase 0 compilation is complete. Prepare for it now by keeping phase APIs pure and query-friendly.  
**Confidence:** High

### Benefit Magnitude

- **Correctness:** Medium. Demand-driven computation can improve architectural discipline by making hidden state and phase coupling more obvious.
- **Maintainability:** High once the compiler has enough moving parts that recomputation and dependency structure matter.
- **Performance:** High for Tide, REPL, LSP, and many-variant agent loops; low for one-shot batch compilation.
- **Capability:** High. Incremental compilation, on-demand analysis, and shared caches are exactly the use cases `salsa` is meant for.

### Migration Cost

- Very high.
- This is not a crate swap. It is a compiler architecture rewrite.
- Public and internal APIs across the workspace would change from linear "call phase N+1 with phase N output" flows to query-oriented interfaces.
- Existing mutable internal state in inference, lowering, or optimization would need to be audited and likely reshaped around pure query functions plus tracked inputs.

### Timing

- **Now:** No.
- **Between Phase 0 and 1:** Only design preparation, not implementation.
- **Phase 2:** Yes. This is the right phase to seriously adopt it.
- **Never:** No. Unlike some other choices, this one does map directly to stated product needs.

### Reasoning

Of the six libraries, `salsa` has the clearest alignment with the roadmap. The repo and spec explicitly point toward interactive workloads where many small variants are compiled and checked repeatedly. That is almost a textbook argument for queries, memoization, dependency tracking, and selective recomputation.

The only reason not to adopt it now is timing. Phase 0 still needs the simplest path to a complete end-to-end compiler. Pulling `salsa` in today would force a major refactor before the backend, AD pass, and MNIST path have stabilized. That is the wrong order.

The right move is to treat `salsa` as a future architecture target and make current decisions that keep the migration feasible:

- keep crate APIs as pure as possible
- keep hidden mutable state localized
- separate file/module/function boundaries clearly
- avoid coupling diagnostics and caching to global side effects

### Alternatives

- Introduce **lighter caches first**:
  - file-content cache
  - parsed-module cache
  - typed-definition cache
  - IR cache keyed by source hash or definition identity
- Build a minimal internal query layer before adopting `salsa`.
- If Tide lands with only shallow interactivity, ad hoc memoization may be enough for longer than expected.

### Interactions

- `salsa` materially strengthens the case for `cranelift`: once front-end work is incremental, backend compile latency matters more.
- `salsa` weakens the case for a parser rewrite right now because parser API needs should be informed by future query boundaries.
- `salsa` has little to do with `petgraph` directly, but it does increase the value of compiler components that are pure and easy to recompute.

## 6. `cranelift`

**Recommendation:** Do not add now. Consider a bounded prototype for Tide or REPL execution after the C backend and interactive surface are real.  
**Confidence:** High

### Benefit Magnitude

- **Correctness:** Low. This is not mainly a correctness play.
- **Maintainability:** Low-to-negative at first because it adds a second backend.
- **Performance:** High for compile-to-run latency in interactive execution; low for batch compilation.
- **Capability:** Medium. It adds a direct native execution path and possible JIT support.

### Migration Cost

- High.
- This is a new backend crate plus shared lowering/codegen abstractions and a test burden for backend agreement.
- The cost is not only implementation. It is also the obligation to keep two execution paths numerically and semantically aligned.
- FFI, runtime calls, memory layout, and external library calls would need explicit handling rather than relying on C as the integration layer.

### Timing

- **Now:** No.
- **Between Phase 0 and 1:** No.
- **Phase 2:** Yes, as a prototype if Tide/REPL latency becomes painful.
- **Never:** Also possible if a cached C path is "fast enough."

### Reasoning

The current strategic backend is still C. That path fits three planned needs well:

- OpenMP-style CPU parallelism
- easy C/BLAS integration
- a natural host-side bridge for future GPU work

`cranelift` is attractive for a different reason: avoiding external compiler startup and getting machine code quickly for interactive runs. That is a real advantage, but it is not the same problem as "build the primary production backend."

There is also an important repo-specific observation: Chelis already has a tensor-aware interpreter in `chelis-ir/src/eval.rs`. That means the very first interactive execution path may not need a JIT at all. For REPL-scale evaluation, an interpreter or cached AOT helper may deliver most of the user experience benefit at much lower complexity.

### Alternatives

- Use the existing IR evaluator as the first interactive execution path.
- Add **persistent C compilation infrastructure** first:
  - object caching
  - content-addressed build artifacts
  - long-lived helper process instead of spawning a full toolchain per evaluation
- Defer native JIT work until there is a concrete Tide workload that justifies it.
- If a JIT is eventually needed, evaluate whether a narrower `cranelift-*` crate set is better than the umbrella crate.

### Interactions

- `salsa` significantly improves the payoff of `cranelift`, because once parsing/type-checking/lowering are incremental, backend latency dominates more of the interactive path.
- `cranelift` should not become the "main backend" before the Phase 1 GPU host/runtime strategy is settled; otherwise Chelis risks splitting its backend story too early.
- `egg` and `cranelift` are largely independent choices.

## Alternatives We Should Also Keep In View

The listed six crates are not the only reasonable options. The missing alternatives that matter are:

- **Stay custom** where Chelis already has compact, well-tested, phase-appropriate code. This is the right answer more often than it first appears.
- **Diagnostics-first front-end improvements** before parser replacement:
  - `ariadne`
  - `codespan-reporting`
  - deeper `miette` integration
- **Lighter identity/arena crates** before `petgraph`:
  - `slotmap`
  - `la-arena`
  - `id-arena`
- **Lighter caching/query infrastructure** before `salsa`:
  - hash-keyed caches
  - module/function memoization
  - a minimal internal query layer
- **Execution shortcuts before `cranelift`**:
  - the existing IR evaluator
  - cached C artifacts
  - a persistent compiler helper process
- **Rewrite alternatives before `egg`**:
  - custom greedy fusion passes
  - a smaller bespoke rewrite engine
  - `egglog` only if Chelis later wants a more declarative incremental rule system

The conspicuously absent category is **diagnostic tooling**. If Chelis wants much better compiler UX in the near term, the highest-ROI additions may be reporting crates, not parser or lexer rewrites.

## Choice Interactions

### `petgraph` ↔ `egg`

- Weak interaction.
- `petgraph` does not make `egg` substantially easier to use.
- If anything, a generalized graph layer risks increasing conceptual distance between Chelis IR and the term language `egg` wants.

### `salsa` ↔ `cranelift`

- Strong positive interaction.
- `cranelift` pays off most when the rest of the compiler can avoid recomputing unchanged work.
- Without incremental queries, a JIT still saves C compiler latency, but it does not solve repeated front-end and middle-end work.

### `salsa` ↔ parser/frontend rewrites

- `salsa` argues for caution.
- Before replacing the front end, Chelis should know what its query boundaries are: file, module, declaration, function body, expression, or some mix.
- That makes "rewrite parser first" the wrong sequence unless parser diagnostics become an urgent product need.

### `egg` ↔ Phase 1 fusion

- Strong positive interaction.
- Phase 0 optimizations are too simple to justify `egg`.
- Phase 1 fusion is the point where rewrite-space exploration, equivalence reasoning, and extraction cost models could become first-order.

### Backend choices ↔ GPU host/runtime direction

- Strong sequencing constraint.
- The C backend aligns naturally with a future "C host plus emitted kernel strings" design.
- A `cranelift` backend should be treated as an additional interactive path, not as a strategic replacement for the host-side architecture that Phase 1 likely wants.

### `logos` ↔ `chumsky`

- Mild positive interaction, but not enough to change either recommendation.
- If Chelis ever rewrites the parser around combinators, a declarative lexer becomes a more natural fit.
- The nested-comment constraint still blocks a clean all-in `logos` story.

## Ranked Shortlist: What to Revisit Soonest

1. **Prepare for `salsa` without adopting it yet.** Keep crate boundaries pure and query-friendly during the remaining Phase 0 work.
2. **Prototype `egg` when Phase 1 fusion design starts.** Do not force it into the current optimizer.
3. **Measure the interactive path before touching `cranelift`.** First test whether the existing IR evaluator or a cached C flow is already sufficient for Tide-scale latency.

## Final Recommendations Table

| Library | Recommendation | Timing |
|---|---|---|
| `logos` | Do not adopt | Never or only after Phase 2 if lexer maintenance becomes painful |
| `chumsky` | Do not rewrite now; possible prototype later | Phase 2 prototype only if recovery/diagnostics become central |
| `petgraph` | Do not adopt | Never unless the IR stops being a simple append-only DAG |
| `egg` | Prototype for fusion, not current optimizer | Phase 1 |
| `salsa` | Adopt later as architectural direction | Phase 2 |
| `cranelift` | Prototype only, as a second backend | Phase 2 |
