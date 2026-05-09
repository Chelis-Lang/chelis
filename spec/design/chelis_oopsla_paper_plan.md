# Chelis Language Systems Paper - Plan

**Working title:** TBD. Something like "Chelis: A Typed Tensor Language with Automatic Differentiation Through Dataframes, Effect-Tracked Stochasticity, and Compiled Fusion"
**Venue (highly tentative):** OOPSLA 2027 (SPLASH)
**Deadlines:** R1 ~Oct 2026, R2 ~Mar 2027 (estimated from historical pattern - 2027 CFP not yet posted)
**Format:** PACMPL, acmsmall, 23 pages max, double-blind
**Status:** Not started. Blocked on Coral and ideally Shoals shipping for the evaluation section.

---

## What This Paper Is

The language design and systems paper for Chelis. The contribution is not the type theory (that's LaCaDiLE at POPL) and not the coding model training methodology (that's the ICLR paper). The contribution is: combining effects + linearity + AD + named dimensions in one compiled language produces practical benefits that no existing system achieves, demonstrated with a working compiler, ecosystem libraries, and benchmarks.

This is the paper that talks about:
- AD through dataframe operations (Coral - no other dataframe library can do this)
- Effect-tracked Monte Carlo reproducibility (no other language enforces this at the type level)
- Compiled fusion beating the pydata stack by 3-7x on compound expressions (measured)
- Greeks for free via `grad` (Shoals)
- Named tensor dimensions catching shape bugs at compile time (without dependent types)
- The Octant workflow (quants write LaTeX, get compiled binaries with provenance)
- The dual-syntax design (Surf for humans, Deep for AI agents) as an AI-native language design choice

None of these belong in the POPL paper (which is pure metatheory) or the ICLR paper (which is about training the coding model). They belong here.

---

## Relationship to Other Papers

**LaCaDiLE (POPL 2027):** Proves the type system is sound. This paper cites it in one sentence: "The type system is formally verified [ref]." Then moves on to the evaluation. The POPL paper proves correctness; this paper proves usefulness.

**ICLR paper (coding model):** Proves the LLM training methodology works. This paper cites it for the Deep syntax design rationale: "The s-expression representation is designed for AI code generation [ref]." The ICLR paper uses Chelis as a testbed; this paper presents Chelis as a contribution.

---

## Contribution Claims (what OOPSLA reviewers will evaluate)

1. **Language design:** The first language that combines algebraic effects, linear types, reverse-mode AD, named tensor dimensions, and dual human/AI syntax in a single coherent system. Each feature exists in isolation elsewhere (Dex has effects + AD, Futhark has linearity, Granule has graded effects + linearity). The novelty is the combination and the domain-specific interactions it enables.

2. **AD through dataframe operations:** Coral's `filter -> gather` and `aggregate -> reduction` pipeline is differentiable. `grad(portfolio_risk)` where portfolio_risk filters rows and aggregates a column produces correct gradients. No existing dataframe library (pandas, Polars, RAPIDS, Spark) supports this. This is a capability that emerges from the design decision to make dataframe columns be tensors on the same lazy DAG as all other tensor operations.

3. **Effect-tracked stochastic computation:** A Monte Carlo simulation carries `Random` in its type. Handling it with `with seed(42) { ... }` guarantees reproducibility - not by convention, by the type system. This is a regulatory requirement in finance (identical results on re-run) that no other language enforces statically.

4. **Compiled fusion vs the pydata stack:** Measured performance. Compound expressions (`normal_cdf(x) * exp(-x^2)`) achieve 3.0-3.6x over numpy at 100k elements because the compiler fuses them into single loops. Per-kernel wins of 2.3-6.8x on special functions and distribution CDFs. 881 scipy-parity assertions validate correctness.

The current benchmarks use the scalar C backend with auto-vectorization only. The SIMD support plan (Levels 1-3) is expected to improve these numbers further: `restrict` annotations unlock auto-vectorization on aliased loops, and Sleef/vForce integration provides SIMD-width math functions (exp, log, sin, erf) in fused kernels. Level 3 is targeted for completion before the paper's benchmark section is finalized.

5. **Implementation evidence:** A working Rust compiler (6 crates), CPU (C + OpenMP) and GPU (HIP/AMD) backends, a standard library covering PyTorch's NN/optim/loss surface, and three ecosystem libraries (Nautilus for numerical computing, Coral for dataframes, Shoals for finance). Not a paper design - a running system.

---

## Paper Structure (sketch)

### Section 1: Introduction (~2 pages)

The problem: tensor programming today is fragmented. Numerical correctness (shapes), memory safety (use-after-free on GPU tensors), stochastic reproducibility (Monte Carlo seeds), and automatic differentiation (gradients) are all managed by convention, not by the language. PyTorch checks none of these statically. JAX checks some via tracing but only at runtime. Julia has no effect system.

The thesis: a language that combines effects, linearity, named dimensions, and AD can check all of these statically, and the same design decisions that enable static checking also enable performance (fusion, safe memory management) and novel capabilities (AD through dataframe operations).

Contributions: (1) language design, (2) AD through dataframes, (3) effect-tracked stochasticity, (4) compiled fusion with measured performance, (5) implementation + ecosystem.

### Section 2: Language Design (~4 pages)

The key design decisions and their interactions. Not the formal calculus (that's LaCaDiLE at POPL) - the practical design choices:

- Named tensor dimensions and how they prevent shape bugs without dependent types
- The effect system (`Random`, `IO`, `Resource`, `Fail`) and how it tracks data provenance and reproducibility
- Linear types and how they prevent use-after-transfer for GPU tensors
- `grad` requiring linearity + effect purity - the practical consequence of the LaCaDiLE formalization
- `vmap` adding a batch dimension via `addDim`
- Dual syntax (Surf/Deep) as an AI-native design choice - humans read Surf, agents generate Deep, compiler fitness scores on Deep

Show the interactions: `grad` requires linear use (linearity) and effect purity (effects). `vmap(grad(f))` gives per-example gradients (AD + vectorization). `with seed(42) { vmap(sample) }` gives reproducible batched sampling (effects + vectorization). These interactions are the paper's intellectual substance.

### Section 3: Ecosystem and Novel Capabilities (~4 pages)

#### AD Through Dataframes (Coral)

The headline result. Show the full pipeline:

```chelis
def portfolio_risk(prices: Frame, threshold: f32) -> f32 = {
  mask = gt(get_float_col(prices, "volatility"), threshold)
  high_vol = filter(prices, mask)    -- gather under the hood
  mean(get_float_col(high_vol, "return"))
}
sensitivity = grad(portfolio_risk, wrt=threshold)(prices, 0.3)
```

Explain why this works (columns are tensors, filter is gather, gather is differentiable). Explain why pandas/Polars can't do this (columns are not on a differentiable computation graph). Show the financial use case: sensitivity of a risk measure to a portfolio parameter.

#### Effect-Tracked Monte Carlo (Shoals)

```chelis
def mc_price(spot: f32, vol: f32, paths: int64) -> f32 ! { Random } = {
  -- pricing logic using normal_sample
}
-- Reproducible: same seed -> same price, guaranteed by the type system
price = with seed(42) { mc_price(100.0, 0.2, 100000) }
```

Explain how `Random` as an effect prevents accidental non-reproducibility. The type system won't let you call `mc_price` without handling `Random`. This is a regulatory-grade guarantee.

#### Greeks for Free (Shoals)

```chelis
delta = grad(black_scholes_price, wrt=spot)(spot, vol, rate, T, strike)
portfolio_greeks = vmap(grad(price_fn))(portfolio_spots, portfolio_vols, ...)
```

No bump-and-reprice. No finite differences. No hand-coded formulas. Show the comparison: traditional quant workflow (implement price, separately implement each Greek) vs Chelis (implement price once, get all Greeks via `grad`).

### Section 4: Compiler and Performance (~3 pages)

- Compiler pipeline: Surf/Deep -> typed AST -> tensor DAG + host IR -> fusion -> C+OpenMP / HIP
- Fusion: how compound expressions compile to single loops
- The two-lane architecture (tensor DAG vs host lane) and why it matters for mixed numeric/string workloads (Coral)

Benchmark results from `BENCHMARK_FINDINGS.md`:
- Compound expression fusion: 3.0-3.6x over numpy at n=100k
- Special functions: `erfinv` 6.5x, `normal_inv_cdf` 6.8x
- Distribution CDFs: 2.3x after convergence optimization
- Solver per-call: 45x (`brent`) to 4000x (`rk4` vs scipy dispatch, with honest caveats)
- 881 scipy-parity assertions validating correctness
- LTO finding: 3.8x improvement from cross-TU inlining - an important workaround for
  generated helper overhead, but not the long-term codegen story for backend library
  dispatch

### Section 5: Related Work (~2 pages)

Comparison table (same structure as the POPL paper's Section 7 but focused on practical capabilities, not formalism):

| System | Effects | Linearity | Dimensions | AD | Dataframe AD | Fusion | GPU |
|---|---|---|---|---|---|---|---|
| PyTorch | x | x | x (runtime) | yes (runtime tape) | x | x | yes (CUDA) |
| JAX | x | x | x (traced) | yes (tracing) | x | yes (XLA) | yes (TPU/CUDA) |
| Dex | yes | x | yes (for-indexed) | yes | x | yes | yes |
| Futhark | x | yes (uniqueness) | x (size) | x | x | yes | yes (OpenCL) |
| Julia | x | x | x | yes (Enzyme) | x | x | yes (CUDA.jl) |
| pandas | x | x | x | x | x | x | x |
| Polars | x | x | x | x | x | yes (query opt) | x |
| **Chelis** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes (HIP)** |

Develop deeply: Dex (closest on effects+AD, no linearity, no dataframes), JAX (closest on tracing+AD+fusion, no static types), pandas/Polars (dataframe state of the art, zero AD).

### Section 6: Limitations and Future Work (~1 page)

Honest:
- f32 precision throughout (f64 support partial)
- Missing scalar builtins (`cos`, `tan`, `abs` - workarounds documented)
- No query optimizer for Coral (Polars wins on complex multi-join analytical queries)
- Single-machine only (no distributed computation)
- OpenMP doesn't parallelize composed functions (only single RISC ops)
- No SIMD vectorization in the C backend (relies on gcc auto-vectorization)
- Cross-function user-defined helpers can currently hide BLAS-equivalent tensor math
  from the backend specializer; clang LTO mitigates helper overhead but does not replace
  a compiler-owned specialization pass
- Small team, early-stage ecosystem, zero production deployments
- GPU backend (HIP) less mature than C backend

Future: distributed runtime (the D1-D4 phasing), cross-function specialization for
BLAS-equivalent library helpers (`cross_function_specialization.md`), complex numbers
(Phase 5f, unblocking Signal), the Octant LaTeX bridge, general-n iterative LinAlg, and
Hull - a self-hosted executable specification shell where the LaCaDiLE typing rules are
implemented as Chelis functions over Deep AST ADTs, enabling differential testing
between the spec and the compiler in the language itself. Hull is a natural OOPSLA
story: the language specifies itself, the spec is checked by the compiler, and the
compiler is tested against the spec. Self-hosted executable specifications are rare in
PL literature and directly reinforce the "designed for AI reimplementation" thesis.

The executable-properties-as-spec pattern (`@property` annotations + `chelis fuzz`) is a natural extension of the compiler fitness story for this section: properties are the user-facing version of what the compiler fitness score does for the RLVR training loop. The compiler checks structural soundness automatically; properties check domain correctness empirically. Together they form a trust stack that no Python-based platform can offer. If `chelis fuzz` with `@property` is shipped before submission, it strengthens Section 3 (novel capabilities). If not, it belongs here in Section 6 (future work). Full design: `chelis_trust_stack.md`.

### Section 7: Conclusion (~0.5 pages)

Restate: first language combining effects + linearity + AD + named dimensions with a working compiler and ecosystem. AD through dataframes is a novel capability. Effect-tracked reproducibility is a regulatory-grade guarantee. Compiled fusion beats the pydata stack. The formal foundations are verified (cite LaCaDiLE at POPL).

---

## Prerequisites (what must be shipped before writing)

| Artifact | Status | Why it's needed |
|---|---|---|
| Nautilus v0.1.0 | Shipped | 881 scipy-parity assertions, benchmark data |
| `BENCHMARK_FINDINGS.md` | Shipped | Performance evaluation section |
| Coral v0.1.0 | In progress | AD-through-dataframes is contribution #2 - can't claim it without the implementation |
| Shoals (at minimum Black-Scholes + Greeks + Monte Carlo) | Planned (3l) | Greeks-for-free and effect-tracked MC are contributions #3-4 - need at least a demo |
| LaCaDiLE submitted to POPL | In progress | Cited as "[our companion paper]" for the formal foundations |

**Coral is the hard blocker.** The paper's most novel claim (AD through dataframe operations) requires Coral to be implemented, tested, and benchmarked. Shoals needs at minimum a Black-Scholes pricer with `grad`-derived Greeks - that's a small amount of code once Nautilus and Coral exist, but it must be a running artifact, not a future plan.

**Octant is nice-to-have but not required.** The LaTeX->Deep workflow is compelling but the paper is strong without it. Include it if it's shipped; omit it if not. Don't hold the paper for Octant.

---

## Anonymization Considerations

Double-blind at OOPSLA. The paper must not name the language or link to the repos. Refer to "our language" and "our compiler" throughout. Cite LaCaDiLE in third person. The Lean proof scripts and compiler source are submitted as anonymous supplementary material.

The risk: Chelis is publicly described in multiple design docs, the Nautilus repo, and potentially conference talks. If a reviewer googles distinctive phrases from the paper and finds the Chelis repos, anonymity is compromised. Mitigation: don't use distinctive names (`Nautilus`, `Coral`, `Shoals`, `Octant`, `Surf`, `Deep`) in the paper. Use generic names (the numerical library, the dataframe library, the finance library, the surface syntax, the s-expression syntax). Rename in the artifact submission.

---

## Timeline (no dates, just dependencies)

```text
LaCaDiLE submitted to POPL
           ↓
Coral v0.1.0 ships
           ↓
Shoals Black-Scholes + Greeks demo
           ↓
Benchmark Coral operations (AD-through-filter-aggregate performance)
           ↓
Write paper (Sections 2-4 are the bulk; Sections 1, 5-7 are straightforward)
           ↓
OOPSLA R1 or R2 submission
```

The writing can begin in parallel with Coral development: Section 2 (language design) and Section 5 (related work) don't depend on Coral. Section 4 (compiler + performance) can use existing Nautilus benchmarks and be extended with Coral benchmarks when available. Section 3 (ecosystem + novel capabilities) is the section that's blocked by Coral and Shoals.

---

## Open Questions

1. **Does the paper need Octant?** The LaTeX->Deep workflow is a strong demo for the OOPSLA audience (practical tool, user-facing impact). But it's a separate system from the core language contribution. Include as a section if shipped; omit if not.
2. **Single paper or split by contribution?** The current plan is one paper covering the full language design + ecosystem + evaluation. An alternative is to split language/compiler and domain-capability material, but the current recommendation is to keep one coherent systems paper unless page limits force a change.
