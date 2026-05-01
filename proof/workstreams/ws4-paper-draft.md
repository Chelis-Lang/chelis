# WS4: Paper Draft

- **Size:** Large. 25 pages.
- **Dependencies:** [WS1](ws1-core-calculus.md) (§3–§5), [WS2](ws2-paper-proofs.md) (§5 proof sketches). §1/§2/§6/§7 can begin in parallel with WS1/WS2.
- **Output:** POPL 2027 submission PDF. LaTeX sources live under [`../paper/`](../paper/).
- **See also:** [decisions.md](../decisions.md), [anonymization.md](../anonymization.md), [ws5-supplementary.md](ws5-supplementary.md).

---

Write the actual POPL submission.

## WS4.1 — LaTeX setup

`acmart` with `acmsmall` option. Double-blind (`anonymous` option).

```latex
\documentclass[acmsmall,screen,review,anonymous,nonacm]{acmart}
```

## WS4.2 — §1 Introduction (~1.5 pages)

The intellectual spine is differential linear logic.

Opening: tensor programs combine numerical computation, stochastic elements, device management, and automatic differentiation. Current systems (PyTorch, JAX, Julia) check none of these statically.

The thesis: Ehrhard and Regnier's differential linear logic established that differentiation and linearity are fundamentally connected — the differential operator is the dual of the exponential modality. LaCaDiLE is the first computational type system that realizes this connection for tensor programs, with algebraic effects providing the additional structure for stochasticity and device management.

The specific connection (set up for §3): in DiLL, the typing rule for the differential combinator requires its argument to be in the linear zone (not under `!`). In LaCaDiLE, `grad` requires its argument to use the differentiated parameter linearly. Same constraint, different formalisms. Side-by-side comparison of the typing rules.

Contributions:

1. LaCaDiLE: a core calculus for tensor computation with algebraic effects, linear types, and named dimension indexing, realizing the DiLL connection between differentiation and linearity
2. Metatheory: five theorems — type soundness, dimension safety, effect correctness, linearity soundness, AD correctness
3. Lean 4 mechanization of the core metatheory, with the final theorem package stated honestly in the supplementary material
4. Implementation evidence: Chelis, a compiler for a full language extending LaCaDiLE, targeting CPU and GPU

## WS4.3 — §2 Motivation and Examples (~2 pages)

Running examples exercising the key interactions:

- *Training loop:* forward pass allocates on GPU (`Resource`), uses dropout (`Random`), computes loss, calls `grad` (requires `Diff` + linear use). The `Accum` effect mediates gradient accumulation. Shows all interactions in one program.
- *Monte Carlo estimation:* repeated sampling (`Random`), handled by `withSeed` for reproducibility. Dimension-typed: `vmap` over samples adds a batch dimension.
- *Multi-device pipeline:* tensors on CPU and GPU with explicit transfers. Linear types prevent use-after-transfer.

For each: (a) the well-typed program, (b) what goes wrong without each property (the bug caught), (c) the type error.

## WS4.4 — §3 Core Calculus (~3 pages)

Formal definition from WS1. Figures for typing rules. Highlight:

- `grad` requiring linearity + `Diff` capability + introducing/handling `Accum` via explicit handler in reduced term
- The tape mechanism: forward pass borrows intermediates, backward pass uses taped borrows. This makes AD memory management visible and typed — contrast with PyTorch's implicit autograd tape (a major source of memory leaks).
- `vmap` with `addDim` meta-function
- `handle` with continuation as linear binding (one-shot falls out of linearity)
- The DiLL side-by-side comparison for `grad`'s typing rule
- `DiffCompat` effect set: `Resource` is differentiation-compatible (allocation doesn't affect the mathematical computation), `Random`/`IO`/`Fail` are not

**Remark on `vmap(grad(f))` vs `grad(vmap(f))`:** Both are well-typed with the same type signature (`tensor[batch, d̄] → tensor[batch, d̄]`) but compute different mathematical quantities — per-example gradients vs. batch-aggregated gradient. The type system guarantees both are well-typed and shape-safe but does not distinguish their mathematical semantics. Worth a paragraph: this is a well-known source of confusion in JAX, and LaCaDiLE makes both compositions typecheck cleanly while the dimension types ensure shape correctness for each.

## WS4.5 — §4 Operational Semantics (~2 pages)

Reduction rules including heap store. Figures. The interesting rules:

- `grad` reducing to `λx. handle[Accum] adjoint(e, x) with {...}` — showing the Accum handler is standard, not bespoke
- The tape: forward-pass borrows saved for backward-pass reference
- Effect handling with one-shot continuation capture
- Store allocation/deallocation tied to linearity; `copy` as physical allocation
- `vmap` should be presented with one explicit shared batch-dimension choice in both typing and stepping; the Lean mechanization now uses that annotated surface directly

## WS4.6 — §5 Metatheory (~3 pages)

All five theorem statements. Proof sketches. Key lemmas: substitution, adjoint typing, `addDim` preserves typing, store invariant, and the runtime-linearity invariant needed to iterate preservation. Full proofs in supplementary appendix.

The current Lean development forces a stronger correction to the paper statement of preservation: unconditional runtime preservation is false, and the first runtime-linear repair is still not the final invariant for handled runtime states. The exported branch boundary has already been narrowed to the residual config theorem surface `runtimeSafeConfig_step_or_residualDebt` / `preservation_runtimeSafeConfig_or_residualDebt`, while the older `_or_debt` theorems remain only as compatibility wrappers; a fully generic `RuntimeSafeConfig` closure claim is still false because of the concrete `handleOpCtx` counterexample, so the final paper theorem still needs either a stronger public surface or unreachability proofs for the residual cases. Separately, the current AD typing gap is now narrower and more precise: `sum` is mechanized, the old concrete `letBind` / `letpair` routing bug is repaired, and the transform now routes typed cotangent seeds through `pair` / `fst` / `snd` / `copy`. The public typed theorem surface is already on the honest slot-threaded existential output shape and now carries explicit `AdjointTermFresh` tracking, but the AD theorem stack is still not closed: the legacy public `adjointFrom` theorem is false for handled product seeds because `handle` still uses the tensor-only clause path, unrestricted `adjointTypedFrom` is false on higher-order raw terms, and the current private shape-only helper is too weak for `mul` because it forgets the source context needed to type replayed operands. The section should therefore:

1. state Theorem 1b in the honest branch shape first: the executable Lean theorem preserves typing for closed, well-scoped, `RuntimeLinear` terms
2. explain the closed well-typed counterexample showing why unconditional runtime preservation is false for arbitrary runtime terms with duplicated explicit locations
3. explain why `ActiveRuntimeLinear` and then `DeepActiveRuntimeLinear` both fail, and introduce `HandlerAwareRuntimeLinear` as the current stronger cross-boundary candidate
4. summarize the actual exported residual config boundary (`runtimeSafeConfig_step_or_residualDebt` / `preservation_runtimeSafeConfig_or_residualDebt`), the failed generic `RuntimeSafeConfig` `handleOpCtx` closure, and the remaining beta/direct-handler/context-freshness residual debt
5. explain that the live public AD proof sketch is already on `adjointTypedFrom` / `adjointTypedClausesFrom`, that the public output-context claim has already been weakened to the honest existential slot surface, and that the remaining repair is to replace the private shape-only helper with a supported-fragment/source-typed one before closing the final `mul` and clause-threading cases
6. send the exact proved-vs-staged split to the supplement rather than letting §5 imply that the runtime or AD side is already fully closed

## WS4.7 — §6 Implementation (~1.5 pages)

Chelis in third person. Rust compiler, six crates, pipeline, CPU + GPU backends. Evidence that LaCaDiLE's guarantees hold in practice. Practical extensions beyond the core calculus: precision tracking, parameterized `Resource(Device)` effects for multi-device placement, `Arc`-based reference counting (semantically equivalent to physical copy for immutable tensors), dual syntax, fitness scoring.

## WS4.8 — §7 Related Work (~1.5 pages)

Comparison table:

| System | Effects | Linearity | Dimensions | AD typing | Tape/AD memory | Formalized |
|---|---|---|---|---|---|---|
| Dex | ✓ (State, Accum, Reader) | ✗ | ✓ (for-indexed) | ✓ (via effects) | Implicit (no linearity, values freely shared) | ✗ |
| Futhark | ✗ | ✓ (uniqueness) | ✗ (size annotations) | ✗ | N/A | Partial |
| Koka | ✓ (row-polymorphic) | ✗ | ✗ | ✗ | N/A | ✓ |
| Granule | ✓ (graded) | ✓ (graded modal) | ✗ | ✗ | N/A | ✓ |
| Linear Haskell | ✗ (monadic) | ✓ | ✗ | ✗ | N/A | ✗ |
| DiLL | ✗ | ✓ (linear logic) | ✗ | ✓ (differential) | N/A (no computation) | ✓ (on paper) |
| Sized types | ✗ | ✗ | ✓ (dependent) | ✗ | N/A | Various |
| **LaCaDiLE** | **✓ (row-poly)** | **✓ (consume/borrow/copy)** | **✓ (named)** | **✓ (Diff + Accum)** | **Explicit (typed borrows in tape)** | **In progress (Lean 4)** |

The Tape/AD memory column highlights a practical consequence of the theory: LaCaDiLE's type system makes AD memory management visible and safe. PyTorch's autograd tape is a major source of memory leaks; LaCaDiLE's linearity + explicit tape borrows prevent this by construction.

Key comparisons to develop deeply:

- **Dex:** closest. Effects + tensors + AD but no linearity. Dex uses `Accum` for gradient accumulation but can't guarantee buffer safety. AD tape is implicit — values freely shared, no ownership tracking. LaCaDiLE adds linearity, completing the picture and making the tape explicit/typed.
- **Granule:** closest on PL theory. Effects + linearity but no tensors/AD. LaCaDiLE is less general (no graded modalities) but domain-specific with domain-specific interactions.
- **DiLL (Ehrhard & Regnier):** the theoretical foundation. LaCaDiLE is a computational realization of DiLL's connection between differentiation and linearity, extended with algebraic effects for stochasticity/devices and named dimensions for shape safety.
- **Futhark:** linearity for arrays but no effects, no AD typing.

## WS4.9 — §8 Conclusion (~0.5 pages)

Restate the contribution without overclaiming the unfinished mechanization status on the active branch. The DiLL connection is realized computationally; the Lean package must be described in the exact proved-vs-admitted terms that ship in the supplement.
