# Risk Register

Cross-cutting risks for Chelis Proof. Updated as workstreams progress.

| Risk | Impact | Mitigation |
|---|---|---|
| Tape/borrow mechanism unsound with linear contexts | **Highest** — blocks adjoint typing lemma, which blocks preservation, which blocks everything | **Resolve first in WS1.** Write the adjoint transformation for `mul(a, b)` with explicit tape borrows. Verify the linear context threading: borrow before consume, taped borrows in scope for backward pass. If the borrow/consume ordering creates an irreconcilable conflict with the linear context algebra, the entire AD formalization needs redesign. |
| `handle` rule unsound with linear contexts | High — requires redesign | The continuation-as-linear-binding design is the mitigation. If `k` is linear, multi-use is a type error. Test by attempting the preservation case for `handle` on paper (WS2.5) before Lean. |
| `grad` adjoint doesn't preserve types | High — preservation fails | WS2.2 (adjoint typing lemma) is a standalone proof obligation. The explicit `Accum` handler + tape borrows must produce a well-typed term. Prove this lemma first. |
| Substitution lemma intractable in Lean | Medium — blocks Lean proofs | `Δ` is non-linear (passes through unchanged), limiting the damage. Only `Γ` requires splitting. If still intractable, use intrinsically-typed terms (de Bruijn) to avoid substitution lemma entirely — different Lean architecture, same theorems. |
| `Accum` interacts badly with other effects in reduced term | Medium — effect row algebra breaks | The explicit handler in the reduced term is standard — `handle[Accum] body with {...}`. `Accum` never appears in user-facing effect rows; it's introduced and immediately handled within the `grad` reduction. |
| AD correctness (Theorem 5) unprovable in Lean | Medium — weakens contribution | Fall back: state in Lean, prove on paper, note in README. Four mechanized + one paper-proved is still strong. |
| Reviewer sees "implementation paper" | Medium — low scores | The DiLL framing prevents this. Lead with theory. Lean mechanization is rigor. Implementation is evidence. |
| Fresh dimension in `vmap` causes capture | Low — `addDim` resolves | `addDim` is a meta-function, not a quantifier. No variable capture. Freshness is a side condition. |
| Paper exceeds 25 pages | Low — fixable | §6 and §7 are compression targets. §3–§4 should not be compressed. |
