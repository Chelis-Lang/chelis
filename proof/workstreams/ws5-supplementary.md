# WS5: Supplementary Material

- **Size:** Medium.
- **Dependencies:** [WS2](ws2-paper-proofs.md), [WS3](ws3-lean-mechanization.md), [WS4 §2](ws4-paper-draft.md) (case studies derive from §2 examples).
- **Output:** Anonymized PDF appendix and Lean tarball, uploaded with the HotCRP submission. The current scaffold starts in [`../paper/appendix-outline.tex`](../paper/appendix-outline.tex).

---

## WS5.1 — Full proofs appendix

Complete proofs of all five theorems. No page limit. Anonymized PDF.

Start from [`../paper/appendix-outline.tex`](../paper/appendix-outline.tex). The appendix should open with an honest theorem-status snapshot, then separate: substitution/progress/addDim, preservation plus the runtime-invariant counterexamples and `HandlerAwareRuntimeLinear` debt split, adjoint typing on the typed transform surface, and the still-denotational AD-correctness obligations.

## WS5.2 — Lean proof scripts

Anonymized tarball of `chelis-metatheory/`. README with: file structure, theorems proved, admitted lemmas, build instructions, Lean version, non-standard axioms. The README must explicitly record the `adjointFrom` versus `adjointTypedFrom` boundary and the current handler-aware runtime debt split.

## WS5.3 — Case studies

Example programs from §2, fully typed, with inferred effects and dimension-checked types.
