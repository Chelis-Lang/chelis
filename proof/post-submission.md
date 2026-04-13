# Post-Submission

## Author Response (Sep 7–10)

Anticipated challenges and prepared lines:

- *"How does this differ from Dex?"* — Linearity. Dex has no linear types. Can't guarantee buffer safety. Dex's `Accum` handles gradient accumulation but doesn't enforce linear use of differentiated arguments. LaCaDiLE requires linearity for `grad`, completing the DiLL picture that Dex only partially realizes.

- *"How does this differ from Granule?"* — Domain specificity. Granule is general-purpose graded modalities. LaCaDiLE is a tensor calculus where the interactions produce domain-specific guarantees (shape safety, AD correctness, reproducibility). No tensor types, no AD, no dimension indexing in Granule.

- *"Is the DiLL connection actually novel?"* — The connection between differentiation and linearity is known (Ehrhard & Regnier). The novelty is: (a) realizing it computationally in a tensor calculus, (b) combining it with algebraic effects for stochasticity/device management, (c) mechanizing the result.

- *"Why not dependent types for dimensions?"* — Named dimensions give shape safety without proof obligations. 90% of the safety for 0% of the proof burden. Practical design choice.

- *"Why one-shot handlers?"* — Multi-shot + linearity is a separate research contribution requiring graded modalities. One-shot is sound by construction via linear continuations. Matches the implementation. Future work.

## Revision (if conditionally accepted, due Oct 26)

Incorporate reviewer-requested changes.

## Artifact Evaluation (after acceptance)

Submit Lean code + compiler. Target "functional" or "reusable" badge.
