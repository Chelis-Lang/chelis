## Context

`spec/05-risc-primitives.md` (v0.2) records the ~12 irreducible tensor operations, their AD
adjoints, the derived built-ins that lower to them, and the standard ML-op lowerings. It also
carries three decided-atom sections: seed determinism (§2.6 [05-RNG-1]), the unsupported-case
response contract (§7), and the observation/formatting contract (§8), some of whose atoms are
eval-lane-conformant with the C lane tracked for a later phase.

## Goals / Non-Goals

**Goals:**
- Capture each primitive family, its precision/AD rules, and the derived-builtin and standard
  lowerings as SHALL requirements with positive and failure scenarios.
- Faithfully record the decided contracts (unsupported-case, observation) as normative
  requirements.

**Non-Goals:**
- Enumerating every AD adjoint row and every reference-implementation loop; the requirement
  captures the governing rule (every primitive has an adjoint; the C reference is the oracle)
  and representative cases.
- Restating the full `RtDim`/runtime-bound materialization matrix of §2.4.1; the requirement
  captures the dual-lane run-time validation contract and the HIP/Metal rejection.

## Decisions

- Group the elementwise-binary/unary, reduction, windowed-reduction, movement, memory, and
  effectful families into distinct requirements aligned to the source's §2 sub-sections, since
  each family has its own precision and AD contract.
- Keep the three decided-atom sections (seed determinism, unsupported-case, observation) as
  their own requirements to preserve their normative status and per-lane conformance framing.

## Risks / Trade-offs

- [Decided-atom lane divergence] → §8 observation atoms are eval-lane-conformant with the C
  lane tracked for a later phase (chelis#732 Phase 2), and §7 has residual per-atom notes. The
  requirements capture the decided contract; the current lane-conformance gaps are recorded in
  Open Questions, not softened.

## Open Questions

- The observation contract (§8) and parts of the unsupported-case contract (§7) are decided but
  only partially implemented (eval lane conformant, C lane tracked for a later phase; per-atom
  issue references). This is a known implementation-vs-spec gap tracked upstream (chelis#730,
  #732), not a spec ambiguity.
- Phase-3h primitive additions (`einsum`, `where`, `sort`, `clamp`, `diagonal`/`trace`) and
  their check-time-vs-runtime value-error rejection are recorded at the family level; the exact
  per-op rejection matrix is deferred to the source and the checker.
