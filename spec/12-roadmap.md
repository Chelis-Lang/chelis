# Roadmap

**Status:** Living document.
This file summarizes the current phase boundaries and project status.
For detailed execution planning, see `spec/design/chelis_project_plan.md`.
Detailed Phase 3 planning lives in `spec/design/chelis_phase3_plan.md`.

## Phases

| Phase | Deliverable | Status |
|---|---|---|
| **0a** | Project scaffold, spec docs, CI, test infra | ✅ Complete |
| **0b** | Deep parser (s-expressions) | ✅ Complete |
| **0c** | Surf parser + Surf→Deep desugaring | ✅ Complete |
| **0d** | Type checker (ADTs, HM inference, precision, named dims) | ✅ Complete |
| **0e** | RISC DAG construction from typed AST | ✅ Complete |
| **0f** | C backend codegen (host + BLAS + OpenMP) | ✅ Complete |
| **0g** | `grad` transformation (reverse-mode AD on DAG) | ✅ Complete |
| **0h** | End-to-end: MNIST on CPU + spec test suite | ✅ Complete |
| **0i** | Tide v0.1 (REPL, `chelis deep`, `chelis surf`, `chelis fmt`, `chelis eval`) | ✅ Complete |
| **1a** | HIP kernel code generation (`chelis build --target hip`) | Implemented; carried-forward backend limits documented |
| **1b** | Kernel fusion (elem→elem, elem→reduce, multi-consumer split) | Implemented; carried-forward backend limits documented |
| **1c** | GPU memory planning (buffer reuse, transfer minimization, peak estimate) | Implemented; carried-forward backend limits documented |
| **1d** | Optimized reductions, staged scalar scratch, hipBLAS matmul specialization | Implemented; carried-forward backend limits documented |
| **1e** | Benchmarks, reference comparison, checked-in results | Shipped for fixed benchmark models |
| **1f** | Executable grammar | Implemented |
| **2a** | Algebraic effects | Initial subset shipped: effect syntax, annotated checked Deep, `Random` via `dropout` + `with seed(...)`, and `Resource(Device)` build-boundary validation |
| **2** | Remaining Phase 2 work: broader effects, linear types, macros, `vmap`, Tide Agent API + MCP, LSP, TUI (`chelis cove`) | In progress; 2e Tide Agent API + MCP, 2f LSP, 2g Cove, and the 2gb Deep pretty-formatting side quest are shipped alongside the initial 2a subset; seed corpus work from the original 2s track was deferred to Phase 4a; detailed design in `spec/design/chelis_phase2_plan.md` |
| **3** | Language completeness: pipe-first style pass, package system (Reef), Python FFI, direct execution, scalar/string foundation, collections/iteration, core numeric primitives, Rust runtime rewrite, data loading/tokenization, `Std.Time`/`Std.Decimal`, SKILL.md v2 | In progress; `3a`, `3b`, `3b-ii`, `3c`, `3d`, `3e`, `3g`, `3h`, `3i`, and `3m` are shipped; the real `3f` redo remains |
| **4** | ML & AI coding: seed corpus, ICL measurement, trajectory collection, local model training, model integration |  |
| **5** | Advanced backends + research: StableHLO + JAX DLPack guarantee, FX Graph, Triton, multi-GPU, sparse tensors, complex numbers, research type features, Lean formalization, host-lane scalar AD (`grad` over `f32 -> f32` functions; design locked in `spec/design/phase5_host_scalar_ad.md` — recommendation: forward-mode dual numbers; deferred until a real driver appears) |  |
| **M** | Metal backend (`chelis build --target metal`) — Apple Silicon GPU peer of HIP. Pure string emission of Objective-C++ host + MSL kernels; `[MTLDevice newLibraryWithSource:]` is the hiprtc analog. Design: `spec/design/chelis_metal_backend_plan.md`. | In progress; M0 spec-sync underway, M1–M7 to follow |

## Red Team Checkpoints

- after 0a
- after 0d
- after 0h
- after each major phase

Each checkpoint reviews spec compliance, test coverage, architectural debt, and
remaining design ambiguity before the project moves forward.

For Phase 0h specifically, the authoritative milestone validation is the release-mode
MNIST runner in `crates/chelis-e2e`, not `cargo test --workspace` alone.

Phase 2 begins with the documented carry-forward fixes from the shipped Phase 1
boundary: the remaining HIP `pad`/`shrink` work, symbolic normalized-axis support for
`layer_norm`/`mean` when needed, and the Deep dotted-path round-trip gap.

## Trust stack expansion and reef distribution

Two parallel tracks emerged from a red-team review of the shipped trust-stack
surface and a state-of-the-world survey of reef. The trust-stack review found
that the architectural foundation is correct but the effect taxonomy is
narrower than the broader trust story needs (five variants today, no
`Network` or `Filesystem`). The reef survey found that reef has the
foundational pieces (content-addressed local registry, validated install
path, lockfile + SHA256 verification) but no remote-fetch path; the dev team
currently has to clone four shell repos and publish them manually in
dependency order. Both tracks layer on top of existing language work without
changing language semantics. Detailed designs in
`spec/design/effect_taxonomy_expansion.md` and
`spec/design/reef_distribution.md`.

| Phase | Deliverable | Status |
|---|---|---|
| **A** | Distribution unblock — `chelis reef install --from-github`, `--bootstrap`, auto-fetch during build, lockfile remote-origin field. After Phase A: a fresh dev's onboarding is `git clone + GITHUB_TOKEN + chelis reef build`. Includes the architectural correction that `chelis-std` is the language runtime (compiler-bundled, recorded as `LockSource::Bundled` in lockfiles for auditability), not a network-distributable shell. | ✅ Complete |
| **B** | Effect taxonomy expansion — add `Network` and `Filesystem` variants to the `Effect` enum, annotate `Std.Io` and shells, ship `chelis audit --effects` CLI plus matching MCP tool. ~1.5 weeks. | Planned |
| **C** | Capability enforcement — `chelis run --refuse Network,Filesystem` (signature-based pre-flight refusal, not runtime sandboxing) plus reef-side effect manifests at install time (`chelis reef install --print-effects`, `--refuse`). Depends on Phase B. ~1 week. | Planned |
| **D** | Signing and registry server — artifact signing with publisher keys; public registry server replacing GitHub Releases as the artifact backend. Demand-driven; do not start without a specific driver. | Demand-driven |

Phase A is independent of the language phases. Phase B is independent of
the language phases but its annotation pass touches every shell. Phase C
depends on Phase B. Phase D is post-launch.

Cross-references: `spec/design/effect_taxonomy_expansion.md` for the
detailed design of Phases B and C; `spec/design/reef_distribution.md` for
Phase A and the post-launch Phase D registry-server endgame.

## Differentiable programming (committed scope)

Seven-phase plan for end-to-end differentiable programming: AD that
composes through arbitrary control flow, user-defined data, effects,
fixed points, and implicit constructs, with differentiability expressed
at the type level. The scope is committed (canonical document below);
the strategic decision on when to begin Phase 1 is separate. Detailed
design in `spec/design/differentiable_language.md`.

| Phase | Deliverable | Status |
|---|---|---|
| **D0** | Spec lock — `spec/design/differentiable_language.md` | ✅ Complete |
| **D1** | Control-flow AD (`if`, `match`, `while`, `for`, recursion). Depends on the IR-SelectOp-F1 and IR-MatchLowering-F1 §5 entries in `docs/gap_synthesis.md`. | Planned |
| **D2** | ADT and record gradients (field-wise extension + higher-order). Depends on IR-FirstClassFn-F1. | Planned |
| **D3** | Effect-aware AD (state, raises, capability, stochastic sample-effect dispatch for reparam / REINFORCE / pathwise). | Planned |
| **D4** | Implicit differentiation (`fix`, `argmin`, `solve` markers + IFT-derived gradients). | Planned |
| **D5** | Differentiability typing (`Differentiable` / `PartiallyDifferentiable` / `NonDifferentiable` type-level marker, inference, property attachment). | Planned |
| **D6** | Documentation, examples, `chelis-diff` shell library, on-ramp for PyTorch/JAX users. | Planned |

Phases D1–D5 are sequential because each builds on the prior; D6 can
develop in parallel with D5 once D4 lands. Committing to D1 reclassifies
IR-SelectOp-F1, IR-MatchLowering-F1, and IR-FirstClassFn-F1 from
"surface-when-forced" to required prerequisites.

## Hydronnx — ONNX shell (committed scope)

Five-phase plan for `Hydronnx`, the Chelis shell that consumes ONNX
model files and exposes them as typed, callable Chelis functions with
dimension types, property attachment, AD composition, and trust-stack
integration. The scope is committed (canonical document below); the
strategic decision on when to begin Phase H1 is separate. Detailed
design in `spec/design/hydronnx.md`. Throughout: `Hydronnx` /
`hydronnx` is the shell; `ONNX` is the upstream interchange format the
shell consumes, owned by the ONNX project, not by Chelis.

| Phase | Deliverable | Status |
|---|---|---|
| **H0** | Spec lock — `spec/design/hydronnx.md` | ✅ Complete |
| **H1** | ONNX protobuf parser + internal IR + `chelis-hydronnx-inspect` CLI utility. | Planned |
| **H2** | Operator translator over the v0.1 core subset (tensor manipulation, elementwise, comparisons, reductions, matrix, activations, normalization, convolution, decomposed Attention/MultiHeadAttention/RotaryEmbedding, Cast, Constant). Per-operator numerical agreement vs ONNX Runtime. | Planned |
| **H3** | Weight loading (TensorProto → Chelis tensor), layout conversion, dtype conversion, Chelis function emission with attached provenance metadata. End-to-end loading via `load_model` / `inspect_model`. | Planned |
| **H4** | Type-discipline integration: dimension types on loaded signatures, call-site type checking, property attachment, AD composition where operators support it, composition with other Chelis code. | Planned |
| **H5** | Documentation, examples per strong-fit category (image classification, object detection, tabular forecasting), property examples, performance framing, ONNX-Runtime → hydronnx migration guide. | Planned |

Dependencies on the IR §5 entries (`IR-FirstClassFn-F1`,
`IR-SelectOp-F1`, `IR-MatchLowering-F1` in `docs/gap_synthesis.md`)
are non-blocking for v0.1: dynamic-graph ONNX operators (If, Loop,
Scan) extend Hydronnx's operator coverage when those entries close
(and when the differentiable-language Phase D1 control-flow AD work
lands), but they are explicitly out of the v0.1 operator subset.
Fusion, kernel authoring, and MLIR-backend work are performance
enhancers that loaded models benefit from automatically without
Hydronnx changes.

## Exploratory: Agent Editing Surface

**Status:** Exploratory. One bounded proof-of-concept tool, then evaluate.
Detailed design in `spec/design/chelis_agent_editing_surface.md`.

The agent-first design of Chelis (small Deep vocabulary, fast compiler with
fitness-oracle output, dual Surf/Deep syntax) suggests structural editing
primitives could be more reliable than text-based file editing for AI
coding agents. This direction explores whether that hypothesis holds
empirically with one bounded proof-of-concept tool
(`chelis_replace_body`), followed by a comparative benchmark. The full
toolset (`chelis_define`, `chelis_change_signature`, `chelis_rename`,
transactions) is gated on those two items succeeding.

Independent of the language phases and of the trust-stack/reef tracks
above. Rides on top of shipped Tide MCP infrastructure and the in-flight
span survival work.
