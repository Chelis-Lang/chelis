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
boundary: symbolic normalized-axis support for `layer_norm`/`mean` when needed,
and the Deep dotted-path round-trip gap. (HIP `pad`/`shrink` codegen, formerly
on this list, is implemented — verified by the `gpu_correctness` oracle; see
`spec/08-backends.md` §3 Phase 1a.)

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
Phase A and the post-launch Phase D registry-server endgame;
`spec/design/chelis_packaging_and_install.md` for the unified install/packaging
end-state — `chelisup` (WS-B), binary distribution (WS-A), and the
`chelis reef setup` orchestrator + unified `reef doctor` (WS-C) — that extends
the Phase-A distribution surface.

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
| **D1** | Control-flow AD (`if`, `match`, `while`, `for`, recursion). Depends on the IR-SelectOp-F1 and IR-MatchLowering-F1 §5 entries in `docs/gap_synthesis.md`. | Planned; static-scrutinee `match` slice shipped (chelis#520), plus static-condition `if` pruning and bounded static recursion unrolling (chelis#620), both in `spec/06-transformations.md` §2.10.1. Runtime-condition control flow (runtime-scrutinee `match`, runtime-condition ADT branches) remains gated on `RiscOp::Select` (chelis#618) |
| **D2** | ADT and record gradients (field-wise extension + higher-order). Depends on IR-FirstClassFn-F1. | Planned; field-wise ADT-gradient slice shipped in the eval lane, including a multi-argument ADT-alongside-tensor payload (`grad(model_forward, wrt=params)(x, params)`) (chelis#520, `spec/06-transformations.md` §2.10.1). Compiled-lane ADT-param export and higher-order ADT gradients remain roadmap work |
| **D3** | Effect-aware AD (state, raises, capability, stochastic sample-effect dispatch for reparam / REINFORCE / pathwise). | Planned |
| **D4** | Implicit differentiation (`fix`, `argmin`, `solve` markers + IFT-derived gradients). | Planned |
| **D5** | Differentiability typing (`Differentiable` / `PartiallyDifferentiable` / `NonDifferentiable` type-level marker, inference, property attachment). | Planned |
| **D6** | Documentation, examples, `chelis-diff` shell library, on-ramp for PyTorch/JAX users. | Planned |

Phases D1–D5 are sequential because each builds on the prior; D6 can
develop in parallel with D5 once D4 lands. Committing to D1 reclassifies
IR-SelectOp-F1, IR-MatchLowering-F1, and IR-FirstClassFn-F1 from
"surface-when-forced" to required prerequisites.

## Hydronnx — ONNX shell (committed scope)

Six-phase plan for `Hydronnx`, the Chelis shell that consumes ONNX
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
| **H2** | Operator translator over the v0.1 core subset (tensor manipulation, elementwise, comparisons, logical, reductions, matrix, activations, normalization, convolution, decomposed Attention/MultiHeadAttention/RotaryEmbedding, Cast, Constant, ConstantOfShape). Per-operator numerical agreement vs ONNX Runtime. | Planned |
| **H3** | Weight loading (TensorProto → Chelis tensor), layout conversion, dtype conversion, Chelis function emission with attached provenance metadata. End-to-end loading via `load_model` / `inspect_model` / `load_model_with_opts`. | Planned |
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

## Kerrent — GPU kernel authorship (committed scope)

Six-milestone plan for `Kerrent`, the Chelis core language feature for
authoring GPU kernels in Chelis source code. Kerrent introduces a new
`kernel` keyword, tile-level primitives, and a new IR layer between
the RISC DAG and Triton IR. v1 targets Triton as the kernel compiler;
FlashAttention is the customer-visible proof point. The scope is
committed (canonical document below); the strategic decision on when
to begin Phase K1 is separate. Detailed design in
`spec/design/kerrent.md`. Throughout: `Kerrent` is the Chelis language
feature; `Triton` is the upstream kernel compiler owned by the Triton
project, not by Chelis — analogous to the Hydronnx-vs-ONNX naming
discipline.

| Phase | Deliverable | Status |
|---|---|---|
| **K0** | Spec lock — `spec/design/kerrent.md` | ✅ Complete |
| **K1** | Kernel syntax and parser: `kernel` annotation and tile-level primitives parse correctly; kernel-annotated functions get a distinct AST representation. | Planned |
| **K2** | Kernel IR layer between the RISC DAG and Triton IR; tile-level operations as first-class IR nodes; type checking against the kernel IR. | Planned |
| **K3** | Triton IR emission: lowering pass from the kernel IR to Triton's MLIR dialect with output validated against Triton's IR specification. | Planned |
| **K4** | Build integration: Triton compiler invocation from the Chelis build pipeline; PTX/AMDGCN kernel artifacts produced and linked. | Planned |
| **K5** | Runtime integration: kernel calls from tensor-level Chelis lower to launches against Triton artifacts via the existing HIP/CUDA backend machinery. | Planned |
| **K6** | First production kernel: FlashAttention-shaped fused attention kernel written in Kerrent, replacing the current attention decomposition. Transformer inference becomes competitive with PyTorch+CUDA for the attention block. | Planned |

Addendums KA–KF (AD through kernels, thread-level addressing, custom
shared memory patterns, MLIR-direct lowering, verified kernel bodies,
and kernel platforms beyond NVIDIA/AMD) are post-v1 extensions with
priority tiers tracked in `spec/design/chelis_project_plan.md`
§Kerrent Track. They are not active phases and do not get roadmap rows
or phase oracles. Addendum KA (AD through kernels) is the highest-
leverage extension and composes with the differentiable-language
D-track.

Kerrent's K3 (Triton IR emission for user-authored kernels) is
orthogonal to Phase 5c (RISC DAG → Triton IR backend, which emits
whole-program tensor IR to Triton). The two operate at different layers
and do not conflate.

## Agent Editing Surface

**Status:** L0/L1 shipped and L2 query/cascade tools added. Detailed design and oracle:
`spec/design/chelis_agent_editing_surface.md`.

The agent-first design of Chelis (small Deep vocabulary, fast compiler with
fitness-oracle output, dual Surf/Deep syntax) supports structural editing
primitives that operate on Deep ASTs rather than text patches. The shipped
tools include `chelis_replace_function_body`, `chelis_add_function`,
`chelis_deep_outline`, `chelis_deep_references`,
`chelis_deep_call_graph`, `chelis_replace_function`,
`chelis_add_property`, `chelis_rename`, and
`chelis_change_signature` on Tide MCP and HTTP. Edit tools return a rewritten
module only after the post-edit whole-module check is clean; parse,
edit-shape, name-resolution, cascade-completeness, stale-preimage, type,
effect, and linearity failures are structured rejections.

Independent of the language phases and of the trust-stack/reef tracks above.
Rides on top of shipped Tide MCP infrastructure. Deep-path diagnostics for
compiler pass errors remain blocked on provenance threading; query/edit-owned
failures identify the addressed function, and rename/signature-change use the
public reference/call-graph seam.
