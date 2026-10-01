# Roadmap

This chapter states the project's decided direction: the tracks that extend the
language and toolchain described by chapters 00 through 11, and what each track
delivers. It is not a status report. GitHub issues track delivery,
`spec/design/chelis_project_plan.md` holds detailed sequencing, and each track's design
document holds its detailed design.

## 1. Foundation

Every track builds on the language and toolchain the numbered chapters define:

- the dual Surf/Deep syntax and canonical formatter (chapters 02 and 03)
- the type checker: ADTs, Hindley-Milner inference, precision, named dimensions,
  effects, linearity, and graded fitness diagnostics (chapter 04)
- the RISC DAG and its transformations, including `grad` and `vmap` (chapters 05 and 06)
- the C reference backend and the HIP and Metal GPU backends (chapter 08)
- Tide: the REPL, the HTTP/JSON compiler service, the MCP server, the language
  server, and the `chelis cove` terminal UI (chapter 09)
- serialization, the Python FFI, and Reef packages (chapters 10 and 11)
- `chelis prove` property verification and the bundled `chelis-std` runtime

Language completeness continues inside those chapters. Each chapter states its full
rule; where an implementation gap would mislead a reader, the chapter links the open
issue that owns it.

## 2. Trust Stack and Distribution

These tracks layer on the existing language without changing its semantics.

| Track | Deliverable |
|---|---|
| **Reef distribution** | `chelis reef install --from-github`, `--bootstrap`, auto-fetch during build, and the lockfile remote-origin field. `chelis-std` is the compiler-bundled language runtime, recorded as `LockSource::Bundled` in lockfiles for auditability, not a network-distributable shell. |
| **Packaging and install** | `chelisup` as installer and version-routing shim, binary distribution, and the `chelis reef setup` orchestrator with a unified `reef doctor`. |
| **Effect taxonomy expansion** | `Network` and `Filesystem` variants of the `Effect` enum, annotations on `Std.Io` and the shells, and a `chelis audit --effects` CLI with a matching MCP tool. |
| **Capability enforcement** | `chelis run --refuse Network,Filesystem` (signature-based pre-flight refusal, not runtime sandboxing) and reef-side effect manifests at install time (`chelis reef install --print-effects`, `--refuse`). Depends on the effect taxonomy expansion. |
| **Signing and registry server** | Artifact signing with publisher keys and a public registry server replacing GitHub Releases as the artifact backend. Demand-driven: it starts when a specific driver appears. |

Designs: `spec/design/reef_distribution.md` (distribution and the registry server),
`spec/design/chelis_packaging_and_install.md` (packaging and install), and
`spec/design/effect_taxonomy_expansion.md` (effects and capability enforcement).

## 3. Differentiable Programming

Differentiation is a research direction inside Chelis's numerical computing scope; it
does not define the language. The direction is AD that composes through arbitrary
control flow, user-defined data, effects, fixed points, and implicit constructs, with
differentiability expressed at the type level. The scope is decided; when each step
begins is a separate decision. Detailed design in
`spec/design/differentiable_language.md`.

| Step | Deliverable |
|---|---|
| **D1** | Control-flow AD (`if`, `match`, `while`, `for`, recursion), using a compiled conditional and ADT branch representation. `spec/06-transformations.md` §2.10.1 specifies the static-scrutinee `match`, static-condition `if`, and bounded static recursion forms; runtime-condition control flow needs `RiscOp::Select` (chelis#618). |
| **D2** | ADT and record gradients: the field-wise extension of `spec/06-transformations.md` §2.10.1 in every lane, then higher-order ADT gradients, which depend on first-class function values in compiled IR. |
| **D3** | Effect-aware AD (state, raises, capability, stochastic sample-effect dispatch for reparameterization / REINFORCE / pathwise estimators). |
| **D4** | Implicit differentiation (`fix`, `argmin`, `solve` markers and IFT-derived gradients). |
| **D5** | Differentiability typing (a `Differentiable` / `PartiallyDifferentiable` / `NonDifferentiable` type-level marker, inference, and property attachment). |
| **D6** | Documentation, examples, and a `chelis-diff` shell library. |

D1 through D5 are sequential because each builds on the prior; D6 can proceed in
parallel with D5 once D4 lands.

## 4. Hydronnx: ONNX Import

`Hydronnx` is the Chelis shell that consumes ONNX model files and exposes them as typed,
callable Chelis functions with dimension types, property attachment, AD composition,
and trust-stack integration. Throughout, `Hydronnx` / `hydronnx` is the shell and
`ONNX` is the upstream interchange format, owned by the ONNX project rather than by
Chelis. The scope is decided; when work begins is a separate decision. Detailed design
in `spec/design/hydronnx.md`.

| Step | Deliverable |
|---|---|
| **H1** | ONNX protobuf parser, internal IR, and a `chelis-hydronnx-inspect` CLI utility. |
| **H2** | Operator translator over the v0.1 core subset (tensor manipulation, elementwise, comparisons, logical, reductions, matrix, activations, normalization, convolution, decomposed Attention/MultiHeadAttention/RotaryEmbedding, Cast, Constant, ConstantOfShape), with per-operator numerical agreement against ONNX Runtime. |
| **H3** | Weight loading (TensorProto to Chelis tensor), layout conversion, dtype conversion, and Chelis function emission with attached provenance metadata. End-to-end loading via `load_model` / `inspect_model` / `load_model_with_opts`. |
| **H4** | Type-discipline integration: dimension types on loaded signatures, call-site type checking, property attachment, AD composition where operators support it, and composition with other Chelis code. |
| **H5** | Documentation, examples per strong-fit category, property examples, performance framing, and an ONNX Runtime to hydronnx migration guide. |

Dynamic-graph ONNX operators (If, Loop, Scan) are outside the v0.1 operator subset.
They extend Hydronnx's coverage once first-class function values, compiled conditional
selection, and ADT branch lowering exist, together with D1 control-flow AD. Fusion,
kernel authoring, and MLIR-backend work are performance improvements that loaded
models inherit without Hydronnx changes.

## 5. Kerrent: GPU Kernel Authorship

`Kerrent` is the Chelis core language feature for authoring GPU kernels in Chelis
source. It introduces a `kernel` keyword, tile-level primitives, and an IR layer
between the RISC DAG and Triton IR; its first version targets Triton as the kernel
compiler. Throughout, `Kerrent` is the Chelis language feature and `Triton` is the
upstream kernel compiler owned by the Triton project. The scope is decided; when work
begins is a separate decision. Detailed design in `spec/design/kerrent.md`.

| Step | Deliverable |
|---|---|
| **K1** | Kernel syntax and parser: the `kernel` annotation and tile-level primitives parse, and kernel-annotated functions get a distinct AST representation. |
| **K2** | A kernel IR layer between the RISC DAG and Triton IR, with tile-level operations as first-class IR nodes and type checking against the kernel IR. |
| **K3** | Triton IR emission: a lowering pass from the kernel IR to Triton's MLIR dialect, with output validated against Triton's IR specification. |
| **K4** | Build integration: Triton compiler invocation from the Chelis build pipeline, producing and linking PTX/AMDGCN kernel artifacts. |
| **K5** | Runtime integration: kernel calls from tensor-level Chelis lower to launches against Triton artifacts through the existing GPU backend machinery. |
| **K6** | A first production kernel: a fused attention kernel written in Kerrent, replacing the attention decomposition. |

Extensions after the first version (AD through kernels, thread-level addressing,
custom shared-memory patterns, MLIR-direct lowering, verified kernel bodies, and
kernel platforms beyond NVIDIA and AMD) are tracked in
`spec/design/chelis_project_plan.md` §Kerrent Track. AD through kernels composes with
the differentiable-programming track (§3).

Kerrent's K3 (Triton IR emission for user-authored kernels) is orthogonal to a
whole-program RISC DAG to Triton IR backend (§6). The two operate at different layers.

## 6. Backends and Research Directions

- StableHLO export, with a JAX DLPack interchange guarantee
- FX graph export for PyTorch ecosystem interop
- a whole-program RISC DAG to Triton IR backend
- multi-GPU execution
- sparse tensors and complex numbers
- research type features
- Lean formalization, building on the mechanized core calculus in LaCaDiLE

Integration backends add targets; they do not replace the C, HIP, and Metal backends
(`spec/08-backends.md` §5).

## 7. Agent Editing Surface

The agent-first design of Chelis (a small Deep vocabulary, compiler output scored as
fitness, and the dual Surf/Deep syntax) supports structural editing primitives that
operate on Deep ASTs rather than text patches. The tools are
`chelis_replace_function_body`, `chelis_add_function`, `chelis_deep_outline`,
`chelis_deep_references`, `chelis_deep_call_graph`, `chelis_replace_function`,
`chelis_add_property`, `chelis_rename`, and `chelis_change_signature`, on Tide MCP and
HTTP (`spec/09-tide.md` §4). Edit tools return a rewritten module only after the
post-edit whole-module check is clean; parse, edit-shape, name-resolution,
cascade-completeness, stale-preimage, type, effect, and linearity failures are
structured rejections.

The direction is Deep-path diagnostics for compiler pass errors, which needs
provenance threaded through the passes; query and edit failures identify the
addressed function. Detailed design: `spec/design/chelis_agent_editing_surface.md`.
