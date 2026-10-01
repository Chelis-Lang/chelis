# Hydronnx: ONNX Shell for Chelis

## Name and convention

The shell is `Hydronnx` as a module path (PascalCase, single component, matching
the §6.3 module-pascal-components rule and the existing shell names Coral,
Nautilus, Octant, Capstone). The crate/package, source files, directory, and
prose references are lowercase `hydronnx`, the same way `coral` and `nautilus`
appear in prose. References to the underlying interchange format remain `ONNX`
(uppercase) — that is the technology Hydronnx consumes, not the shell.

## Goal

Hydronnx is a Chelis shell that consumes ONNX model files and exposes them as typed, callable Chelis functions. The user takes an ONNX file (from PyTorch export, JAX export, Hugging Face, or any framework that produces ONNX), loads it through hydronnx, and gets back a Chelis function with full dimension types, property attachability, cross-backend compilation, and composition with the rest of their Chelis code.

Hydronnx is the on-ramp from the existing AI/ML ecosystem to Chelis. A user with an ONNX model can be running it under Chelis's type discipline and verification machinery without rewriting the model in Chelis from scratch.

## Audience and purpose

The primary audience is users with existing ONNX models who want one or more of: cross-platform deployment from a single source, compile-time shape verification, runtime property verification, composition with verified Chelis code, AD composition through the loaded model, or the trust stack for regulated inference.

The on-ramp is the dominant near-term benefit. A user with a Hugging Face image classifier, a fine-tuned encoder, a tabular model, or an object detector can be running it in Chelis in minutes rather than reimplementing it.

Performance improves over time. v0.1 inference performance is constrained by the lack of fusion in Chelis's optimization pipeline. As the kernel authorship work and MLIR-as-backend work ship, the loaded models benefit automatically — same load, same call, better performance.

Relative to ONNX Runtime: ONNX Runtime gives cross-platform inference today. Chelis with hydronnx gives cross-platform inference plus type discipline plus property verification plus the trust stack. For users who only want the first part, ONNX Runtime stays the right answer. For users who care about the rest, Chelis becomes accessible through their existing models.

## Pre-locked decisions

Locked before any agent dispatches:

**Decision 1: ONNX is the v0.1 interchange format.** GGUF and MLIR-based formats are future considerations. ONNX has the broadest ecosystem support, the most stable spec, and the simplest parser story. Hydronnx targets ONNX specifically; other formats become separate shells if they're added later.

**Decision 2: Inference graphs only for v1.** Training graphs (ONNX Training extensions) are out of scope. The common case is "load a trained model, run inference"; that's what v1 delivers.

**Decision 3: Standard operator set only.** Custom operators (research models using non-standard ops) fail loading with a clear error. Extension support for custom operators is a future workstream.

**Decision 4: Static shapes by default; symbolic dimensions where they match Chelis representations.** ONNX models with fully static shapes are the dominant case and load directly. Models with symbolic batch or sequence dimensions get loaded with the symbolic dimensions preserved when Chelis's dimension type system can express them. Models with dynamic shapes that don't fit Chelis's representation fail loading with a clear error pointing to the specific dimension.

**Decision 5: Decompose-then-recognize for high-level operators.** ONNX has operators like Attention, LayerNormalization, Softmax that are higher-level than Chelis's RISC primitives. The translation decomposes these to their RISC equivalents initially. Later refinements add recognizers in the IR specialization substrate so the decompositions collapse back to fused emissions when fusion infrastructure ships. The translation never produces RISC-level decompositions that the compiler can't recognize as the original high-level pattern.

**Decision 6: Weight loading is part of the shell, not a separate concern.** Hydronnx reads ONNX tensor data, converts to Chelis tensor format, handles dtype mapping, and embeds the weights as constants in the loaded function. The user doesn't manage weight loading separately.

**Decision 7: The shell is named `Hydronnx` and lives in the canonical shell location.** Follows the existing shell naming convention (Coral, Nautilus, Octant). Module path is `Hydronnx` per the ecosystem nomenclature; crate, package, source files, and directory names use lowercase `hydronnx`.

**Decision 8: Initial dtype scope matches Chelis's host-side dtype support.** f32 and f64 ship in v0.1. i32 and i64 ship in v0.1 (for index tensors). bf16, f16, i8, i16 follow when the dtype build-out delivers host-side precision. Quantized formats (Q8, Q4, ONNX QDQ) are out of scope for v0.1.

## Architecture

Hydronnx has four components:

**Parser.** Reads the ONNX protobuf format, extracts the computation graph, weight tensors, input/output signatures, model metadata, and operator set version. Produces an internal representation suitable for translation.

**Operator translator.** Maps each ONNX operator to a Chelis expression. For operators with direct Chelis equivalents (MatMul, Add, Mul), the mapping is a single function call. For higher-level operators (Softmax, LayerNormalization, Attention), the mapping is a decomposition to RISC primitives that the compiler can later recognize.

**Weight loader.** Reads ONNX tensor data, performs dtype conversion if needed, lays out tensors in Chelis's expected memory format, embeds weights as Chelis constants accessible from the loaded function's body.

**Function emitter.** Produces a Chelis function definition with the correct type signature (derived from ONNX input/output specs), body (derived from the translated operator graph), and metadata (model name, ONNX opset version, provenance information).

## Phases

The work decomposes into six phases (Phase 0 spec lock plus Phases 1–5 of implementation and ecosystem).

### Phase 0: Spec lock and ONNX subset definition

Single agent. Produces the canonical spec at `spec/design/hydronnx.md`. Pins:

**The supported ONNX operator subset for v0.1.** Not every ONNX operator. The v0.1 set covers the operators needed for the strong-fit model categories from earlier analysis: image classifiers, encoder-only language models, object detection, segmentation, pose estimation, tabular models, recommender systems. The operator list is concrete and enumerated in the spec. Operators outside this set produce a clear "unsupported operator: <name>" error at load time.

**The supported ONNX opset versions.** The spec pins specific opset versions (likely opset 17 or 18 as the current stable). Models exported with older opsets may work but aren't guaranteed; models exported with newer opsets fail loading until hydronnx catches up.

**The shape-handling rules.** Specific rules for how ONNX's shape system maps to Chelis's dimension types. Static shapes map directly. Symbolic dimensions map to Chelis dim variables. Dynamic shapes that don't fit Chelis's representation produce a load-time error with the specific dimension identified.

**The dtype mapping.** Per-dtype mapping from ONNX's type system to Chelis's. f32 → f32. f64 → f64. i32 → i32. i64 → i64. Other ONNX dtypes (uint8, bf16, fp16) produce errors at load time pointing to the dtype build-out's planned coverage.

**The error model.** What kinds of failures produce errors, what kinds produce warnings, what produces structured diagnostics. Concrete categories: unsupported operator, unsupported opset version, unsupported dtype, unsupported shape pattern, weight conversion failure, internal inconsistency in the ONNX file.

**The function signature derivation rules.** How ONNX input/output specs become Chelis function signatures. Single-input models become single-arg functions. Multi-input models become functions taking a record of inputs (the ONNX input names become record field names). Single-output models return the output directly. Multi-output models return a record.

**The provenance metadata.** What gets attached to the loaded function: ONNX file path, model name from ONNX metadata, opset version, ONNX operator inventory, weight tensor inventory, load timestamp. Accessible via `inspect_model` and via metadata on the Chelis function.

### Phase 1: Parser and IR

Single agent. Builds the ONNX file parser and the internal representation.

**Scope.**

- Add ONNX protobuf dependency to the hydronnx crate. Use the standard prost-based parser (or equivalent in the Chelis-toolchain Rust ecosystem).
- Define the internal representation (`hydronnx::ir::Model`, `hydronnx::ir::Graph`, `hydronnx::ir::Node`, `hydronnx::ir::Tensor`, etc.) that the parser populates and the downstream translator consumes. The types are path-qualified under the `hydronnx::ir` module rather than carrying an `Onnx` prefix.
- Implement the parser: read the protobuf, validate the structure, populate the internal representation, surface errors for malformed inputs.
- Implement a `chelis-hydronnx-inspect` CLI utility (or library function) that prints the model's inventory: input/output names and shapes, operator list, weight tensor list, opset version, file size. Useful for debugging and for verifying the parser before the rest of the pipeline ships.

**Tests.**

- Parse a known-good ONNX model (a small image classifier like MobileNet or a simple linear regression). Inventory matches what `onnx.checker.check_model` produces in Python.
- Parse a model with each supported operator class to verify the parser handles all of them.
- Parse a model with errors (truncated file, unsupported opset, corrupted weights) and produce the expected diagnostic per the error model.

**Output.** A working ONNX parser with no translation logic yet. Standalone milestone; the downstream phases consume the parser's output.

### Phase 2: Operator translator (core subset)

Single agent. Implements the mapping from ONNX operators to Chelis expressions for the core operator subset.

**Scope.**

The core subset for v0.1, in priority order based on the strong-fit model categories:

- Tensor manipulation: Reshape, Transpose, Squeeze, Unsqueeze, Concat, Split, Slice, Gather, Scatter, Where, Identity
- Elementwise arithmetic: Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tan, Floor, Ceil, Round, Clip, Sign, Pow
- Comparisons: Equal, Greater, Less, GreaterOrEqual, LessOrEqual
- Logical: And, Or, Not, Xor
- Reductions: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ArgMax, ArgMin
- Matrix: MatMul, Gemm
- Activations: Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Elu, Selu, Hardsigmoid, Hardswish, Mish, Gelu
- Normalization: BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization
- Convolution: Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool
- Higher-level (decomposed): Attention, MultiHeadAttention, RotaryEmbedding when present in opset
- Type conversion: Cast
- Constant: Constant, ConstantOfShape

Each operator gets a translator function that takes the ONNX node's inputs and attributes and produces the corresponding Chelis expression. For direct-mapping operators (MatMul, Add), this is a single primitive call. For decomposable operators (Softmax, LayerNormalization), this is the decomposition into RISC primitives that follows the spec's decomposition recipes.

**Decomposition recipes.** Each high-level operator has a canonical decomposition pinned in the spec. Softmax decomposes one way; LayerNormalization decomposes one way; Attention decomposes one way. The recipes are stable so future recognizers in the IR specialization substrate know what tag tree to match against.

**Tests.**

- Per-operator numerical agreement: load a small ONNX model exercising each operator, run inference under both ONNX Runtime and Chelis, assert outputs match within precision-appropriate tolerance.
- A representative model from each strong-fit category: a small image classifier, an encoder-only model, an object detector, a tabular model. End-to-end correctness verified against ONNX Runtime baseline.

**Output.** Translation works for the v0.1 operator subset. Loaded models run correctly. Performance is constrained by the lack of fusion (the decomposed operators don't get fused into single kernels yet); correctness is verified.

### Phase 3: Weight loading and function emission

Single agent. Implements the weight conversion and the final Chelis function emission.

**Scope.**

- Weight tensor conversion: read ONNX TensorProto data, validate against expected dtype and shape, convert to Chelis tensor format, embed as a Chelis constant accessible from the loaded function's body.
- Layout conversion: ONNX uses NCHW for image convolutions; Chelis may have layout conventions that differ. Hydronnx handles the conversion (or fails clearly if the conversion isn't supported).
- Dtype conversion: weights stored as one dtype in ONNX may need conversion to a different dtype when loaded. f32 weights with f64 model body: cast at load time. Mismatched dtypes that can't be cleanly converted produce a clear error.
- Function emission: given the translated graph and loaded weights, emit a Chelis function definition. Signature derived per Phase 0 rules; body is the translated operator sequence with weights as constants; metadata attached.

**The user-facing API.** Concretely:

```
import Hydronnx (load_model, inspect_model)

-- Load with default settings
my_classifier = load_model("path/to/classifier.onnx")

-- Inspect before loading
info = inspect_model("path/to/classifier.onnx")

-- Load with explicit overrides
custom_loaded = load_model_with_opts("path/to/model.onnx", {
  precision: f64,
  batch_dim: 32,
  ...
})
```

**Tests.**

- Load a small model with weights; the loaded Chelis function produces correct outputs when called.
- Load a model with `inspect_model`; the inventory matches the model's actual structure.
- Load with explicit options; the overrides take effect (precision change, batch dim binding).
- Load failure cases: model with custom op (clear error), model with unsupported opset (clear error), corrupted weights (clear error).

**Output.** End-to-end ONNX loading works. A user can load a model, call it, get correct outputs.

### Phase 4: Type discipline integration

Single agent. Integrates the loaded function with Chelis's full type system so that dimension types, property attachment, and compile-time checking work properly.

**Scope.**

- The loaded function's signature carries proper dimension types. If the ONNX model has input shape `[batch, 3, 224, 224]` with `batch` symbolic, the Chelis function's signature is `tensor[batch, 3, 224, 224, f32] -> tensor[batch, 1000, f32]` with `batch` as a dim variable.
- Call-site type checking: invoking the loaded function with a wrong-shape input fails at type-check, not at runtime.
- Property attachment: a user can attach properties to the loaded function the same way they'd attach properties to a hand-written function. The harness verifies the properties by sampling inputs.
- AD integration: the loaded function admits AD where the operators support it. For models where some operators don't have gradient rules yet, AD applied to the model fails with a clear error pointing to the specific non-differentiable operator.
- Composition with Chelis code: the loaded function can be called from other Chelis functions, wrapped in transformations, composed in pipelines. Standard Chelis function semantics.

**Tests.**

- Wrap a loaded model with a function that takes a specific input shape; type-check catches shape mismatches at the call site.
- Attach a property (e.g., "output probabilities sum to 1 within tolerance") to a loaded classifier; the harness samples inputs and verifies the property.
- Apply AD to a loaded function with all-differentiable operators; gradient computation produces correct results.
- Apply AD to a loaded function with at least one non-differentiable operator; the operation fails with a diagnostic identifying the operator.
- Compose a loaded model with preprocessing and post-processing functions; the composition type-checks and runs correctly.

**Output.** Loaded ONNX models behave as first-class Chelis functions with full type discipline and verification capability.

### Phase 5: Documentation, examples, and ecosystem integration

Single agent. The capability shipped in Phases 1-4 needs discovery and onboarding.

**Scope.**

- User-facing documentation: how to install hydronnx, how to load a model, how to inspect a model, how to handle common error cases, how to attach properties to loaded models, how to compose loaded models with other Chelis code.
- Examples directory: at least one worked example per strong-fit model category. Image classification (load a ResNet variant, run inference on sample images). Object detection (load YOLOv8 or similar, draw bounding boxes). Tabular forecasting (load a TabNet or gradient-boosted model). Each example is a complete Chelis program demonstrating the loading-and-using pattern.
- Property examples: at least one example per category showing what properties are worth attaching. Image classifier with confidence-bound properties. Object detector with box-coordinates-in-image properties. Tabular model with input-distribution properties.
- Performance documentation: honest framing of where hydronnx is performance-competitive (correctness-critical inference, type-discipline-required deployment) and where it's not (raw throughput on large generative models). The honest framing helps users know when Chelis is the right tool versus when they should use ONNX Runtime directly.
- Migration guide: for users moving from ONNX Runtime to hydronnx, what's similar, what's different, what extra capability they get.

**Output.** A researcher or engineer with an ONNX model can find the documentation, follow the examples, load their model, and start using it under Chelis's type discipline within minutes.

## Dependency on other Chelis workstreams

Hydronnx composes with other ongoing Chelis work; some phases benefit from features in those workstreams without strictly requiring them:

**Dtype build-out.** Phase 2's operator translation requires f32, f64, i32, i64 to be working end-to-end. The current dtype work delivers these. Future dtype support (bf16, f16, i8, i16, quantized) extends hydronnx's operator coverage when those land.

**IR architectural workstreams.** First-class function values, compiled
conditional selection, and ADT branch lowering do not strictly gate Hydronnx,
but some ONNX operators (If, Loop, Scan) require that IR work. The v0.1 spec
excludes those operators; later versions add them as the IR work ships.

**Kernel authorship and MLIR backend.** Hydronnx ships without these dependencies. Performance is what it is until those land. Loaded models automatically benefit when the optimization infrastructure ships; no changes to hydronnx required.

**Property verification (hull/c-earchin).** Hydronnx composes with the existing property machinery. Phase 4's property-attachment integration uses the existing hull infrastructure; doesn't require new work in hull.

## What's not in v1 scope

**Custom operators.** ONNX models can declare custom operators. v1 doesn't support them; the model fails loading with a clear error identifying the custom operator. Custom operator support is a future workstream where users register Chelis implementations of their custom ops with hydronnx.

**Training graphs.** ONNX Training extensions are out of scope. Hydronnx loads inference graphs only. Users wanting to fine-tune a loaded model do so in Chelis (using the loaded model as the forward pass and Chelis AD for gradients) rather than through ONNX training.

**Dynamic graph operations.** Some ONNX operators (Loop, Scan, If, Recurrent operations) involve control flow at the graph level. These depend on the broader Chelis support for control flow (the differentiable-language work's Phase 1) and ship after that infrastructure.

**Quantized formats.** ONNX supports quantization-aware quantization (QDQ format) and standalone quantized models. v1 loads as f32 only. Quantized loading is a future workstream tied to Chelis's quantization story (which doesn't exist yet).

**Multi-device or distributed inference.** The loaded function runs on whatever backend Chelis targets. Sharding a loaded model across multiple devices is a future workstream.

**Round-trip export.** v1 loads ONNX into Chelis. It doesn't export Chelis code back to ONNX. The reverse direction is a separate workstream that would let users develop in Chelis and deploy via ONNX Runtime.

## Operator translation details

The translation logic per operator is the bulk of hydronnx's implementation. The spec at `spec/design/hydronnx.md` contains the full table; this section captures the principle.

**For direct-mapping operators**, the translation is straightforward:

```
ONNX MatMul(A, B)       → matmul(A, B)
ONNX Add(A, B)          → add(A, B)
ONNX Relu(X)            → relu(X)
ONNX Reshape(X, shape)  → reshape(X, shape)
```

**For decomposed operators**, the translation produces the canonical RISC-level decomposition:

```
ONNX Softmax(X, axis) → 
  let m = ReduceMax(X, axis)
  let e = Exp(Sub(X, Broadcast(m)))
  let s = ReduceSum(e, axis)
  Div(e, Broadcast(s))

ONNX LayerNormalization(X, gamma, beta, axis, epsilon) →
  let m = ReduceMean(X, axis)
  let centered = Sub(X, Broadcast(m))
  let v = ReduceMean(Mul(centered, centered), axis)
  let normed = Div(centered, Sqrt(Add(Broadcast(v), epsilon)))
  Add(Mul(normed, gamma), beta)
```

The canonical decomposition matters because the IR specialization substrate will eventually recognize these specific tag trees and rewrite them to fused emissions. Future recognizers know exactly what shape to match.

**For high-level operators with multiple valid decompositions** (e.g., Attention can be written several ways), the spec pins one canonical form. The translation always produces that form. Variants are recognized by future passes but not produced by the translator.

## Error model

Hydronnx's error model is concrete:

**Load-time errors.** Surfaced at `load_model` invocation. Categories:

- File not found, file corrupted, file not valid ONNX
- Unsupported opset version (with the supported range named)
- Unsupported operator (with the operator name and ONNX source location)
- Unsupported dtype (with the dtype and the planned support timeline)
- Unsupported shape pattern (dynamic shape that can't fit Chelis's representation)
- Internal inconsistency in the ONNX file (operator references a nonexistent input, weight shape doesn't match operator expectation)

Each error includes the specific issue, the location in the ONNX file (operator name, node index), and a pointer to the resolution (supported versions, supported operators, planned support).

**Type-check-time errors.** Surfaced when Chelis code uses the loaded function with mismatched types. Standard Chelis type-check errors with the specific mismatch identified.

**Runtime errors.** Surfaced during inference. Standard Chelis runtime errors; hydronnx doesn't introduce new runtime error categories beyond what Chelis already has.

**Warnings.** Surfaced at load time for non-fatal concerns:

- Operator decomposed to RISC primitives (potential performance impact until recognizers fire)
- Opset version older than recommended (may still work)
- Model uses experimental ONNX features (may need verification)

## The release shape

Version numbers follow the standard Chelis shell `vX.Y.Z` SemVer convention; the
milestone labels and the package version increment in lockstep.

v0.1.0: Phases 0-4 implemented. Documentation and examples for one model category (recommend image classification as the simplest).

v0.2.0: Phase 5 documentation and examples expand to cover all strong-fit categories.

v0.3.0: Recognizers added to the IR specialization substrate so that decomposed operators (Softmax, LayerNormalization, Attention) get collapsed to fused emissions when fusion infrastructure ships. Hydronnx itself doesn't change; the loaded models get faster automatically.

v0.4.0 and beyond: Operator coverage expansion (Loop, Scan, If when IR control flow lands), quantization support when Chelis has a quantization story, custom operator support, multi-device deployment.

## Net

Hydronnx is a bounded, well-scoped workstream that delivers concrete user value: any ONNX model becomes a Chelis function with the type discipline and verification machinery applied. v0.1 covers the operators and model categories that match Chelis's strong-fit profile (everything except large generative models, which wait for kernel authorship).

The six-phase structure decomposes the work into independent, testable milestones. Phases 0-3 build the core capability. Phase 4 integrates with Chelis's broader type system. Phase 5 produces documentation and examples.

Dependencies on other Chelis workstreams are limited and clearly specified. Hydronnx composes with those workstreams as they ship rather than blocking on them.

Users with existing models in the PyTorch/JAX ecosystem can bring them into Chelis without rewriting them, and check them with Chelis's dimension types and properties.
