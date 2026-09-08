## Why

`crates/chelis-types/src/infer.rs` contains approximately 23,000 lines and 751 functions. Its size increases navigation cost, merge conflicts, and review risk in the lowest compiler layer above Deep.

## What Changes

- Replace `infer.rs` with an `infer/` module tree that groups inference code by responsibility.
- Split the `infer_app` dispatcher into small handlers for application classes and operation families.
- Keep one public inference entry point and preserve existing internal visibility where practical.
- Preserve type inference results, diagnostic variants, diagnostic text, source spans, fitness scores, and checked-tree metadata.
- Add structural tests or lint rules that prevent a return to one monolithic inference file.
- Keep the type-system requirements and all public APIs unchanged.

### Non-Goals

- Do not change Algorithm W, unification, generalization, dimension rules, precision rules, linearity, or opacity rules.
- Do not add a builtin registry or change builtin semantics.
- Do not change compiler, runtime, CLI, backend, package, or generated-code behavior.
- Do not refactor `chelis-ir` lowering or evaluation in this change.
- Do not change normative text in `spec/**` or `openspec/specs/type-system/spec.md`.

## Capabilities

### New Capabilities

- `type-inference-architecture`: Define internal module boundaries, source-size limits, and parity evidence for type inference.

### Modified Capabilities

None. The existing `type-system` requirements and language behavior do not change.

## Impact

The primary impact is in `crates/chelis-types/src/infer.rs` and new files under `crates/chelis-types/src/infer/`. Tests in `chelis-types` will provide behavioral and diagnostic parity evidence.

The authoritative acceptance oracle is `cargo test -p chelis-types`. Workspace compilation and relevant integration tests provide additional evidence.
