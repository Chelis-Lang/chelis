## 1. Baseline Evidence

- [x] 1.1 Add accepted parity fixtures for generic calls, shape operations, collections, records, patterns, and transforms.
- [x] 1.2 Record checked Deep, metadata, and inference statistics for each accepted fixture.
- [x] 1.3 Add rejected parity fixtures for numeric restrictions, invalid shapes, collection callbacks, records, patterns, and transforms.
- [x] 1.4 Record ordered diagnostic kinds, messages, spans, and hints for each rejected fixture.
- [x] 1.5 Add a compile test that imports every public item from `chelis_types::infer`.
- [x] 1.6 Add positive and negative unit fixtures for the source architecture guard core.

## 2. Mechanical Module Move

- [x] 2.1 Move `src/infer.rs` to `src/infer/mod.rs` without logic edits.
- [x] 2.2 Make sure that the public imports and the baseline evidence remain green after the move.

## 3. Program and Validation Modules

- [x] 3.1 Extract stack and recursion protection into a focused child module.
- [x] 3.2 Extract checked-program construction, metadata ownership, and totality finalization into focused child modules.
- [x] 3.3 Extract program entry points, declaration collection, dependency analysis, and inference schedules into focused child modules.
- [x] 3.4 Extract IR validation, static-value checks, and type annotation into focused child modules.
- [x] 3.5 Run the baseline evidence after each extraction and correct all parity differences before the next extraction.

## 4. Expression Modules

- [x] 4.1 Extract the expression dispatcher and common expression helpers.
- [x] 4.2 Extract function, definition, let, conditional, and pipe inference.
- [x] 4.3 Extract match and pattern inference.
- [x] 4.4 Extract tuple, record, access, update, and cast inference.
- [x] 4.5 Extract gradient and vector-map inference.
- [x] 4.6 Run the baseline evidence after each extraction and correct all parity differences before the next extraction.

## 5. Application Modules

- [x] 5.1 Reduce the application root to generic call inference and operation-family dispatch.
- [x] 5.2 Extract numeric operation rules and precision diagnostics.
- [x] 5.3 Extract tensor and shape operation rules, including operation-specific signature checks.
- [x] 5.4 Extract collection operation rules and constructor application rules.
- [x] 5.5 Extract static shape readers and application helper functions.
- [x] 5.6 Run the baseline evidence after each extraction and correct all parity differences before the next extraction.

## 6. Architecture Guard and Documentation

- [x] 6.1 Restrict child-module visibility to private or `pub(super)` where practical.
- [x] 6.2 Implement the repository source guard assertion, but keep the assertion inactive.
- [x] 6.3 Make sure that every source file under `src/infer/` satisfies the source architecture specification.
- [x] 6.4 Update `ARCHITECTURE.md` with the internal module map for type inference.
- [x] 6.5 Format the changed Rust and Markdown files with the repository tools.

## 7. Acceptance and Review

- [x] 7.1 Run `cargo test -p chelis-types` as the authoritative completion oracle.
- [x] 7.2 Activate the repository source guard for the legacy path, required roles, and the 3,000-line limit.
- [x] 7.3 Rerun the authoritative completion oracle with the active source guard.
- [x] 7.4 Run `cargo check --workspace` as additional compilation evidence.
- [x] 7.5 Run relevant `chelis-ir`, `chelis-compiler-api`, and `chelis-cli` integration tests.
- [x] 7.6 Run `openspec validate --strict modularize-type-inference` and correct every diagnostic.
- [x] 7.7 Run an independent adversarial review of API, diagnostic, metadata, and source-boundary parity.
- [x] 7.8 Correct each confirmed finding and rerun the authoritative completion oracle.
- [ ] 7.9 Request hosted CI after fresh approval for the remote state change, then record its result.
