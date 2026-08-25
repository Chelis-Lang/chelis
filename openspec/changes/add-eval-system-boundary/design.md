## Context

See `proposal.md` for the motivation. `runtime/eval.rs` directly uses `std::fs`, `std::path::Path::exists`, and `std::process::Command`.

The evaluator supports seven filesystem builtins and `process_run`. The compiled C lane also supports the filesystem builtins through the runtime C ABI.

`process_run` is evaluator-only. `chelis-ir` records that restriction in `EVAL_ONLY_HOST_BUILTINS`.

The language contract uses one `IO` effect for these operations. `spec/04-type-system.md` §7 and `spec/05-risc-primitives.md` §2.6 control that behavior.

#1170 tracks compiled-lane host-operation support. #729 owns a separate operation, dtype, and lane matrix for numeric conformance.

Neither issue decides whether evaluator system access is permitted. #267 tracks compiled support and sandbox work for `process_run`.

`EvalContext` has two construction sites. Program evaluation creates one in `runtime/mod.rs`, and invariant revalidation creates one in `runtime/invariant.rs`.

## Goals / Non-Goals

**Goals:**

- Isolate evaluator filesystem and subprocess access behind one typed port.
- Make refused operations fail before any system access.
- Preserve all existing public evaluator behavior under the default policy.
- Support deterministic tests with fake filesystem and process responses.
- Prevent direct system access from returning to evaluator dispatch.

**Non-Goals:**

- Change the language-visible `IO` effect or handler vocabulary.
- Add `Network`, `Filesystem`, `Process`, or `Console` to Chelis effect rows.
- Add `chelis run`, `--refuse`, or another CLI option.
- Change a public compiler API signature.
- Mediate generated artifacts or the runtime C ABI.
- Implement a true memory map for evaluator `mmap_file`.
- Replace #1170's compiled-lane inventory or #729's numeric conformance matrix.
- Implement #267's compiled subprocess support or runtime sandbox.
- Move proof subprocesses or other crates onto this port.

## Decisions

### D1: Use evaluator-specific names

The new types use the `EvalSystem` prefix. This name distinguishes the port from the `HostProgram` CPU lane and language `EffectKind` handlers.

The closed capability type contains `Filesystem` and `Process`. It does not contain empty future categories.

Alternatives included `HostEffect`, `HostOps`, and `RuntimeEffectHandler`. Each name conflicts with an existing Chelis concept or implies broader runtime coverage.

### D2: Use typed methods instead of one request and response pair

The port provides one method for each evaluator operation class. Its expected shape is:

```rust
trait EvalSystem {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError>;
    fn write_file(&mut self, path: &Path, contents: &str) -> Result<(), EvalSystemError>;
    fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError>;
    fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError>;
    fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError>;
    fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError>;
    fn load_mapped_file_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError>;
    fn run_process(&mut self, program: &str, args: &[String])
        -> Result<EvalProcessOutput, EvalSystemError>;
}
```

These methods make an invalid request and response pair unrepresentable. A single `perform(request) -> response` method needs repeated response checks in evaluator dispatch.

Pure transformations remain outside the adapter. These transformations include line splitting, output decoding, exit-code conversion, and `RuntimeValue` construction.

The `list_dir` adapter returns lossy filenames in the order from `std::fs::read_dir`. It aborts on the first directory-entry error.

`load_mapped_file_bytes` describes the current evaluator action. The builtin error still uses the public `mmap_file` name.

### D3: Apply policy in a mandatory wrapper

`EvalSystemPolicy` stores a typed decision for `Filesystem` and `Process`. The program default permits both capabilities.

A private policy wrapper owns the policy and the selected `EvalSystem` adapter. Every `EvalContext` constructor must supply that wrapper.

Program evaluation supplies a permissive wrapper. Invariant predicate evaluation supplies a deny-all wrapper because `spec/04-type-system.md` §2.5.1 rejects effects there.

Each wrapper method checks its fixed capability before it calls the adapter. A refusal returns `EvalSystemError::Refused` with the operation and capability.

The wrapper does not substitute a value after refusal. It also does not retry another adapter.

Alternatives included checks at all eight match arms. That design permits a new call site to omit its check.

### D4: Define the default adapter contract

The default adapter owns all direct Rust standard library calls. Program evaluation constructs it with the permissive policy.

The adapter preserves `Path::exists` behavior, including its false result for inaccessible metadata. `list_dir` preserves source iteration order and adds no sort.

A `list_dir` entry error aborts the operation. It retains the same error prefix as a directory-open error.

The process adapter uses direct argument vectors and no shell. It returns raw output bytes and the optional exit status to pure evaluator code.

The evaluator converts a missing exit status to `-1`. It also uses lossy UTF-8 conversion for both output streams.

`EvalSystemError` has a closed operation identity, a refused capability variant, and a system-error variant. The system-error variant owns the underlying `std::io::Error`.

The error does not reduce the source to `ErrorKind`. Its one display conversion emits the exact templates in `specs/eval-system-boundary/spec.md`.

### D5: Inject the port for one evaluation lifetime

The program evaluator accepts a mutable boundary reference and stores it on `EvalContext`. The reference exists for one evaluation only.

The `eval_compiled` path constructs the permissive default. Its public functions, prepared handles, signatures, and result schemas remain unchanged.

The invariant revalidation path constructs a deny-all boundary locally. It does not add a parameter to `decode_with_tables` or another decode API.

Tests call a crate-private program entry with a fake adapter. The fake records calls and returns configured values or errors.

The boundary check occurs only at a covered builtin call. Ordinary expression evaluation does not perform a policy lookup.

### D6: Guard the boundary with a focused source check

`scripts/eval_system_guard.py` scans evaluator runtime sources and excludes the one adapter module. It rejects direct filesystem, path-existence, and process construction forms.

The guard fails if it cannot read or classify a source file. Its unit tests include accepted adapter code and rejected bypass fixtures.

The negative fixtures include:

- `std::fs` and imported `fs` calls
- `std::process::Command` and imported `Command` calls
- `std::path::Path::exists` and imported `Path::exists` calls
- method-form `.exists()` calls on a path

The focused oracle is `<managed-python> scripts/eval_system_oracle.py`. It runs guard tests, the guard, compiler API tests, and named CLI parity suites.

The repository gate adds this focused oracle. Hosted CI remains the final workspace evidence, while the focused oracle controls completion for this change.

### D7: Correct stale design statements without changing semantics

The documentation pass corrects active statements that say file builtins do not carry `IO` or that `process_run` does not exist.

The pass preserves the planned effect-taxonomy work as a separate decision. It does not convert that planning document into normative language.

OpenSpec remains planning and review evidence. The numbered specifications remain authoritative for language behavior.

## Risks / Trade-offs

- **[Risk] The boundary can be described as a sandbox.** → State the evaluator-only scope in the proposal, specification, design, and user documentation.
- **[Risk] Error text changes during extraction.** → Freeze positive and negative parity cases before the adapter replaces direct calls.
- **[Risk] Invariant revalidation constructs an ungoverned evaluator.** → Require its context to use a deny-all wrapper before expression evaluation.
- **[Risk] The source guard misses an alias or method form.** → Add one negative fixture for every accepted Rust spelling in the guard inventory.
- **[Risk] The guard rejects harmless text in comments or strings.** → Scan parsed code tokens or strip comments and literals before pattern checks.
- **[Risk] Policy checks add cost to each AST node.** → Perform checks only at the eight covered builtin dispatch arms.
- **[Risk] The port appears to cover compiled filesystem access.** → Keep the compiled lane unchanged and include an explicit scope test.
- **[Trade-off] The restrictive policy has no public selector in this change.** → Keep the policy private until a separate CLI or embedding contract gains approval.

## Migration Plan

1. Add red fake-adapter tests, refusal tests, parity tests, and source-guard fixtures.
2. Add the typed port, capability policy, default adapter, and both context policies.
3. Route each covered evaluator operation through the permissive boundary.
4. Remove direct filesystem and process imports from evaluator dispatch.
5. Correct stale active design documents.
6. Add the focused oracle to the repository gate.
7. Run the focused oracle, strict OpenSpec validation, and the repository acceptance path.

The program policy activates with the first routed operation and remains permissive. No data, cache, wire, or package migration is necessary.

Rollback removes the port and restores the previous direct calls. No persistent state requires recovery.
