## Why

The compiler API evaluator performs filesystem and subprocess operations directly in `runtime/eval.rs`. This structure prevents deterministic fake-system tests and leaves no single policy boundary for evaluator host access.

Active design documents also state that `process_run` does not exist and that `IO` excludes file access. The implementation and numbered specifications contradict those statements.

## What Changes

- Add a closed evaluator capability type for the current `Filesystem` and `Process` operation classes.
- Add an injected evaluator system port for all current file and subprocess operations.
- Add a typed allow/refuse policy at that port. Program evaluation selects the permissive policy.
- Give invariant predicate evaluation a deny-all policy because `spec/04-type-system.md` §2.5.1 forbids effects in invariant predicates.
- Move operating-system access into one default adapter that preserves current results and diagnostics.
- Add fake adapters for positive, failure, and refused-operation tests without real filesystem or process access.
- Add a source guard that rejects direct filesystem or process access in the evaluator module.
- Correct active design documents that contain stale claims about file access and `process_run`.

This change does not alter the language-visible `IO` effect. It does not add `Network` or `Filesystem` effect variants.

This change does not add a CLI refusal flag or change a compiler API signature. It does not mediate generated binaries or the `chelis-runtime` C ABI.

This change does not replace #1170's compiled-lane host-operation inventory. It also remains separate from #729's numeric conformance matrix and #267's compiled subprocess work.

## Capabilities

### New Capabilities

- `eval-system-boundary`: Defines the injected evaluator system port, typed evaluator capabilities, policy behavior, parity requirements, and scope limits.

### Modified Capabilities

None.

## Impact

The main implementation area is `crates/chelis-compiler-api/src/runtime/`. The change also adds a focused architecture guard and its positive and negative tests.

The permissive default preserves existing compiler API, CLI, test, Hull, and package behavior. No dependency, wire format, backend, generated-code, or package format changes.

The controlling language rules remain in `spec/04-type-system.md` §7 and `spec/05-risc-primitives.md` §2.6. This change requires no numbered-language-spec amendment because `IO` and `process_run` semantics remain unchanged.
