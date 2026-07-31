## Context

`chelis-types` is the lowest compiler crate above `chelis-deep`. Its inference module defines the checked-Deep boundary for all later compiler stages.

The current `infer.rs` file contains approximately 23,000 lines. The `infer_app` function alone contains approximately 3,700 lines and several operation families.

This change is an internal source refactor. The existing `type-system` capability and `spec/04-type-system.md` remain the authorities for language behavior.

## Goals / Non-Goals

**Goals:**

- The inference implementation uses modules with clear responsibilities.
- Application inference separates generic call logic from operation-specific rules.
- The public API and all inference outputs remain unchanged.
- A source guard prevents the return of one monolithic inference file.
- Each extraction step keeps the crate in a reviewable state.

**Non-Goals:**

- This change does not alter type rules or diagnostic policy.
- This change does not introduce an inference context object.
- This change does not add a builtin registry.
- This change does not refactor unification, linearity, lowering, evaluation, or backend code.
- This change does not modify a normative language specification.

## Decisions

### 1. Use a module tree with role boundaries

The implementation will replace `src/infer.rs` with `src/infer/mod.rs` and child modules. The root module will own the public facade and shared result types.

The child modules will cover these roles:

- stack and recursion protection
- checked-program construction and metadata ownership
- program orchestration, declaration collection, and inference schedules
- IR annotation, static checks, and totality checks
- expression-form inference
- application inference and operation-specific rules

Application inference will use separate modules for numeric operations, tensor operations, shape operations, and collection operations. Its root dispatcher will retain generic function application logic.

**Alternative:** Keep one file and add section comments. This option does not reduce merge conflicts, navigation cost, or function size.

**Alternative:** Create one file for each builtin. This option creates excessive modules and hides shared rules between operation families.

### 2. Extract code in behavior-preserving stages

The first source change will move `infer.rs` to `infer/mod.rs` without logic edits. Later changes will extract one responsibility at a time.

Each extraction will preserve statement order, argument evaluation order, diagnostic order, and mutation order. Helper signatures will remain unchanged unless Rust privacy requires a visibility change.

**Alternative:** Introduce new state types during extraction. This option mixes architecture changes with logic changes and increases parity risk.

### 3. Keep visibility narrow

Existing public items will keep their names, signatures, and module paths. Child modules will use private or `pub(super)` visibility where practical.

The refactor will not expose current private helpers through the crate API. Test-only mutation helpers will retain their current effective visibility.

### 4. Record parity before source movement

Before the source move, tests will record representative accepted and rejected programs. Accepted cases will compare checked Deep, metadata, and inference statistics.

Rejected cases will compare ordered diagnostic kinds, messages, spans, and hints. The cases will cover generic calls, shape operations, numeric restrictions, collections, records, patterns, and transforms.

Existing tests remain the main semantic evidence. The new parity cases target extraction boundaries that the current file hides.

### 5. Add a source architecture guard

A crate-local guard will reject these states:

- `src/infer.rs` exists.
- A production Rust file under `src/infer/` contains more than 3,000 physical lines.
- A required role module is absent.

The guard will use repository files and standard Rust library functions. It will not add an external parser or build dependency.

The guard core will accept an in-memory path and line-count list. Unit tests will provide positive and negative fixtures without temporary repository changes.

**Alternative:** Rely on review alone. This option does not prevent later consolidation into another large file.

### 6. Use one acceptance oracle

The authoritative acceptance oracle is:

```text
cargo test -p chelis-types
```

`cargo check --workspace` and selected downstream integration tests provide additional evidence. Local results do not claim hosted CI state.

## Risks / Trade-offs

- **Risk: Diagnostic order changes after helper extraction.** → Keep control flow and mutation order unchanged, then compare ordered parity fixtures.
- **Risk: Rust visibility expands across sibling modules.** → Prefer `pub(super)` and keep public exports unchanged.
- **Risk: A large move obscures logic changes.** → Separate the mechanical move from each extraction step.
- **Risk: The line limit becomes a target instead of a design aid.** → Require role modules and review responsibilities in addition to the limit.
- **Risk: Parallel work conflicts with moved code.** → Rebase before the move and extract one role at a time.

## Migration Plan

1. Add the parity fixtures and the source-guard tests.
2. Move `infer.rs` to `infer/mod.rs` without logic changes.
3. Extract checked-program and program-orchestration code.
4. Extract IR annotation and static-check code.
5. Extract expression forms.
6. Split application inference by operation family.
7. Activate the repository source guard.
8. Run the acceptance oracle and additional evidence commands.

A rollback restores `src/infer.rs` and removes the new module declarations. This change has no data, package, or runtime migration.

## Open Questions

None.
