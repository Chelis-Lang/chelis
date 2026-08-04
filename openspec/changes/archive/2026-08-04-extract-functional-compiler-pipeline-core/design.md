## Context

PR #1013 created one typed semantic pipeline in `chelis_compiler_api::pipeline`. The module owns type analysis, effect checks, linearity checks, lowering, root metadata, and typed phase artifacts.

The module sits inside `chelis-compiler-api`. That crate also depends on Reef, source preparation, backends, caches, schemas, serialization, and standard-library bundles.

`chelis-reef` is below `chelis-compiler-api` in the dependency graph. Its package artifact path repeats type, effect, and linearity stages because it cannot use the current owner.

`ARCHITECTURE.md` and `docs/investigations/compiler_pipeline_inventory.md` record this exception. The active capability assigns its removal to issue #1012.

This change defines implementation architecture only. The numbered specifications retain authority for language semantics, pass order, wire formats, packages, runtimes, and backends.

## Goals / Non-Goals

**Goals:**

- Put typed semantic transitions in an unpublished dependency-bottom crate.
- Make Cargo enforce the boundary between semantic code and shell dependencies.
- Preserve the current `chelis_compiler_api::pipeline` import surface.
- Remove the direct Reef semantic sequence.
- Preserve all accepted products, rejected diagnostics, root order, bytes, and exit codes.
- Extend the source guard, dependency guard, compile-fail evidence, and compiler pipeline oracle.
- Record concrete blockers for future `#![no_std]` support.

**Non-Goals:**

- This change does not add `#![no_std]` support.
- This change does not claim referential transparency for lower compiler crates.
- This change does not alter a compiler stage or its order.
- This change does not move source preparation, Reef resolution, caches, schemas, diagnostics, host policy selection, or backend selection into the core.
- This change does not add generic stage traits.
- This change does not create a backend abstraction.
- This change does not change the public machine-facing API.

## Decisions

### 1. Add one unpublished dependency-bottom crate

The workspace will add `crates/chelis-pipeline-core` with `publish = false`. The root workspace manifest will expose it as `chelis-pipeline-core`.

The core will depend directly on these lower compiler crates:

- `chelis-deep`
- `chelis-types`
- `chelis-effects`
- `chelis-ir`

The core will not depend on `chelis-compiler-api`, `chelis-reef`, source crates, macro crates, backends, caches, schemas, or wire libraries.

`chelis-compiler-api` and `chelis-reef` will depend on the core. This graph removes the current cycle that blocks Reef from the canonical transitions.

The crate name describes typed value transitions. It does not claim `#![no_std]`, purity inside lower crates, or freedom from standard-library allocation.

**Alternative:** Keep the module in `chelis-compiler-api`. This option preserves the unenforced boundary and the Reef exception.

**Alternative:** Put the pipeline in `chelis-reef`. This option reverses the package and compiler dependency direction.

### 2. Start the core at owned expanded Deep

The core will own `PreparedProgram` and its `Vec<DeepExpr>`. A constructor will accept already expanded Deep without source parsing or entry policy.

`chelis-compiler-api` will retain these operations:

- source-kind conversion
- Surf parsing
- Deep parsing
- Surf desugaring
- macro expansion
- entry pruning

The facade will apply those operations before it constructs `PreparedProgram`. Reef will construct the same carrier after its current link and expansion steps.

This boundary keeps source libraries and Reef policy out of the core. It also gives both upper crates one typed semantic input.

**Alternative:** Start at Surf source. This option adds parser, macro, schema, and source-policy dependencies to the core.

**Alternative:** Start at `CheckedProgram`. This option leaves type analysis and the Reef duplicate outside the core.

### 3. Move phase artifacts and semantic transitions

The core will own these artifact groups:

- prepared type-analysis products
- checked compilation products
- lowered compilation products
- semantic rejection products
- root names and root metadata
- exact root maps and forward-node indexes

The core will own isolated and contextual type analysis. It will also own effect checks, linearity checks, root derivation, and lower-phase construction.

`compose_checked` will move into the core with the narrow visibility that cross-crate use requires. The compiler API will not expose that helper as a new public facade function.

The core will keep concrete artifact types. It will not add a generic stage trait or an abstract pipeline framework.

**Alternative:** Move only functions and leave artifact types in the facade. This option creates a dependency from the core to the shell crate.

**Alternative:** Duplicate carrier types across crates. This option creates conversion seams and permits state drift.

### 4. Keep dispatch and presentation in the facade

`PipelineRequest`, `PipelineGoal`, `PipelineOutcome`, `PreparationError`, and `PipelineRejection` will remain in `chelis-compiler-api`.

The facade will retain `run_source` and `run_prepared`. These functions will select the required core transition and map native errors to the current rejection variants.

`SemanticRejection` will move into the core because it contains only effect or linearity errors. A narrow core lowering error will contain lower diagnostics and root-count failures.

The facade will map the core errors into the current `PipelineRejection` variants. Error text, order, stage names, and cancellation results will remain unchanged.

Cooperative cancellation checks will remain in the facade before and after long core transitions. Reef will retain its current behavior without a cancellation policy.

The facade will select `LoweringMode`. The core will apply the explicit mode only to construct valid root artifacts.

The core will not inspect CLI target names or import a backend. Thus, backend selection and host policy selection remain outside the core.

**Alternative:** Move the complete public dispatcher into the core. This option imports source schemas, preparation errors, and shell policy into the dependency-bottom crate.

**Alternative:** Return formatted strings from the core. This option weakens the typed error boundary and risks diagnostic drift.

### 5. Preserve the compiler API through re-exports and wrappers

`chelis_compiler_api::pipeline` will re-export the moved artifact types and pure phase functions. Existing consumers will not require a direct core dependency.

Facade wrappers will preserve functions that need preparation, cancellation, dynamic dispatch, or error conversion. Compile fixtures will use only `chelis-compiler-api` as a direct dependency.

The current machine-facing schema types remain in `chelis-compiler-api`. Core artifacts will not gain Serde wire implementations.

Core compile-fail doctests will lock private construction and distinct root-map roles. Existing compiler-API doctests will lock facade compatibility and non-wire artifact types.

**Alternative:** Require all current consumers to import the core. This option creates an unnecessary public migration and exposes implementation placement.

### 6. Remove the Reef exception without moving Reef policy

`chelis-reef::checked_program_with_effects` will install its current linked-program guard. It will then pass owned expanded Deep through the core transitions.

The Reef adapter will preserve its current string format for type, effect, and linearity failures. Positive and negative package fixtures will compare exact artifact and schema output.

Reef will retain package resolution, graph links, name mangling, source expansion, archive production, and schema assembly. Only the direct semantic sequence will move.

After the migration, the source guard will include `crates/chelis-reef/src`. The documented exception and its issue reference will leave active architecture documents.

**Alternative:** Move Reef link or package logic into the core. This option mixes package policy with compiler semantics.

### 7. Enforce the crate and source boundaries

A dependency guard will inspect the workspace graph and the core manifest. It will use a closed allowlist for direct production dependencies.

The guard will reject a dependency path from the core to these upper areas:

- compiler API
- Reef
- source preparation
- macros
- backends
- schemas
- wire adapters
- caches

An in-memory negative fixture will plant a forbidden dependency. The guard must reject the fixture with the dependency name and path.

The source architecture guard will inspect these production roots:

- `crates/chelis-pipeline-core/src`
- `crates/chelis-compiler-api/src`
- `crates/chelis-reef/src`
- `crates/chelis-cli/src`
- `crates/chelis-e2e/src`

Only the core owner can call two or more canonical semantic stages. Other guarded production functions must call the core or the compiler API facade.

The source guard will retain its current path, alias, macro, method, branch, and loop analysis. A planted Reef duplicate must fail the guard.

**Alternative:** Use review guidance without executable guards. This option cannot prevent a later dependency or sequence copy.

### 8. Record `#![no_std]` blockers without changing them

A focused investigation will list direct standard-library use in the core. It will also list transitive blockers in lower compiler crates.

The inventory will distinguish these classes:

- standard collections and allocation
- thread-local or global state
- stack-growth support
- panic and unwind behavior
- filesystem or process dependencies, if present
- dependency features that require `std`

The inventory will not claim a schedule for removal. It will state that the extracted core still requires `std`.

### 9. Use one authoritative oracle

The authoritative completion oracle will remain:

```text
.venv/bin/python scripts/compiler_pipeline_oracle.py
```

The oracle will add these controls:

- core unit and integration tests
- core compile-fail doctests
- compiler API facade compatibility tests
- the dependency guard and its negative fixture
- the extended source guard and its Reef negative fixture
- accepted and rejected Reef artifact parity
- current compiler API, CLI, edit, cache, and E2E parity

The canonical gate will run doctests for both compiler pipeline crates. OpenSpec validation will remain structural evidence only.

Local evidence will not claim hosted CI status. Hosted macOS Smoke, Docs, and changed-crate jobs remain final platform evidence.

## Risks / Trade-offs

- **Risk: Re-exports change an existing Rust path or visibility.** → Add compile fixtures before the move and use no direct core dependency.
- **Risk: Error conversion changes public diagnostics.** → Freeze each rejection class and compare exact text, order, stage, and bytes.
- **Risk: Cancellation moves across a long operation.** → Keep before-and-after facade checks and retain the current cancellation fixtures.
- **Risk: Host policy leaks into the core.** → Let the facade select the explicit mode and forbid backend dependencies in the core.
- **Risk: Reef error text or artifacts change.** → Add accepted and rejected Reef parity before the direct sequence leaves.
- **Risk: A dependency guard accepts a transitive upper path.** → Inspect the resolved graph and add a planted transitive path fixture.
- **Risk: The new crate creates two semantic owners.** → Activate the expanded source guard before removal of the old owner code.
- **Risk: The crate name implies current `#![no_std]` support.** → Publish the blocker inventory and keep the explicit non-goal.

## Migration Plan

1. Add facade import tests, core compile-fail stubs, Reef parity, and negative guard fixtures.
2. Add the core crate and its dependency guard before semantic code moves.
3. Move artifacts and root construction, then re-export them from the facade.
4. Move isolated and contextual semantic transitions behind facade wrappers.
5. Preserve cancellation and lowering policy in the facade adapters.
6. Migrate Reef to the core and remove its direct semantic sequence.
7. Expand the source guard and remove the documented Reef exception.
8. Add the `#![no_std]` blocker inventory and update architecture documents.
9. Run the authoritative oracle and the local gate.
10. Run a fresh local red-team agent and correct each confirmed major finding.
11. Record hosted CI results for the exact final commit.

Each move will preserve facade wrappers until its parity tests pass. A rollback can restore the prior module implementation without a data migration.

## Open Questions

None. Private module names can change during implementation without a requirement change.
