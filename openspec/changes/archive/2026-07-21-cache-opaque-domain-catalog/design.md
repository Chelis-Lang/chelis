## Context

`opaque-domain-construction` needs declarations from sibling Surf files to detect cross-module construction. Today `check_surf` calls `collect_surf_catalog(ctx.root)` for every checked Surf file. That helper starts an independent, unfiltered `WalkDir`, then reads and parses every `.ch` file it finds. A repository lint therefore performs one full repository walk and one full Surf-corpus parse per checked Surf file; nested build and environment trees are traversed even though the main lint walker excludes them.

The fix must satisfy Chelis-Lang/chelis#603 without weakening the rule's existing CR-9, CR2-7, or CR3 behavior. It must also remain correct when the CLI reuses rule objects across multiple targets or repeated `lint` calls during `--fix` convergence.

## Goals / Non-Goals

**Goals:**

- Make opaque catalog discovery use exactly the entries admitted by the canonical lint walker.
- Build the corpus catalog once per `chelis_lint::lint` invocation and share it across that invocation's Surf checks.
- Keep prepared state invocation-local so repeated lint calls observe filesystem changes.
- Preserve all current Surf, Deep, and single-file diagnostics.
- Lock both performance invariants and semantic parity with positive and negative tests.

**Non-Goals:**

- Persisting catalogs across CLI invocations or introducing filesystem invalidation.
- Parallelizing lint dispatch.
- Changing opacity/type-system semantics or diagnostic wording.
- Optimizing unrelated rules or eliminating the checked file's normal parse.

## Decisions

### 1. Add invocation-scoped rule preparation to the lint driver

The driver will materialize the canonical walker result once, then give each rule one preparation opportunity before per-entry dispatch. The `Rule` trait will provide a default no-op preparation path and a default prepared-check path that delegates to the existing `check`, so unaffected rules retain their current implementation.

Prepared state will be immutable, type-erased, and paired with its owning rule only for the duration of the `lint` call. `OpaqueDomainConstruction` will override preparation to build a `Catalog` from admitted `Surface::SurfSource` entries and override prepared checking to consume that catalog.

This is preferred over a process-global or root-keyed lazy cache because persistent caches become stale after edits and can cross-contaminate concurrent or repeated lint invocations. It is preferred over special-casing the rule in `lint` because the lifecycle remains a general rule capability with a testable default.

### 2. Reuse the canonical walker result instead of starting a second walk

`lint` already receives a vector from `walker::walk(root)`. It will resolve walker errors and retain the admitted entries before rule preparation and dispatch. Opaque catalog preparation will filter those entries by `Surface::SurfSource`; it will not call `WalkDir` or duplicate `is_skip_dir`.

This gives catalog discovery the exact same exclusion semantics as lint dispatch and limits the invocation to one filesystem traversal. Candidate source read or Surf parse failures will continue to be skipped, matching the current fail-soft catalog behavior and the driver's unreadable-source behavior.

### 3. Split immutable corpus data from file-local shadow state

`Catalog.current_file_module_less_leaves` is currently mutated for every checked file. The refactor will keep only corpus-wide opaque types and named-module declaration leaves in the prepared `Catalog`. `surf_module_less_leaves` will remain a per-checked-file value passed separately through the Surf checking functions.

This avoids cloning or mutating the shared catalog and preserves CR2-7: top-level module-less declarations shadow constructions only in their own file, while named-module declarations remain corpus-wide.

### 4. Keep direct and single-file rule behavior available

The prepared path is authoritative for `chelis_lint::lint`. The existing direct `Rule::check` path will remain usable for focused rule tests and one-off callers by preparing from the supplied root once for that direct call. A file root is represented by the canonical walker as one admitted Surf entry, so the style gate remains self-contained.

Deep checking will bypass the prepared Surf catalog and continue deriving its catalog from the checked Deep expressions.

### 5. Test observables rather than elapsed time

Regression tests will prove the complexity boundary mechanically:

- a test rule will verify that preparation is invoked once while checks run for multiple entries;
- opaque-rule tests will verify that skipped-tree `.ch` declarations cannot influence the catalog and admitted declarations still can;
- repeated invocations with the same rule objects will verify that catalog state is rebuilt rather than retained;
- existing allow/reject, module-less shadow, and Deep tests will remain semantic parity coverage.

The authoritative acceptance oracle for this change is the following two-command runner (the crate-local CLI regression test requires an explicit current-worktree binary):

```text
CARGO_TARGET_DIR=target/agents/issue-603 cargo build -p chelis-cli --bin chelis
CARGO_BIN_EXE_chelis="$PWD/target/agents/issue-603/debug/chelis" CARGO_TARGET_DIR=target/agents/issue-603 cargo nextest run -p chelis-lint
```

## Risks / Trade-offs

- **[Risk] The new preparation hook broadens the public `Rule` trait.** → Provide default methods so existing rules and downstream implementations remain source-compatible unless they opt into prepared state.
- **[Risk] Type-erased state can be paired with the wrong rule.** → Keep preparation and dispatch vectors index-aligned inside `lint`; opaque prepared checking will use an explicit downcast with an invariant assertion and a safe direct-check fallback.
- **[Risk] Precollecting admitted entries retains path metadata for the invocation.** → The walker already returns a vector, so this does not materially increase current memory use.
- **[Risk] Catalog and file checking still parse each admitted Surf file separately.** → This change removes the quadratic catalog parses; sharing full parsed ASTs is a separate optimization with wider ownership and API consequences.
- **[Trade-off] `--fix` may rebuild the catalog on each convergence pass.** → Rebuilding is required for correctness because earlier passes can edit source; each pass remains linear and filtered.

## Migration Plan

1. Add failing lifecycle and filtered-catalog regression tests.
2. Add the default rule-preparation API and adapt `lint` to prepare once from its canonical entry vector.
3. Move `OpaqueDomainConstruction` catalog construction to prepared state and split file-local shadow data from the corpus catalog.
4. Run the crate acceptance oracle, then the repository local gate if implementation changes extend beyond `chelis-lint`.

Rollback is a normal source revert; there is no persisted data or format migration.

## Open Questions

None. The invocation boundary is the existing `chelis_lint::lint` call, which also gives repeated `--fix` passes the required fresh catalog.
