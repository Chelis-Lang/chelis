## Context

See `proposal.md` for the user-visible fault. The current build path expands the selected program and immediately calls `drop_unreachable_eval_only_defs`.

The layered cache path still analyzes the original declarations. It returns `Ok(Some(...))` only after complete semantic success.

`Ok(None)` combines semantic rejection with valid cache fallback cases. The caller then runs a monolithic check on the post-drop program.

That fallback input causes chelis#1184. An invalid definition can disappear before the authoritative diagnostic path sees it.

The design must preserve three earlier fixes:

- chelis#334 permits a well-typed unused `process_run` dependency to stay outside a compiled artifact.
- chelis#1176 removes the complete unreachable eval-only wrapper chain and preserves cache-state parity.
- chelis#1168 keeps the clean warm build path on the layered cache.

OpenSpec records the plan and review evidence. `spec/04-type-system.md` and `spec/05-risc-primitives.md` remain the normative authorities.

Structured suggestions operate after this semantic gate produces a diagnostic. Their repair model does not depend on this pipeline-order change.

The SNAFU migration can type operational failures from this path. It cannot replace the `Ok(None)` fallback or source diagnostics.

## Goals / Non-Goals

**Goals:**

- The selected program obtains complete semantic success before build-specific pruning.
- A semantic error in an unreachable eval-only-tainted definition fails build.
- A well-typed unreachable eval-only-tainted definition stays outside backend checks and code emission.
- Clean warm builds retain the layered cache path without a new monolithic full-program check.
- Cold, warm, and cache-disabled build errors remain byte-identical.
- If structured repairs exist, cache modes preserve the same repair data for the same diagnostic.
- The checked emission target remains the only input to backend checks and code emission.

**Non-Goals:**

- The design does not add termination analysis or a new semantic pass.
- The design does not change the eval-only builtin roster.
- The design does not change `tensor_scan`, test assertions, or object-mode liveness rules.
- The design does not change a cache key, cache file, Reef graph, or generated artifact format.
- The design does not make `chelis build` analyze a file outside the selected target.
- The design does not define a structured repair producer or change SNAFU ownership.

## Decisions

### D1: Keep the selected program until its semantic gate succeeds

`cmd_build` retains the expanded full program as `selected_deep_exprs`. It does not overwrite this value with the eval-only removal result.

The build obtains a complete `CheckedCompilation` for `selected_deep_exprs` before it prepares the emission program. This checked value proves type, effect, and linearity success.

The build sequence becomes:

1. Source preparation creates `selected_deep_exprs` and `entry_deep_exprs`.
2. The layered or monolithic path checks `selected_deep_exprs` completely.
3. The separate whole-program `tensor_scan` build gate checks that selected program.
4. Eval-only removal transforms the accepted selected program.
5. General reachability pruning creates the emission program.
6. The pipeline checks the changed emission program for its backend input.
7. Other backend gates and emitters consume only the checked emission program.

The second semantic check remains necessary when pruning changes the program. It creates checked metadata and root data for the exact backend input.

If pruning changes nothing, the build reuses the checked selected program. It does not repeat semantic analysis.

**Alternative rejected:** A monolithic full-program check always runs before pruning. This option restores correctness but removes the clean warm-cache benefit from chelis#1168.

**Alternative rejected:** Build calls the `chelis check` command as a subprocess. This option duplicates source preparation and changes diagnostic and performance behavior.

### D2: Keep layered fallback, but run it against the selected program

`check_layered_for_build` keeps its current return contract:

- `Ok(Some(checked))` proves complete selected-program success.
- `Ok(None)` requests the existing byte-identical monolithic diagnostic path.
- `Err(error)` reports the existing standard-library context failure.

The caller handles `Ok(None)` with `checked_compilation_with_effects(&selected_deep_exprs)`. It never uses a post-drop value for this fallback.

`Ok(None)` remains an expected fallback request, not an operational error. A SNAFU migration can wrap only the existing `Err(error)` branch.

A disabled cache takes the same monolithic selected-program path. A valid clean layered result avoids that monolithic path.

This choice fixes the fault without a new cache result enum. The final user diagnostic still comes from the established monolithic formatter.

**Alternative rejected:** The layered path returns its own semantic errors directly. That option changes diagnostic bytes and mixes cache suitability failures with final program rejection.

### D3: Apply eval-only policy after semantic success

After D1 succeeds, `drop_unreachable_eval_only_defs` keeps its current transitive closure. It removes each unreachable direct user and each unreachable dependent.

A reachable eval-only call remains in the emission program. The shared compiler gate rejects it through the existing `Unsupported` channel.

A well-typed unreachable eval-only call does not reach a backend. Therefore, the build does not reject that call only because of backend availability.

This split matches the Rust model for a selected crate target. Semantic checks cover included definitions, while dead-code removal precedes code generation.

The linked Reef requirement defines the package case. It does not remove the existing loose-source path or its host-library surface preservation.

The separate `tensor_scan` whole-program gate runs after semantic success and before pruning. This design does not merge its roster with `EVAL_ONLY_HOST_BUILTINS`.

### D4: Preserve exact cache and diagnostic parity

The negative fixture uses one Reef dependency with an unreachable eval-only chain. It places semantic errors at direct and transitive positions.

Each fixture runs in these modes:

- cache disabled
- cache cold
- cache warm

All modes must reject with byte-identical build stderr. `chelis check` must reject the same diagnostic kind and source location.

If structured suggestions exist, all modes must preserve equivalent help and repair data for that diagnostic.

The positive fixture keeps a well-typed depth-three wrapper chain. All build modes must accept it and emit byte-identical C.

A reachable eval-only fixture must retain the existing `Unsupported` diagnostic. An unreachable ordinary type-error fixture must remain rejected.

### D5: Use one focused completion oracle

The authoritative completion oracle is:

```text
CARGO_TARGET_DIR=target/agents/check-before-build-pruning cargo nextest run -p chelis-cli --test library_cache_oracle --no-fail-fast
```

This suite carries the direct, transitive, cache-parity, diagnostic, and generated-C evidence for this change.

`openspec validate check-before-build-pruning --strict` provides structural evidence only. It does not replace the executable oracle.

If structured suggestions exist, final validation also runs their application oracle. This check protects pre-prune candidate validation.

The required hosted `Integration Tests (Linux)` check supplies final hosted evidence. A fresh local red-team pass supplies adversarial evidence before completion.

## Risks / Trade-offs

- [A fallback build checks more source than before] → This stricter result is the intended breaking change and matches `chelis check`.
- [The clean warm path gains a duplicate full check] → The branch reuses `Ok(Some(checked))`, and the focused oracle covers warm-cache use.
- [Eval-only removal stops before the transitive closure] → A direct unit fixture verifies the complete chain and paired `defsig` removal.
- [The selected and emission programs use the wrong checked product] → Backend code receives only the product for the exact emission program.
- [Diagnostic bytes vary by cache state] → The layered rejection still routes through one monolithic formatter on `selected_deep_exprs`.
- [Repair data varies by cache state] → Conditional fixtures compare help and structured repairs across all modes.
- [SNAFU converts fallback into failure] → Keep `Ok(None)` as control flow and type only the `Err(error)` branch.
- [The change accidentally alters `tensor_scan`] → A negative fixture keeps the entry-unreachable whole-program rejection.
- [Local evidence differs from hosted evidence] → The local oracle records behavior, while the required Linux integration check confirms the hosted environment.

## Migration Plan

No data or cache migration is necessary. Existing valid programs and generated artifacts remain unchanged.

A package with a dormant semantic error in selected code becomes a build failure. The source error is the migration path.

The change activates with the compiler release. It uses no feature flag or compatibility mode.

Rollback reverts the pipeline order, tests, and normative text together. A partial rollback is not valid because it restores the documented divergence.
