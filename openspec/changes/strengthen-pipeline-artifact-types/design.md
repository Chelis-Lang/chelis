## Context

The canonical compiler pipeline now separates preparation, type analysis, semantic success, and lowered success. Several smaller artifacts still use shapes that permit invalid states.

Edit success uses a public Boolean that is always true. Root metadata uses adjacent vectors and maps with identical element types.

`complete_checks` returns rejection variants that it cannot produce. `LayeredCheck` can contain effect and linearity errors at the same time.

`DiagnosticSink::iter_from` accepts any `usize` and uses that value as a slice boundary. Its callers get valid offsets from `len()`, but the type does not record this source.

This change defines implementation architecture only. `spec/04-type-system.md` remains the authority for pass order and language behavior.

`spec/10-serialization.md` remains the authority for wire behavior. Existing schema types stay at machine-facing boundaries.

## Goals / Non-Goals

**Goals:**

- Make successful edit validation a value that callers cannot forge.
- Give each root collection and root map a distinct Rust type.
- Bind normal tensor root names to DAG roots through one fail-closed constructor.
- Replace the lowered tuple with a named product.
- Remove impossible semantic error branches from direct callers.
- Make layered semantic outcomes exclusive.
- Replace raw diagnostic offsets with sink-issued checkpoints.
- Preserve all existing behavior and machine-facing bytes.

**Non-Goals:**

- This change does not alter language semantics, diagnostics, pass order, or lower policy.
- This change does not alter runtime, backend, package, generated-code, CLI, or wire behavior.
- This change does not add `UnitInterval`, `StdLibCacheKey`, or another context hash type.
- This change does not add `EntryName` or root-count wrapper types.
- This change does not transfer authority from the numbered specifications.

## Decisions

### 1. Use `ValidatedModule` as the edit success proof

The fragment API will add this opaque type:

```rust
pub struct ValidatedModule(Vec<Expr>);

impl ValidatedModule {
    pub fn as_exprs(&self) -> &[Expr];
    pub fn into_exprs(self) -> Vec<Expr>;
}
```

Only `check_whole_module_edit` will construct `ValidatedModule`. The function will return `Result<ValidatedModule, EditValidationError>`.

`ReplacementReport` will contain `ValidatedModule` and no Boolean marker. The report can retain its name for future replacement metadata.

The opaque field prevents callers from creating false success. Cloning the proof remains valid because the clone contains the same checked expressions.

**Alternative:** Keep `checks_clean` private and add an accessor. This option still stores no information and does not prove which module passed.

**Alternative:** Return `Vec<Expr>` directly. This option cannot distinguish checked expressions from unchecked expressions.

### 2. Use one root name type and four collection types

The pipeline will add these opaque domain types:

```rust
pub struct IrName(String);
pub struct AllRootNames(Vec<IrName>);
pub struct TensorRootNames(Vec<IrName>);
pub struct NamedRoots(BTreeMap<IrName, NodeId>);
pub struct ForwardNodeIndex(BTreeMap<IrName, NodeId>);
```

`IrName` is a namespace type, not a syntax validator. It must support internal load aliases and dotted tuple names.

The collection types will provide narrow iteration, lookup, slice, and consuming conversion methods. Raw `String` conversion will occur only at existing consumer and schema boundaries.

`RootMetadata` will contain `AllRootNames` and `TensorRootNames`. Accessors will return those types instead of `&[String]`.

`NamedRoots` will contain declared tensor outputs only. `ForwardNodeIndex` will start from those outputs and add internal load aliases.

The CLI sites that call `tensor_names()` will use tensor-specific local names. This rename removes the current `all_root_names` naming drift.

**Alternative:** Keep `String` elements inside each opaque collection. This option prevents collection swaps but loses the name namespace at map boundaries.

**Alternative:** Add different name element types for each collection. This option blocks legitimate movement of one canonical name between pipeline stages.

### 3. Construct normal root bindings through exact alignment

`NamedRoots` will have one crate-owned constructor for normal DAG output. The constructor will compare the tensor-name count with the DAG-root count.

If the counts differ, the constructor returns the existing `RootCount` rejection. It will not create a partial map with `zip`.

The existing empty host fallback remains explicit. That branch will call a separate empty constructor only after the current lower policy selects the fallback.

This split preserves host behavior and makes normal alignment fail closed. It also rejects a DAG with unnamed roots when the pipeline did not select the fallback.

**Alternative:** Put the count check in each caller. This option repeats the invariant and still permits direct `collect` calls.

**Alternative:** Store separate root counts. Counts do not prove that names and node identifiers align.

### 4. Return `LoweredParts` instead of a tuple

`LoweredCompilation::into_parts` will return a `LoweredParts` structure with named fields. The fields will use `NamedRoots` and `ForwardNodeIndex`.

The structure will use `#[non_exhaustive]`. Callers can read named fields, but external callers cannot construct a false lowered product.

This shape prevents positional map swaps in compiler API and E2E consumers. It also makes future field additions source-compatible for non-destructuring callers.

**Alternative:** Keep the tuple and change only the map types. The map types stop direct swaps, but a named product gives clearer ownership transfer.

### 5. Narrow direct semantic outcomes

`complete_checks` will return `Result<CheckedCompilation, SemanticRejection>`. `SemanticRejection` will contain only `Effects` and `Linearity` variants.

`run_prepared` and other full pipeline owners will convert `SemanticRejection` into `PipelineRejection`. Preparation, type, lower, and root-count errors stay in the wider type.

Direct callers can use an exhaustive two-variant match. The current `unreachable!` arms for impossible pipeline stages will disappear.

`LayeredCheck` will become an enum with these states:

- `Clean`
- `EffectRejected`
- `LinearityRejected`

Each state will carry the common fitness and typed-program product. Rejected states will carry only their applicable error collection.

**Alternative:** Keep two error vectors and add constructor checks. Public fields still permit contradictory values after construction.

**Alternative:** Return `PipelineRejection` from all functions. This option hides the smaller state space from direct callers.

### 6. Issue diagnostic checkpoints from the sink

`DiagnosticSink` will add `checkpoint()` and `iter_since()` methods. `DiagnosticCheckpoint` will contain a private offset.

The inference code will request a checkpoint before body inference. It will pass that checkpoint back to the same active sink.

The sink only appends errors while inference holds the checkpoint. Therefore, the stored offset remains a valid boundary for that sink.

The old `iter_from(usize)` method will be removed. The existing `len()` method can remain only if another caller needs the count.

**Alternative:** Use `get(start..).unwrap_or_default()`. This option hides an invalid offset and does not encode its source.

**Alternative:** Use a saturated offset. This option changes an invalid state into an empty diagnostic range.

### 7. Keep machine-facing types unchanged

The new types will not derive `Serialize` or `Deserialize`. Existing schema structs will continue to use strings, integers, vectors, and maps.

Compiler API adapters will convert domain types at the current wire boundary. CLI JSON and Tide JSON will remain byte-identical.

No new dependency is necessary. The conversion code uses standard collection traits and explicit accessors.

### 8. Defer the remaining newtypes

`FitnessReport::score` stays as `f64` in this change. CLI code mutates the field after effect and linearity errors.

A separate review will inventory public construction, mutation, tests, and wire conversion before a `UnitInterval` proposal. That review must also examine the numeric-surface rules.

`ContextHash`, `stdlib_cache_key`, and `LayeredCheck` do not share one construction boundary. The hash changes remain separate from this pipeline follow-up.

The `EntryName` and root-count wrapper ideas remain rejected. They do not make the stated invalid states impossible.

### 9. Extend the existing compiler pipeline oracle

The authoritative completion oracle remains:

```text
.venv/bin/python scripts/compiler_pipeline_oracle.py
```

The script will include the new unit, integration, compile-fail, and consumer-parity tests. Its own tests will lock the command list.

Compile-fail tests will cover forged `ValidatedModule`, swapped root map types, and contradictory layered outcomes. Runtime tests will cover alignment errors and diagnostic ranges.

The existing CLI, cache, compiler API, and E2E parity tests will prove that public behavior does not change. OpenSpec validation proves artifact structure only.

Local oracle results do not prove hosted CI status. The final acceptance record will list local evidence and hosted evidence separately.

## Risks / Trade-offs

- **Risk: Public Rust callers fail to compile.** → Migrate all repository consumers in one change and document the new accessors.
- **Risk: Root alignment rejects an existing host fallback.** → Keep a separate empty constructor behind the current fallback decision.
- **Risk: Name conversion changes ordering.** → Preserve vector order and `BTreeMap` ordering in parity tests.
- **Risk: The layered enum changes CLI report assembly.** → Freeze clean, effect, and linearity JSON bytes before migration.
- **Risk: A compile-fail test does not run in CI.** → Keep compiler-API doctests in the authoritative oracle command.
- **Risk: `UnitInterval` expands the change.** → Complete a separate API review and proposal before code changes.

## Migration Plan

This change uses one acceptance phase and one authoritative oracle.

1. Add positive tests, negative tests, and compile-fail stubs for all new requirements.
2. Add `ValidatedModule` and `SemanticRejection`. Migrate fragment and direct semantic callers.
3. Add the root name types, root collection types, exact alignment, and `LoweredParts`.
4. Migrate compiler API, CLI, and E2E root consumers. Keep schema conversions at their current boundaries.
5. Add `DiagnosticCheckpoint` and migrate the body-inference diagnostic query.
6. Replace `LayeredCheck` with exclusive variants and migrate CLI report assembly.
7. Record the separate `UnitInterval` API review without a fitness code change.
8. Run strict OpenSpec validation and the authoritative compiler pipeline oracle.
9. Run a fresh-context adversarial review and correct each confirmed finding.
10. Use hosted CI as final evidence for the full workspace and platform jobs.

Each type family can revert with its consumer migration because no persistent data changes. A rollback restores the prior Rust API and does not need data repair.

No policy activation occurs in this change. All enforcement comes from Rust types and tests after the implementation lands.

## Open Questions

None. The deferred `UnitInterval` work requires a separate proposal after its API review.
