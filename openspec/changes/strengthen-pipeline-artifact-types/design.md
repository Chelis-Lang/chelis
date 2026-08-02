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

### 3. Construct DAG-backed root bindings through exact alignment

`NamedRoots` will have one crate-owned constructor for DAG-backed output. The constructor will compare the tensor-name count with the DAG-root count.

If the counts differ, the constructor returns the existing `RootCount` rejection. It will not create a partial map with `zip`.

A private root-binding enum will identify these construction causes:

- exact DAG-backed output
- selected successful host-backend output
- accepted nonfatal lowering rejection

Strict mode and host-only mode will use exact alignment after successful lowering. This rule includes successful empty DAGs.

`AllowHostBackend` can accept a successful empty DAG after the CLI selects its host backend. The C host lane then emits the tensor-typed output without a DAG root.

This host result will use the explicit empty constructor. It is not a lower rejection fallback.

An accepted nonfatal lower rejection will also use the explicit empty constructor. The selected policy will accept that rejection before construction.

This split preserves host behavior and makes DAG-backed alignment fail closed. It rejects unnamed DAG roots unless a typed host mode owns the result.

**Alternative:** Align every successful empty DAG. This option breaks the target C host-lane build contract.

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

The conversion code adds no dependency. The source guard adds the existing transitive `syn` package as a direct development dependency.

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

Compile-fail tests will cover forged `ValidatedModule`, swapped root maps, contradictory layered outcomes, and raw checkpoint offsets. Runtime tests will cover alignment errors and diagnostic ranges.

The source-guard suite will include direct stage sequences, helper-composed sequences, and focused one-stage helpers. It will reject orchestration through helper calls.

The existing CLI, cache, compiler API, and E2E parity tests will prove that public behavior does not change. OpenSpec validation proves artifact structure only.

Local oracle results do not prove hosted CI status. The final acceptance record will list local evidence and hosted evidence separately.

### 10. Propagate source stages through local helper calls

The source guard will build one callable and import inventory for all guarded workspace files. It will propagate stage sets through each crate-local graph.

A fixed-point calculation will handle helper chains. A production function will fail when one path reaches two or more canonical semantic stages.

Each call site will remain in the path model. This rule lets two calls to one conditional helper reach different stages on one execution path.

Function identities will include their crate, module, and implementation or trait owner. Canonical stages will use their module and function identity.

Qualified calls will resolve local path prefixes and imported aliases. Shared final function names will not merge stage identities.

Named imports, glob imports, and local function-value aliases will resolve to their original callable. `extern crate` aliases will use the same import map.

Direct, typed, parenthesized, assigned, `if`, and `match` callable expressions will use one abstract-value resolver.

Each execution path will retain its own alias bindings. A lexical shadow will hide the outer binding only in its lexical block.

A local helper's callable parameter will contribute stages only when the helper invokes it. Higher-order calls will substitute callable arguments into the helper summary.

Typed receiver bindings will resolve crate-local methods. Local type aliases will resolve before method lookup.

Qualified-method calls will remove the explicit receiver before callable argument substitution.

Invoked nested functions and closures will contribute their stages. Local macros will resolve direct canonical calls and canonical imports from their definition scope.

Macro identities will include nested block scopes. Lookup will restore an outer macro after an inner block ends.

The loop model will calculate a fixed point across repeated iterations. It will retain zero-iteration exits for `while` and `for`.

Known array iterables will retain their element values and exact iteration counts. Stable Boolean values will restrict branch selection on each loop path.

Return, break, and continue outcomes will remain distinct. Labeled control will terminate only the path for its target loop.

Pattern bindings in `if let`, `while let`, `match`, and `for` will shadow outer aliases only in their lexical scope.

Uninvoked closure bodies and nested function bodies will not contribute stages to the enclosing function.

Only canonical stage paths and imports from those paths will create direct stages. Full import targets will distinguish local and external aliases.

A focused helper that reaches one stage will remain valid. Tests will cover cross-file helpers, imports, path-specific aliases, higher-order calls, traits, macros, and termination.

**Alternative:** Reject every direct stage helper outside the owner. This option blocks valid focused consumers that need only one stage.

### 11. Integrate with the current target branch before acceptance

The implementation branch predates the typed Deep AST changes on `main`. The rebase will replace removed `Atom::Symbol` use with the current `Atom::Name` or typed AST accessors.

The root collector uses one tagged-child view for transitional `List` values and typed `Node` values. This view preserves ordered tuple-root expansion.

Strict parsing identifies unknown tags and malformed known tags in typed fallbacks. Nested structural lists remain valid.

Lenient fragment parsing uses typed nodes for valid input. It retains a transitional diagnostic form if the stamp pass fails, except for a function without a body.

Canonical `deftype` and `typealias` parameter lists use syntax roles. Their type bodies and variants retain their current roles.

The macro expander reads compiler-internal definitions from the typed fallback. It resolves typed variable nodes at macro call sites.

The CLI retains target `.dp` ingestion through `parse_and_stamp_file`. The compiler API wire adapter retains the `Node::to_list` bridge.

The shared Deep verification consumers parse through the typed file boundary. A heap worklist converts the complete tree for their transitional list dispatch.

These consumers include property discovery, producer obligations, constant probes, and constructor probes.

The Deep validator uses one complete tagged-list view for transitional `List` and typed `Node` values. Identity checks use this view.

Deep lint uses a borrowed view of both representations.

The trace fixture walker visits every typed expression variant. It retains nested span IDs without a raw representation fallback.

Compiler API authoring reads declaration tags, binders, and metadata from both representations. The insertion helper calculates a declaration-relative index before it applies the representation offset.

Opaque-value decode collectors use one borrowed view of both representations. The field table supplies the field names for invariant evaluation.

The conflict in `compiler.rs` overlaps the #912 realizability manifest observation. The conflict resolution will retain that observation around the migrated pipeline path.

The authoritative oracle will run only after the rebase compiles against the target branch. Evidence from the stale branch does not satisfy final acceptance.

**Alternative:** Merge first and repair later. This option cannot pass the target build and can delete target-branch observation code during conflict resolution.

## Risks / Trade-offs

- **Risk: Public Rust callers fail to compile.** → Migrate all repository consumers in one change and document the new accessors.
- **Risk: Root alignment rejects an existing host fallback.** → Keep a separate empty constructor behind the current fallback decision.
- **Risk: Name conversion changes ordering.** → Preserve vector order and `BTreeMap` ordering in parity tests.
- **Risk: The layered enum changes CLI report assembly.** → Freeze clean, effect, and linearity JSON bytes before migration.
- **Risk: A compile-fail test does not run in CI.** → Keep compiler-API doctests in the authoritative oracle command.
- **Risk: A successful empty DAG bypasses policy.** → Require an exact or selected host-backend root-binding mode.
- **Risk: Helper composition bypasses the source guard.** → Test cross-file calls, path-specific aliases, higher-order calls, traits, and macros.
- **Risk: Syntax creates false findings.** → Track path termination and exclude uninvoked callable bodies.
- **Risk: A typed fallback hides an invalid Deep tag shape.** → Reject unknown and malformed tags while retaining nested structural lists.
- **Risk: The typed stamp pass removes checker diagnostics for malformed fixtures.** → Retain the malformed fragment only at the lenient diagnostic boundary.
- **Risk: Typed declaration roles reject canonical generic ADTs.** → Keep parameter lists as syntax and retain typed variants.
- **Risk: Typed macro nodes hide an internal macro call.** → Read fallback definitions and typed variable callees.
- **Risk: Typed Deep verification disappears from a list-based consumer.** → Parse through the typed boundary and bridge the complete tree.
- **Risk: Typed Deep identity checks accept a forgery.** → Use one complete tagged-list view for both representations.
- **Risk: Typed Deep lint or trace traversal skips a subtree.** → Visit metadata and children for each typed expression variant.
- **Risk: Typed Deep authoring uses a raw-list insertion offset.** → Calculate the declaration index before applying the representation offset.
- **Risk: Typed Deep decode omits invariant data.** → Collect field types, field names, invariants, and constants through one representation view.
- **Risk: Final-name matching merges stage identities.** → Classify each stage with its owning module and function.
- **Risk: Receiver syntax hides a higher-order helper.** → Resolve typed local receivers and test qualified-method argument positions.
- **Risk: A loop model loses repeated stages.** → Calculate a fixed point over iteration and labeled control outcomes.
- **Risk: A loop model creates an impossible path.** → Preserve stable Boolean facts and known finite iteration counts.
- **Risk: A branch pattern leaks an outer alias.** → Bind patterns before branch analysis and remove them at scope exit.
- **Risk: A macro alias hides a stage.** → Resolve canonical imports from the macro definition scope.
- **Risk: An inner macro replaces an outer macro.** → Include block identity in the lexical macro key.
- **Risk: The stale branch loses target changes during conflict resolution.** → Preserve typed ingestion, the wire bridge, root support, and the #912 observation.
- **Risk: `UnitInterval` expands the change.** → Complete a separate API review and proposal before code changes.

## Migration Plan

This change uses one acceptance phase and one authoritative oracle.

1. Add positive tests, negative tests, and compile-fail stubs for all new requirements.
2. Add root tests for strict empty-DAG mismatch, selected host-backend success, and an accepted nonfatal fallback.
3. Add source-guard tests for helper composition and focused one-stage helpers.
4. Rebase onto the current target branch and preserve the #912 realizability manifest observation.
5. Add `ValidatedModule` and `SemanticRejection`. Migrate fragment and direct semantic callers.
6. Add the root name types, root collection types, exact alignment, and `LoweredParts`.
7. Route every DAG-backed lower result through exact root alignment and type the selected host-backend result.
8. Propagate source stages through the local call graph.
9. Migrate compiler API, CLI, and E2E consumers. Keep schema conversions at their current boundaries.
10. Add `DiagnosticCheckpoint` and migrate the body-inference diagnostic query.
11. Replace `LayeredCheck` with exclusive variants and migrate CLI report assembly.
12. Record the separate `UnitInterval` API review without a fitness code change.
13. Run strict OpenSpec validation and the authoritative compiler pipeline oracle against the target branch.
14. Run a fresh-context adversarial review and correct each confirmed finding.
15. Add repeated-helper, qualified-helper, unrelated-receiver, typed-root, and wire-shape regression tests.
16. Run a second fresh-context review after remediation.
17. Add cross-file, alias, trait, macro, and executable-path source-guard tests.
18. Run a third fresh-context review after source-guard remediation.
19. Use hosted CI as final evidence for the full workspace and platform jobs.

Each type family can revert with its consumer migration because no persistent data changes. A rollback restores the prior Rust API and does not need data repair.

No policy activation occurs in this change. All enforcement comes from Rust types and tests after the implementation lands.

## Open Questions

None. The deferred `UnitInterval` work requires a separate proposal after its API review.
