## Context

Chelis has several production owners for the same front-end sequence. The main copies are in compiler API compilation, CLI checks, edit validation, and E2E compilation.

The CLI check path calls `check_ir_fitness` and `check_typed_program` in sequence. Both calls run type inference over the same Deep program.

`compile_source_scoped` uses `check_ir_program`, effects, linearity, and lowering. `chelis_e2e::compile_surf` repeats that sequence and derives root names separately.

The layered Reef path adds a valid cache optimization. It also duplicates fitness formulas and semantic stage transitions.

The pass order remains controlled by `spec/04-type-system.md`. Wire compatibility remains controlled by `spec/10-serialization.md`.

Backend separation remains controlled by `spec/08-backends.md`. This OpenSpec change records implementation architecture and does not replace those authorities.

## Goals / Non-Goals

**Goals:**

- One compiler-API module owns production orchestration for each semantic stage.
- Named pipeline goals make early phase stops explicit.
- Fitness and a typed program come from one inference product on each selected path.
- Typed outcomes make rejected, checked, and lowered states distinct.
- Existing consumers preserve behavior while they move to the shared pipeline.
- Contextual and monolithic checks use the same semantic state transitions.
- One source guard prevents new production copies of the full sequence.

**Non-Goals:**

- This change does not alter language semantics or pass order.
- This change does not alter diagnostic policy or wire formats.
- This change does not replace Reef preparation or cache policy.
- This change does not move style policy into the compiler API.
- This change does not create a generic backend interface.
- This change does not hide target-specific backend failures.
- This change does not prohibit direct pass calls in focused unit tests.

## Decisions

### 1. Put the canonical pipeline in `chelis-compiler-api`

A new `chelis_compiler_api::pipeline` module will own source preparation, semantic stage transitions, and optional lowering.

Lower crates will keep their stage primitives. `chelis-types` will own type analysis, `chelis-effects` will own effect checks, and `chelis-ir` will own lowering.

This placement preserves dependency direction. `chelis-types` cannot own effects or lowering without dependency inversion.

**Alternative:** Add a new workspace crate for orchestration. This option adds another public boundary without removing a current dependency cycle.

**Alternative:** Keep orchestration in the CLI. This option leaves Tide, Python, edits, and E2E with separate compiler behavior.

### 2. Separate preparation, semantic checks, and target emission

The pipeline will use three boundaries:

1. Preparation converts Surf or Deep input into expanded Deep and records source metadata.
2. Semantic checks produce a typed and fully checked program.
3. Optional lowering produces a DAG and canonical root metadata.

Reef graph preparation, linked-name policy, and cache selection remain outside boundary one. They supply prepared declarations or a checked context.

CLI style policy remains before the pipeline. Target selection and backend emission remain after the pipeline.

This split lets C, HIP, and Metal keep their different emission models. The shared pipeline ends before target-specific emission.

### 3. Use named goals instead of one always-full operation

The pipeline will accept a closed goal vocabulary:

- `TypeAnalysis` stops after type analysis and fitness production.
- `FullCheck` adds effect and linearity checks.
- `Lower` adds DAG lowering and root metadata.

Compatibility adapters will select the goal that matches their current contract. The compiler API `check` operation will retain its current phase boundary.

CLI checks and edit validation will select `FullCheck`. Compilation, build, evaluation, and E2E paths will select `Lower` where they require a DAG.

A goal can stop work early, but a consumer cannot reorder stages. This structure preserves compatibility without duplicate orchestration.

**Alternative:** Run every stage for every operation. This option changes current check behavior and adds unnecessary lowering work.

**Alternative:** Expose independent booleans for each stage. This option permits invalid orders and unsupported stage combinations.

### 4. Add one combined type-analysis entry point

`chelis-types` will add an analysis entry that computes structural fitness and runs one IR inference session.

The entry will return a closed outcome:

```rust
pub enum TypeAnalysisOutcome {
    Rejected { fitness: FitnessReport },
    Accepted {
        fitness: FitnessReport,
        program: CheckedProgram,
    },
}
```

The accepted fitness report will use `CheckedProgram::infer_stats()`. The rejected report will derive from the failed inference product.

The entry will preserve the current recursion and cycle guards. A base-case-free recursion group must reject before unbounded work.

Existing public fitness and typed-check functions can remain as compatibility functions. Production orchestration will not call them in sequence.

A test-only inference counter will prove one session per selected monolithic path. The layered error fallback remains an explicit second selected path.

**Alternative:** Cache by source text between the two current calls. This option keeps two APIs authoritative and creates invalidation complexity.

### 5. Model phase success with distinct types

`CheckedCompilation` will contain expanded Deep, fitness data, a fully checked program, inferred signatures, and canonical root metadata.

`LoweredCompilation` will contain a `CheckedCompilation` and its DAG. Rejected outcomes will carry a typed stage rejection and no success payload.

The implementation will not represent phase state with unrelated optional fields. A rejected outcome cannot expose a `CheckedProgram` or `Dag`.

The internal types will not derive `Serialize` or `Deserialize`. Existing schema types remain the only machine-facing wire models.

**Alternative:** Extend `CompiledSource` with optional diagnostics and optional products. This option permits contradictory states and repeats downstream checks.

### 6. Keep diagnostics typed until each existing adapter boundary

The pipeline will preserve native check, effect, linearity, and lowering errors in separate rejection variants.

Compiler API, CLI, edit, and E2E adapters will map those variants to their current result types. The pipeline core will not format stage errors as strings.

The CLI will retain `assemble_check_json` during this change. Issue #886 remains the owner of the separate JSON type and layout decision.

Parity fixtures will freeze diagnostic kind, text, order, severity, spans, inferred signatures, JSON bytes, and exit codes.

**Alternative:** Consolidate diagnostics and JSON in the same change. This option couples an architecture refactor to a public wire decision.

### 7. Preserve layered cache behavior and centralize fitness construction

The layered path will call the same semantic core with a checked library context. The cache will still avoid repeated chelis-std inference.

The clean path will still reconstitute whole-program fitness. The error path will still use the monolithic fallback for exact report parity.

Fitness weights and clean-report construction will move behind one `chelis-types` API. `layered.rs` will not copy the formula constants.

The linked-program guard and trusted linker boundary remain unchanged. This change will not redesign package identity.

### 8. Derive root metadata once from checked Deep

The pipeline will derive root names, tensor roots, tuple flattening, and node mappings from the checked program.

Compiler API and E2E results will consume that canonical metadata. E2E will stop deriving roots from the Surf declaration tree.

Parity fixtures will compare root names, order, tuple suffixes, and DAG node IDs on representative programs.

### 9. Add a scoped source guard

A test-only guard in `chelis-compiler-api` will inspect production Rust sources in this repository.

The guard will reject a production file outside the owner module that orchestrates two or more canonical semantic stages.

The guard will exclude tests, examples, and lower-layer stage implementations. A focused production helper can still call one stage for one purpose.

The guard core will accept an in-memory source inventory. Positive and negative fixtures will test the guard without repository mutation.

A repository test will run the core against the actual workspace. This test prevents a new CLI or E2E copy from passing review unnoticed.

### 10. Use one acceptance oracle

The authoritative acceptance oracle will be:

```text
.venv/bin/python scripts/compiler_pipeline_oracle.py
```

The script will run named parity tests for `chelis-compiler-api`, `chelis-cli`, and `chelis-e2e`. It will also run the source guard.

The oracle will test accepted and rejected inputs. It will test type, effect, linearity, lowering, cache, root, JSON, and exit-code paths.

The repository local gate and hosted CI remain additional evidence. Local results do not claim hosted CI state.

OpenSpec validation proves artifact structure only. It does not prove compiler correctness.

## Risks / Trade-offs

- **Risk: Diagnostic output changes during adapter migration.** → Freeze ordered diagnostics and CLI bytes before code movement.
- **Risk: The combined type entry drops an early safety guard.** → Add bounded recursion and cycle rejection fixtures before migration.
- **Risk: Layered and monolithic reports diverge.** → Keep the error fallback and run cold, warm, and cache-disabled parity fixtures.
- **Risk: Root metadata changes E2E behavior.** → Compare names, order, tuple roots, and DAG mappings before consumer removal.
- **Risk: The source guard rejects focused code.** → Reject only multi-stage orchestration and keep a narrow owner allowlist.
- **Risk: The pipeline becomes a backend abstraction.** → End shared ownership at the lowered DAG and retain backend-specific errors.
- **Risk: A large migration hides behavior changes.** → Move one consumer group per phase and keep compatibility adapters until parity passes.

## Migration Plan

1. Add frozen positive and negative baseline fixtures for all consumer groups.
2. Add the combined `chelis-types` analysis outcome and its one-session evidence.
3. Add the compiler-API pipeline types and monolithic semantic core.
4. Migrate compiler API check, lower, compile, and evaluation helpers.
5. Migrate contextual, layered, cache, and whole-module edit paths.
6. Migrate CLI check and build semantic orchestration without a JSON change.
7. Migrate E2E compilation and root metadata to the compiler API.
8. Activate the source guard after every listed production consumer delegates.
9. Run the acceptance oracle and the local repository gate.
10. Run a fresh-context adversarial review and correct confirmed findings.

Each consumer migration can revert to its prior adapter while the shared API remains additive. The change has no data or package migration.

## Open Questions

None. The implementation can refine private type names without changing the requirements above.
