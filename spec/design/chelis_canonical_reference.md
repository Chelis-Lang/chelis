# Chelis: Canonical Project Reference

**Status:** Active project-level reference.
Use this document to align README text, roadmap updates, design docs, and agent-authored
documentation.
Language semantics still belong in the numbered spec documents.

---

## 1. What Chelis Is

Chelis is a functional programming language for AI research.
It is designed for a workflow where a coding agent is the primary author and a human is
the supervisor.
Programs written in Chelis are themselves AI systems: models, training loops,
evolutionary search spaces, and learned functions.

Chelis is not a general-purpose language, not a systems language, not a web framework,
and not a Python replacement.
It targets model authoring, training, deployment, and program synthesis for AI
workloads.

**Current status:** Phase 0 is complete. Phase 1 and Phase 2 have shipped their
planned compiler surfaces. Phase 3 is the active language-completeness and shell
ecosystem phase: the compiler/runtime foundations through `3j` Nautilus are shipped,
while Coral (`3k`), Shoals (`3l`), Octant Part A/B (`3n`/`3o`), native Chelis testing
(`3t`), and the final SKILL.md v2 refresh (`3f`) remain active work.

Phase completion claims still follow the repository completion bar: a phase is not
closed by crate-local green alone. The authoritative oracle must be named, current
docs must match shipped behavior, and the required fresh-context red-team checkpoint
must run before a phase is declared complete.

---

## 2. Core Bet

AI development benefits from a language whose type system, representation, and
compilation model are designed around AI primitives instead of being bolted onto Python
or a systems language after the fact.

---

## 3. Design Principles

When tradeoffs appear, apply these in order:

1. **Unambiguity over ergonomics.**
2. **Composition over special cases.**
3. **Inference over annotation.**
4. **Machine generation first.**
5. **Additive sugar only.**
6. **Explicit over implicit.**
7. **Small language, big library.**
8. **Future-proof without over-building.**

Practical consequences:

- no implicit broadcasting
- no implicit precision promotion
- no implicit currying or partial application
- Deep is canonical; Surf is supervision-friendly sugar

---

## 4. Dual Syntax Architecture

### Surf

Surf is the human-facing syntax stored in `.ch` files.
It is the supervisory interface: modern functional syntax with ADTs, pattern matching,
pipes, and type annotations.

### Deep

Deep is the machine-facing syntax stored in `.dp` files.
It is the compiler's canonical textual representation and the generation target for AI
agents.

Every Deep node has the same shape:

```lisp
(tag {} children...)
```

- `tag` comes from the closed Deep tag vocabulary
- `{}` is the metadata map and is always present in canonical form
- `app`, `var`, and `lit` are explicit structural nodes

Deep is the single source of truth.
Surf desugars losslessly to Deep.
The default Deep-to-Surf decompile path returns formatter-canonical Surf for
the supported round-trip surface; verbose decompile remains the explicit
best-effort debug path.
Current Phase 1 caveat: the compiler may emit dotted Deep module/import paths that
`chelis validate` accepts, but the compiler-side Deep parser does not yet fully
reparse that emitted shape. This round-trip gap is tracked as a Phase 2 bug because it
matters for decompilation and LLM repair workflows, but it is not a Phase 1 blocker.

**Macro expansion boundary:** LLMs interact exclusively with expanded Deep.
Macro invocations are expanded before any LLM-facing operation (generation, training,
fitness scoring, error reporting), and before CLI paths such as `chelis deep`,
`check`, `build`, and `eval`.
Provenance metadata in the `{}` slot traces expanded nodes back to their macro source
(e.g., `{source: (relu input)}`).
The 62-tag vocabulary is the complete LLM-facing grammar regardless of how many macros
exist in the ecosystem.
Macros are a human authoring convenience that compiles away before LLMs touch the code.
Compiler-internal pre-expansion forms such as `defmacro` and `macro-invoke` are not
public Deep and are rejected by strict Deep validation.

The same properties that make Deep a stable generation target for agents
also make it a stable editing target. The 62-tag closed vocabulary, the
3-tuple uniformity, and the metadata-map slot for provenance mean
structural edits (replace a function body, rename a symbol, change a
signature) are well-defined operations rather than character-level
gambles. This is the architectural foundation for the Agent Editing Surface
direction (`spec/design/chelis_agent_editing_surface.md`).

### Serialization

Chelis currently has three named artifact forms:

- `.ch` for Surf source
- `.dp` for Deep source
- `.chb` for Shell metadata in the Reef package system

`.chb` now exists as an implementation-owned binary package interface.
The name and role are stable; the low-level wire format is intentionally not frozen in
the public spec yet.

---

## 5. Ecosystem Names

All project-level naming follows the turtle/ocean metaphor.

| Concept | Name | Rationale |
|---|---|---|
| Surface syntax | **Surf** | The surface — what you see |
| Canonical s-expression syntax | **Deep** | The depths — what's underneath |
| Packages | **Shells** | Turtles have shells; self-contained |
| Package registry | **Reef** (reef.chelis.ch) | Where shells live |
| Interactive mode | **Tide** (`chelis tide`) | Comes and goes; interactive |
| TUI coding environment | **Cove** (`chelis cove`) | Sheltered workspace where Surf meets shore |
| Project manifest | `reef.toml` | A project's place in the reef |
| Numerical methods shell | **Nautilus** | Mathematical precision — the chambered nautilus is nature's logarithmic spiral |
| Dataframe shell | **Coral** | Structured colonies built from the reef |
| Finance shell | **Shoals** | Where the currents of capital run shallow |
| LaTeX ↔ Deep bridge shell | **Octant** | Navigational instrument bridging celestial observation (math) and positional computation (code) |
| Classical ML shell | **School** | A school of fish learning together — and the ML sense of *learning* |
| Evolutionary algorithms shell | **Darwin** | Natural selection — survival of the fittest programs, mutated and crossed over the Deep AST |
| Language specification shell | **Hull** | The hull defines the shape of the vessel — the spec defines the shape of the language |
| Automated static analysis shell | **Hydrostatic** | A pressure test proving the hull holds before the vessel sails |
| Bound-propagation verification shell | **Beacon** | A guiding light marking safe passage through proven bounds |

Chelis is pronounced **CHEL-is**.
The domain is **chelis.ch**.

### Shell Ecosystem

Requirements for shell repos — scaffolding, pin hygiene, capability-surface
docs, upstream-issue discipline, blocker probes, skills vendoring — are
normative in [`shell_repo_contract.md`](shell_repo_contract.md), mechanized by
`chelis reef conform` in the toolchain.

The **machine-readable** shell registry is `chelis_conformance::registry::REGISTRY`
(the `chelis-conformance` crate), ground-truthed to the
`.github/workflows/ecosystem-drift.yml` canary matrix by the
`registry_matches_drift_matrix` tripwire. The active set it tracks (nautilus,
coral, shoals, school, hull, whale, octant, calcify, c-earchin, hydronnx,
hello-chelis) is the authority for the live ecosystem; the prose table below is
a narrative view and may lag it (reconciling the two into a single generated
table is tracked follow-up).

**Runtime vs shells.** `chelis-std` is the language **runtime**, not a shell.
It version-marches with the compiler, ships bundled with the toolchain, and
cannot be substituted independently — programs depend on it implicitly the
same way Rust programs depend on `core`/`std`. Every other entry below is a
**shell** in the substitutability sense: a library that someone could in
principle write an alternative to (alternative scipy/pandas/etc.). Shells are
fetched and installed via reef from network releases; the runtime is not.

Lockfile entries for the runtime use `LockSource::Bundled
{ compiler_version }` to make the distinction explicit and auditable. The
entry is synthesized into every project's lockfile based on the project's
own `compiler =` pin, regardless of whether `chelis-std` appears in
`[dependencies]`: the pin IS the runtime declaration. A declaration of
`chelis-std = { version = "X" }` in a downstream `reef.toml` soft-verifies
against the compiler's bundled runtime version: matching is recorded as
`Bundled` (idempotent with the synthesized entry), mismatching surfaces a
typed error naming both versions. The runtime bytes themselves are
compile-time-embedded into the chelis binary via `include_bytes!()` in
`crates/chelis-std-bundle`; the loader serves them directly without
consulting the local registry. Auto-fetch from GitHub is intentionally
disabled for the runtime; `reef install --bootstrap chelis-std` is
rejected with a typed error explaining the runtime is compiler-bundled.

Shells layered on the runtime. `nautilus` and `coral` are independent and can land in
parallel; `shoals` depends on both. `octant` Part A (LaTeX ↔ Deep bridge, parser +
deterministic lowering + rendering + provenance) has the same prerequisite as
`shoals` (namely `nautilus` green) and runs **in parallel with `shoals`**; `octant`
Part B (finance-notation lowering through `shoals`, Greek rendering, notebook) is
sequential after `shoals`. `school` is **active** (the ML shell — see its
table row). `darwin` (evolutionary
algorithms), `hull` (executable language specification), `hydrostatic` (automated
static analysis on the tensor DAG), and `beacon` (IR-native bound-propagation
verification) are post-Phase-3 stubs, as is `octant-docs` (full
LaTeX document ingestion, Octant Phase 4).

| Package | Kind | Depends On | Status | Contents |
|---|---|---|---|---|
| `chelis-std` | **Runtime** (compiler-bundled) | (core) | Active | `Std.Init` (Kaiming, Xavier, trunc_normal), `Std.Io` (files, mmap, safetensors, CSV, JSON), `Std.Tensor` (including `Std.Tensor.Mask`), `Std.Index`, `Std.Scan`, `Std.Sort`, `Std.Process`, `Std.Tokenizer`, `Std.Time`, `Std.Decimal`, `Std.Test` (assertion functions for Chelis-native tests). The ML modules moved to the `school` library as of chelis-std 0.4.0: `Std.Nn.*` → `School.Nn.*`, `Std.Loss.*` → `School.Loss.*`, `Std.Optim` → `School.Optim`, `Std.Schedule` → `School.Schedule`. |
| `nautilus` | Shell | `chelis-std` | Active (`v0.5.0` released) | Numerical methods — stats, distributions, linear algebra (nalgebra-backed with hand-written AD adjoints), convex optimization, ODE/SDE solvers, roots, integration, interpolation, special functions (`erf`, `log_gamma`, …), distances. The scipy competitor. `Nautilus.Signal` stubbed until complex numbers (Phase 5f). |
| `coral` | Shell | `chelis-std` | Phase 3k | Typed dataframes — numeric columns are tensors (lazy, GPU-accelerable, fusible via the DAG), string columns are host-side lists (eager). AD through dataframe operations. Column selection, filtering, sort-by, group-by, joins, pivot/melt, rolling windows, NaN handling built into `Coral.Frame`, Parquet I/O via `parquet2`, DataFrame-aware CSV/JSON. The pandas competitor. No query optimizer — numeric optimization comes from the tensor compiler's fusion. |
| `shoals` | Shell | `chelis-std` + `nautilus` + `coral` | Phase 3l | Options pricing, risk measures, yield curves, stochastic processes, order books |
| `octant` | Shell | `chelis-std` + `nautilus` required; `shoals` required only for the Part B SDE / MC / curve lowering | Phase 3n (Part A) ∥ Phase 3l, Phase 3o (Part B) after Phase 3l | LaTeX ↔ Deep bridge for quantitative finance. Parses a bounded LaTeX subset, lowers to Deep deterministically (arithmetic, derivatives, special functions, integrals, matrix ops) in Part A plus LLM-assisted lowering (SDE, Monte Carlo expectation, calibration, yield curves) in Part B, round-trips through the compiler with type overlays, and carries provenance spans on every Deep node. Ships an interactive cell-based notebook in Part B. **NOT a CAS** — notation adapter only, no symbolic integration or simplification. |
| `school` | Shell | `chelis-std` (+ `nautilus` + `coral` planned re-adds) | **Active** (P0–P5 shipped; pinned `=0.7.23`) | Machine learning. Sole home of the NN surface since chelis-std 0.4.0 (`School.Nn.*`, `School.Loss.*`, `School.Optim`, `School.Schedule`): layers, activations, norms, attention, losses, 9 optimizers, schedules, HPO, data utilities, training loop, six-model zoo. Intent is a general deep-learning framework (School `spec/vision.md`); the classical-ML scope (regression, trees, SVMs, clustering, pipelines, cross-validation) remains roadmap. Reference implementation for [`shell_repo_contract.md`](shell_repo_contract.md). |
| `darwin` | Shell | `chelis-std` + `nautilus` required, `coral` optional | **Stub** (post-3) | Evolutionary algorithms — GA, genetic programming over the Deep AST, evolution strategies, population-based training, neural architecture search. Uniquely natural fit because Deep is homoiconic: program mutation and crossover are typed AST operations, and the compiler's 0–1 fitness scoring is literally the fitness function for evolutionary search. `coral` is optional for evolving feature-engineering pipelines over tabular data. |
| `hull` | Shell | `chelis-std` | **Stub** (post-3) | Executable language specification. Self-hosted reference type checker and evaluator implementing the LaCaDiLE typing rules and operational semantics as Chelis functions over Deep AST ADTs. Differential testing against the real compiler. Spec-driven random well-typed program generation. The spec of Chelis, written in Chelis, checked by Chelis. |
| `hydrostatic` | Shell | `chelis-std` + compiler DAG IR | **Future** | Automated static analysis: value range inference, div-zero detection, overflow detection, NaN propagation, bounded output verification. Input ranges specified by user; output ranges inferred. Pre-deployment gate (minutes, not milliseconds). Inspired by Astree (Airbus A380 flight control verification). Trust stack Level 3. |
| `beacon` | Shell | `chelis-std` + compiler DAG IR | **Future** | IR-native, sound bound-propagation verification engine (CROWN lineage: interval, zonotope, then linear relaxation with branch-and-bound). Proves output bounds for numerical programs; the same engine bounds finance pricing graphs and neural networks (the latter via Hydronnx ONNX-to-IR). Plugs into the in-core verification orchestrator through the discharge-engine interface and runs standalone for the VNN-COMP path. Design: `verification_stack_master_plan.md`, `beacon_plan.md`. |

Design rule: the chelis-std runtime covers what every Chelis program may need
(tensors, neural primitives, time, decimal); the substitutability criterion
keeps it out of the shell taxonomy — programs cannot opt out of it any more
than a Rust program can opt out of `core`. `nautilus` owns general numerical
methods. `coral` owns tabular data. `shoals` is finance-only. `school` owns
machine learning — the NN surface migrated from chelis-std 0.4.0 plus the
classical-ML roadmap. `darwin` is evolutionary search only. If it's about the language's own specification and
conformance testing, it goes in `hull`. If it's about automated static analysis on the
DAG (range inference, overflow detection, numerical stability), it goes in `hydrostatic`.
If it's about discharging a verification goal by sound bound propagation over the DAG
(proving output ranges, bounded Greeks, or neural-network properties), it goes in `beacon`.
`octant` is a notation bridge layered on top of `nautilus` and (optionally) `shoals` —
it consumes their APIs and adds no numerical capabilities of its own. Time and decimal
stay in `chelis-std` because every domain needs dates and exact arithmetic.

### Cross-Cutting Design Decisions

**Stability labels on exported APIs.** Every function in every shell's SKILL.md API surface table carries a stability label: `stable` (signature will not change between releases — safe for AI training corpus inclusion) or `alpha` (signature may change — exclude from training data or down-weight). This serves the AI coding pipeline: the RLVR training loop (Phase 4) needs to know which functions are safe to teach the model. It also serves human consumers: a function marked `alpha` comes with an explicit warning that the API may change.

**Persistent data structures for frame-like containers.** Coral's DataFrame uses a persistent dictionary (HAMT) for the column map, so that operations like `with_column`, `drop_column`, and `rename` produce new frames sharing column references with the original via structural sharing. This is a performance requirement for AD through frame pipelines: `grad(fn_with_10_frame_ops)` produces intermediate frames on the backward pass, and structural sharing keeps memory cost at O(num_operations) rather than O(num_columns * num_operations). Pure-Chelis HAMT preferred over Rust-side HAMT for AD compatibility (the persistent dict must be transparent to the AD system).

**Instruments as dicts in Shoals, not closed ADTs.** Financial instruments are open-ended (new payoff structures are invented continuously). Representing instruments as `Dict[String, f32]` lets new instrument types be added as data without code changes. The pricing function dispatches on a key, not a pattern match over a closed enum. This also makes instrument definitions AI-friendly: an agent writes a dict literal (within current LLM capability), not a new ADT variant (requires understanding the type system's extension points).

**Fast `chelis eval` as a pre-Phase 4 investment.** The RLVR training pipeline needs sub-second program evaluation with package-aware imports. `chelis eval` must resolve reef package imports and return results in under 200ms for the training loop to be practical. This also serves agent-driven development (sub-second feedback during Coral/Shoals/Octant construction).

**Verified error messages with per-property explanations.** Compiler diagnostics explain which property the rejection protects and suggest a fix. Not "type mismatch" but "mul requires dimension-wise equality: expected [batch, hidden] got [hidden, batch] — did you mean permute(b, [1, 0])?" Directly improves the RLVR reward signal: better errors = more informative feedback = faster agent repair = faster training convergence. Does not require Lean — the existing type checker has the information, it just needs better formatting.

**Structured fitness score with per-property components.** The 0-1 fitness score is broken into components in the fitness JSON: dimension score, effect score, linearity score, differentiability score, syntax score. Agents see which property failed and focus repair on that specific issue. The aggregate score is still computed for RLVR reward; the components are exposed for agent introspection and trajectory analysis.

**Reproducibility manifests.** `chelis manifest program.ch` extracts all `Random`-effect-annotated operations from the typed AST into a structured JSON report: which operations introduce randomness, which seed handlers cover them, and whether the computation is fully reproducible. `chelis manifest --check` exits 0/1 for CI gating. Finance product feature for model validation teams. Full design: `chelis_reproducibility_manifests.md`.

**Executable properties as spec (trust stack Level 2).** Properties are first-class Chelis functions annotated with `@property`. They define what "correct" means for the implementation they accompany. `chelis prove` discovers properties, generates type-directed random inputs, and verifies each property holds. Scalar interval/order guards additionally direct the generator into their declared machine-representable domain, with strict spacing computed at the binder dtype; unsupported, inconsistent, or unrepresentable scalar guard shapes fail closed, and machine records disclose accepted/attempted/rejected counts. Three categories: domain invariants (output bounds, conservation laws), spec correspondence (optimized impl matches simple reference impl), and behavioral constraints (monotonicity, continuity, symmetry). Properties are the primary artifact the customer interacts with for verification of AI-generated code. Generated code is not reviewed directly — properties are reviewed, and the toolchain enforces agreement. Full design: `chelis_trust_stack.md`.

**Canonical domain properties ship with domain shells.** Every domain shell includes a `properties/` directory containing reference `@property` functions for the domain's standard invariants. These are onboarding templates, credibility artifacts, and documentation-by-example. They are co-located with the implementation code they verify, NOT packaged as separate shells. `chelis prove src/` runs all properties against the shell's exports. Convention applies to Shoals (finance invariants — put-call parity, delta/gamma bounds, Monte Carlo convergence, no-arbitrage), Octant (round-trip and provenance invariants), and any future vertical shell. A standalone "properties" package with no implementation is an empty vessel; the convention exists so no future agent creates one.

**Canonical references and properties co-located with domain shells.** Every domain shell that targets standard, well-defined models ships two co-located artifact directories: `references/` (simple, obviously-correct reference implementations) and `properties/` (invariants and `matches_reference` checks). These are not separate packages. Customers verify their own (or AI-generated) optimized implementations against the shell's references via `chelis prove`. Customers write their own references only for proprietary models. Convention applies to Shoals (finance), Octant (LaTeX bridge), and any future vertical shell. Full design: `chelis_reference_implementations_spec.md`.

**Effect-polymorphic test handlers.** Standardized pattern for replacing effects with test doubles: `with seed(n)` for Random (already used), `with_deterministic_random(sequence)` for exact output testing, `with_mock_io(trace)` for IO, `with_cpu_fallback` for Resource(GPU). The effect system guarantees substitution safety. Library functions in `Std.Test`, documented in SKILL.md.

**Lazy list fusion (future compiler optimization).** The tensor DAG fuses elementwise tensor operations. The host lane (lists, strings) is eager and creates intermediate allocations for chained `map`/`filter`/`fold`. A future compiler pass could fuse host-lane list operation chains into single-pass traversals, eliminating intermediates. Same principle as tensor fusion, applied to the host lane. Low priority — becomes relevant when profiling shows list allocation as a bottleneck in Coral string columns or Hull AST processing.

**Chelis-native testing as the default.** All reef package tests are written in Chelis and run via `chelis test`, except for cross-language parity tests (comparing Chelis output against an external oracle) which use Python. This is a hard rule, not a guideline. Python test infrastructure exists only for parity verification against external libraries: scipy/numpy for Nautilus, pandas for Coral, sympy/latex2sympy2 for Octant LaTeX parsing correctness, QuantLib for Shoals if needed. `Std.Test` provides assertion functions (`assert_eq`, `assert_close`, `assert_close_tensor`, `assert_true`, `assert_false`, `fail`); `chelis test` discovers `tests/*.ch` files and runs them via the evaluator — no C compiler, no linking, no runtime library required. The `Test` effect (or runtime builtin) tracks assertion pass/fail. Reef package layout: `tests/` for Chelis-native tests, `parity/` for Python oracle comparison scripts. Full design: `chelis_native_testing_plan.md`.

**SIMD support (four-level plan, future).** Level 1: `restrict` + `const` + alignment + pragmas in generated C (leverages linearity — the type system proves no aliasing, justifying `restrict`). Level 2: hand-written SIMD reductions in the runtime (sum/max/min/argmax/argmin, AVX2 + NEON). Level 3: vectorized math library integration (Sleef on Linux, Accelerate vForce on macOS) for SIMD-width math in fused kernels — highest impact item, targeted before OOPSLA benchmarks. Level 4: full SIMD-width-aware codegen (only if Levels 1-3 leave gaps). Full design: `chelis_simd_plan.md`.

---

## 6. CLI Surface

The planned user-facing command set is:

```text
chelis build app.ch
chelis build app.ch --target hip
chelis build app.ch --target metal
chelis check app.ch                         # fitness report (JSON) with per-property components
chelis deep app.ch
chelis deep --flat app.ch
chelis surf program.dp
chelis eval expr
chelis manifest app.ch                      # generate reproducibility manifest (JSON)
chelis manifest app.ch --check              # exit 0 if reproducible, exit 1 if not
chelis tide
chelis tide serve --port 8080
chelis tide mcp
chelis tide lsp
chelis cove
chelis fmt app.ch
chelis fmt app.dp --check
chelis lint                              # naming/style rules per spec/01-nomenclature.md
chelis lint --check                      # exit non-zero on any violation (CI mode)
chelis reef init demo --module-prefix Demo
chelis reef build
chelis reef publish
chelis validate --surf app.ch
chelis validate --deep app.dp
chelis validate --desugar app.ch
chelis test tests/                    # discover and run Chelis-native test files
chelis test tests/foo.ch              # run a specific test file
chelis test tests/ --filter erf     # run only tests matching "erf"
chelis test tests/ --timeout 10       # per-test wall-clock timeout (seconds, default 30)
chelis test tests/ --batch-mode auto  # default suite batching for eligible files
chelis test tests/ --batch-mode file  # force per-file worker isolation
chelis test tests/ --jobs auto        # worker concurrency cap; `--jobs 1` preserves serial file mode
chelis test tests/ --json             # emit newline-delimited JSON records instead of plain text
chelis prove                           # discover properties in current package, run all
chelis prove src/                      # explicit path
chelis prove src/pricer.ch             # discover @property annotations, test on random inputs
chelis prove src/ --samples 1000       # control sample count (default 100)
chelis prove src/ --only delta      # filter to properties matching "delta"
chelis prove src/ --seed 42            # reproducible property run
chelis prove src/ --json               # machine-readable output for CI integration
```

`chelis reef build` is byte-reproducible for unchanged inputs. Its source
archive uses lexical UTF-8 member order, normalized regular-file metadata
(`0644`, uid/gid `0`), and `SOURCE_DATE_EPOCH` as the member mtime (Unix epoch
`0` when absent); malformed epoch values fail the build. The CHB embeds the
canonical archive SHA-256. See `spec/design/reef_distribution.md` for the
complete artifact contract.

`chelis manifest` and `chelis prove` are demo-blocking for the first commercial CProof prospect. Full CLI surface and JSON schemas: `chelis_manifest_spec.md`, `chelis_property_spec.md`.

This is the intended stable surface for project-level documentation.
`chelis deep` defaults to canonical pretty Deep; `--flat` is the explicit flat-output
escape hatch.

### 6.1 Style gate

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` invoke `chelis fmt --check` plus `chelis lint --check`
on the input file before the front-end pipeline runs. Style failures
fail the command and emit one diagnostic per issue on stderr. The
flag `--allow-style-violations` bypasses the gate (with a stderr
warning) for emergency builds; the env var
`CHELIS_STYLE_GATE_DISABLE=1` does the same process-wide and is
reserved for the integration-test corpus. Production CI must use
neither.

User-facing CLI documentation lives at `docs/book/src/cli.md`; the
gate's behavior contract is locked in `crates/chelis-cli/tests/style_gate.rs`.

## 6a. Surf Style

Project-facing Surf should read like human-written model code, not typed Deep debug output.

- prefer `def ... -> T = ...` for typed function definitions
- put input types on parameters instead of top-level load-style bindings
- use symbolic dimensions for runtime-varying axes such as `batch` and `seq`
- keep fixed architecture dimensions concrete
- avoid redundant intermediate type ascriptions when inference already determines the type
- prefer meaningful intermediate names over mechanically naming every primitive step

Planned public-style target for Phase 3:

- use block bindings such as `x = expr`; Surf no longer has a separate `let` surface
- prefer pipe-first composition for eligible linear flows
- use multiline pipes for long or many-stage chains, breaking after `=` and before every
  `|>` when the flat form exceeds the width budget or the chain becomes visually dense
- apply the same flat-first, width-threshold philosophy in Surf that Deep already uses
  for pretty printing

---

## 7. Type System Scope

### In scope for v1 / Phase 0

- ADTs with exhaustive pattern matching
- Hindley-Milner inference
- numeric precision tracking
- named tensor dimensions
- graded fitness scoring with repair suggestions

### Phase 2a shipped subset

- effect annotations on Surf `sig` / `def` and Deep `t-fn` metadata
- checked-program upgrade: downstream passes consume annotated Deep with type metadata
- algebraic-effect boundary handling for `Random` and `Resource(Device)`
- `with seed(...)` for seeded stochastic regions and `with device(...)` for resource regions

### Phase 2 surface beyond the initial 2a subset

Shipped after the initial 2a subset:

- linear types for tensors with borrowing rules and explicit `copy` (now the implicit
  copy/drop linearity model — see `spec/design/implicit_linearity.md`)
- lightweight uniqueness / alias tracking before any full heavy ownership-and-lifetimes model
- macro expansion before all LLM-facing operations, with provenance in metadata
- algebraic-effect, linearity, macro, and `vmap` tooling
- constrained name-preserving rank polymorphism (Tier-2 / Tier-3, chelis#339) — see
  `spec/04-type-system.md` §4.5.3 and `spec/design/rank_polymorphism.md`

Designed but not yet shipped (the checker implements a bounded subset; see
`spec/04-type-system.md` §7.1):

- broader effect inference/checking beyond the initial `Random` / `Resource(Device)` subset
- `Diff` as a fully specified effect surface (it remains a compiler capability today)
- `Accum` as a user-visible effect surface (it remains internal-only today)

### Phase 3 shipped foundations and remaining shell work

Phase 3 is the language-completeness phase rather than the research-extension phase.
The practical compiler/runtime foundations that now ship are:

- first-class scalar `Int` / `Float` / `Bool` values outside tensors
- first-class immutable `String` values with practical non-tensor operations
- collection types such as `List[T]` and `Dict[K, V]`
- functional iteration over variable-length host-side data
- core numeric primitives such as `einsum`, `concat` / `split`, `gather` / `scatter`,
  `where`, `cumsum`, `sort`, `diagonal` / `trace`, and `clamp`
- a Rust runtime rewrite that replaces the old C runtime implementation and cleans up
  the compiled host-value ABI before more host/library work lands
- data-loading and tokenization support that removes the mandatory Python
  preprocessing step
- standard-library host modules such as `Std.Time` and `Std.Decimal`

The remaining active Phase 3 work is shell ecosystem and test-surface work: Coral,
Shoals, Octant Part A/B, Chelis-native testing, and the final SKILL.md v2 refresh.

### Deferred to Phase 5+

- sparse tensors
- complex numbers
- distribution types
- equivariance constraints
- optimization-property annotations
- full ILP/AUTOMAP-style rank-polymorphism and related research type features (the
  constrained name-preserving Tier-2 / Tier-3 rank polymorphism has already shipped —
  see `spec/04-type-system.md` §4.5.3; only the general research version is deferred)
- Lean mechanized formalization of the core type system

---

## 8. Computational Model

Chelis lowers typed programs to a RISC DAG built from a small set of primitive tensor
operations.
High-level operations such as `matmul`, `softmax`, and `relu` are library-facing names
that lower into primitive compositions during compilation.

Phase `3h` added the practical tensor surface real model code expects: `einsum`,
`concat` / `split`, `gather` / `scatter`, `where`, `cumsum`, `sort`, `diagonal` /
`trace`, and `clamp`. `School.Nn.Embedding` (moved to the `school` library in chelis-std
0.4.0) remains the explicit public shell/library surface over `gather`.

Core transforms remain first-class:

- `grad(f)` for reverse-mode AD
- `vmap(f)` for vectorization
- `jit(f)` as a future compilation boundary marker
- the shipped executable 2d subset supports direct `vmap(f)(args...)` and
  `vmap(grad(f))(args...)` applications for named defs and inline lambdas, lowering the
  transform away before ordinary DAG codegen
- the shipped executable `vmap(grad(f))` path now also supports flat tuple-valued
  gradient payloads from multi-parameter `grad(..., wrt=(...))`
- first-class stored/returned transformed function values remain out of the executable
  path for now

Implementation-surface note:

- `min_elem` is part of the specified derived built-in surface.
- `block` is part of the Deep syntax vocabulary and Surf block desugaring.
- `normalize` currently exists in the type checker built-in environment, but it is not
  yet a stable specified/lowered built-in.
  Treat it as provisional implementation surface until `spec/05` and the IR lowering
  are aligned.

The language is built around programs-as-data, but mutation and evolution operators are
left to user space rather than embedded as special language primitives.
Programs-as-data operations (`quote`, `unquote`, `splice`) work on expanded Deep.
Macros have already been resolved — the AST an agent inspects or transforms contains
only base tags.

`vmap` remains a core compiler transform in the same sense as `grad`: a DAG rewrite
whose semantics compose with the rest of the lowering pipeline rather than a library
helper.

Current 2d performance boundary:

- batched `matmul` is correct on both backends and specializes to runtime-sized BLAS when
  the operands have contiguous trailing matrix slices
- the C backend emits a host loop over `cblas_sgemm` for batched matmul; HIP defaults to
  `hipblasSgemmStridedBatched` on uniformly strided batched layouts (Perf-F1, shipped)
  and retains the per-batch hipBLAS helper loop only as a fallback for broadcasted
  leading axes or otherwise non-uniform leading strides

---

## 8.5. Scope Boundaries

The cut line between core language and library is whether the compiler needs to know
about it.

### Core (ships with the compiler)

Everything the compiler has special knowledge of.
The ~12 Tier 1 RISC primitives (`add`, `mul`, `exp`, etc.) are language-native because
the compiler decomposes them, the AD engine has adjoint rules for them, and the C/HIP
backends emit specialized code for them.
The Tier 2 derived built-ins (`relu`, `sigmoid`, `softmax`, `matmul`, `layer_norm`,
`conv2d`) are in the core because the compiler recognizes them by name and decomposes
them to RISC primitives during IR lowering.
The type checker knows their signatures.
The optimizer can fuse them.
In the shipped compiler, ordinary user-space functions cannot teach the AD engine a new
adjoint. Backend specialization is narrower but no longer strictly intraprocedural:
the C backend can derive BLAS-equivalent summaries for simple pure tensor helpers and
wrappers, so user-defined `matmul` helpers hit the BLAS path without relying on native
LTO. This is not a general user annotation mechanism and does not yet cover arbitrary
library abstractions, gather summaries, or HIP callsite summary emission; the active
scope is tracked in [`cross_function_specialization.md`](cross_function_specialization.md).

The core transforms (`grad`, `vmap`, `jit`) are also compiler-intrinsic for the same
reason: they require compiler cooperation to implement.

Compiler options such as clang LTO can reduce some helper-call overhead after C code has
already been emitted, but they are only a workaround for this limitation.
They are not the Chelis codegen story for BLAS/cuDNN-equivalent user abstractions,
because backend dispatch must be selected before generated C/HIP reaches the native
compiler.

This is roughly the scope of PyTorch's `torch` namespace — the fundamental tensor
operations, basic neural network layers, loss functions, and optimizers that are
implemented in C++/CUDA underneath.
In Chelis, they are implemented as compiler-recognized patterns that lower to RISC DAG
subgraphs.

### Standard library (`Std.*`)

Ships with Chelis but is implemented in Chelis itself.
The compiler does not know these names.
They ship as Shells (Chelis packages) in the `Std` namespace.

Expected contents:

- common initializers (Xavier, Kaiming, normal, uniform)
- data loading utilities
- tokenizer utilities
- basic I/O (tensor serialization, checkpoint save/load)
- time/date helpers (`Std.Time`)
- exact-decimal helpers (`Std.Decimal`)

The neural-network building blocks (such as `School.Nn.Embedding`), optimizers beyond
SGD (Adam, AdamW, LAMB), learning rate schedulers, metric computation (accuracy, F1,
AUC), and loss functions (cross-entropy, KL, BCE, focal, hinge) moved to the `school`
library as of chelis-std 0.4.0 (`School.Nn.*`, `School.Optim`, `School.Schedule`,
`School.Loss.*`).

### External libraries

Anything that expresses an opinion about model architecture, training methodology, or
domain.
These are Chelis programs that depend on the core and standard library but add domain
knowledge the compiler does not need.

Examples by analogy:

- **scikit-learn equivalent** (`chelis-ml`): classical ML algorithms, preprocessing
  pipelines — compositions of tensor ops with specific algorithmic structure
- **HuggingFace Transformers equivalent** (`chelis-transformers`): pre-built
  architectures (GPT, BERT, LLaMA, ViT) and pretrained weight loaders
- **torchvision/torchaudio equivalents**: domain-specific dataset loaders, augmentation
  pipelines, and model architectures
- **Probabilistic modeling** (`chelis-diffusion`): denoising schedules, noise prediction
  architectures, sampling algorithms

### The decision principle

If removing it would make the compiler produce worse code (cannot optimize, cannot
differentiate, cannot fuse), it belongs in the core.
If removing it just means the user has to write it themselves from the primitives, it
belongs in a library.

### Grey area: shipped Phase 2a effects vs later extensibility

The shipped Phase 2a surface is intentionally closed and compiler-known:
`Random`, `Accum`, `IO`, and `Resource(Device)` live in the type layer. `Random` and
`Resource(Device)` are the Phase 2a boundary-checked effects; `IO` is the shipped Phase
3 host-side debugging/logging effect.
This is narrower than the longer-term design space.
User-defined effects remain deferred; the current compiler knows both the effect
mechanism and the concrete built-in effect vocabulary it ships.

### Phase 3 practical note

Everything currently lives in the core repo.
The shipped Phase 3 foundations already cover public Surf style, Reef packaging, and
Python interop. The remaining Phase 3 work is not more ecosystem polish or research
prestige work; it is the language-completeness gap between "tensor programs compile"
and "a full AI workflow can run in pure Chelis."

As part of that practical gap, Phase `3m` rewrites the runtime in Rust and cleans up
the host-value ABI: `chelis_tensor` stays layout-visible for generated numeric code,
while strings, collections, and other host values move to opaque runtime-managed
handles with accessors and explicit ownership.

That means the next practical surfaces are:

- scalar/string programming
- collections and iteration
- core numeric primitives beyond the original minimal tensor surface
- the Rust runtime rewrite that moves host-value runtime work out of
  `chelis_runtime.c`
- file/config/data loading
- tokenization and batching
- standard-library time and exact-decimal support

Broader hosted registry work, research type features, and Lean formalization remain
later work.

---

## 9. Backend Strategy

### Phase 0

Portable C code generation with BLAS and OpenMP.
This is the reference backend and numerical oracle for future backends.
Generated programs include `chelis_runtime.h`; the runtime implementation behind that
header is now expected to ship as a Rust static library rather than a hand-maintained C
implementation file.

### Phase 1

Futhark-style GPU compilation using C host code plus embedded **HIP** kernel strings,
compiled with `hiprtc` at runtime.
There is no separate CUDA backend plan.
HIP is the single GPU code generation path.

### Later

StableHLO, FX, and Triton are additive integration layers for TPU and PyTorch/NVIDIA
ecosystem access. They do not replace the C/HIP story. The Python interop stack now
includes CPU-only
PyTorch DLPack plus PyO3 compiler bindings from `bindings/python`, and `3b-ii` adds
direct execution via `compile_and_load` / `load` plus the NumPy DLPack guarantee. The
JAX DLPack guarantee remains deferred to the StableHLO phase.

### Rejected

- no OpenCL-first backend plan
- no separate JIT backend on the roadmap
- no Cranelift adoption unless measured latency proves that cheaper options fail

---

## 10. Interactive Execution

Tide and `chelis eval` use the IR evaluator in `chelis-ir/src/eval.rs` as the default
interactive execution path.
Interactive execution does **not** compile through C by default.

If latency later becomes a real bottleneck, the escalation order is:

1. IR evaluator
2. cached C artifacts
3. persistent compiler helper process
4. JIT only if the first three fail on measured workloads

This is a performance policy, not an open design question.

---

## 11. Compiler Implementation

Chelis is implemented as a Rust workspace with six primary crates:

- `chelis-deep`
- `chelis-surf`
- `chelis-types`
- `chelis-ir`
- `chelis-backend-c`
- `chelis-cli`

Compiler infrastructure is intentionally hand-written where it matters:

- hand-written lexers
- hand-written Pratt / recursive-descent parsing
- custom DAG
- explicit optimization passes

Settled library decisions:

### Adopted or planned

| Library | Decision |
|---|---|
| `egg` | Prototype for Phase 1 fusion only; adopt only if fusion search is genuinely combinatorial |
| `salsa` | Planned for Phase 2 incremental compilation use cases |
| `ariadne` / `miette` | Evaluate later for diagnostics UX |

### Rejected for now

| Library | Reason |
|---|---|
| `logos` | Existing lexer works; hybrid handling would erase the declarative win |
| `chumsky` | Existing Pratt parser works; rewrite cost is too high for current value |
| `petgraph` | Custom DAG is smaller and better matched to compiler needs |
| `cranelift` | Not planned; IR evaluator and cached C paths are cheaper interactive options |

### Architectural discipline

To keep a future `salsa` migration cheap:

- each compilation stage should remain a pure function
- no global mutable compiler state
- public crate APIs should take inputs and return outputs

---

## 12. AI Coding Assistance

Chelis has two explicit first-party tracks for AI code generation.

### Track 1: SKILL.md + Frontier Models

Chelis ships a first-party `SKILL.md` for frontier models operating through the Tide MCP
server.
This is the Phase 2 coding-assistance story: compiler-in-the-loop generation, no local
training requirement, and immediate usefulness for agents that already have strong
general coding ability.
The full-surface `SKILL.md` v2 refresh belongs to late Phase 3, after the shipped
public Surf idiom and the remaining language-completeness surfaces are stabilized:
pipe-first chains, short-form block bindings, scalar/string code, collections,
iteration, core numeric primitives, tokenization/data-loading workflows, and the later
`Std.Time` / `Std.Decimal` host-program surfaces.

Current validation result:

- the checked-in SKILL workflow validated at **9/10** tasks against the compiler on a
  local Qwen 35B MoE setup
- the remaining miss was a Deep repair execution failure, not a language-design or
  skill-content failure

This means the skill file is real project infrastructure, not aspirational promptware.

### Track 2: Local Model Ships With Toolchain

Chelis also requires a local coding model as a Phase 4 deliverable.
This is not optional and not a speculative fallback.

Target shape:

- a local 4B-8B-class coding model
- quantized GGUF artifacts that run on consumer hardware
- integration with `chelis cove --assist` and related local workflows

Training pipeline:

1. **SSD for distributional shaping**
2. **Trajectory collection with compiler feedback**
   Compute `nesting_depth × operation_count` as a complexity proxy for each generated
   program and log it.
   Stratify by complexity band post-collection; let the ICL prerequisite measurement
   determine the effective band rather than pre-committing.
3. **Fine-tune, method chosen empirically**
4. **Quantize and ship as GGUF**

SSD matters because it is the cheap bridge between "model has never seen Deep" and
"model can emit something the compiler can score."
It uses the model's own outputs and directly targets the structural-validity gap seen
in SKILL evaluation, where smaller or local models may reason correctly about Deep yet
still fail to emit the canonical form.

Step 3 starts with LoRA as the default.
If forgetting is measured, switch to SDFT instead.
RLVR remains available as an optional final polish step if the quality bar still is not
met.
Anti-forgetting is a hard constraint: the model must preserve PyTorch/JAX semantic
knowledge through training.
Before any training, measure the ICL effect by running the SKILL evaluation with and
without the spec in context.
That quick measurement gates whether distillation-style methods are worth trying at
all.

The local model generates and is trained on expanded Deep exclusively.
Macro invocations never appear in training data, generation targets, or compiler
feedback sent to models.

Product framing:

- a language for AIs that does not include an AI is an incomplete product
- Track 1 is the frontier-model path
- Track 2 is the shipped local-model path

**ChelisBench:** A 50-task ML programming benchmark comparing LLM code generation in
Chelis vs PyTorch on equivalent tasks. Measures whether a language designed for LLMs
produces better ML code than the standard approach. Serves double duty as a measurement
tool and a trajectory source for model training.

**Type-driven property testing:** the shipped surface is `chelis prove` and the
`chelis_prove` MCP tool, which run property checks over Surf/Deep inputs using the type
information the compiler already has — tensor shapes, dtypes, and dimension constraints —
to verify properties such as output shapes, determinism (for pure functions), and
gradient correctness (for differentiable functions). There is no `POST /test` HTTP
endpoint, and this is distinct from `chelis test`, the Chelis-native runner that
discovers and executes `tests/*.ch` files. The shipped Tide HTTP/MCP surface is
enumerated in §6 and [`spec/09-tide.md`](../09-tide.md). The aspiration is full
type-signature-as-test-specification generation; the shipped `chelis prove` is the
current step toward it.

### Editing Surface

Tracks 1 and 2 above cover code *generation*. A separate structural
*editing* surface modifies existing Chelis source through Deep-AST operations
rather than text patches. The shipped Tide MCP and HTTP editing tools are
`chelis_replace_function_body`, `chelis_add_function`,
`chelis_deep_outline`, `chelis_deep_references`,
`chelis_deep_call_graph`, `chelis_replace_function`,
`chelis_add_property`, `chelis_rename`, and
`chelis_change_signature`. They accept Deep strings at the public boundary,
perform structured Deep AST queries/edits internally, and edit tools return
canonical Deep only after the rewritten whole module passes the compiler-owned
validation pipeline. Any parse, edit-shape, name-resolution,
cascade-completeness, preimage, type, effect, or linearity failure returns the
normal structured failure envelope and no edit result.

Detailed design and the current oracle live in
`spec/design/chelis_agent_editing_surface.md`. Future structural-edit
extensions must name an executable oracle in the same change set as any
implementation claim.

---

## 13. Documentation Hierarchy

Use the docs in this order:

1. `spec/design/chelis_canonical_reference.md` for project-level truth, current
   status, naming, ecosystem boundaries, and cross-doc alignment
2. numbered specs `spec/00-12*.md` for active language, CLI, serialization, backend,
   and roadmap contracts
3. `spec/design/chelis_project_plan.md` and active phase plans such as
   `spec/design/chelis_phase3_plan.md` for phased execution and explicit acceptance
   oracles
4. focused active design docs under `spec/design/` for current implementation
   contracts not yet folded into numbered specs
5. `README.md` and `docs/book/` for repository orientation and developer-facing usage
   docs; they should follow the hierarchy above, not redefine it

Historical design notes belong under `spec/design/archive/` and must be treated as
rationale, not current guidance. If an active doc and an archived note disagree, update
or cross-reference the active doc rather than adding a third explanation.
