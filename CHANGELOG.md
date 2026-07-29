# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Locked Nix packages expose the Chelis compiler, C runtime, and `chelisup`.**
  The root flake supports `x86_64-linux` and `aarch64-darwin` with native checks.
  Pinned `crate2nix` crate derivations share dependency outputs across the product packages.
  It also provides `chelis` and `chelisup` applications for `nix run`.
  The Nix `chelisup` launcher roots its closure before it installs release shims.
  Nix remains an additive source-build channel. `chelisup` still routes release toolchains.
- **A tracked Devenv shell supplies the contributor toolchain on Linux and macOS.**
  The shell pins Rust, Python 3.11, test tools, and the platform C toolchain.
  Linux uses GCC, OpenBLAS, and Valgrind from Nixpkgs. macOS maps `gcc` and
  `g++` to the Nixpkgs clang wrapper from `pkgs.stdenv.cc`.
  `devenv test` checks the tool versions and compiler warnings.

### Fixed

- **Compiled list combinators grow accumulators in place (part of chelis#943).**
  The C backend now gives `map`, `filter`, `scan`, `partition`, and `flat_map`
  exclusive, pre-sized accumulators instead of rebuilding a list for every
  element. Runtime guards preserve immutable-list semantics and reject invalid
  capacities, shared destinations, and self-extension.

## [0.18.1] — 2026-08-01

### Added

- **Sound structural induction for supported recursive integer models
  (chelis#978).** `chelis prove` now dispatches eligible structurally recursive
  properties through explicit base and step obligations. Unsupported recursion
  remains fail-closed with no sampling or `ASSUMED` fallback. The default
  `--tier auto` selects this terminal lane before ordinary SMT/fuzz dispatch
  whenever the checked property reaches a recursive model, and Tide exposes
  the same honest result. The executable `induction_bond.ch` example and its
  three-runtime parity test demonstrate a general-size bond recurrence.
- **Target-aware root realizability manifests (chelis#912).** Root
  observations now carry explicit target/lane realizability, roots are always
  labelled at the observation boundary, and `chelis eval --target` exposes the
  target choice for cross-lane comparison.

### Changed

- **Invalid Deep AST states are unrepresentable.** Closed tag vocabulary and
  typed construction replace stringly invalid tag states, with exhaustive
  consumers kept synchronized across the compiler.

### Fixed

- **Root-boundary lowering covers concat and previously dropped realizable
  forms (chelis#820).** Realizability is inferred transitively and checked
  independently from effects before backend lowering.

## [0.18.0] — 2026-07-31

### Added

- **`compile_and_load` executes f64 elementwise programs end to end
  (chelis#919, chelis#920).** `emit_fused_elem` is parameterized on the
  IR-pinned dtype instead of being f32-hardcoded behind a `panic!`, so
  `sigmoid`, `tanh`, `gelu`, and composite elementwise expressions such as
  `exp(x) * x` lower and run at f64: data pointers, step variables, and the
  libm symbols all follow the dtype, with an explicit reinterpreting cast
  where the emitted C indexes `chelis_runtime.h`'s `float *data`. The Sleef
  AVX2 path stays f32-only, so f64 chains take the scalar OMP SIMD loop.
  f32 emission is byte-identical.

  `supported_execution_dtypes(target)` is now the single admit list every
  dtype-dependent site in `chelis-python` derives from. The C target admits
  f32 and f64; HIP stays f32-only and rejects explicitly, because its
  kernels remain f32-hardcoded. Reduced floats and integers reject through
  the `Result` channel as branded `unsupported:` diagnostics instead of
  crossing the pyo3 boundary as a `PanicException`.

  `emit_fused_reduce` is not widened: an f64 fused reduction, reachable
  from ordinary Surf as `sum(exp(x), 0)`, still rejects as a diagnostic.
  That surviving half is tracked as chelis#951.

### Changed

- **`chelis-types` type inference is split into modules.** The single
  25k-line `crates/chelis-types/src/infer.rs` becomes an `infer/` module
  tree (declarations, expressions, application, shape, and annotation
  lanes) with `infer/mod.rs` preserving the existing public paths.
  Behavior is unchanged; `ARCHITECTURE.md` records the new layout.

### Fixed

- **A compiled kernel no longer writes through its caller's buffer
  (chelis#933).** Fused in-place reuse treated `reusable_input` — a
  linearity fact about a value being dead after the op — as permission to
  write through storage the caller owns. Calling a compiled model
  overwrote its own NumPy argument, made repeat calls on one array
  non-idempotent, and wrote to arrays marked `writeable=False` without
  error. `MemoryPlan::borrows_caller_storage` now reports whether a node's
  bytes are the caller's, following metadata-view sources to their root so
  a `reshape` of an input is still recognized as a window onto the same
  bytes, and `fused_in_place_spec` refuses in-place reuse when they are.
  Chains reading an owned intermediate keep the optimization. The fix
  lands in the backend, so `chelis build` whole programs are covered on
  the same terms as the bindings, and both dtype lanes at one site.
- **The compiled output's dtype comes from the runtime tag (chelis#920).**
  `NativeTensor::dtype` returned the literal `"float32"` and the DLPack
  capsule hardcoded `bits: 32`. Both were accidentally correct only while
  the artifact gate rejected every non-f32 artifact one layer earlier;
  widening it by one dtype turned them into silent corruption, with a
  numerically correct f64 buffer described as float32 and read at a 4-byte
  stride. `spec_dtype_mapping`, `numpy_dtype_name`, and `dlpack_bits` no
  longer have a default arm, and decode through `RuntimeDType::decode_id`
  rather than dispatching on a raw `i32`.
- **Unloading an executed compiled artifact no longer aborts the process
  (chelis#963).** On Linux, dropping a `NativeCompiledModel` whose kernel
  had run took the interpreter down with SIGSEGV after returning correct
  results: emitted elementwise loops carry `#pragma omp parallel for simd`,
  and executing one starts libgomp's thread pool, which registers
  thread-local destructors pointing into the artifact's code segment that
  `dlclose` then unmaps. Compiled artifacts are opened with
  `RTLD_NODELETE`, so the drop is a refcount decrement and the mapping
  stays resident; `RTLD_NOW` makes an unresolved symbol a load error
  rather than a crash on first call.
- **Pure host-lane arrow-form value roots surface in `chelis eval --file
  --json` (chelis#947).** An arrow-form value def `def name -> T = body`
  desugars to a nullary thunk, which the host runtime's eager
  value-binding order skips while still listing it as a display root, so
  such roots reported `{"roots":[]}`. A surfacing pass applies the
  zero-arg host-lane thunk and records its value. The pass is gated to
  effect-free roots read from the checker's effect annotation, so no
  effect runs at display time; already-bound roots keep their existing
  binding, tensor-lane roots continue to surface through the DAG, and a
  failed application stays unsurfaced.
- **A reserved word heading a value binding is named in the diagnostic
  (chelis#915).** `sig = 0.2f64` failed with `expected identifier, found
  Eq at byte 4`, naming neither the reserved word nor its position:
  `sig` is a valid declaration head, so the parser consumed it and failed
  one token later at the `=`. A shared guard at the three declaration
  heads that are plausible value names (`sig`, `type`, `dim`) keys on the
  next token being `=` and points the offset at the keyword.
  Diagnostic-only: no grammar change and no token reclassification, so
  every program that parsed before still parses.

## [0.17.5] — 2026-07-31

### Added

- **`chelis-vocab` defines a closed vocabulary for the physical representations of runtime elements.**
  `RuntimeDType::repr()` maps each dtype to one `Repr` value.
  `RuntimeDType::byte_width()` derives its value from that representation.
  The current bool representation remains a four-byte binary32 payload.
  `chelis-vocab` now uses `no_std` and declares no dependencies.
  The crate uses no allocation or unsafe code.

  An executable C probe compares every valid dtype tag with the Rust width.
  An invalid tag causes an unsuccessful process exit.

### Changed

- **The Rust APIs for effect errors and C-header output changed.**
  `EffectKindDecodeError<'a>` now borrows an unknown symbol from the input.
  `render_runtime_dtype_c_header()` moved from `chelis-vocab` to `chelis-runtime::dtype_header`.
  This change breaks Rust code that imports the old path or names `EffectKindDecodeError` without a lifetime.
  The C ABI values and generated header bytes did not change.

  For source migration:

  - Replace `chelis_vocab::render_runtime_dtype_c_header` with `chelis_runtime::dtype_header::render_runtime_dtype_c_header`.
  - Add an explicit lifetime to each `EffectKindDecodeError` type.

### Fixed

- **Root-scoped eval ignores dead symbolic inputs from unrelated dependency
  modules (chelis#991).** Symbolic-dimension discovery now follows the live
  evaluation slice, so importing a calculation from a package with unrelated
  generic declarations no longer invents external inputs. Live shape-only
  dependencies remain required and continue to fail closed when absent.
- **Risk-guard fuzzing is constraint-directed and non-vacuous (chelis#977).**
  The prover derives candidate inputs from guard constraints, records observed
  in-domain coverage, and preserves evaluator/backend parity instead of
  accepting a property from samples that never enter its guarded domain.
- **The undispatched Tier-D induction scaffold fails closed (chelis#978).**
  Its placeholder base and step helpers no longer return `Proved` with
  `ASSUMED` evidence; unsupported induction obligations remain unsupported.
  A sound dispatched general-induction lane remains open under chelis#978.
- **Quantile monotonicity contracts bind to the real linked Nautilus surface
  (chelis#979).** Tier-B Reef proofs recognize only the dependency-owned linker
  declaration for `Nautilus.Stats.quantile_vec`, preserve compiler AST identity
  for the tensor operand, and couple levels only for calls over the same
  dataset. Spoofs, missing trusted calls, different datasets, and unbridged
  range/boundary contracts fail closed as unsupported.

- **The runtime and generated C now decode native int32 storage through signed 32-bit pointers.**
  Runtime comparison, `where`, scatter-add, `cumsum`, `trace`, `clamp`, and `einsum` now use native `i32` access.
  Generated C uses `int32_t` for int32 element access and keeps `float` for F32 and the current Bool payload.
  Generated C max and min operations compare int32 values without a binary32 conversion.
  This correction preserves the dtype IDs, element widths, storage representation, and public C signatures.

## [0.17.4] — 2026-07-30

### Fixed

- **Bare file diagnostics now drive native expected-failure suites
  (chelis#967).** `chelis test --expect neg|blocked` preserves file-level
  compile and check failures even when a probe declares no `test_*` function,
  so `.expect` substrings can match the diagnostic instead of receiving a
  recordless `config-error`. Mismatches retain the complete diagnostic for
  plain and NDJSON reporting; genuinely clean testless files remain
  fail-closed configuration errors. Ordinary non-`--expect` testless files
  retain their legacy zero-record behavior. Exact regressions lock both modes.
- **Layered stdlib checks report the same honest checked-node counters as
  monolithic checks (chelis#973).** Cached checked programs now retain the
  inference product's `typed_nodes` and `total_nodes`; layered fitness reports
  add those counters across their partition instead of substituting a
  structural AST count.
- **Package-owned source-file checks and builds repair a missing or malformed
  `reef.lock` even on a warm prepared-graph cache hit (chelis#971).** A
  successful `chelis check <package-source>` or `chelis build
  <package-source>` can no longer omit the resolved lockfile merely because
  graph preparation came from cache.
- **Imported rank-generic functions instantiate independently at every call
  site (chelis#968).** Environment free-variable analysis now protects a
  scheme's quantified type, dimension, and rank variables from unrelated
  global substitutions, including alias chains, so an earlier concrete tensor
  extent cannot leak into a later use.
- **Reef package artifacts are byte-reproducible across unchanged builds
  (chelis#970).** Source archives now use lexical UTF-8 member ordering and
  canonical regular-file metadata (mode `0644`, uid/gid `0`, and
  `SOURCE_DATE_EPOCH` mtime with a fixed zero default). Invalid epoch values
  fail loudly. Repeated builds produce identical `.tar.zst` and `.chb` bytes,
  while the CHB continues to embed the canonical archive SHA-256.
- **Reef now validates complete canonical CHB envelopes (chelis#972).**
  CHB decoding rejects trailing bytes, truncation, noncanonical encodings and
  malformed or ambiguously ordered metadata, including fields installation
  does not otherwise consume. Every Reef install path applies that validation
  before registry mutation and still checks the paired archive against the
  embedded SHA-256. The new read-only
  `chelis reef verify-artifact --shell <CHB> --archive <ARCHIVE>` command
  exposes the same boundary to downstream release gates; `--json` emits a
  JSON-only report with `valid == errors.is_empty()` and matching exit status.
- **Concrete return contexts now specialize nullary generic ADT
  constructors before C ABI projection (chelis#935).** A layout-free
  constructor may retain its named generic ADT term through host-expression
  lowering until the checked enclosing expectation supplies applied type
  arguments. Zero-argument generic constructor wrappers are emitted through
  concrete call-site specialization, so the exact `Box[a]`/`Empty`
  reproducer C-builds without inventing a type. Unconstrained uses still fail
  at the resolved-type boundary, while wrong ADT names and arities continue
  to reject before emission.
- **Generic ADT match patterns now receive their checked concrete application
  at bounded call sites (chelis#936).** Ordinary stored type parameters are
  distinguished from tensor dimensions nested through ADTs (chelis#940), so
  `Hamt[a]` follows ordinary specialization while `Column[n]`/`Frame[n]`
  retain rank specialization. Invoked self- and mutually recursive ordinary
  generics reject before C emission pending memoized monomorphized symbols
  (chelis#941).
- **Host generic classification now consumes checker-owned signature and ADT
  records instead of reconstructing them from authored syntax (chelis#948).**
  Alias-resolved constructor fields survive `CheckedProgram` serialization,
  composition, effects annotation, and lowering. Mixed
  `Frame[n,a] -> Hamt[Column[n,a]] -> tensor[n,a]` programs erase only the
  dimension parameter while specializing the stored dtype, including after
  beta reduction removes an expression-local type stamp.
- **`fold` materializes the checked callback accumulator type onto unresolved
  initializers (chelis#939).** An empty `[]` initializer now resolves when the
  callback declares the accumulator, without defaulting genuinely
  unconstrained lists.
- **Integer bitwise and shift builtins now emit real scalar C expressions
  (chelis#682).** `bitand`, `bitor`, `bitxor`, `shl`, and `shr` have promoted
  eval/C parity locks at `int8`, `int16`, `int32`, and `int64` instead of
  reaching the compiled-lane unsupported boundary. Shift lowering uses
  declared-width unsigned helpers rather than undefined signed C shifts;
  boundary counts and negative-count traps are locked under UBSan.

## [0.17.3] — 2026-07-30

### Fixed

- **`chelis test` now has a hard whole-suite deadline (chelis#927).**
  The public command supervises Reef preparation, batch and file workers,
  output collection, and finalization in a dedicated process group.
  `--suite-timeout` defaults to 600 seconds and is independent of the
  per-test `--timeout`; expiry terminates the suite and all descendants
  without retrying through another execution mode. Plain output reports an
  incomplete-suite failure. NDJSON retains completed test rows, replaces any
  child-produced perfect summary, and ends with an explicit timeout record
  plus `summary.incomplete:true`, so a wedged worker cannot leave CI hanging
  or emit a false green. Flushed auto-batch rows and completed `--expect`
  verdicts are retained on timeout, with a mode-correct incomplete summary.
  Non-Unix targets fail closed before starting the suite because equivalent
  descendant-cleanup guarantees are not yet implemented there. The public
  command cannot be switched into worker mode through an inherited environment
  variable; batch progress uses a dedicated temporary record file, while
  ordinary stderr remains byte-preserving and separate from the protocol.
  Captured-output forwarding is deadline-bounded under consumer backpressure,
  and very large accepted timeout values no longer overflow `Instant`.
  A zero-second suite deadline is rejected. The TERM grace period is contained
  within the advertised deadline. On Unix the supervisor forks the suite
  directly instead of exposing a hidden worker subcommand; an inherited
  lifecycle pipe removes the progress file and kills the suite process group
  if the public supervisor disappears. If the suite leader itself is killed,
  the supervisor kills remaining descendants before collecting output and
  emits an explicit incomplete-suite record instead of hanging on inherited
  output descriptors or returning an empty runner error.

## [0.17.2] — 2026-07-30

### Added

- **`chelis prove --json` reports compiler-owned declaration dependencies
  (chelis#922).** Reef-package summaries now carry a structured
  `dependency_graph` with stable declaration IDs, package/module/source-span
  ownership, and stable-ID edges derived from linker-resolved Surf AST.
  Complete empty analysis is distinct from explicit `unavailable` analysis;
  collisions, shadowing, imports, cycles, unused declarations, dimensions,
  types, constructors, aliases, macros, and invariant references retain their
  compiler identity. The legacy name-only `dependency_edges` field remains
  for one compatibility release.
- **SMT proofs can lower supported scalar `grad` applications
  (chelis#923).** Applied, single-target gradients of pure `f32`/`f64`
  functions with scalar floating inputs now reach cvc5 instead of being
  categorically unsupported. Named and inline functions share the compiler's
  result-type restrictions; unsupported control flow, casts, effects,
  recursion, transforms, intrinsics, and non-floating results fail explicitly
  before any green verdict is emitted.

### Fixed

- **Reef package proofs reuse a race-safe prepared graph and check only the
  reachable declaration closure (chelis#924).** Cache identity covers the
  exact source-root inventory and source bytes, including additions,
  deletions, renames, dependency sources, and escaping symlinks. A stable
  pre/post snapshot prevents caching a graph built from different bytes than
  its determinant, while lookup independently rechecks the live inventory.
  Reachability follows functions, types, dimensions, constructors, patterns,
  macros, and invariants, so unrelated dependency declarations no longer
  dominate package-proof latency without hiding reachable errors.
- **Lint traversal exclusions are structured and configurable (chelis#740).**
  `walker.rs` no longer hard-codes generated, dependency, build, or
  infrastructure directory names. `chelis-lint` now composes a shipped
  baseline with the nearest strict `chelis-lint.toml`; every repository entry
  has a typed class and resolvable spec cross-reference. BurntSushi's `ignore`
  engine performs matching and pruning with ambient `.gitignore`, `.ignore`,
  parent/global Git, Git-exclude, and hidden-file filters disabled, so local
  and CI scope is identical. Policy and spec links are resolved inside the
  policy root; non-file paths and broken or escaping links fail closed instead
  of importing machine-local policy. Explicit targets remain lintable, and all
  rules plus prepared catalogs consume one policy-admitted corpus. Ancillary
  Cargo manifests used by `doc-filename-convention` now come from admitted
  entries or pass a parent-aware policy check. Admitted sibling workspace
  manifests remain visible to subdirectory and explicit-file lint targets,
  including crates exposed through internal directory symlinks, while external
  directory links, excluded crates, and machine-local ancestors above the
  policy root cannot suppress an admitted documentation violation. Discovered symlink
  targets are resolved through the same policy: internal admitted regular-file
  links remain usable, while special files and aliases into excluded or
  machine-local content cannot affect prepared catalogs or ancillary metadata.
  Depth-zero targets override exclusion matching only; special-file roots and
  symlinked directory roots escaping the policy boundary are rejected before
  traversal. Ancillary governance follows the lexical path, so an above-root
  link cannot gain authority by pointing inward. Rule modules are tripwired
  against independent walker and directory-discovery primitives. A crate-local
  canonical `AGENTS.md` plus `CLAUDE.md` symlink locks the rule-registration and
  canonical-traversal protocol for both agent entry points at edit time.
- **Repeated CLI builds can replace staged runtime archives on read-only-artifact
  hosts.** `chelis build` now restores owner-write permission on an existing
  staged `libchelis_runtime.a` before copying the current Cargo artifact. This
  preserves stale-runtime replacement and lets repeated builds target the same
  output directory on macOS, where Rust static libraries are emitted read-only.
- **`opaque-domain-construction` repository linting is linear and honors
  canonical skip filters (chelis#603).** The rule now prepares its Surf
  declaration catalog once per `chelis_lint::lint` invocation from the main
  walker's admitted entries, then reuses that immutable catalog for every
  checked Surf file. It no longer starts an unfiltered `WalkDir` per file or
  descends into `target/`, `.git/`, `.venv*`, dependency trees, agent
  worktrees, and generated opaque-invariant programs. Opaque definitions are
  indexed by type leaf and defining module, so each construction, cast, or
  update lookup is independent of unrelated declarations instead of rescanning
  the catalog. Prepared state remains invocation-local so repeated `--fix`
  passes and reused rule objects observe source edits without stale-cache
  behavior.
- **Rejected explicit lint roots fail loudly.** An explicitly named lint root
  that exists but fails depth-zero admission — a socket or FIFO, a link
  resolving to a different entry kind, an escaping link, or a broken link —
  now fails `chelis lint` and the built-in style gate with the root path and
  rejection reason instead of exiting 0 with no entries, matching the
  nonexistent-root failure. Discovered inadmissible entries below an admitted
  root remain silently omitted. The standalone CLI absolutizes targets without
  resolving symlinks so a link's identity reaches the traversal-policy
  boundary check. BREAKING for callers that relied on a silent empty walk of
  a rejected explicit root.
- **Repository lint policy discovery is cwd-insensitive.** `chelis-lint`
  resolves relative lint targets against the invocation working directory
  before searching ancestors for `chelis-lint.toml`, so
  `cd sub && chelis lint --check .` applies the same repository exclusions as
  the absolute spelling. One traversal policy load is now shared per lint
  invocation across the walker and every rule's `prepare_run` hook.

## [0.17.1] — 2026-07-23

The first cut of the numeric-remediation "loud checking" work: cases that
used to substitute a plausible value or silently skip checking now fail
loudly. Some programs that previously compiled or scored a perfect check are
now rejected at the offending site — see **Changed** for the migration.

### Changed

- **Unsupported cases fail loudly instead of substituting a value**
  (chelis#730 Phase 1). `chelis build` / `chelis eval` now return a branded
  `unsupported:` error where a stage used to emit a plausible default (a
  literal `0`, a dropped operand, an f32 kernel over an int64 buffer, an
  empty window, a discarded effect handler). A program that "worked" by
  relying on one of these silent substitutions now errors at that site.
- **`with seed(...)` requires an int64-suffixed integer literal**
  (chelis#731 Phase 1). `with seed(42)` is rejected; write
  `with seed(42i64)`. The seed width is now explicit in the source
  (spec/02 §P10a); an unsuffixed literal defaults to `int32` and is a type
  error naming the suffix.
- **`with seed` / `with device` bodies are now type-checked** (chelis#709,
  chelis#710). An ill-typed body inside a handler — previously invisible to
  the checker, so `chelis check` reported a perfect score — is caught, and
  the enclosing function's declared return type is enforced. Malformed
  `fn` / `let` / `if` / `app` forms and unknown effect kinds are rejected
  with `MalformedForm` / `UnknownForm` diagnostics rather than a silent
  `Type::Error`.
- **`chelis eval` output is dtype-faithful** (chelis#732 Phase 1). Integers
  print as integers (not `750.0`), `bool` prints `true` / `false`, and floats
  print shortest-round-trip for their own width. The compiled-C lane is
  brought to byte-identical rendering in a later release.

### Fixed

- **Correctly-rounded f32 `sqrt`** in the C backend; the Accelerate `vvsqrtf`
  path is dropped (chelis#719).
- **`uniform_like` is bit-identical across the eval, C, and HIP lanes** — a
  single correctly-rounded FMA, independent of `-ffp-contract` (chelis#770,
  chelis#771, chelis#776). Literal `with seed` values are read at full i64
  width so every lane derives identical seeds. A `uniform_like` whose range
  bounds are wrapped in a non-static expression now resolves them at lowering
  time or fails loudly, never silently defaulting to `[0, 1)` (chelis#776).
- **Labeled roots for `def`-call-valued top-level bindings** in compiled C
  (chelis#750).
- **Nominal typing through tuple projection `.N`** is restored (chelis#707).
- **Top-level `def`s apply past the lowered tensor-root shadow** (chelis#721).
- **Bare, non-tail expression statements are rejected** with a targeted
  diagnostic instead of being silently dropped (chelis#706).
- Shape-computed builtin overrides return `Error` on `Error` operands, and
  sibling arguments past an `Error`-typed argument are still checked
  (chelis#773; conv2d annotation-clobber hotfix).
- **`chelis prove --capabilities` no longer reports `beacon` as dispatchable**
  (chelis#673).
- The stale `WireDag` schema-version references are corrected to the current
  v3 (chelis#701).
- The std init self-test corpus migrates to `seed(Ni64)`, and the build-cache
  oracles skip the host-only `Std.Test` module, now documented as eval-only
  (chelis#796, spec/05 §3.6.1).

### Added

- **`chelis prove` gains Beacon-certified contract discharge** (chelis#831).
  Normal-CDF contract obligations can be proof-discharged by Beacon's
  certified envelope (`DischargeMethod::CertifiedEnvelope`), promoting
  Black-Scholes composites from `ProvenModuloFuzzValidatedContract` to
  `ProvenModuloCertifiedEnvelope`; plus monotonicity relational injection
  into the SMT lowering.
- The **numeric-remediation design set** — five class plans (dtype semantics,
  loud unsupported, checker totality, faithful observation, spec provenance),
  the capability-table schema, and the sequencing roadmap with its
  release-slicing plan (chelis#729–#733, #740; docs and spec atoms), together
  with the Phase-0 detector harnesses (the `Type::Error` census + totality
  invariant, the loud-unsupported census + token tripwire, the observation
  round-trip harness, and the dtype domain-validity checker).
- **[05-OBS-1..5]**, the faithful-observation atoms that define the eval
  rendering contract (chelis#732).
- **[05-RNG-1]**, the per-lane with-seed determinism atom (chelis#735).

## [0.16.1] — 2026-07-12

### Added

- **`grad` through static control flow over ADT/list values (chelis#620).**
  An `if` whose condition const-folds at lowering time now prunes to the
  taken branch, whatever its value type — tensor, tuple, ADT constructor, or
  list — instead of dying in "if then branch expected a single tensor value".
  This makes the eps fail-guard idiom (`if eps <= 0.0 then fail(...) else
  body`) and constructor-valued guards differentiate, in both the eval and
  compiled lanes. Recursive builders whose base case prunes statically
  (`if k >= n then [] else concat([row], recurse)`) unroll under grad,
  bounded by depth caps (512 per callee, 1024 total) with a loud diagnostic
  past the cap; the unrolled `Cons`/`Nil` list value flows through list
  append and tensor `concat` (the same Pad+Add construction as the
  expression-level path). Compiler-inserted linearity `copy`/`drop` over
  tuple and ADT values lower structurally (one node per tensor leaf), fixing
  the "copy input expected a single tensor value, got an ADT value" failure
  on the curried single-argument closure `grad(fn (p) -> loss(p, x, y))`.
  Acceptance oracle: `crates/chelis-cli/tests/issue_620_static_if_adt_grad.rs`.
  Runtime-condition control flow over ADT values remains rejected loudly and
  is tracked in chelis#618 (`RiscOp::Select`).

### Fixed

- **Deep DAGs from the bounded unroll no longer overflow the stack in
  consumer passes (chelis#620 red team).** Lowering's per-level
  `stacker::maybe_grow` protected the unroll itself, but the passes that
  consume the resulting DAG inside the same entry (reverse-mode grad
  construction, DCE, verify) recursed unprotected, so a ~484-level
  combining builder (the cap's own im2col justification) lowered fine
  and then SIGABRT'd BELOW the 512 cap. Every lowering entry now runs on
  a grown 512 MiB stack segment (the chelis-types `with_grown_stack`
  boundary pattern); the whole cap window returns values or the loud cap
  diagnostic, never a crash.
- **The static `if` fold refuses non-finite condition intermediates
  (chelis#620 red team).** The lowered comparison composition evaluates
  NaN opposite to the IEEE comparisons both forward lanes apply
  (chelis#666), so folding e.g. `gte(div(0.0, 0.0), 0.0)` would prune to
  a branch the forward pass never takes. Non-finite intermediates now
  fall to the runtime path: float branches keep the pre-existing mask
  behavior, ADT/list branches get the loud runtime-condition rejection
  instead of a silent wrong-arm gradient.
- **A function parameter named `params` (or any reserved Deep tag that
  is not also a Surf keyword) now
  binds correctly in DAG lowering (chelis#620).** Such names desugar
  through chelis-surf's MetaExpr param wrapper, which the lowering's
  param-name walk silently dropped: the parameter never bound, body
  references lowered to bogus `Load` placeholders, and
  `grad(loss, wrt=(params))(...)` over a struct conventionally named
  `params` failed with a misleading "match on a runtime scrutinee"
  diagnostic (reproduced on 0.16.0).
- **Concat result typing (chelis#631, spec §4.5.4).** The checker types
  `concat(list, axis)` from the concat-axis VALUE and — for statically
  enumerable lists, including `let`-bound list literals — the element
  COUNT: an axis-0 concat of two `[1, m]` rows now types `[2, m]`
  (previously the ELEMENT type with the last axis wildcarded — a wrong
  concrete extent the host-program C lane baked into tensor-helper
  signatures, aborting guarded `if`/`fail` forward binaries at run
  time). Non-enumerable lists honestly wildcard the CONCAT axis, a
  runtime axis wildcards every axis, and an out-of-bounds literal axis
  is a check-time error. Guarded forward programs now have full
  eval-vs-C build-and-run parity
  (`issue_631_guarded_forward_concat_c_parity.rs`). Also fixes the
  misplaced-wildcard symptoms of chelis#594.
- **C-lane runtime guards for consumers of runtime-wildcard extents
  (chelis#664).** Same-shape elementwise ops (`add`/`mul`/`div`/
  `trunc_div`/`floor_div`/`max_elem`/`cmplt`) now emit an operand-shape
  agreement abort when any involved extent is non-static, and the
  reshape numel guard fires for ANY non-static reshape (Sym-resolved
  targets like `[shape(x, 0)]` and literal targets over runtime-sized
  inputs), not only Node-valued targets. Before, a movement-op runtime
  wildcard beside a differently-sized sibling was read out of bounds and
  the binary exited 0 with wrong values while `chelis eval` rejected —
  the silent-divergence class. Pre-existing #616-era gap (reproducible
  through runtime-bounded `shrink`); rank-0-vs-rank-N operands (the
  scalar-broadcast idiom) are exempt by design, and rank-divergent
  operands with both ranks > 0 stay unguarded — tracked as #668. Pins:
  `issue_664_runtime_wildcard_consumer_guards.rs` (five error-parity
  cases plus two no-false-abort twins); fully static codegen is
  byte-identical.
- **Checker movement typing is identity-only for symbolic dims
  (chelis#632).** A non-identity `stride`/`pad` axis (literal step != 1,
  non-zero padding) over a symbolic dim now types a fresh
  runtime-guarded extent instead of passing the input's symbol through —
  the extent genuinely changes (`ceil(d/step)`, `d + lo + hi`), so the
  pass-through was an annotation-level lie that also falsely rejected
  `sig f: tensor[n, f32] -> tensor[u, f32]` over `stride(x, 2)` via the
  §4.4.1 rigidity guard (that program now checks, builds, and has
  eval-vs-C parity: `issue_632_literal_stride_under_sig_symbols_matches_c`).
  Identity axes (stride step 1, zero pad) still pass the symbol through
  (the `issue_513` contract); literal axes keep exact arithmetic.
- **Anonymous dims are no longer dim-substitution keys (chelis#632,
  partial).** `tensor_dim_substitutions` recorded `"*" -> <actual>` and
  repainted every wildcard-typed node in a helper DAG with one symbol,
  conflating distinct runtime extents (a direct-return `shrink ->
  stride` chain under `sig ... -> tensor[u]` aborted its C binary at
  the runtime-dim equality guard while eval computed correctly). The
  declared return now retypes the helper ROOT positionally, on
  op-declarable axes only; the formerly-pinned degenerate has full
  eval-vs-C parity (`issue_632_direct_return_movement_chain_eval_matches_c`).
  The checker-side movement-typing residue of chelis#632 is tracked on
  that issue.
- **`fail`-reaching forward bodies stay in the host lane (chelis#631).**
  The DAG lane lowers `fail` to a mask-selected zero placeholder
  (grad-lane semantics); a forward tensor helper built from a
  fail-reaching body compiled into a binary that returned ZEROS where
  `chelis eval` aborts with the user's message. Such bodies now lower
  through host-lane `if`/`fail` control flow (`chelis_fail`), keeping
  error parity (`issue_631_data_dependent_fail_branch_aborts_in_c`).
  Known residual (chelis#662, pre-existing): the `grad`/`vmap` exemption
  is whole-expression, so a forward `fail` BESIDE a grad call in one
  body still keeps mask semantics in C.
- **Ragged direct-literal concat sums per-element extents (chelis#594,
  spec §4.5.4 rule 1).** A literal list at the concat site now re-infers
  each element and SUMS literal concat-axis extents: `concat([a, b], 0)`
  over `tensor[2, f32]` and `tensor[3, f32]` types `tensor[5, f32]`
  (previously the §4.5.2 join widened the ragged literals to `*` before
  the concat rule could see them). Conscious boundary flip on the direct
  path: a wildcard element now makes the sum honestly unknown instead of
  inheriting the head-biased join times the count — the head bias
  remains observable only through a binding, where just the joined
  element type and the literal length survive (uniform extents only;
  let-bound ragged lists keep the honest wildcard).

### Changed

- A return-only symbolic dim declared on a concat axis that chelis#631
  now types concretely (e.g. `-> tensor[k, 2, f32]` over an axis-0
  concat of two `[1, 2]` rows) is rejected by the §4.4.1 return-dim
  rigidity rule, since `k` pins to the derived literal. Use the concrete
  extent (`tensor[2, 2, f32]`) or an explicit wildcard (`tensor[*, 2,
  f32]`); the symbolic spelling was the pre-#631 workaround for
  chelis#594.
- **Constant-condition `if` now prunes instead of mask-blending
  (chelis#620).** Previously both branches were always lowered and blended
  arithmetically even when the condition was a compile-time constant, so a
  NaN/Inf in the untaken branch leaked through `0 * NaN`, and a non-float
  constant-condition `if` was rejected outright. Pruning lowers only the
  taken branch and admits any branch type. Runtime-condition `if` behavior
  is unchanged (float mask blend; non-tensor branches rejected).
- **Self-recursion in DAG lowering errors loudly instead of silently
  collapsing (chelis#620).** The old Inlining-F1 guard made a recursive call
  fall through to a fallback that returned its last argument — silently
  wrong values in the build lane. Recursion now unrolls when statically
  bounded and otherwise reports the unroll-cap diagnostic naming the callee.

## [0.16.0] — 2026-07-10

### Added

- **Transcendental discharge via certified special-function envelopes
  (chelis#434).** A property whose goal carries a non-whitelisted transcendental
  (`erf`, `normal_cdf`) can now be discharged through a machine-certified
  envelope: the standard-normal CDF lowers to the exact `erf` identity
  (`½(1+erf(x/√2))`), each `erf`/`exp`/`log`/`sqrt` sub-term whose argument is
  soundly boundable is abstracted to a fresh variable ranging over its certified
  envelope hull (Gappa-proved for `erf`; Arb mean-value enclosure, cross-checked
  `<= naive`, for `exp`/`log`/`sqrt`), and the residual is solved over the reals.
  Argument bounding is sound interval arithmetic over the whole goal expression
  (affine forms exact; nonlinear `mul`/`div`-with-nonzero-divisor/`abs`/`min`/
  `max`/`ite` and nested transcendentals soundly bounded; fail-closed on
  unbounded leaves, zero-spanning divisors, and unsupported ops). A discharge
  through this lane wears the distinct honest verdict
  **`proven_modulo_certified_envelope`** — strictly weaker than
  `proven_modulo_real_arithmetic` (it discloses the extra envelope
  over-approximation), and an envelope-free over-reals proof never carries it.
  The lane returns a green result ONLY on a residual proof; a residual
  counterexample (possibly spurious under the over-approximation) declines rather
  than reporting a false disproof.

### Scope / known limitation

- **Black-Scholes call-price positivity remains `unsupported` (chelis#637).**
  The transcendental-envelope capability ships, but the flagship BS-positivity
  goal is NOT reachable by it: abstracting the coupled `normal_cdf(d1)` /
  `normal_cdf(d2)` into independent envelope-bounded variables discards the
  functional coupling the property depends on, so the residual is falsifiable and
  the goal stays honestly `unsupported`. The successor lanes (correlated/
  relational abstraction, or whole-expression `BoxRange` interval evaluation) are
  tracked in chelis#637.

## [0.15.2] — 2026-07-10

### Fixed

- **`reef conform` downstream-adoption friction (chelis#651–#655).** Five fixes
  surfaced by the first real downstream `conform bump` (Chelis-Lang/school#171 →
  #172):
  - **§8 repo-local domain skills (#651).** A shell declares skills it owns in
    `reef.toml` under `[conform] local_skills = [...]`; `conform sync` preserves
    them and `conform audit` §8 exempts them, instead of silently deleting a
    shell's domain skill (e.g. School's `chelis-std`). An undeclared extra skill
    is still pruned, now with a warning.
  - **§8 shell-specific overrides (#653).** A trailing
    `<!-- shell-local:begin -->…<!-- shell-local:end -->` block in a shared
    `SKILL.md` is shell-owned: `conform sync` regenerates the toolchain body above
    it verbatim (upstream edits still propagate downstream) and preserves the
    block; §8 byte-checks only the managed span; `conform bump` flags a skill
    whose upstream body changed underneath a block.
  - **§4 whitespace-tolerant citations (#652).** `chelis #316` (with a space) now
    matches a `chelis#316` cite on both the `src/` and `docs/UPSTREAM_BUGS.md`
    sides, ending a false "uncovered narrowing citation".
  - **`conform audit --explain` (#654).** Names the citation site(s) and the
    per-candidate coverage reasoning behind a failing §4 row; the evidence is also
    always emitted in `--json`.
  - **Categorized `conform bump` report (#655).** The post-bump audit separates
    bump-owned failures (its own pins, managed-block stamps, and skills — exit 1,
    the shell is not in a clean bumped state) from author-follow-up rows (CI
    wiring, pre-existing doc fixes, the prose Pin-Bump-Checklist/Scaffolding-Drift
    headings the bump does not restamp); a clean bump that leaves only follow-up
    exits 0 and lists the remaining steps.

## [0.15.1] — 2026-07-10

### Added

- **Downstream shell-repo conformance in the toolchain (#628).** New
  `chelis reef conform <audit|init|sync|bump|bump-check>`, backed by the
  `chelis-conformance` crate, so the shell-repo contract
  (`spec/design/shell_repo_contract.md`) is enforced and propagated
  mechanically from the pinned toolchain instead of via hand-copied scripts
  that drift. Two arms on one command group: **convention conformance**
  (`audit` checks the contract's §11 rows offline and hermetically —
  managed-block stamps, lockstep workflow pins, narrowing coverage,
  materialized-skill fork detection — with `init`/`sync` scaffolding and
  regenerating the pointer managed blocks and shared skill set), and
  **version propagation** (`bump` mechanizes the Pin Bump Checklist;
  `bump-check --base <ref>` fails any diff that moves the reef pin without a
  green audit, so a raw direct-to-main cascade cannot land). The contract's
  §11 table (`MANIFEST`) and shell registry (`REGISTRY`) become
  tripwire-locked single sources of truth (the `gate.py`/`test_gate.py`
  pattern), ground-truthed to the `ecosystem-drift.yml` canary matrix, which
  gains an advisory issue-only `conform-audit` leg.
- **`chelis test --expect neg|blocked`.** Native expected-failure suites,
  replacing the shells' vendored `run_negative_tests.py` /
  `run_blocked_probes.py`. An empty `--expect` suite is a hard error (a
  zero-probe directory can no longer silently disable the guard), and
  `--filter` combined with `--expect` is rejected.
- **`chelis-version` crate.** A zero-dependency leaf crate owning the strict
  `X.Y.Z` semver and safe-path-component predicates, so the chelisup
  installer and the conformance auditor cannot drift on what a valid pin is
  (the auditor can no longer bless a pin the shim would reject).

### Fixed

- **Hull conformance manifest pin rides the release bump (#634).** The
  gate's `STALE CORPUS` check compares
  `tests/conformance/hull/manifest.json`'s `chelis_version_pinned` to the
  live binary, so the previously-manual post-release pin bump opened a
  red window on `main` at every release (0.14.0 and 0.15.0 both hit it).
  It is now category 6 of `scripts/bump_compiler_pins.py` — rewritten in the
  release change set, validated by that PR's own conformance-gate run, and
  locked by `test_pinned_version_matches_workspace_version`.

## [0.15.0] — 2026-07-10

### Added

- **Runtime-symbolic movement bounds and reshape targets (chelis#616,
  #627).** A `shrink`/`stride`/`pad` bound or a `reshape` target extent can
  now be a runtime rank-0 integer scalar (a `shape()` read or integer
  arithmetic over one, including an inlined window-count parameter),
  represented as a node-valued `RtDim` and resolved in both the eval and C
  lanes with matching value AND error paths (negativity, range, zero-size,
  and reshape-numel runtime guards). The movement adjoints are
  runtime-capable on the same representation, so the runtime-symbolic-window
  `avgpool1d` gradient — the chelis#368/#513 residual — now computes exactly
  (`[0.5, 0.5, 0.5, 0.5]`, locked by analytic + finite-difference +
  forward-parity oracles in eval and by the compiled C binary). Runtime dims
  are *op-declared*: the owning op declares the extent inline in C and the
  evaluator binds it mid-evaluation, with runtime equality aborts across
  sites. `--target hip`/`--target metal` reject runtime bounds and targets
  with a clean diagnostic naming `--target c`. See
  `spec/05-risc-primitives.md` §2.4.1 and `spec/06-transformations.md`
  §2.7.1; residuals are tracked as chelis#631 (host-lane list-concat typing)
  and chelis#632 (movement-chain over-unification), both loud.
- **`RiscOp::Shape` runtime shape-value read (#558, #617).** A scalar
  `shape(x, axis)` VALUE read lowers to a real rank-0 integer node instead
  of the bogus `Load { name: "shape" }` fallthrough (which silently
  mis-evaluated under `grad`). AD-transparent (zero cotangent); the C
  backend reads the extent from the runtime input tensor.
- **Multi-argument ADT gradients + multi-target grad display (chelis#520,
  #614, #619).**

### Fixed

- **Per-axis movement typing (soundness, chelis#616 red team).** Type
  inference no longer collapses every axis of a `shrink`/`pad`/`stride` to a
  wildcard when one bound is runtime; a multi-axis shrink mixing a runtime
  axis with a literal-bounded axis previously produced a silently mis-sized,
  wrong-valued C tensor. Literal axes keep their precise infer-time
  validation; the C backend additionally aborts if a runtime extent
  disagrees with a statically-typed dim (defense in depth).
- **Leading-axis symbolic concat heap overflow (chelis#593) closed
  structurally.** The C backend's anonymous-dim renaming no longer copies an
  operand's dims wholesale over an extent-altering movement output; the
  formerly fail-closed reproducers now build and run correctly.
- **Zero-size runtime shrink error parity.** The C runtime guard now rejects
  `end <= start` exactly as the evaluator does (previously it silently
  produced an empty tensor where eval errored).
- **`fail(...)` in a DAG-lowered `if` branch** lowers to a properly-shaped
  zero placeholder instead of a phantom rank-0 `Load { name: "fail" }` (which
  broke the C lane and mixed ranks in the mask arithmetic).
- **Zero-fill adjoint-free tensor slots in multi-target grad (#625).**
- **rustc 1.97 toolchain compatibility.** Three new clippy lints on
  pre-existing code, and the chelis-types small-stack guard tests no longer
  drop their deep test chain on the 1 MiB worker thread (1.97's larger drop
  frames overflowed it; the guard itself was measured sound at the existing
  128 KiB red zone).

### Changed

- **Wire schema version 2 → 3 (chelis#616).** `WireRiscOp::Reshape` now
  serializes `Vec<WireRtDim>` (a bound-tagged value: `lit` / `node` / `sym`)
  instead of `Vec<WireDimInfo>`, and `WireRtDim` gained `Sym`. Pinned
  consumers (Beacon, offline verifiers) must observe the bump.

## [0.14.0] — 2026-07-03

### Added

- **Error localization in `chelis check` diagnostics (#615).** The
  structured `errors` array now carries `span_offset` (byte offset in
  source) and `span_id` (e.g. `"surf:23..45"`) for PrecisionMismatch,
  ArityMismatch, DimensionMismatch, UnboundVariable, and
  UnknownConstructor errors. Downstream tools can identify WHERE an error
  occurs without heuristic post-processing. FlukeBall can drop its
  approximate error localizer.
- **`ConstTensor` RISC op for in-core tensor literals (#615).**
  `to_tensor([1.0, 2.0, 3.0])` with all-literal arguments now lowers to
  a single `ConstTensor` node instead of a 145-node add-tree. C/HIP
  backends emit a static `memcpy` from an initialized constant array.
  Beacon-provable with trivial known range `[min(data), max(data)]`.
  FlukeBall can relax the #38-B add-tree avoidance steer.

## [0.13.0] — 2026-07-02

### Added

- **`chelis reef setup` one-command provisioning, unified `reef doctor`,
  and the cross-version unknown-subcommand hint (#602).** The WS-C
  orchestration layer: `reef setup` ensures the pinned toolchain
  (auto-installing by subprocessing the real `chelisup` binary), installs
  source packages and binary artifacts from `reef.lock`, syncs
  `[chelis-src]` source crates, and prints the `reef doctor` summary.
  `reef doctor` now reports toolchains from the consolidated chelisup
  store (`$CHELIS_HOME/toolchains/<ver>`) with a machine-wide header
  (home, shim, recorded default) and covers `[artifacts]` binaries.
  Unrecognized-subcommand errors name this chelis's version, the pin
  source that routed the invocation, and the `chelis +<ver>` override.
- **chelisup bootstrap release assets (#604).** `release.yml` now
  publishes `chelisup-<slug>` bare executables (plus sha256 sidecars) and
  `chelisup.sh` itself, so the bootstrap one-liner has a canonical
  latest-release URL. The linux chelisup builds in the glibc-2.31
  container so the first binary a bare machine runs loads on the oldest
  supported glibc.
- **Deep authoring L2 substrate (#588).** Build-out of the Deep authoring
  layer-2 substrate.

### Fixed

- **Soundness Wave-2: symbolic-dim grad/eval machinery (#590).** Covers
  chelis#551, #513, #523, and #383.
- **Form-3 runtime expand-size resolution: fail-closed reject +
  deterministic kernel ABI (#469, #596).**
- **Loud-reject inline `expand` sizes (#530) and the runtime `vmap` axis
  (#524) (#563).** Previously-silent unsound accepts now reject loudly.
- **Eval host negative-axis parity (#522) and chelis-ir integer division
  zero-trap (#550) (#567).**
- **backend-c: `cmplt` dtype-correct read (#517); named-axis reduction
  family admitted in `..r` bodies (#340) (#565).**
- **`chelis prove` soundness hardening (#564).** chelis#463/#434 oracles
  plus #496 canonical attribution.
- **IR by-position named-axis recovery re-validated against reorder
  (#549) (#562).**
- **Cache-proof the bundled chelis-std lock-hash invariant (#585,
  #598).** The lock-hash desync class fixed in 0.12.1 is now guarded
  against cache-staleness regressions.

### Internal

- ci(smt): durable prebuilt cvc5 as a Release asset with a rustc-free
  cache key and cache pruning (#591); stable cvc5 cache key (#583, #584);
  SMT Full Prove runs nightly/dispatch only, not on PRs (#595).
- test(types): lock the #458 mixed `(f32, i32)` numeric-op rejection
  consistency (#566).

## [0.12.1] — 2026-06-30

### Added

- **reef source-crate dependency manager (`reef src` + `reef doctor`)
  (#571).** Class-(c) source-crate sourcing so shells that link chelis
  compiler crates (`chelis-ir`, `chelis-types`) as Cargo path deps can
  build side by side on different chelis pins without colliding on a
  single `../chelis` sibling clone. Adds the manifest `[chelis-src]`
  section and a version-keyed source store.
- **chelisup rustup-style toolchain installer + shim (#164, #576).** New
  `crates/chelisup` member: a single binary that, via `argv[0]`, runs the
  installer CLI when invoked as `chelisup` and the fast pin-resolving
  shim when invoked as `chelis`. Store layout lives under `$CHELIS_HOME`
  (default `~/.chelis/`): `bin/`, `toolchains/<ver>/`, and a recorded
  `default`.
- **reef first-class binary artifact distribution (#468, #577).** Per-
  platform, SHA-pinned binary release artifacts: declared once in a
  manifest `[artifacts]` section, recorded in `reef.lock`, and installed
  by `reef install --from-lockfile`, which downloads, fail-closed
  SHA-256-verifies, extracts, and places a runnable binary at
  `$CHELIS_HOME/bin/<name>`.

### Fixed

- **`chelis prove` linked-program guard on the property path (#581).**
  `chelis prove` on a reef package carrying a `@property` rejected the
  linker's own internal-mangled names (`pkg__<pkg>__<Module>__<def>`)
  with `ReservedLinkerName`: the property path
  (`run_surf_linked_properties_shared`) never installed the
  linked-program provenance guard that the obligation path
  (`check_linked_decls`) already installs, so `detect_forged_linker_names`
  saw `linked_program() == false` and rejected the linker's own output.
  Install `install_linked_program_guard()` on the linked property path.
- **Re-synced the committed reef.lock SHA-256s with the embedded bundle
  (#585).** `packages/chelis-std/reef.lock` and the `release_pipe_stage`
  fixture lock carried stale `archive_sha256`/`shell_sha256` after #559
  regenerated the embedded chelis-std bundle without re-committing the
  locks. The release bump regenerates both locks, restoring
  `bundled_chelis_std_lock_hashes_match_embedded_artifacts` to green.

### Docs

- **Packaging & install north-star roadmap (#574).** Adds
  `spec/design/chelis_packaging_and_install.md` (the three-layer install
  model, the lockfile-ownership rule, store consolidation under
  `~/.chelis/`, the chelisup #164 spec, and the reef setup orchestrator)
  and `reef_distribution.md` Item 11 (binary distribution #468), framing
  source crates (#571), binary distribution (#468), and chelisup (#164)
  as one coherent install tool plus package manager.

### Internal

- Raise the SMT Feature Build (Linux) timeout 20->40m so cold PR-branch
  cvc5 builds finish and warm the cache instead of being cancelled at the
  20m cap (#582).
- Split the full prove lane from the required SMT smoke job and share its
  warm cache namespace (#560).
- Add a live Beacon shim round-trip gate (#561).

## [0.12.0] — 2026-06-29

### Changed

- **BREAKING: `div` is now float-only; integer division uses the new
  `floor_div` / `trunc_div` primitives (#178, #511).** The single-op
  two-semantics overload (`div(7, 2) == 3` for ints, `== 3.5` for
  floats) matched none of torch/JAX/numpy and was a footgun, so `div`
  (and the `/` operator that desugars to it) now requires float
  operands and an integer operand is a type error whose diagnostic
  cites `spec/05-risc-primitives.md` §2.1 and points at the
  replacements:
  - `floor_div(a, b)` rounds the quotient toward -inf (Python `//` /
    torch / JAX / numpy `floor_divide`); admits integer **and** float
    operands.
  - `trunc_div(a, b)` rounds toward zero (the C/Rust integer `/`
    quotient); integer-only. This is what `Std.Decimal` arithmetic
    needs.

  Both new primitives are Tier-1 and are wired through every dispatch
  site: type checker (direct and polymorphic-wrapper rejection),
  linearity, IR (`RiscOp`/`FusedStepOp`, lower, tier2, grad, fuse,
  verify, specialize, eval), the wire schema (bumped to version 2 with
  a `chelis-prove` producer tripwire), and the C and HIP backends (with
  remainder-sign correction and a zero-divisor guard). `Std.Decimal`'s
  bundled `decimal.ch` was migrated from `div(int)` to `trunc_div` and
  the bundled `chelis-std` artifact regenerated. **Migration:** replace
  integer `div(a, b)` with `floor_div(a, b)` for Python-style flooring
  or `trunc_div(a, b)` for C-style truncation; float `div` is unchanged.

### Added

- **Verification stack — Beacon discharge engine + transformation
  layer.** The Phase 2 push that takes the verification substrate from
  a registered-in-tests shim to a live engine that closes real proof
  goals end to end:
  - **Beacon wired into live dispatch with end-to-end verified Greeks
    (#539).** `BeaconShim` is registered in the live engine registry,
    and the transformation-layer Stage 1 substrate lands (`Transformation`
    type, pipeline, and a soundness harness that rejects unsound
    transformations).
  - **Beacon verified-zonotope selector (#538, #532).** A
    `BeaconOracleMode` lets the shim keep its default `oracle: null` or
    explicitly request `oracle: "zonotope_verified"`, with exact
    request-shape and no-laundering tests.
  - **Abstract-subterm transformation: envelope + polynomial
    decomposition (#540).** The first transformation through the
    soundness harness replaces a transcendental subterm (`erf`) with a
    fresh variable bounded by the WI-13 certified envelope, leaving a
    residual polynomial goal that routes to Z3/cvc5. The bound is
    evaluated soundly over the argument's entire range (hull across all
    intersecting envelope boxes), declines on unboundable arguments, and
    tags its result `SpecialFunctionCertified` rather than `Exact`.
  - **Vectorized Black-Scholes pricer WireDag root (WS-9, #541, #440).**
    A pure-tensor-DAG BS call pricer (`bs_vec.ch`) that emits a named
    WireDag root Beacon can verify, with byte-seam round-trip,
    deterministic-lowering, and non-finite-float-precondition tests.
    Approximation coefficients are passed as tensor parameters pinned to
    point intervals at the Beacon boundary.
  - **Goal splitting with lattice-aware recombination (#542).** A
    `GoalSplit` transformation splits a conjunctive postcondition into
    independent sub-goals (each retaining the full variable set and
    precondition list, so coupled variables stay sound) and recombines
    the sub-discharges by a soundness lattice: soundness is the `min`
    across sub-discharges, the qualifier set is their union, and
    `Disproved` / `Error` / `Timeout` dominate so partial green never
    reads as green. Hand-built as an integrity core with an 11-row
    exhaustive recombination truth-table test.
  - **`div(x, x)` one-fabrication elimination + corpus coverage tooling
    (#546, #547).** The `bs_vec.ch` pricer no longer fabricates `1.0`
    via `div(x, x)` (replaced with `add(half, half)`), so its WireDag
    carries no `Div` node; and `scripts/prove_corpus/harvest_coverage.py`
    harvests `@property` goals from downstream shells and measures
    discharge coverage with differentiated fuzz-tier buckets.
- **Verification stack handover milestone docs (#554, #555).** A
  single-source handover artifact documenting what the stack proves,
  what falls to fuzz and why, the honesty-layer integrity core, the seam
  contract state, build commands, and the frontier — validated by a
  fresh-clone build and a round-tripping proof.

### Fixed

- **`len` / `index` borrow their `List` / `Dict` argument (#527,
  #531).** Both queries consumed their container, so the idiomatic
  read-then-reuse shape stopped type-checking once consume-tracking
  began enforcing tag-colliding `List[...]` parameters. Their runtime
  backings take a `const` pointer and never free, so they are now
  classified as borrows in the linearity checker; a genuine consume
  (explicit `drop`, or moving into an owned parameter) still makes a
  later `len` / `index` a use-after-consume. The explicit `len(&xs)` /
  `index(&xs)` surface form is rejected with a diagnostic that the query
  auto-borrows.
- **`grad` through a Tier-3 named-axis reduction reached via a
  concrete-rank intermediate (#373, #515).** The reverse-mode lane
  inlines the differentiated body into one DAG, monomorphizing the
  named anchor (`seq`) to a concrete rank and tripping a
  `rank-spread anchor absent` lowering error even though the forward
  lane evaluated correctly. Anchor positions are now recorded only when
  no spread precedes the named dim, and `extract_rank_var_bindings`
  falls back to the recorded position (range-checked) when the name is
  absent from the monomorphized actual. (A by-position recovery
  staleness caveat under intervening axis-reorder ops is tracked as
  chelis#549.)
- **Differentiable `concat` grad + grad-capture eval parity (#368,
  #377, #514).** `grad` through a windowed reduce over `concat` hit the
  host-only `concat` catch-all and dropped the windowed rows; `concat`
  along a constant axis now lowers to the adjoint-carrying `Pad`+`Add`
  cascade so the gradient reaches the windowed inputs (finite-difference
  validated, including the symbolic-non-concat-axis case via a
  `SHRINK_TO_END` sentinel resolved before eval). Separately, the
  host-runtime transform lane now serves captured and top-level tensor
  bindings to a `grad`/`vmap`-captured def, closing an eval-vs-C-backend
  parity gap. (Correct `vmap`-with-captures broadcast stays a tracked
  residual, #377.)
- **`trace` reduces its diagonal in the stride-4 ILP cascade for torch
  parity (#170, #505).** `torch.trace == torch.sum(diagonal)`, but both
  the evaluator and the f32/f64 runtime used a strict left fold, diverging
  from torch by 1 ULP for diagonals longer than 16 f32 elements. Both
  lanes now route through the verified stride-4 cascade. The
  accompanying audit documents why `matmul`/`einsum` (BLAS GEMM order),
  softmax `sum_exp` (already host==lowered consistent), and `cumsum`
  (inherent prefix order) deliberately do **not** take the cascade and
  keep their f64 reference accumulator.
- **Deep replacement-authoring substrate hardening (#548).**
  `chelis_replace_function_body` now reports malformed / type / effect /
  linearity failures over both MCP and HTTP with no replacement result
  on rejection; duplicate `defsig` declarations are rejected in both
  check and validate (Chelis has no same-name overload dispatch); and
  default `chelis surf` / `decompile` output is formatter-canonical, with
  the agent-editing-surface specs and oracles synced to the shipped tool.

## [0.11.1] — 2026-06-26

### Added

- **WI-13 Sollya+Gappa erf proof-term envelope (#533).** Machine-checkable
  polynomial approximation of `erf` for the central arm (16 sub-interval
  Gappa proofs + whole-box f64-Horner rounding proof) and Arb-certified
  saturation tails. Unblocks transcendental finance property discharge.
- **WI-14 Arb/FLINT rigorous erf-enclosure oracle** (`--features arb`).
  Independent numerical cross-check of the committed envelope.
- **Default-lane eps-backing gate.** The committed envelope `eps` is now
  independently verified against `.gappa` goal lines in every default CI
  run (no `--features arb` required). Closes the RT-WS7b proof-integrity
  HIGH finding.
- **Arb CI lane** in the SMT Feature Build job.

## [0.11.0] — 2026-06-26

### Removed

- **`Std.Tensor.Reduce` (`min`, `prod`, `argmax`, `argmin`) removed
  (#333).** The four were bodyless sigs declaring a *runtime* `int32`
  axis, but the `*_reduce` builtins they would forward to require a
  *compile-time-constant* axis, so the module was unimplementable as
  declared and never had a runtime function — calling any of them was an
  `unknown runtime name` at eval. Shipping a std signature the repo
  cannot honor violates the honesty invariant, so the module and its
  (quarantined) corpus test are removed rather than faked. Migration:
  call the builtins directly with a const axis —
  `min_reduce(x, cast(1, int32))`, `prod_reduce`, `argmax_reduce`,
  `argmin_reduce`.

### Fixed

- **SOUNDNESS: two calls of an ITE-bodied def at a goal site no longer
  false-prove (#426).** A property comparing or subtracting two calls of
  the same def whose body itself calls an `if`/`then`/`else`-bodied
  helper (e.g. an `fmax`-based max-over-actions), with different
  arguments, collapsed to the all-arguments-equal corner during the
  Surf->SMT property lowering: the nested-call inlining bound the
  helper's parameters to its raw argument expressions and re-lowered
  them in the callee scope, dropping the outer call-site's bindings, so
  both calls produced identical SMT terms and a mathematically false
  goal reported `passed / smt / proven` with no counterexample. The
  inline substitution now binds each parameter to its argument already
  lowered in the caller scope, so two call-sites with different
  arguments lower to distinct terms: a false goal refutes with a
  counterexample and a true goal still proves. `let`-bindings lower
  their values under the same scope discipline.

## [0.10.0] - 2026-06-23

### Added

- **WI-16 Carcara audit of cvc5 Alethe proofs (#450).** A Carcara
  engine that re-checks cvc5's Alethe proofs in-process
  (Confirmed / ConfirmedModuloRewrites with trusted-hole disclosure /
  Failed / Unavailable) as write-only audit evidence that cannot
  promote a badge: `classify` binds the hedged
  `SoundApproximate`/`RealArith` before the audit runs and `badge()` is
  blind to the audit result. `render_real_lit` is aligned to the
  exact-f64 rational matching the #444 literal rendering. The SMT
  Feature Build CI job now also builds and tests `--features carcara`
  (reusing the cvc5 build; gmp/mpfr added to the apt step).
- **Beacon subprocess discharge-engine shim (#439).** A thin in-tree
  `DischargeEngine` shim for `GoalShape::BoxRange` that shells out to an
  out-of-tree `chelis-beacon` binary and maps its `CheckReport` to a
  `Discharge`, using content-addressed inline `WireDag` byte transport
  via a dispatch-site-owned `WireDagByteStore` (zero frozen-surface
  change to `IrHandle` or the trait). The soundness guard is
  fail-closed: only a Beacon-proved report yields
  `SoundApproximate`+`SoundOverApproximation` and only a verified-refuted
  report yields `Disproved`+`SoundApproximate`; every other report
  (unverified, error, timeout, store-miss, hash-mismatch, or a
  self-contradictory proved/oracle-unverified report) maps to
  `Untrusted` with an empty guarantee set. Transport is deadlock-safe
  (auto-fallback to a temp file above a 32 KiB inline ceiling; a 256 KiB
  payload hard-kills in ~0.5 s). Registered in tests only for now;
  production dispatch wiring is a later change.
- **Symmetric DisprovedModuloRealArithmetic honesty hedge (#447).** An
  SMT-over-reals disproof now renders
  `CompositeVerdict::DisprovedModuloRealArithmetic` and discloses
  `real_arithmetic`, symmetric to the #445 `ProvenModuloRealArithmetic`
  hedge: a reals counterexample may be a false counterexample at f64, so
  a definite `Failed` would overclaim — the hedged failure dominates
  greens but loses to a definite `Failed`. A fuzz disproof (a real f64
  witness) stays a definite `Failed`. Reuses `Qualifier::RealArith` (no
  new qualifier); both the `property_runner` and `obligation_engine`
  paths thread the hedge, and the MCP surface inherits it.

### Changed

- **WI-3 graph extraction scoped to a single entry's reachable defs
  (#440).** Entry-scoped reachable-defs pruning for the WI-3 producer
  plus a shared `chelis-compiler-api::prune` module that the CLI build
  path now delegates to (the duplicate traversal is removed). The
  entry-scoped DAG is a structurally-identical subgraph of the
  whole-program DAG and the build-path output is byte-identical.

## [0.9.0] - 2026-06-23

### Added

- **Verification stack Phase 1: substrate + honesty layer (WI-1..WI-8)
  (#427).** The engine-independent verification substrate every future
  proof engine plugs into: `WireDag.schema_version` IR op-subset
  versioning with a typed reject of unknown versions plus the
  wildcard-free `RiscOp::is_verifier_targetable` classifier (WI-2); the
  discharge-engine seam with the engine-independent `Goal` / `Discharge`
  types (WI-4/WI-5); and the verification-stack design-doc cluster. The
  static-analysis shell `beacon` was renamed `hydrostatic`, freeing the
  `beacon` name for the verification engine registered as an ecosystem
  shell.
- **Verification stack Phase 2 Wave 1: discharge integrity primitive
  (#428).** `Discharge::new` enforces a per-`Qualifier`
  minimum-soundness map over all seven guarantee kinds
  (Exact/CertificateBearing -> Exact;
  DeltaComplete/SpecialFunctionCertified/SoundOverApproximation ->
  SoundApproximate; Fuzz -> Empirical; Axiom -> Untrusted), the
  no-laundering invariant every later guarantee rests on, locked by an
  exhaustive 7x4 table-driven oracle with a closed-set guard.
- **WI-3: graph-extraction seam producer (#431).** The content-addressed
  `IrHandle { dag_hash, root_index }` addressing a serialized WireDag v1
  by sha256 plus root index, and the box/range `Goal` producer the
  verification dispatch layer and the out-of-tree Beacon shell build on.
- **WI-9: discharge-engine registry + fitness-based dispatcher (#432).**
  `DischargeRegistry` with a public `register(...)` entry point so an
  out-of-tree shell engine registers without touching the crate; boxed
  trait objects rather than a closed enum. `dispatch(&Goal, timeout)`
  scans registered engines in registration order and takes the first
  whose `fitness(goal)` is true, replacing the hardcoded cvc5 selection.
- **WI-10: AD-as-verification-target rail (#433).** Makes the gradient
  (adjoint) graph dispatchable as box/range goals so a
  verified-bounded-sensitivities goal routes through the WI-9
  dispatcher: `grad_goals_from_request` calls the public
  `chelis_compiler_api::compiler::grad(...)` once and emits one scalar
  box/range goal per requested target.
- **Deep Authoring L0 + the `chelis_replace_function_body` wedge
  (#429).** Foundation of the Deep authoring stack (path addressing,
  fragment-scoped validation, Deep manipulation) plus one vertical edit
  tool end to end in chelis-tide. Fragment-scoped validation type-checks
  a single spliced function body against a module context built once,
  without recompiling the module, at full type + effects + linearity
  parity with `chelis check`. Deep-native: the tool consumes and
  produces canonical Deep and persists nothing.

### Changed

- **`chelis prove`: honest-verdict taxonomy -- a green now discloses its
  verification method and approximations on the verdict (chelis#422).** The
  `composite_verdict` is the single WEAKEST badge token plus a new structured
  `qualifiers:[...]` array carrying the full disclosed caveat set, on BOTH
  prove-JSON surfaces (`chelis prove --json` and the tide MCP tool). Three
  soundness situations are seated on one non-launderable lattice:
  - A fuzz-only BASE (no SMT proof underneath -- a non-smt build, a `--tier
    auto` fuzz fall-through, or `--tier fuzz-only`) renders `fuzz_validated`
    with `qualifiers:["fuzz_base"]`, NEVER a `proven_*` badge. Previously such
    a pass false-greened as `proven_modulo_fuzz_validated_contract`,
    misreporting an unsampled measure-zero falsehood as proven.
  - An SMT proof is over the REALS (int widths -> unbounded integer sort,
    `f32`/`f64` -> `Real`; no overflow/NaN/rounding), so every
    `proof_tier:"smt"` green now renders `proven_modulo_real_arithmetic` with
    `real_arithmetic` in `qualifiers`, disclosing the machine-arithmetic gap on
    the verdict itself rather than only the legacy `arith_model:"real"` side
    field (retained as a mirror). Plain `proven` is reserved for a future
    exact-machine-arithmetic lowering.
  - A sound over-approximation base (e.g. an out-of-tree interval engine such
    as Beacon, via `Qualifier::SoundOverApproximation`) renders
    `sound_approximate`. The consumer seam was fixed so the dispatch
    `Discharge`'s `(soundness, qualifiers)` is threaded into the verdict
    instead of being discarded (`property_runner`/`obligation_engine`), which is
    what lets such a discharge reach `composite_verdict` honestly; the
    signed-off `DischargeEngine::discharge` trait shape is unchanged.

  The legit `proven_modulo_fuzz_validated_contract` badge (an exact SMT base
  modulo a fuzz-validated CONTRACT) keeps its token and additively gains
  `real_arithmetic` in `qualifiers`. Exit codes are unchanged. The `wi6`
  NxN byte-identity lattice oracle is extended to the new badges and stays
  green (a weaker base or qualifier can never launder into a stronger badge).
  NEW accepted `composite_verdict` tokens for downstream allowlists:
  `proven_modulo_real_arithmetic`, `sound_approximate`, `fuzz_validated`; NEW
  field: `qualifiers:[...]` (snake_case strings: `real_arithmetic`,
  `fuzz_base`, `fuzz`, `sound_over_approximation`, `axiom`, ...).

### Fixed

- **`chelis prove`: cvc5 lowers `RealLit` to the exact f64 value, not the
  decimal string (#444)**: a CRITICAL soundness fix. cvc5 lowered a float
  literal via `mk_real_from_str(format!("{value}"))`, the exact DECIMAL
  (`0.1` -> `1/10`), reasoning about a value the runtime f64 never holds, so
  it could PROVE a float property FALSE at runtime (it proved
  `0.1 + 0.2 == 0.3`). `RealLit` now lowers to the exact f64 value
  (`BigRational::from_float` -> num/den -> `mk_real_from_str`), so cvc5, Z3,
  and the runtime f64 evaluator reason about the same number; the
  literal-driven prove-false is closed (`0.1 + 0.2 == 0.3` is now Disproved,
  matching the evaluator) with zero corpus flips. `num-rational` is
  smt-gated; the non-finite guard is preserved.

- **`chelis prove`: call-form predicates lower to SMT instead of silently
  fuzzing (chelis#422)** — operator-form `(x * x) >= 0.0` lowered to SMT and
  proved, but call-form `gte(mul(x, x), 0.0)` returned `None` from the
  predicate-position lowering and silently dropped to a Tier C fuzz pass, so a
  measure-zero-false call-form rendered a fuzz green. Call-form comparison
  primitives (`gt`/`gte`/`lt`/`lte`/`eq`/`neq`, and the desugared `cmplt`) now
  lower to `SmtExpr::Cmp` and call-form arithmetic (`add`/`sub`/`mul`/`div`)
  to interpreted `SmtExpr::Arith` nodes, making call-form fully equivalent to
  operator-form. A measure-zero-false call-form is now SMT-refuted with a
  counterexample, not fuzz-passed. Oracles in
  `crates/chelis-cli/tests/prove.rs`
  (`call_form_predicate_proves_at_smt_like_operator_form`,
  `measure_zero_false_call_form_is_refuted_at_smt_not_fuzz_passed`) and unit
  tests in `crates/chelis-prove/src/property_runner/smt_lower.rs`.

- **C backend: function-body heap temporaries and nested-tuple printing no
  longer leak (completes #406)** — #412 freed the heap temporaries the
  program-root `main` allocates, but two sibling "definitely lost" classes
  of the same shape survived, each a `chelis_*_from_values` allocation
  generated `main` never released transitively. (1) A heap host value
  (tuple / list / dict / adt / string) built as an *intermediate* inside a
  compiled function body — e.g. a `let`-bound `p = (a, b)` consumed by the
  body — was never released: #412 only enabled scope-release tracking for
  `emit_main`, whose flat-scope guard skips the deeper indent a function
  body's block introduces. The `let` lowering now releases a block's heap
  bindings at block close, retaining any binding the block result aliases
  (directly or through an `if` / `match` arm, or a `b = a` binding-to-
  binding alias) so the caller keeps exactly one reference — implemented as
  a `LetReleaseScope` stack plus `retain_transferred_result` /
  `binding_release` in `chelis_backend_c::host_emit`. (2) A tuple field
  that is itself a heap container (a *nested* tuple / list / adt) leaked at
  the labeled-root printer, because `chelis_tuple_get` retains the boxed
  element it returns and the printer only read it; the printer now releases
  the retained handle after printing the field. Acceptance oracles in
  `crates/chelis-cli/tests/cli.rs`:
  `build_c_function_body_heap_temp_has_zero_definitely_lost_under_valgrind`,
  `build_c_function_body_list_temp_has_zero_definitely_lost_under_valgrind`,
  `build_c_function_body_tuple_transfer_has_zero_definitely_lost_under_valgrind`,
  and `build_c_nested_tuple_print_has_zero_definitely_lost_under_valgrind`
  each build the reproducer to C, gcc-link it, run it under
  `valgrind --leak-check=full` with no suppressions, and assert
  `definitely lost: 0 bytes` (red on the post-#412 tree, green after this
  change).

- **C backend: `main` no longer leaks the list / tensor temporaries it
  allocates (#406)** — a `chelis build --target c` program leaked two
  "definitely lost" blocks under `valgrind --leak-check=full`: the
  `to_tensor([...])` list built by `chelis_list_from_values` and the
  result tensor allocated via `chelis_contiguous`/`chelis_alloc`, both
  reachable from generated `main`. The runtime helpers are correct (they
  return owned values); the leak was that generated `main`, the program
  root, never released the heap allocations it owns. `emit_main` now
  tracks each owned heap allocation declared at the function's top level
  (global binding values plus top-level list / tuple temporaries) and
  emits the matching refcounted release (`chelis_free`,
  `chelis_list_release`, `chelis_tuple_release`, …) before `return 0`.
  Block-scoped temporaries inside `map` / `filter` / `fold` / `if`
  bodies are excluded so cleanup never references an out-of-scope
  identifier. Compiled-function bodies already emitted their own
  per-local cleanup and are unchanged. Acceptance oracle:
  `build_c_grad_program_has_zero_definitely_lost_under_valgrind` in
  `crates/chelis-cli/tests/cli.rs` builds the grad_quadratic reproducer
  to C, gcc-links it, runs it under valgrind with no suppressions, and
  asserts `definitely lost: 0 bytes`.

- **Type system: a body wildcard narrows to a param-bound declared
  symbolic dim, fixing the `chelis build` reshape ICE (chelis#405, #430)**:
  `const_col`-style code (`nn = shape(spots, 0)` ->
  `to_tensor(map(.., range(0, nn)))` -> `reshape([nn, 1])`) returned
  `tensor[*, 1]` (shape-erased wildcard) at the call site rather than
  `tensor[n, 1]`. The post-defsig narrowing in `narrow_wildcards_with`
  narrowed a body `Dim::Wildcard` to a declared dim only when that dim
  was a `Dim::Lit`, never a `Dim::Var`, so the literal axis narrowed but
  the symbolic axis stayed `*`; downstream the vmap kernel referenced an
  undeclared symbolic dim and the lowering guard panicked. Narrowing now
  also resolves a declared `Dim::Var`, so the symbolic axis is recovered
  and the build no longer ICEs.

## [0.7.27] — 2026-06-17

### Fixed

- **Eval value renderer no longer leaks package-mangled ADT constructor
  names (#399)** — a regression from #386. `chelis eval` (both the human
  renderer and the `eval --json` `ExecutionValue` ABI) emitted the
  internal `Pkg__<pkg>__<Module>__<Ctor>` form for reef-linked ADT values
  where it should show the bare display name; #386 applied de-mangling to
  diagnostics but not to the value renderer. Both render sites
  (`render_value` and `runtime_value_to_schema`) now call
  `chelis_types::demangle_ident` (a no-op on bare / builtin constructors),
  restoring the bare name and keeping the eval encode/decode round-trip
  consistent (decode already keys on bare constructor names). Surfaced by
  the Chelis-Lang/flukeball typed-ABI consumer during the 0.7.26 cascade.

## [0.7.26] — 2026-06-16

### Changed

- **Opaque types with declared invariants (Option 1.5, #386)**: a type
  declared `@opaque` is constructible and inspectable only inside its
  defining module — the type checker rejects out-of-module record
  construction, constructor application, field access, pattern
  destructuring, casts, and `.dp` literal forgery, returning the true
  type so no error cascades. An optional `@invariant(p) <expr>`
  predicate over the representation is recorded as Deep metadata,
  invisible to `chelis check`, and consumed by `chelis prove`: every
  exported producer of the type gets a derived producer obligation
  (covered-or-rejected — an unsupported return container is an error
  naming the producer, never silently skipped), and the invariant is
  injected as an assumption on opaque-typed inputs. Guarded-`Option`
  constructors prove at the SMT tier (`proof_tier:"smt"`);
  equality-constrained invariants (e.g. a simplex's sum-to-one) route to
  constructor-based generation rather than starving rejection sampling.
  Decode of an ADT value at a codec boundary revalidates the invariant
  and fails closed (never repairs). Tier B cvc5 lowering is now total
  (every term checks arity + sort) and runs in a crash-isolated
  subprocess so an uncatchable solver abort cannot take down the prove
  run. The lint rule `opaque-domain-construction` remains as fast
  per-file feedback; the typing judgment is the guarantee. Explicitly
  not refinement typing: no predicates in the typing judgment, no solver
  in `chelis check`. The new `kind:"obligation"` JSON records are
  additive — downstream admission parsers should update at pin time.
- **Return-position declared-dim rigidity is now enforced (#370/#273,
  spec/04-type-system.md §4.4.1)** — *potentially breaking*. A declared
  dim parameter that appears only in a function's RETURN type is now
  rigidity-checked, closing a soundness hole where a body could silently
  pin a "for all k" return dim to a caller-visible input (e.g.
  `def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a` pinned `k := 2`
  while promising polymorphism). The rule distinguishes two roles: an
  **output-inferred** return dim — produced from body-internal data, e.g.
  `def main() -> tensor[n, f32]` that genuinely builds a `tensor[3, f32]`
  — stays legal, while an **input-coupled** one (pinned to a concrete
  literal occurring in a parameter position, or collapsed with a
  param-position declared dim parameter) is rejected with
  `DimensionMismatch`, matching the param-position rigidity that shipped
  in 0.7.9 (#272). This can reject previously-accepted but falsely
  rank-polymorphic signatures. **Migration:** re-declare affected
  signatures honestly — make genuinely-distinct dims distinct, let
  body-determined output dims be output-inferred or dynamic (`*`), and
  drop unused/false `[..]` dim parameters. Oracle:
  `crates/chelis-cli/tests/issue_273_return_dvar_rigidity.rs`.
- The reef package linker's reserved-name predicate is now a single
  shared definition (`chelis_types::is_linker_format_name`): the reef
  entry/test reserved-name reject (RFC v6) and the checker's
  `ReservedLinkerName` rule no longer keep two copies that could drift
  (CR-7).
- `chelis prove` now warns on stderr when a default (non-`smt`) build
  proves a module declaring invariant-carrying opaque types: the
  producer-obligation machinery is gated behind the `smt` feature, so a
  default build checks no obligations. The one-line warning names the
  count and the `--features smt` remedy; it never touches the stdout
  NDJSON stream or the exit code, so machine consumers are unaffected.
- `chelis prove` now type-checks the module on EVERY build and errors
  (exit 3, a `{kind:"error", stage:"check"}` record under `--json`) on a
  type-broken module, on both the Surf and Deep paths. Previously the
  whole-module type-check rode only on the `smt`-gated obligation engine,
  so a default (non-`smt`) build silently passed a type-broken module that
  the `smt` build rejected. The default build does what it can without a
  solver (type-check + Tier-C fuzz of user properties) and messages what it
  cannot (the obligation warning above); the up-front check uses the same
  bare desugar + `check_typed_program` the obligation engine uses, so the
  default and `smt` builds agree on what is type-broken.

### Fixed

- **Builtin-shadowing defs rejected at check time (#353)**: a top-level
  `def` or `sig` whose name appears in the closed builtin vocabulary
  (`def sum`, `sig relu: ...`) is now a hard front-end error
  (`BuiltinShadowing`, spec/04-type-system.md §8.6) on every lane.
  Pre-fix the reproducer checked clean, hit the builtin's arity error
  under eval (call dispatch is builtin-first by name), and segfaulted
  on the C backend. The rejection derives from the same
  `BUILTIN_NAMES` table the evaluator and IR lowering dispatch on, so
  the sets cannot drift; reef package modules are exempt by
  construction (their decls are internal-name-rewritten before
  checking, so a package-scoped `def sum` still resolves to the user
  def). Style-gate bypasses (`--allow-style-violations`,
  `CHELIS_STYLE_GATE_DISABLE`) do not unlock it.

<!-- opaque-types prove-obligation review-5 fixes -->
- The shared Deep property discoverer now classifies a `@property` def's
  source kind exactly as the CLI discoverer does (review 5 / F6): a
  `chelis_role: "property"` def with an absent or invalid
  `property_source_kind` is an ERROR on both surfaces, not silently skipped
  (which previously made the tide tool report `total:0` / `ok:true` while
  the CLI errored), and a `bridge:c-earchin` property is skipped by the
  shared runner (the CLI bridge path owns it). A user `@property` is now
  discovered identically on the CLI and tide surfaces.
- The signed-integer-width decision is now genuinely single-source across
  every prove site (review 5). The type system owns the int-width set
  (`chelis_types::types::Prim::is_integer`), the per-width representable
  range (`Prim::integer_range`), and the fuzz-sampling window
  (`Prim::integer_fuzz_bounds`); the prove layer's `is_int_width` /
  `int_sample_bounds` are thin name-keyed wrappers, and every recognition /
  sort / sampling / literal site (the obligation engine's `sample_scalar` +
  `scalar_lit`, the property runner's `sample_value` + `unsupported_type`,
  the injection path's `classify_binder` + `sample_scalar` + `scalar_lit`,
  the opaque record-value `int_lit`, and the CLI's `unsupported_type` +
  `sample_value`) routes through them. Previously several sites spelled the
  width set independently and admitted only int32/int64, so an int8/int16
  field, param, or constant was sampled as a float or built as a mistyped
  literal -- now every width samples as a width-clamped integer (an int8 in
  [-128, 127]) and builds a correctly-cast literal everywhere.
- Tier B cvc5 lowering is now a single TOTAL pass and can no longer hand
  cvc5 a term that aborts the process. A prior whitelist GATE and the term
  builder were two enumerations of cvc5's rules that diverged: the gate
  checked operand sorts but not cvc5's arity / kind-domain requirements, so
  it admitted an empty or single-child `and`/`or`, a non-binary `implies`,
  an integer `/` feeding a comparison (cvc5 division returns Real), and a
  non-finite literal -- all of which the builder then aborted cvc5 on. The
  gate is removed; `chelis_prove::tier_b::lower_to_cvc5` now returns
  `(Term, SmtSort)` and verifies cvc5's requirement (operand sorts AND
  arity) at every `mk_term` call site, returning a clean Tier C `Error`
  rather than building an aborting term. Degenerate connective arities
  normalize to their logical identity (empty `and` is true, single-child is
  the child) so a single-conjunct invariant still proves at the SMT tier.
  Further guards at their call sites: a non-finite literal, an empty
  quantifier binder list, a non-Bool precondition / postcondition, an
  interior-NUL variable name, and an `SmtExpr` past a depth bound (the
  recursive walks would overflow the stack) all route cleanly to Tier C.
- Tier B now runs every cvc5 solve in an ISOLATED child process when a host
  enables it (the `chelis` binary does). Because cvc5 fails by uncatchable
  process abort, the in-process guards above close every KNOWN cause but
  cannot be proven exhaustive; `chelis_prove::worker` makes it moot --
  `solve_property` re-execs a short-lived worker that solves and returns its
  result over a pipe, and ANY way the child can fail (a cvc5 C++ abort, a
  cvc5-internal assertion, a stack overflow, an OOM kill, a panic, a hang
  past the deadline) becomes a clean Tier C result in the parent instead of
  taking `chelis` down. Tests solve in-process (no spawn); the end-to-end
  isolated path, including recovery from a worker that crashes on every
  solve, is locked by the `prove_isolation` integration test.

<!-- opaque-types prove-obligation review-4 fixes -->
- The CLI `.dp` path now runs USER `@property` declarations through the
  shared `chelis_prove::property_runner::run_deep_source_properties` -- the
  SAME engine the tide MCP tool drives (F6) -- so a CLI prove and a tide
  prove of the same Deep module agree on every verdict (including the
  `--tier smt-only` case, which both now report unsupported). The
  c-earchin bridge deep properties keep the CLI-local path with their
  span/requirement rendering. Previously the CLI `.dp` path used a local
  deep runner while tide used the shared one, the parallel-path divergence.
- The deep property path honors the `--tier` contract (F7): `--tier
  smt-only` on a `.dp` module is now unsupported (a Deep body has no
  Surf->SMT lowering path) instead of silently running the fuzz loop and
  reporting a pass.
- Tide and the CLI now render a property's status through one shared
  `is_pass`-bucketed `PropertyOutcome::display_status` (F8). The tide MCP
  JSON previously rendered the raw status, so a zero-sample `Passed`
  sentinel reported "passed" through tide while the CLI emitted
  "unsupported" for the identical outcome. Both surfaces now report the
  same status for every property.
- The int-width to SMT-sort decision is now single source
  (`chelis_prove::opaque::prim_to_smt_sort` / `is_int_width` /
  `INT_WIDTHS`), consulted by every site (the opaque field sort, the
  producer-param sort, the `@property` param sort, the in-module constant
  recognizer, the field value class, and the c-earchin bridge converter)
  (F3). The rework had widened only some sites to int8/int16, so an
  int8/int16 field or param compared against an int8/int16 constant lowered
  as an integer literal against a real-sorted variable and routed to Tier C
  (a regression). int8/int16 opaque fields are now in the value class and
  lower consistently with int32/int64.
- Comparison-operand sort reconciliation is now one shared function
  (`chelis_prove::opaque::reconcile_cmp_operands`) used by BOTH the
  flattened-predicate path and the cvc5-feeding producer-body path
  (`tier_b_lower::lower_pred_bool`) (F4). The producer-body path previously
  had no reconciliation, the parallel-path divergence the unification
  missed. The U3 operand-sort pre-check remains the universal backstop.
- The in-module constant alias chain is now followed to full depth with
  cycle detection instead of a four-hop cap (F5). A longer integer alias
  chain previously dropped silently to a real sort; a cyclic chain now
  terminates instead of mis-resolving.
- The Tier B sort pre-check no longer lets a `min`/`max` over a mixed
  Int-vs-Real operand pair abort the cvc5 process (F1). cvc5 lowers
  `min`/`max` to an internal `ITE(LT(a,b),a,b)` that aborts on mismatched
  operand sorts; the pre-check now clash-checks all `min`/`max` operands
  pairwise (and the two branches of an `ITE`), and infers a `min`/`max`
  result sort from its operands, so a mixed-sort intrinsic routes the
  whole property to a clean Tier-C result rather than crashing the prove
  process. A sound all-Int or all-Real `min`/`max` still proves at Tier B.
- The Tier B sort pre-check no longer spuriously routes `neg` of an
  integer to Tier C (F2). Unary `neg` is represented with a dummy real
  placeholder second operand that cvc5 ignores; the pre-check now skips
  the placeholder, so `neg(int_field)` proves at Tier B.

<!-- opaque-types prove-obligation unification (review 3) -->
- User `@property` verification now runs through ONE shared property
  runner (`chelis_prove::property_runner`) that BOTH the CLI prove path
  and the chelis-tide MCP `chelis_prove` tool drive (U4). The tide tool
  previously gated user-property dispatch on a binding literally named
  `property`, so a property with any other name was silently never proved
  (the response reported ok:true / total:0); it parsed every source with
  the Surf parser even for Deep modules; and it folded a
  zero-sample-validated property (the smt-only timeout sentinel) as a
  pass. The shared runner discovers `@property` declarations the SAME way
  for `.ch` (parse + flatten) and `.dp` (Deep metadata scan) -- with no
  hardcoded name -- runs each through the same Tier B (SMT) -> Tier C
  (fuzz) engine with assumption injection for invariant-carrying opaque
  binders, and reports ok=true ONLY when every property AND obligation is
  a genuine pass (Proved, or statistically validated with samples>0); a
  Disproved/Rejected/unsupported/errored property or obligation, a
  zero-sample sentinel, a parse failure, or a type-check failure makes the
  response not-ok. The CLI renders the shared runner's outcomes as its
  NDJSON property records, so a CLI prove and a tide prove agree on the
  same module (test-locked parity). The CLI's own Surf->SMT lowering and
  injection modules were retired in favour of the single runner.
- Tier B now rejects an operand-sort mismatch (an Int term compared
  against or combined with a Real term) as a clean
  `TierBResult::Error` before any cvc5 term is built (U3). cvc5's
  `mk_term` ABORTS THE PROCESS on a sort-mismatched comparison/op, which
  surfaces to a JSON consumer as an empty-stdout bare exit. U2 makes such
  a mismatch unconstructible on the obligation path; this pre-check is the
  backstop that guarantees ANY mismatch from ANY caller routes to Tier C
  instead of aborting the prove process. The pre-check is conservative
  (an undetermined sort unifies with anything), so it never rejects a
  sound term: consistent all-Int and all-Real properties still solve.
- Module-constant SMT lowering is now ONE type-aware function
  (`chelis_prove::opaque::lower_const_ref`) used by BOTH the
  producer-body path and the invariant/precondition path (U2). The
  invariant-application path previously hardcoded a constant as a Real
  literal, so within one property the SAME constant lowered as an integer
  on the producer body but a Real in the invariant; compared against an
  integer-sorted field var, cvc5 ABORTED the process ("Subexpressions
  must have the same type: Int/Real"). The shared resolver preserves the
  declared numeric type for EVERY integer width (int8/int16/int32/int64
  -> Int; f32/f64 -> Real), reads the authoritative declared return type
  from the constant's `defsig`, and follows a constant whose body
  references another constant transitively to the literal that carries
  the type tag.
- Produced-value invariant validation now routes through ONE chokepoint
  (U1). A produced opaque value -- scalar-only, tensor-bearing, or
  nested-record -- is always validated structurally through
  `validate_produced_env`, which walks EVERY representation leaf and
  rejects fail-closed if any is non-finite (NaN OR Inf) before the
  predicate runs. The historical all-scalar branch evaluated the
  invariant through the host runtime, which never reached a finiteness
  guard, so a scalar NaN field under a `!=`/`not(==)` invariant shipped
  as a PASSING obligation (`NaN != C` is true under strict IEEE). The
  generator's sample validation shares the same finiteness helper
  (`chelis_prove::opaque::any_non_finite`), so a non-finite leaf can no
  longer slip through one path while the other rejects it. The
  host-runtime predicate-eval branch is removed.

<!-- opaque-types prove fixes (review 2) -->
- The `chelis prove` non-SMT obligation warning now fires in every build
  where obligations were not SMT-verified (CR2-5). It was gated on the
  absence of the optional `chelis-prove` dependency, so a
  `chelis-prove`-without-`smt` build -- which compiles the obligation
  machinery and runs it at Tier C (fuzz), not cvc5 -- suppressed the
  warning even though no formal SMT verification occurred. The gate is
  now the actual capability (`not(feature = "smt")`), and the message
  says the obligations were not SMT-verified. It remains stderr-only and
  never touches the stdout NDJSON stream or the exit code.

- Tier B constant inlining now preserves a constant's declared numeric
  type (CR2-4). An int-typed module constant
  (`def n() -> int32 = 3`, or the value binding `n = 3`) used in a
  producer guard was inlined as an f32 literal -- silently retyped --
  because the resolved-constant environment carries only an `f64`. The
  inliner now reads the constant's declared literal type from the module
  and emits an integer literal for an `int32`/`int64` constant (so it
  lowers to `SmtSort::Int`) while keeping the f32 path for float
  constants.

- The chelis-tide `chelis_prove` MCP response now folds every non-pass
  property status into `ok:false` (CR2-3). Only `Proved` and
  `StatisticallyValidated` are passes; `Disproved`, `Rejected`, and
  `NotAmenable` now lower the response (previously only `Disproved` did,
  so a rejected or non-amenable property reported `ok:true`). The
  non-pass status is also bucketed into the summary by kind (Disproved
  => failed, NotAmenable => unsupported, Rejected => error), matching the
  CLI's status mapping. As part of this, the tool no longer emits a
  phantom `unbound variable: property` rejection for a module that
  defines no `property` binding: the user-property dispatch runs only
  when the module actually defines `property`, so a clean
  obligation-only module reports `ok:true` and an empty `properties`
  array.

- Invariant validation now fails closed on a non-finite (NaN/Inf)
  representation field (CR2-2). The strict evaluator gives `NaN != C`
  as `true` under IEEE, so a `!=`/negation-shaped invariant accepted a
  NaN tensor field fail-OPEN (the obligation reported `passed`). Both
  validation chokepoints -- the producer-obligation path
  (`validate_with_predicate` in the obligation engine) and the
  injection-sampling path (`validate_env` in `opaque`) -- now reject any
  non-finite field unconditionally, before the predicate runs, because a
  non-finite value is never a valid inhabitant of the opaque domain
  regardless of the predicate's shape.

- Tier B SMT lowering no longer panics on `log` (CR2-1). `log` is a
  valid predicate intrinsic that the concrete evaluator (Tier C)
  supports, but cvc5 has no LOG kind, so the lowering had no arm for it
  and `panic!`ed -- a grammar-admitted `log(x)` predicate could crash
  the prove process under `--features smt`. The cvc5-lowerable intrinsic
  set is now one constant (`CVC5_LOWERABLE`, plus the transcendental
  subset that selects `QF_NRAT`); the arity guard, the lowering match,
  and the transcendental-logic classifier all derive from it, so a
  function admitted by one can no longer diverge from another. A
  whitelisted-but-not-lowerable name (`log`) is rejected as a clean
  `TierBResult::Error` so dispatch falls through to Tier C, and
  `lower_to_cvc5` now returns `Result` (defense in depth) rather than
  panicking on an unhandled name.

- The advisory `opaque-domain-construction` lint now keys opaque types
  by (type, defining module) instead of bare leaf name (CR-9). A
  non-opaque type that shares a leaf name with an opaque type in an
  unrelated module is no longer falsely flagged when constructed in
  its own module, and a same-leaf opaque type can no longer suppress
  the check; the genuine out-of-module forge is still flagged. The
  authoritative checker enforcement was already correct -- this is a
  fast-feedback lint-quality fix.
- The `opaque-domain-construction` lint no longer collapses distinct
  module-less files into one shared shadow bucket (CR2-7). Across a
  corpus walk, a top-level (module-less) `type L` in one file used to
  falsely suppress the forge lint for a DIFFERENT module-less file
  constructing an opaque `L`; the module-less local-type shadow is now
  scoped to the currently-checked file, while named-module shadows
  stay corpus-wide so a module split across files still resolves.
- The `opaque-domain-construction` lint now catalogs opaque types only
  from a NAMED module (CR3). `@opaque` requires a named enclosing
  module -- the checker rejects a module-less `@opaque` as a
  declaration error -- so a module-less opaque type keyed to the shared
  `None` module, collapsing distinct module-less files and falsely
  flagging a same-leaf construction in an unrelated module against an
  invalid declaration. The lint now defers the invalid module-less
  `@opaque` to the checker.

<!-- opaque-types RT-3 soundness fixes (prove layer) -->
- **RT3-F1 (CRITICAL, D-PRODUCER / D-SOUND)**: a record field spelled with
  a type alias of an invariant-carrying opaque type silently defeated the
  producer set (`type TA = T; type Wrapper = { inner: TA }`), so a
  violating value escaped the module boundary unobligated and `chelis
  prove` returned exit 0. The record-field producer path read field types
  syntactically from Deep without resolving aliases; it now resolves
  `(typealias ...)` chains so a record reaching the opaque type through any
  alias spelling is covered-or-rejected, identical to the direct-type case.
  This also closes the same-root unsound injection-pass (a property over
  the type passing while the producer escapes).
- **RT3-F2 (HIGH, D-PRODUCER)**: a module with an unrelated type error
  passed `chelis prove` with exit 0, wiping the checker-inferred signatures
  (empty producer set) and hiding a rejectable producer. `chelis prove`
  (and the chelis-tide `chelis_prove` tool) now surface the check
  diagnostics and report an Error (exit 3) on a type-broken module; a
  `prove` exit 0 warrants the module type-checked and every obligation was
  discharged (the strict downstream prove-compat / FlukeBall guarantee).
- **RT3-F3 (MEDIUM, D-STARVE)**: constructor-based generation false-starved
  for tight SCALAR-field invariant bands because the producer-result reader
  handled only tensor values; a rank-0 scalar field access returns
  `Float64`/`Int64`/`Bool`, which was dropped. Scalar field reads now
  extract the scalar value, so a producer that always lands in a tight
  scalar band serves the binder instead of being reported `unsupported`.
- **RT3-F4 (verification, D-CHECK)**: confirmed declaration errors
  (`OpaqueTypeViolation`, amenability mismatch) are visible on `chelis
  check` (non-zero exit with the error listed) and gate `chelis build` /
  `chelis eval --file`; `chelis validate` is a structural validator that
  does not run the type/opacity checker. Documented in spec/04 §2.5.
<!-- end opaque-types RT-3 soundness fixes -->

<!-- opaque-types xhigh-review fixes (prove layer) -->
- **CR-15 (HIGH, D-PRODUCER)**: a producer returning a MULTI-VARIANT ADT
  whose non-first variant wrapped the opaque type was silently missed
  (the record-field collector read only the first variant). All variant
  payloads (record fields and positional types) are now collected, so any
  variant wrapping the type is covered-or-rejected.
- **CR-1 / CR-4 / CR-6 (HIGH, D-OBLIG)**: the tensor-field obligation
  Tier C verdict ignored the inner produced position for `Option[(T, f32)]`
  / nested Option (reading record fields off a tuple) and used a NaN-filled
  record as the None sentinel (so a `Some(record)` with a legitimate NaN
  representation passed VACUOUSLY). The produced value is now validated
  structurally over the `ExecutionValue` tree (Option as `Adt{Some|None}`,
  tuples as `Tuple`, records as `Adt`), applying the inner position and
  using the real None discriminant; a NaN representation fails the
  invariant, fail-closed.
- **CR-2 / CR-5 / CR-10 (MEDIUM-HIGH, D-STARVE)**: invariant-sample
  validation used the fuzz evaluator's `1e-10`-tolerant `==`/`!=`,
  contradicting the strict-acceptance contract. Validation now uses
  `eval_bool_strict` (exact IEEE `==`/`!=`); the fuzz tolerance stays on
  the user-property postcondition path.
- **CR-8 (MEDIUM, D-TIERB)**: a Tier B producer guard comparing against an
  in-module zero-arg constant lowered with the constant unresolved (and
  paniced the solver on the undeclared variable for a value binding). The
  Tier B `reduce` pass now resolves module constants; the engine resolves
  all in-module zero-arg scalar defs.
- **CR-13 (robustness)**: the unary intrinsics in `concrete_eval` indexed
  `a[0]` with no arity guard, so a malformed zero-arg `exp()` paniced;
  they now guard `len == 1` and yield `NaN` on wrong arity.
- **CR-12 (HIGH, D-PARITY)**: the chelis-tide `chelis_prove` tool
  hardcoded `ok: true` and derived its summary only from the single user
  property, so a failed/unsupported/errored producer obligation never
  lowered the MCP response. The obligation outcomes are now folded into the
  response `ok`/summary exactly as the CLI folds them into the prove exit
  status.
- **CR-14 (cleanup, D-PRODUCER)**: a record field of type `&T` was mapped
  to an inert placeholder, hiding the borrowed opaque type; `t-ref` now
  recurses so a borrow of the type is covered-or-rejected.
- **CR-11 (cleanup, RFC L5)**: inlineability consumes
  `chelis_pred::INTRINSIC_WHITELIST` instead of a duplicate array.
<!-- end opaque-types xhigh-review fixes -->

<!-- opaque-types RT-5 final fixes (prove layer) -->
- **RT5-F1 (HIGH, completes CR-13)**: the Tier B SMT lowering applied no
  arity check to the unary intrinsics (`exp`/`log`/`sqrt`/`sin`/`cos`/
  `abs`), so a wrong-arity transcendental -- a zero-arg `exp()` or a
  "valid-looking" two-arg `exp(a, b)` (the checker treats `exp` as
  variadic) -- built an invalid cvc5 term that aborted the solver with
  empty stdout and bare exit 1, a machine-contract violation for `prove
  --json` consumers. The SMT lowering now arity-validates every intrinsic
  application before building any cvc5 term and routes a wrong-arity or
  unsupported application to a clean `TierBResult::Error` (mapped to an
  obligation `unsupported`/Tier-C fallback), never letting a bad term reach
  cvc5. CR-13 had fixed only the parallel concrete-eval path.
- **test hygiene**: two `chelis prove` CLI tests asserted fuzz-tier
  behavior and so failed under `--features smt` (where a trivial property
  auto-proves at the SMT tier); they now pin `--tier fuzz-only`. The
  solver-free corpus gate (`zero cvc5 symbols`) is skipped under
  `--features smt` (where the binary links cvc5 by design), keeping the
  load-bearing default-build assertion intact.
<!-- end opaque-types RT-5 final fixes -->

<!-- opaque-types: runtime IR audit skips declaration metadata -->
- The runtime IR audits (`assert_ir_typed` / `assert_ir_lowerable` in
  `crates/chelis-ir/src/lower.rs`) no longer descend into the metadata of
  declaration nodes (`deftype` / `defsig` / `typealias`). A `deftype`'s
  declared `@invariant` predicate lives in that metadata map and is spec
  metadata consumed only by `chelis prove`; it is never lowered to runtime
  IR (the main `lower_top_level` already returns early for those tags). The
  audit walked into it anyway and rejected a tensor-field invariant such as
  `sum(p.weights)` with "shape-sensitive IR app nodes must carry explicit
  type metadata before lowering", so `chelis eval --file` and (on some
  paths) `chelis build` failed on an opaque type whose invariant is over a
  tensor field, while the equivalent scalar-field invariant passed only by
  accident. The audits now mirror the lowering skip for these declaration
  tags; genuine runtime `def` bodies are still audited. With the fix, the
  `Simplex` tolerance-band example `eval`/`build`s cleanly and is promoted
  from `examples/illustrative/` to `examples/opaque_invariants_simplex.ch`
  (library-only-executable, like `Probability`). The §2.3 span-coverage
  audit invariant is unaffected: declaration metadata carries no input def
  body spans.
<!-- end opaque-types runtime IR audit -->

### Added

<!-- opaque-types W3/W4 (prove layer) -->
- Derived producer obligations for invariant-carrying opaque types in
  `chelis prove` and the chelis-tide `chelis_prove` MCP tool (design
  record `spec/design/opaque_invariants_rfc.md` D-PRODUCER / D-OBLIG /
  D-TIERB / D-PARITY / D-STARVE; record schema in
  `spec/design/chelis_property_spec.md`). For every exported producer of
  an `@opaque @invariant(...)` type, `chelis prove` synthesizes and
  discharges a `for all inputs, where the producer succeeds, the
  invariant holds` obligation through the existing three-tier dispatch:
  the canonical guard-then-`Option` constructor proves at the SMT tier
  (`proof_tier:"smt"`) via new Tier B record beta-reduction and
  case-of-known-constructor reduction; other producers fall to validated
  sampling (`proof_tier:"fuzz"`). The producer set is computed from
  checker-**inferred** return types (an unannotated exported def cannot
  escape it) and is covered-or-rejected: a return reaching the type
  through an unsupported container (list/record/function/other generic),
  or an exported signature handing caller-supplied code an unobligated
  value, is a declaration error naming the producer and channel. Additive
  NDJSON `{kind:"obligation", ...}` records and a summary `obligations`
  count; `ProofArtifact` gains a serde-additive `obligation` field
  (strict downstream admission parsers must add the new record kind at
  pin time). SMT-tier artifacts carry `arith_model:"real"` and the
  spec documents that Tier B proves over the reals, not floats. New
  `--invariant-min-rate` flag (D-STARVE floor) on `chelis prove`. The
  obligation collection, synthesis, lowering, and execution live in
  `chelis-prove` and are shared by the CLI and tide (locked by a
  cross-surface parity test). Oracle:
  `crates/chelis-cli/tests/prove_invariant_obligations.rs` (run with
  `--features smt`).
<!-- end opaque-types W3/W4 -->

<!-- opaque-types W4 (injection + starvation) -->
- Assumption injection and tiered validated generation for
  invariant-carrying opaque types (`spec/design/opaque_invariants_rfc.md`
  D-INJECT / D-STARVE / D-SOUND). A `@property` binder whose type is an
  invariant-carrying opaque type is verified only over
  invariant-satisfying binder values, generated by a tiered generator
  (rejection sampling, then constructor-based generation that evaluates an
  exported producer on sampled raw inputs); every accepted sample is
  predicate-validated. Update-shaped producers (an opaque input parameter)
  are now supported: their input invariant is injected as a Tier B
  precondition, so an invariant-preserving update proves at the SMT tier
  GIVEN the input assumption (the D-SOUND inductive step) and an unguarded
  twin is disproved. Tensor-input producers and the tolerance-band simplex
  obligation (`sum(p.weights)` within `1.0 +/- eps`) are now supported
  rather than reported `unsupported`. When both generation tiers starve
  below the `--invariant-min-rate` floor (default `0.01`) the property /
  obligation is `unsupported` (exit 2) with a generator-starvation
  diagnostic naming the type, per-method accepted/attempted counts, the
  rate, floor, predicate-shape classification (equality-atoms vs
  band-width), and the recommended route; `--invariant-min-rate 0.0`
  disables the classifier and restores the legacy exhaustion (`error`)
  path. The producer-set covered-or-rejected rule now resolves named
  record wrappers: a producer returning a non-generic record that wraps
  the opaque type in a field is a declaration error, never a silent miss.
  Oracle: `crates/chelis-cli/tests/prove_injection_starvation.rs` (run
  with `--features smt`).
<!-- end opaque-types W4 -->

<!-- opaque-types W7 (differential corpus + solver-free gate) -->
- Opaque-invariants differential corpus + solver-free regression gate
  (design record `spec/design/opaque_invariants_rfc.md` D-CORPUS). An
  in-tree, mechanically generated corpus
  (`tests/corpus/opaque_invariants/`) exercises the shipped opaque-types
  rejection, well-formedness, obligation, injection, and
  generator-starvation paths across two lanes: a `check` lane (default
  non-smt binary) and a `prove` lane (`--features smt`). Diagnostic
  coverage is MEASURED, not assumed: `coverage_runner.py` runs every
  program against the live binary and fails the gate if any targeted
  `CheckErrorKind` / well-formedness-message / obligation-status token is
  hit by zero programs, and pins each program's exit code. The central
  deliverable is an executable solver-free regression
  (`solver_free.py`): the default `chelis` binary links zero cvc5 symbols
  (the `--features smt` build links ≈ 33k, a discriminating control),
  `chelis check`s the whole corpus to its pinned exits, and produces
  byte-identical check verdicts to the smt build — proving `chelis check`
  reaches no solver. Oracle:
  `crates/chelis-cli/tests/opaque_corpus_gate.rs` (default gate: solver-free
  + check-lane coverage) and `crates/chelis-cli/tests/opaque_corpus_smt.rs`
  (full both-lane coverage + prove performance sanity, run with
  `--features smt`). The cross-build check-identity arm and Hull-side
  reference support are documented out-of-default-gate follow-ups in the
  corpus README.
<!-- end opaque-types W7 -->

<!-- opaque-types W6 (docs + worked example) -->
- Worked opaque-invariants example and language-book documentation
  (`spec/design/opaque_invariants_rfc.md`). `examples/opaque_invariants.ch`
  is an executable `Probability` unit-interval type: a guard-then-`Option`
  base constructor plus two update-shaped producers, whose three derived
  obligations all discharge at the SMT tier, and an injected property. The
  tolerance-band `Simplex` companion
  `examples/opaque_invariants_simplex.ch` checks and proves clean (its
  producer obligation at Tier C, its binder served by constructor-based
  generation) and is library-only-executable like `Probability`: its
  `sum`-over-a-tensor-field invariant is declaration metadata for
  `chelis prove`, never lowered to runtime IR, so it `eval`/`build`s clean.
  New book chapter `docs/book/src/opaque-invariants.md` teaches the
  declare-invariant-export-prove workflow against real `chelis prove
  --json` output, including a prominent "What this feature does NOT do"
  section (no refinement typing, the invariant is invisible to `chelis
  check`, the argument-egress trust caveat). Oracle:
  `crates/chelis-cli/tests/opaque_invariants_example.rs` (run with
  `--features smt`).
- Schema-doc consolidation in `spec/design/chelis_property_spec.md`:
  corrected the obligation `counterexample` example to the shipped
  positional-placeholder keying (`__arg0` rather than the source
  parameter name) and documented the invariant-binder starvation path
  (the `--invariant-min-rate` floor and the `unsupported`-vs-`Error`
  distinction from user-precondition exhaustion).
<!-- end opaque-types W6 -->

- Checker-enforced opaque types (`@opaque`, renamed from the
  unreleased baseline's `@chelis_opaque`; design record
  `spec/design/opaque_invariants_rfc.md`, normative spec
  `spec/04-type-system.md` §2.5). A `deftype` carrying `opaque: true`
  metadata is constructible and inspectable only inside its defining
  module: record literals, positional constructor application, bare
  constructor references, `pat-record`/`pat-ctor` patterns, field
  access, Deep `record-update`, casts into and out of the type, and
  `{type: (t-adt ...)}` literal ascriptions are rejected outside it
  with the new `CheckErrorKind::OpaqueTypeViolation`, as are
  out-of-module references to unexported bindings of the defining
  module whose signatures mention the type. `@opaque` requires a
  named enclosing module. Module identity covers both encodings
  (lexical `module` wrappers and reef package-linked internal
  names); reef now preserves `export` decls (with internal names)
  through the rewrite so the checker can recover package export
  sets. `record`/`access`/`record-update` are now actually inferred
  (previously silently untyped), closing the latent bogus-field
  hole, and a top-level irrefutable match arm (`| x =>`,
  `| q @ x =>`) now covers the match. The
  `opaque-domain-construction` lint rule is retained as
  defense-in-depth. Compiled-context cache formats bumped
  (`CHELIS_CTX_V5`, stdlib cache v2) for the registry shape change.
  Re-opening a named module with a second `(module ...)` wrapper in
  one check unit is a `DuplicateModule` error (rejected by `check`,
  `build`, and `validate --deep`); it would otherwise forge the
  defining module's identity. Hand-authored programs (raw `.ch` /
  `.dp`) may not use the reef linker's reserved internal-name format
  (`Pkg__<pkg>__<Module>__<Name>`) for declaration names -- it is the
  linker's private output and forging it bypasses module identity
  (`ReservedLinkerName`, gated by an in-process link-provenance flag
  so genuine reef packages still check clean). `chelis test`,
  `chelis eval`'s in-context path, and the batch/module-init test
  paths -- which keep user-authored entry/test decl names through the
  eval rewrite (so roots resolve by name) -- reject the reserved
  format at the entry-decl rewrite boundary directly, so a forged
  mangled test def can no longer self-key to a victim module. The
  sixth rejection now also fires for
  bare un-imported cross-module references in reef packages (the
  reference is canonicalized to its binding key the way inference
  resolves it).
  Reef-surface violation messages render user-facing (de-mangled)
  type/module/producer/def names instead of the internal
  `Pkg__<pkg>__<Module>__<Name>` forms.
  The decompiler (`chelis surf`) round-trips opaque modules: it
  PascalCases every module-path segment (`module Stats.Prob`,
  not `Stats.prob`), renders record variants with braces
  (`| Probability { value: f32 }`), and uses `:` (not `=`) in
  record construction.

- Declared invariants on opaque types (`@invariant(<binder>) <expr>`,
  design record `spec/design/opaque_invariants_rfc.md` D-SYNTAX /
  D-META / D-WF / D-PRED; normative spec `spec/02-surf-syntax.md`
  §P16a, `spec/03-deep-syntax.md` §2.2, `spec/04-type-system.md`
  §2.5.1). An `@opaque` type may carry one boolean predicate over a
  single representation binder, declared between `@opaque` and `type`.
  It desugars to the additive Deep metadata keys `invariant`
  (the predicate as a `(fn {} (params {} <binder>) <body>)` node) and
  `invariant_amenability` (`"linear"`/`"polynomial"`/`"transcendental"`/
  `"opaque"`, derived data recomputed on every desugar). The new
  `chelis-pred` leaf crate (`PredAmenability`, `classify_predicate`,
  `predicate_free_vars`, `predicate_in_grammar`, `INTRINSIC_WHITELIST`)
  is the predicate grammar and amenability classifier, consumable by
  `chelis-surf`, `chelis-types`, and `chelis-prove` with no dependency
  cycle. The checker runs a declaration-time well-formedness pass
  (covering `.ch` and `.dp`): invariant requires `@opaque`; exactly one
  record-shaped variant with every field in the V1 value class; the
  predicate is in grammar with free vars scoped to the binder and
  in-module constants and boolean-shaped at the top; and the recorded
  amenability must match a recomputation (protecting hand-written
  `.dp`). The invariant is invisible to type checking -- it is recorded,
  never evaluated, so a program whose invariant is violated by an
  in-module constructor still type-checks (this is not refinement
  typing; the invariant is consumed by `chelis prove` in later
  workstreams). `chelis fmt` and the decompiler round-trip the
  `@invariant` line (the amenability key is derived and not decompiled).
  Two advisory (non-blocking) lint rules ship: `opaque-without-invariant`
  and `invariant-float-equality` (exact `==` over a representation field
  starves generation by design; use a tolerance band).

- **W5 — decode revalidation (`opaque_invariants_rfc.md` D-DECODE).**
  Experimental public decode chokepoint
  `chelis_compiler_api::decode_adt_value` (and the typed-error variant
  `try_decode_adt_value`, returning `DecodeError`). It converts an
  `ExecutionValue::Adt` payload to a `RuntimeValue` with a structural
  check (constructor declared; field arity/order/type match the declared
  representation) and then revalidates the declared invariant by
  evaluating the predicate through the evaluator's own interpreter --
  `chelis-compiler-api` does not depend on `chelis-prove`. A NaN or
  non-finite value in any numeric representation field is rejected
  *before* predicate evaluation (fail-closed; `not (p.value > 1.0)` is
  true on NaN). A structural mismatch is reported distinctly from an
  invariant violation, both in the `DecodeError` enum and in the message
  prefix. Decode of a violating payload is a failure, never a repair (no
  clamping). The normative rule is documented in `spec/10-serialization.md`
  §4 with the cross-link to `spec/01-nomenclature.md` §12.1. V1 reality
  (survey §7): no external ADT-value payload codec exists yet
  (`EvalRequest.bindings` is tensors-only; `ExecutionValue::Adt` is
  output-only), so the chokepoint ships experimental with the conformance
  suite (`crates/chelis-compiler-api/tests/invariant_decode.rs`) as its
  only caller. The runtime value type `RuntimeValue` is now exported from
  the crate root for the chokepoint's return type. (CR-3) Decode now
  collects the module's in-module zero-argument constant defs into the
  predicate evaluation context, so an invariant referencing a constant
  (e.g. a tolerance band `p.value <= 1.0 + eps` over `def eps() -> f32 =
  0.001`, permitted by the D-WF grammar) resolves the constant instead of
  failing on an unknown name; a valid payload of such a type now decodes,
  and a violating one is still rejected. (CR2-6) The constant collector now
  also handles the bare value-binding Deep form `(def {} eps <lit>)` (body
  is the value directly, not a `(fn {} (params {}) ...)` wrapper). The Surf
  desugarer always wraps def bodies in `fn`, but hand-authored Deep -- which
  the chokepoint accepts and `validate --deep` admits -- can declare a
  constant in the bare form, so a `.dp` invariant referencing it now
  resolves instead of failing on an unknown name. (Review-3) The
  bare-value-binding arm is restricted to genuinely constant-foldable
  bodies (a literal, a predicate-grammar arithmetic/intrinsic/comparison/
  boolean application over constant arguments, or a reference to another
  genuine constant); a value-binding whose body is a `(var ...)` to a
  non-constant or unbound name, or any other non-foldable expression, is
  no longer registered, so a non-constant binding cannot pollute the
  decode constant table or mis-resolve.

## [0.7.25] — 2026-06-11

### Fixed

- **`chelis eval` named-axis reductions (#338, PR #346)**: the host
  runtime routes named-axis work through the IR lowering + forward-DAG
  lane (the grad/vmap machinery), restoring the eval-vs-backend
  agreement invariant for the Tier-3 surface. The Tier-3 corpus now
  runs the full agreement oracle at ranks 2/3/4 with non-square
  operands plus a parity-corners suite (List params, top-level
  reductions, let-blocks, closure aliases, pipes, scalar returns,
  grad over named-reduce defs).
- **Elementwise output types in rank-polymorphic inline bodies
  (PR #346)**: unary and Tier-2 elementwise lowering arms took their
  output type from body annotations whose symbolic dims survive
  rank-poly inlining unsubstituted; `sum(exp(x), seq)`-class bodies
  silently miscompiled on BOTH lanes (C backend garbage since #337,
  wrong-axis eval numerics). All arms now derive output dims from the
  lowered operand (`elementwise_out_ty`), pinned executable.
- **Grad through symbolic-dim sig wrappers (#345, PR #363)**: the
  0.7.24 regression ICE ("symbolic dim referenced by a non-Load node")
  is fixed and pinned by an 8-row regression matrix (sig/inline/
  quantifier/shim wrappers, gelu, rank-2) verified failing on the
  0.7.24 baseline. The `dag.rs` guard now also scans op-internal
  symbolic refs (`Expand::size`, `Reshape::new_shape`, `BlasMatmul`
  dims), which immediately caught and fixed a stale-`Sym` producer in
  `actualize_tensor_helper_types`. Downstream: unblocks School from
  its `=0.7.23` pin (re-probe of the #318/#319/#320 surfaces).

### Added

- **`gate.py --local` (#360, PR #362)**: official local/CI gate split.
  Local pre-push runs workspace clippy, fmt, `chelis lint`, and
  per-crate nextest derived from changed paths (manifest-accurate
  package mapping); the workspace suite is CI-owned (macOS Smoke
  authoritative).
- **Downstream shell-repo contract (PR #361)**:
  `spec/design/shell_repo_contract.md`, normative for every shell in
  the ecosystem table (pin hygiene with offline consistency guards,
  `CHELIS_SURFACE.md` capability inventories, upstream-issue
  discipline with the narrowing-citation rule, expected-to-fail
  blocker probes, negative-test sidecars, pin-bump checklist), with
  `Chelis-Lang/school` as the reference implementation.
- **Dev-environment tooling (#348/#349/#356, PRs #354/#355/#358)**:
  build-concurrency contract + `scripts/reap_orphans.py`; macOS
  first-exec assessment runbook (`docs/local_macos_environment.md`)
  + `scripts/preflight_exec_probe.py` with slow-admission detection
  (exit 3); verified Developer Tools exemption as the durable
  workstation fix.

### Changed

- **`chelis-compiler-api` runtime split (#350, PR #357)**: the
  8,018-line `runtime.rs` is now `runtime/{mod,eval,named_axis,
  transforms,host_ops,tests}.rs`; mechanical move (verified by
  line-multiset proof), no behavior change.

## [0.7.24] — 2026-06-09

### Added

- **Tier-2 rank polymorphism — identity tier (#258, #286)**: `..r`
  rank-variable spread syntax (`d-rank` Deep node), `Dim::Rank` with
  unitary rank unification, and the Body Discipline, so a single `def`
  is generic over tensor rank for identity/erasure shapes (the
  relu/silu/gelu activation family and full reduces as one def each).
- **Tier-3 name-preserving rank polymorphism — named-axis reduction
  (#258, #337)**: a single `def` reduces a named axis at any rank and
  the surviving named axes carry through symbolically, e.g.
  `def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) ->
  tensor[..pre, ..post, f32] = sum(x, seq)`. Includes call-site rank
  monomorphization and named-axis lowering on the C backend.
- **Module-qualified constructor references and patterns (#316, #321,
  #322)**: `Mod.Ctor` parses and resolves in expressions and in match
  patterns, completing the #157 cross-module constructor surface.
- **glibc 2.31 linux-x86_64 release variant (#330)**: the release
  workflow now also builds `chelis-v<ver>-linux-x86_64-glibc2.31.tar.gz`
  in a Debian 11 container, so the linux binary loads on any glibc >=
  2.31. The default ubuntu-latest build carries a hard `GLIBC_2.39`
  verneed record and fails to load on older distros (Debian 11/12,
  Ubuntu 20.04/22.04).

### Changed

- **chelis-std 0.4.0 — ML modules cut to School (#331)**: the
  `Std.Nn.*`, `Std.Loss.*`, `Std.Optim`, and `Std.Schedule` modules are
  removed from chelis-std; they live on in the standalone School ML
  library under the `School.*` prefix. Init/Io/Tensor/Sort/Scan/
  Process/Decimal/Test/Time/Tokenizer are unchanged. The bundle
  embedded in the chelis binary is regenerated at 0.4.0.
- **CI (#332, #328)**: the chelis-std self-test corpus now runs
  nightly, and a red nightly opens a `nightly-failure` tracking issue
  (closed again on green); Linux jobs reclaim runner disk to stop
  intermittent link/build failures.

### Fixed

- **Out-of-scope cross-module constructor is a check-time error (#317,
  #327)**: referencing another module's constructor without importing
  it — at construction sites, applied heads, record literals, and match
  patterns — is rejected with `UnknownConstructor` at check instead of
  silently mis-resolving through the terminal-segment fallback and
  surfacing as a runtime non-exhaustive match.
- **grad through a cross-module precision-polymorphic attention verb
  (#319, #326)**: separate-`sig` def bodies now infer against the
  declared sig param types (not bare type variables), and renamed body
  precision variables bind to the call-site precision only when the
  call is fully precision-monomorphic, preserving the
  no-implicit-promotion invariant.
- **grad through windowed mean / max_reduce / gather (#320, #325)**:
  runtime-extent mean divisor and collapsed-operand rank recovery.
- **grad backward for shape-derived expand-of-scalar broadcast (#318,
  #324)**.
- **chelis-std softmax self-test rank mismatch (#332)**:
  `test_softmax_normalizes_to_one` compared a rank-0 sum against a
  rank-1 expectation, failing the file's typecheck; the unimplementable
  `Std.Tensor.Reduce` self-test is quarantined as #333.

## [0.7.23] — 2026-06-04

### Added

- **Module-scoped constructor resolution (#157)**: the reef linker now
  mangles, rewrites, and exports ADT constructor names as module
  symbols, so a downstream program can import a constructor across
  modules (e.g. `import Std.Io.Json (JsonInt)`) and pattern-match on it.
- **chelis-std bundle regenerated to expose std constructors
  cross-module (#311)**: the prebuilt bundle now carries the
  constructor-mangled, exported symbols from #157, with a cross-module
  constructor acceptance test wired as a manual gate.

### Fixed

- **grad-tuple projection receiver in the C backend (#309)**: a
  multi-`wrt` `grad(f)(x, w)` projected with `.N` now reads from a
  correctly-typed `chelis_tuple` instead of emitting `chelis_tuple_get`
  over a `chelis_tensor*`. Multi-root grad helper calls are typed as a
  tuple sized from the helper's actual root count; single-root and
  zero-root helpers are unchanged.
- **Zero-length arrays for nullary ADT variants and empty tuples
  (#310)**: `assign_adt_construct` and `assign_tuple_literal` no longer
  emit ISO-C-illegal `chelis_value[0]` arrays; they pass `NULL` with
  count 0, mirroring the list and tensor-helper guards.
- **const-broadcast tensor binding (#300)**: the C backend emits valid C
  for a constant-only broadcast tensor binding in a scalar-return body.
- **grad backward for shrink and stride movement ops (#291)**: the
  reverse pass now lowers `shrink` and `stride`.
- **grad backward through the expand-of-scalar constant idiom (#288)**.
- **precision-variable monomorphization through the grad sub-context
  (#289)**: precision type variables are now resolved when lowering a
  grad call.
- **non-literal reduce/expand axis diagnostic (#259)**: a targeted error
  replaces a confusing failure when a reduction or expand axis is not a
  literal.
- **PascalCase `def` quantifier names (#293)**: promoted to type
  variables during desugaring rather than being treated as types.
- **function-typed arrow-argument grouping parens (#290)**: the
  formatter and printers preserve parentheses around function-typed
  arguments in arrow types.
- **Deep `comment` rule (#167)**: made atomic so leading `;` comment
  lines parse.
- **reef metadata-step 404 disambiguation (#147)**: a metadata-step 404
  is distinguished from an auth-privacy 404.
- **Lossy float emission in HIP / C backend code generation (#250, #251,
  #252)**: three remaining `%.8`-style decimal format strings emitted
  float literals into generated HIP / C source that could not round-trip
  to their exact bit pattern, mirroring the failure mode PR #243 (#189)
  and PR #249 (#248) already fixed in the C backend's production emit
  paths. The HIP `emit_const` F32 fill (#250) and the `uniform_like`
  `low` / `high` args (#251) now narrow to f32 and reconstruct each value
  from its exact bit pattern via the `chelis_f32_from_bits` static inline
  helper (with an `f32::to_bits()` / `f64::to_bits()` codegen step); the
  HIP F64 const fill routes through `chelis_f64_from_bits`. The C backend
  test-harness input fill (#252) is migrated to the same bit-pattern
  emission. Denormals such as `1e-40` no longer collapse to `0.0f`, and
  values like `0.1f32` / `1.0 / 3.0` round-trip to identical bits. The
  same sibling-sweep also closes the remaining `%.8` / `{:.17e}` decimal
  fills in the C, HIP, and Metal correctness harnesses (the gcc-gated
  SIMD-reduction static-array initializer, which now emits exact C99
  hexadecimal-float constants; the C backend's Sleef fused/single-op
  test-harness fills; and the HIP / Metal / cross-library GPU-gated
  driver fills) so the generated device input is byte-identical to the
  Rust reference the bit-exact and tight-ULP assertions compare against.

## [0.7.22] — 2026-06-03

### Added

- **reduce_window_{max,min,sum,mean} (#254)**: windowed reductions land
  across the type checker, IR, and backends as first-class RISC
  primitives with positive and negative coverage.
- **host-runtime `tensor_scan` (#257/#264)**: a prefix-scan primitive
  implemented in the host runtime, closing issue #257.
- **Hull Phase 5b conformance gate (#278)**: a vendored, frozen,
  version-stamped Hull conformance corpus
  (`tests/conformance/hull/`) plus a CI gate (`conformance.yml`) that
  re-renders the corpus against the just-built compiler and FAILS on
  any `CompilerUnsound`, unexplained `Disagree`, or `EvalDisagree`. The
  manifest now pins chelis `0.7.22`. Hull is documented as shipped
  (v0.1.0) as the compiler-vs-spec differential layer (#283).
- **SMT-prove tier features (`chelis prove`)**: `--tier auto` now wires
  through SMT Tier B, function-body inlining feeds the SMT dispatch,
  and the dispatcher selects `QF_NRA` vs `QF_NRAT` based on
  transcendental content of the goal.
- `chelis check <file>.dp` and `chelis eval --file <file>.dp` now ingest
  standalone Deep (`.dp`) IR directly. Previously both fed Deep
  s-expressions to the Surf parser, which reported a bogus
  `expected declaration (def, sig, ...), found LParen at byte 0` parse
  error (check) or propagated it as a process error (eval). `check`
  parses `.dp` through `chelis_deep::parser::parse_str_strict` (the same
  closed-vocabulary tag gate `build`, `fmt`, and `cost` use) and runs
  the identical fitness / type / effect / linearity pipeline the `.ch`
  path uses, emitting the same `CheckResult` JSON. `eval --file .dp`
  routes the source through the engine's `SourceKind::Deep` path
  (strict-parsed first for vocabulary parity), emitting the same
  `EvalResult` shape. A `.dp` and its byte-equivalent-meaning `.ch`
  produce field-for-field identical check reports. `chelis check <dir>`
  now also discovers `.dp` files in a mixed directory.
- `chelis test --batch-mode auto|file`, with `auto` as the default.
  Auto mode compiles one shared context and one combined batch probe
  for eligible files, then reports the same NDJSON/per-test rows as
  the file-worker path. The default now amortizes shell/module graph
  compilation across a suite: the CLI batches eligible test files into
  one internal worker, preserves per-file subprocess isolation for
  files that need it, and falls back to the previous
  one-worker-per-file path on batch compile/runtime failures. Use
  `--batch-mode file` to force the old execution strategy.
- Internal `__test_batch` worker support and crash/timeout fallback
  coverage so batch execution cannot hide a file-level failure.

### Fixed

- **#285**: suppress a wildcard `defsig` from overwriting an explicit
  `sig`, preserve the `def` effect-row, and fix the std `generate`
  signatures.
- **#258**: reject a duplicate same-name `def` with a clear diagnostic
  instead of silently shadowing the first definition.
- **#248 (#189 follow-up)**: the C backend's `emit_uniform_like` was
  the third lossy `%.8`-format-string site in the same class as the
  F32/F64 `emit_const` arms that PR #243 closed. `low` / `high` were
  narrowed to f32 and then baked into the emitted
  `chelis_uniform_sample_f32(..., {:.8}f, {:.8}f)` call, drifting up
  to one ULP for ordinary values and collapsing sub-normal-range
  inputs (e.g. `1e-40`) to `0.0f`. The fix computes
  `f64_to_f32_truncate(...).to_bits()` at codegen time and emits
  `chelis_f32_from_bits(0x...u)` for each argument, where
  `chelis_f32_from_bits` / `chelis_f64_from_bits` are new
  `static inline` helpers in `chelis_runtime.h` that bit-cast a
  `uint32_t` / `uint64_t` back to `float` / `double`. Symmetric with
  PR #243's `chelis_fill_f32_bits` mechanism but for per-call scalar
  args rather than buffer fills.

### Docs — harmonize Hull / trust-stack / project-plan with shipped reality

Corrected stale status framing across the design docs to match what the repo
now ships:

- `chelis prove` is described as shipped (V1, v0.7.1) rather than "planned" in
  `chelis_trust_stack.md`, `chelis_project_plan.md`, and the Hull prerequisite
  table — with the V1 scope (scalar binders + fixed-shape tensors) and the
  pending items (symbolic-dim binders, counterexample minimization) stated.
- The Hull spec's string-operations prerequisite reflects the shipped scalar/
  string foundation (`String` primitive + `string_*` builtins, `to_int`/
  `to_float`).
- The Hull spec's `Effect` ADT now mirrors the shipped enum
  (`Random`/`Accum`/`Io`/`Test`/`Resource(String)`) instead of listing a
  non-existent `Fail` and omitting `Test`.
- The Hull LaCaDiLE/timing notes drop the stale "POPL Jul 9" milestone in
  favor of the OOPSLA-targeted, stabilizing status.

## [0.7.19] — 2026-05-26

Wave-1 follow-up release. Closes six issues filed during the 0.7.13–0.7.18
cycle (#185–#208 sprint plus targeted follow-ups #218/#219/#229/#232/#233/#237)
that the released-binary downstream tooling surfaced: lossy C-backend f32
constant emission, doc-filename lint retroactive flip, AD-CLI silent
zero-grad, `chelis check` exit-code contract gap, runtime-shape docs gap,
and the GitHub Actions Node-20 → Node-24 deadline. Wave-1 red-team pass
found 4 findings (2 MEDIUM, 2 LOW) which were folded into PR #247 ahead
of this tag.

### Fixed - Wave-1 red-team follow-ups: #207 parse-error JSON + empty-file rejection, §4.7.5 spec direction, #188 artifact actions

Four findings from the Wave-1 red-team pass on PRs #188 / #189 / #190 /
#197 / #207 / #208:

- **M1 (#207 follow-up)**: parse errors in `chelis check` used to
  short-circuit through the generic error arm in `main`, exiting 1
  with empty stdout. The documented invariant (`exit != 0 iff
  json.errors.len() > 0`) requires that any failure mode emit a JSON
  report with a populated `errors[]`. `cmd_check_one` now catches
  errors from both `chelis_reef::prepare_program_for_file` and the
  raw `chelis_surf::parser::parse_str` branch, synthesizes a minimal
  JSON report with a single `Other`-kind entry carrying the error
  message, and lets the caller map that to exit 2. Reef checksum and
  missing-export errors now surface on stdout (JSON) rather than
  stderr; `reef_check_rejects_tampered_registry_shell_exports` was
  updated accordingly.
- **M2 (#207 follow-up)**: an empty or whitespace-only `.ch` used to
  report `score=1, errors=[]` and exit 0 in `chelis check`, while
  `chelis build` accepted it and emitted a no-op C function. Both
  surfaces now reject a zero-declaration program with the same
  canonical message `empty program: no declarations found` (kind
  `Other` in `check`'s JSON, boxed error in `build`).
- **L1 (#208 §4.7.5)**: the spec narrative previously described
  `reshape(x, [shape(x, 0), 4])` as "TYPE ERROR: int32 vs int64" and
  implied a diagnostic of "expected int64, got int32". The actual
  type-checker emits `precision mismatch: expected int32, got int64`
  (the unification-order artifact of the shape-list elementwise check
  against `List<Int64>`). The spec text now matches the diagnostic
  exactly, with a short note explaining the unification direction so
  the contract is unambiguous in either reading.
- **L2 (#188)**: bumps `actions/upload-artifact@v4` (twice) and
  `actions/download-artifact@v4` in `.github/workflows/release.yml` to
  `@v7` (the current node-24-bundled release). Sweep over all other
  workflow files confirmed no remaining `actions/*@v4` pins.

### Documentation - canonical runtime shape semantics in spec §4.7 (#208)

`spec/04-type-system.md` gains a new §4.7 "Runtime Shape Semantics"
that pins which `shape` / `expand` / `reshape` call shapes preserve
symbolic dims and which fall back to `(d-name {} *)`. The canonical
patterns from chelis#208 (`bias_broadcast` via runtime-sized
`expand`, `flatten_batch` via `cast(shape(x, axis), int64)` in
`reshape`) live as a fully-checked illustrative fixture at
`examples/illustrative/runtime_shape_semantics.ch`, and §4.7 cites
that file. Fall-back cases (cross-tensor shape source, arithmetic
wrappers, non-`var` reshape input) are documented so downstream
tools (Hydronnx, Calcify, other shells emitting Chelis) can emit
only the recognized syntactic forms instead of reverse-engineering
behavior from stdlib examples. Pure docs / fixture change; no
compiler or language behavior changed.

### Changed - CI: bump GitHub Actions to Node-24-compatible versions (#188)

Bumps the four pinned JavaScript actions in `.github/workflows/` to releases
that target Node 24, ahead of GitHub's 2026-06-02 default-runtime cutover:

- `actions/checkout@v4` -> `@v6` (ci.yml, release.yml, heavy-e2e.yml)
- `softprops/action-gh-release@v2` -> `@v3` (release.yml)

`docs/maintenance_schedule.md` updated to reflect the migration completing
and to record the verified-Node-24 adjacent pins (`Swatinem/rust-cache@v2`,
`astral-sh/setup-uv@v8.1.0`).

### Fixed - `doc-filename-convention` retroactive flip (#190)

`crates/chelis-lint/src/rules/doc_filename_convention.rs` previously
classified every `.md` under `docs/` by walking ancestors for a
`book.toml` marker, which made the §8.3-vs-§8.5 verdict depend on
unrelated filesystem state. Dropping a `book.toml` into `docs/`
flipped every narrative `docs/foo_bar.md` from accepted to rejected
(and removing the file flipped them back). The discriminator is now
path-based: §8.5 applies only to paths whose components include a
directory literally named `book` (most commonly `docs/book/` in the
chelis layout). `book.toml` is no longer read by the lint.

Spec §8.5 in `spec/01-nomenclature.md` updated to match. Invariant
coverage in `crates/chelis-lint/tests/issue_190_doc_filename_path_based.rs`.

### Fixed - C backend f32/f64 Const emission no longer drifts vs evaluator (#189)

The `emit_const` arms for F32 and F64 in `crates/chelis-backend-c/src/emit.rs`
previously rendered the IR's f64 source value through a decimal format
string (`{:.8}f` for F32, `{:.17}` for F64). Both format specifiers
print decimal-places-after-the-point rather than significant digits,
so values like `0.000000123456789` (issue #189 reproducer) parsed back
to a different f32 bit pattern -- about 3% off -- and `f64` values
below `1e-17` collapsed to zero. The same lossy pattern lived on the
Pad-fill arm.

The C backend now emits the source value's exact bit pattern
(`f32::to_bits()` / `f64::to_bits()`) and dispatches through new
`chelis_fill_f32_bits` / `chelis_fill_f64_bits` runtime helpers. The
helpers bit-cast the integer pattern back to the IEEE 754 value before
filling, so the emitted constant is bit-identical to the eval-path
representation (for f64) or the closest-f32 narrowing (for f32). The
same architectural shape as the existing bf16 / f16 bit-pattern fill
helpers.

### Fixed - CLI grad path now surfaces `AdError::NotSupported` for non-differentiable ops (#197)

The CLI `chelis build` / `chelis check` / `chelis eval` grad path routed
through `chelis_ir::grad::grad_dag` (the unchecked variant), so:

- `argmax_reduce` / `argmin_reduce` in a gradient path surfaced as a
  generic "grad requires a scalar floating output" message instead of
  the structured `AdError::NotSupported { op: "argmax",
  reason: IntegerIndexOutput }` rejection that
  `grad_dag_checked` already implements at the IR layer.
- `floor` / `ceil` in a gradient path silently zero-graded with no
  diagnostic — a wrong-result mode. The grad rule arms returned
  `Const { value: 0.0 }` and the build succeeded, emitting a kernel
  that filled the gradient tensor with zeros.

The fix is in two parts:

1. **Route through `grad_dag_checked`.** `compile_source::grad`
   (`crates/chelis-compiler-api/src/compiler.rs`), the `grad(...)`
   lowering arm in `crates/chelis-ir/src/lower.rs`, and the
   `vmap(grad(...))` lowering arm in the same file all switch to
   `grad_dag_checked`; the legacy `grad_then_fuse` carrier gains a
   sibling `grad_then_fuse_checked` so the fused path stays
   parallel.
2. **Mark AD-rejection diagnostics as fatal.** `LowerDiagnostic`
   grows a `fatal: bool` field. The host-fallback boundary at
   `host::try_lower_compiled_program`, the sub-lowering call sites
   in `host::lower_host_function` and
   `host::lower_tensor_helper_dag`, and the build CLI's
   `lower_checked_for_cli` (`crates/chelis-cli/src/main.rs`) all
   propagate fatal diagnostics instead of silently falling through
   to the host path. Without this the host fallback emitted an
   undefined-symbol call to the unlowered grad function — a
   compile-clean build that failed opaquely at gcc-link time.

Locked by `crates/chelis-cli/tests/grad_errors.rs` (5 tests:
argmax, argmin, floor, ceil negatives + matmul-grad positive
parity).

### Fixed - `chelis check` exit code mirrors JSON `errors` array (#207)

`chelis check` previously exited `0` even when its JSON report
contained `TypeMismatch`, `DimensionMismatch`, validator, effect, or
linearity errors. Downstream CI gates that treat exit `0` as success
silently accepted programs that `chelis test` later rejected with
exit `2`. Issue #207 inverts the RT-205 F7 contract: the exit code
now mirrors the errors array — exit `0` iff `errors[]` is empty,
exit `2` otherwise (matching `chelis test`'s convention for
"compile test context" failures). The machine-facing JSON shape is
unchanged.

This is a user-visible behavior change. Shell-script consumers that
relied on the previous "always exit 0" contract should either parse
the JSON `errors[]` array (the recommended path for tools that need
finer detail) or accept the new exit code. Test helpers and
fixtures in the in-repo CLI corpus were updated alongside the fix.

The exit-code invariant is locked by
`crates/chelis-cli/tests/issue_207_check_exit_code_invariant.rs`,
which sweeps four error categories plus the clean-program control.

## [0.7.18] — 2026-05-25

Hotfix release. Closes a zero-offset spurious-consume linearity bug class
across multiple consumer paths (closure capture + pipe-stage auto-borrow
inference) that 0.7.17 still surfaced on downstream shells (coral, hydronnx).
The sibling-sweep also folds in chelis#229.

### Fixed - zero-offset spurious-consume linearity bug across closure capture + pipe-stage paths (#237, #229, PR #239)

`crates/chelis-types/src/linearity.rs::check_fn` unconditionally structurally
consumed every captured tensor-carrying name in a closure body regardless of
how the body actually used it, producing zero-offset `was already consumed by
closure capture` diagnostics on any reuse of the captured variable. Same root
shape as #226: the consumer pass was mis-classifying without consulting the
inner callee's borrow signature.

`crates/chelis-types/src/infer.rs::pipe_consumes_param` had the parallel bug
that #229 documented: it only recognized the bare-var pipe-stage shape, so
explicit-arg pipe stages (`x |> add(k)`) became synthesized `__chelis_pipe`
lambdas it mis-classified as consuming uses.

The fix factors a shared `crates/chelis-types/src/pipe_stage.rs` helper for
callee resolution and wires both consumer passes through it. `check_fn` now
calls `param_has_consuming_use` instead of blanket-consuming captures;
`check_app` (direct call) and `arg_is_borrowed` consult `signature_inference`
for inferred-readonly callees.

Locked by `crates/chelis-types/tests/issue_237_linearity_sweep.rs` with 9
tests (6 closure-capture + 3 pipe-stage).

## [0.7.17] — 2026-05-25

Lint cleanup. Removes the `module-pascal-components` rule and its
`KNOWN_SINGLE_WORDS` allowlist. No language or compiler behavior
changes; only the lint surface shrinks.

This is the second cut on 2026-05-25; v0.7.16 (commit `9ec2e55`,
"Add std wrappers for scan index and sort") was tagged in parallel
with PR #234 and consumed the 0.7.16 version slot.

### Removed - module-pascal-components rule

The `module-pascal-components` (§6.3) lint rule and its supporting
`KNOWN_SINGLE_WORDS` allowlist are deleted. The rule's correctness
depended on the allowlist absorbing every long single English word
any shell in the ecosystem might use as a module component
(`Optimization`, `Comprehension`, `Conditional`, `Exception`,
`Translation`, `Calendar`, `Calibration`, …) — a list that grows
monotonically with the corpus and never converges. Demoting to
advisory (the 0.7.12 hotfix) reduced the blast radius but did not
fix the structural problem of a dictionary-dependent lint.

Removed:

- `crates/chelis-lint/src/rules/module_pascal_components.rs`
  (entire file)
- `pub mod module_pascal_components;` in
  `crates/chelis-lint/src/rules/mod.rs`
- Registration in `crates/chelis-lint/src/registry.rs::non_blocking_rules`
- The `KNOWN_SINGLE_WORDS` allowlist that the rule consumed
- Stale `KNOWN_SINGLE_WORDS` cross-reference in
  `crates/chelis-cli/src/main.rs`

Kept:

- The flattened-compound concern (`Linalg` → `LinAlg`,
  `Hellotensor` → `HelloTensor`) is still covered by reviewer
  attention plus the `module-compound-titlecase` rule, which
  checks a precise, narrowly-scoped list of known compounds rather
  than guessing from a lowercase-run length.
- The shared `module_decl` extractor stays (other rules consume it).

Closes `spec/upstream-bugs/module-pascal-components-flags-single-words.md`.

## [0.7.15] — 2026-05-25

Hotfix release. Closes a latent typing bug in `argmax_reduce` /
`argmin_reduce` output dtype that hydronnx PR #2 surfaced when it
retargeted from 0.7.13 to 0.7.14. The bug was present in every 0.7.x
release back to 0.7.0 — not a regression introduced by any of the
five 0.7.13 → 0.7.14 commits — but the practical impact was the
same: any downstream Surf program declaring the canonical
`tensor[..., int64]` return type for an argmax / argmin call hit a
`TypeMismatch`. The full diagnosis lives in
[`docs/investigations/issue_230_argmax_reduce_output_dtype_diagnosis.md`](docs/investigations/issue_230_argmax_reduce_output_dtype_diagnosis.md).

### Fixed - argmax_reduce / argmin_reduce return int64, not input dtype (#230)

`crates/chelis-types/src/infer.rs::check_reduction_signature`
defaulted every non-`sum` reduction's result precision to the input
precision. `argmax_reduce` and `argmin_reduce` emit element indices,
not reduced operand values, so the std-package signatures in
`packages/chelis-std/src/tensor/reduce.ch` already pin the canonical
output as `tensor[b, int64]`:

```text
sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]
sig argmin: &tensor[a, b, p] -> int32 -> tensor[b, int64]
```

Before the fix, `argmax_reduce(tensor[2, 3, f32], 1)` typed as
`tensor[2, f32]` and a user-declared `tensor[2, int64]` signature
was rejected:

```
def 'forward' body doesn't match declared signature:
  body has type `... -> tensor[Lit(2), f32]`,
  declared type is `... -> tensor[Lit(2), int64]`
```

After the fix, `check_reduction_signature` returns
`TensorPrec::Concrete(Prim::Int64)` for argmax_reduce / argmin_reduce
regardless of the input precision. The shape rule
(`out_dims.remove(axis)`, rank `n` → `n-1`) is unchanged.

The host-runtime / IR evaluator continues to store integer-valued
floats internally per the Phase 3j-pre Batch 1 caveat documented on
`RiscOp::Argmax`; the int64 type-system label is independent of the
storage layout. The `TODO(phase3j): widen backend runtime to carry
Int64 tensors natively` in `crates/chelis-ir/src/dag.rs` tracks the
eventual storage widening.

Sibling sweep: `sum`, `max_reduce`, `min_reduce`, `prod_reduce`, and
`mean` continue to preserve input dtype per spec §5.7.1 — only the
argmax/argmin family changes dtype.

Acceptance oracle:
`crates/chelis-types/tests/issue_230_argmax_reduce_output_dtype.rs`
(11 tests: positive on f32 / int32 / int64 inputs, negative on wrong
declared dtype, and a sibling-sweep regression guard) plus
`crates/chelis-compiler-api/tests/issue_230_argmax_argmin_runtime.rs`
(4 host-runtime parity tests pinning integer-valued index outputs
on axis 0 and axis 1 for both ops). The existing
`crates/chelis-ir/tests/negative_axis_normalization.rs` reduction
loops are updated to expect `int64` for argmax / argmin while the
other reductions stay `f32`.

## [0.7.14] — 2026-05-24

Hotfix release. Closes the release-blocking linearity-pass regression
that 0.7.13 surfaced in downstream reef shells (school, hydronnx, c-earchin,
calcify) whose source contains a tensor-carrying record destructured in a
function body that then uses the destructured field as the head of a pipe
into a borrow-arg builtin call. Also bundles the four follow-up fixes
that landed against 0.7.13's surface.

### Fixed - conv2d validator cascade-suppression is one-level-deep (#212)

The cascade-suppression added in PR #205 to dedupe `conv2d` validator
diagnostics only propagated one level past the root failure. A let-chain
of length 3 or more rooted on a non-concrete dim emitted a phantom
"concrete tensor argument metadata" diagnostic for every other
downstream let-binder, producing 2 diagnostics for a length-3 chain
and a "skip one, fire one" pattern beyond.

Root cause: the `failed_let_names` marker insertion in
`crates/chelis-types/src/infer.rs::validate_ir_expr` (let arm) required
`errors.len() > errs_before` to detect a failed RHS. Once cascade
suppression activated for the level-2 RHS, its diagnostic was silenced,
the count did not grow, and the level-2 name was never marked failed,
so the level-3 consumer fired its own cascade error.

Fix replaces the diagnostic-count guard with a structural recognition
predicate `let_rhs_is_recognized_shape_sensitive` that returns true
when the RHS is one of:

- a recognized shape-sensitive IR builtin (`matmul`, `conv2d`,
  `softmax`, `mean`, `layer_norm`, reductions, movement ops);
- a unary shape-passthrough builtin wrapping such a form
  (`relu`, `tanh`, `gelu`, etc.);
- a binary shape-passthrough builtin (`add`, `sub`, `mul`, etc.)
  with such a form on either operand.

Paired with `derive_ir_builtin_output_type` returning `None`, this
detects "should have derived but couldn't" without depending on the
per-level diagnostic count, so the suppression marker propagates
unboundedly through the chain.

The predicate stays narrow: a clean user-defined fn call as a let RHS
is not a recognized shape-sensitive form, so independent failures in
later let-binders still produce their own diagnostics (locked by
`rt205_r4_three_independent_failures_not_suppressed`).

Locked by `crates/chelis-types/tests/issue_212_cascade_suppression_depth.rs`,
which pins the issue body's 3-level reproducer, a 4-level chain, the
non-concrete-spatial / non-concrete-channel root variants, a relu-wrapped
multi-level chain, and the three-independent-failures negative parity.
Existing `red_team_205_round2_f2_sum_wrapped_chained_conv2d_still_rejected`
broadens its assertion: the new suppression silences the validator-level
metadata error in favour of HM's more informative rank-mismatch
diagnostic. Pure diagnostic-quality fix; correctness (which programs
chelis check accepts vs rejects) is unchanged.

### Fixed - spurious `UseAfterConsume` on pipe-stage borrow-arg calls (#226)

`crates/chelis-types/src/linearity.rs::check_pipe` only resolved the
effective callee of a pipe stage when the stage was a bare `(var f)`
reference. For any stage carrying explicit arguments (`x |> shape(0)`,
`x |> add(y)`, `x |> mul(k)`, `x |> matmul(w)`, etc.) the desugarer
emits a synthesized lambda
`(fn (params __chelis_pipe) (app callee args... (var __chelis_pipe) args...))`,
and the pre-fix code looked up `arg_is_borrowed` against the lambda
itself rather than the inner callee. The result was a structural-consume
classification of the piped value, even when the inner callee was a
known borrow-arg builtin. Any later read of the same variable then
tripped `UseAfterConsume` with a malformed "pipe into stage at offset 0
from offset 0" message (zero offsets because the synthesized lambda
carries no source span).

The regression manifested only after PR #183 (in 0.7.13) stamped the
resolved field type onto destructured record-pattern bindings, removing
the fresh-type-variable fallback that had been accidentally suppressing
the bug for the destructure-then-pipe shape used heavily by
`School.Nn.PosEmbed.pos_embed_forward` and adjacent layers.

`check_pipe` now invokes a new `resolve_pipe_stage_callee` helper that
peers through the synthesized `__chelis_pipe` lambda to reach the inner
callee and the piped value's actual arg index, then asks
`arg_is_borrowed` about that callee. The helper falls back to the
historical bare-var contract for any unrecognized stage shape so
`(var f)` pipe stages continue to behave exactly as before.

Locked by
`crates/chelis-types/tests/issue_226_linearity_pipe_borrow_stages.rs`,
which pins three positive cases (pipe into `shape`, pipe into `add`,
composed `mul` chain) and three negative-parity cases (pipe into
`realize`, pipe into a user fn whose first parameter is owned-linear,
pipe into a user fn with explicit extra args).

### Fixed - `extract_dim_list` matched Cons-chain tag symbol as a dim name (#220, PR #224)

`extract_dim_list` in `crates/chelis-ir/src/lower.rs` walked `list.elements`
directly and matched any `Atom::Symbol` as a dim name, including the literal
`"app"` tag symbol that Surf-source dim lists desugar through. Rewritten to
peel the Cons chain via `collect_cons_chain` and interpret each head via
`extract_int_for_dim` or `symbolic_dim_var_name`. Defense-in-depth against
the `dag.rs:1108` ICE.

### Fixed - cast-aware integer extraction across 15 infer-time sites (#216, PR #225)

PR #214 extended cast-aware `extract_int_for_dim` to the movement-op family
(`shrink`/`stride`/`pad`/`permute`/`expand`). The remaining 15 call sites of
`extract_int_literal` across reductions, conv2d helpers, gather/scatter,
softmax, shape, split, vmap, and grad-wrt now also use the cast-aware
extractor so `cast(N, int32)` boundaries trip infer-time checks rather than
deferring to host runtime.

### Added - `to_tensor` literals work in differentiable function bodies (#218, includes #219 Option A)

The architectural piece that the PR #211 R1→R4 red-team cascade signalled
was needed before Part 2 of #199 could ship. Bundles four coordinated
changes:

- **#219 Option A**: `unify_dim` now binds `Dim::Name(_) ↔ Dim::Lit(_)` so the
  canonical style-guide pattern `def f(x: tensor[batch, hidden, p])` accepts
  concrete callers like `f(to_tensor([[1.0, 2.0, 3.0]]))`. Spec amendment in
  `spec/04-type-system.md §4.1`.
- **`to_tensor` source fix**: nested-list literals emit concrete `Dim::Lit(n)`
  via a new `static_to_tensor_shape` walker, rather than the all-`Wildcard`
  fallback that propagated through every downstream consumer.
- **Per-axis Cons join**: list-of-tensor element unification at the Cons site
  uses per-axis-joined element types (`Lit ∩ Lit = Lit` if equal, else
  `Wildcard`) so `concat([to_tensor([[1,2,3]]), to_tensor([[4,5,6],[7,8,9]])], 0)`
  type-checks instead of failing element-type unification.
- **`expr_requires_host_runtime` exemption**: a `to_tensor` literal in a
  function body destined for `grad` no longer forces host-only routing.
  `emit_literal_tensor` lowers the static literal as a `Const`/`Pad`/`Add`
  cascade reachable by the AD pass.

Acceptance oracle: all 19 R1→R4 failure modes type-check + run + finite-diff
agree at 1e-3 tolerance. Locked by
`crates/chelis-cli/tests/issue_218_to_tensor_in_grad_body.rs`,
`crates/chelis-types/tests/issue_219_name_vs_lit.rs`,
`crates/chelis-ir/tests/issue_218_finite_difference.rs`.

## [0.7.13] — 2026-05-24

Cut to ship six downstream-blocking fixes that the Hydronnx H3.x ONNX-emitter
spike surfaced, plus an end-to-end acceptance lock against the H3 shapes.
Part 2 of issue #199 (to_tensor literals in differentiable function bodies)
remains deferred to follow-up issues #218 / #219 / #220 after the PR #211
red-team rounds signalled architectural redesign was needed; see the
`feedback_rround_cascade_is_design_signal` discipline note.

### Fixed - `conv2d` validator rejects every well-formed call (#186, PR #205)

`crates/chelis-types/src/infer.rs::validate_ir_builtin_symbolic_requirements`
rejected every well-formed `conv2d` call at `chelis check`. Two root causes:
`app_result_type_is_concrete` read `:type` from app metadata that the
inference annotation pass writes after the validator runs, so the
output-dims check was structurally always-false for any Surf source; and
`expr_tensor_type_is_concrete` did not peel `(borrow {} ...)` wrappers, so
the idiomatic `conv2d(&x, &k, ...)` form (and the wrappers in
`packages/chelis-std/src/nn/conv.ch`) defeated the tensor-metadata
concreteness check.

The replacement derives output concreteness from the args (`input` /
`kernel` dims, integer-literal `stride` / `padding`) instead of from
app-level `:type` metadata, evaluates the canonical
`floor((in + 2p - k) / s) + 1` spatial formula to catch non-positive
outputs at check time, and routes tensor-metadata lookups through a new
`arg_tensor_type_expr` helper that peels `(borrow {} ...)`. The sibling
sweep extended the same borrow peeling to the `mean` and `layer_norm` arms
and added range checks for zero / negative / overflowing stride and padding
values that previously panicked or silently emitted nonsense C. Symbolic
batch dims in `conv2d` input now type-check (per spec §4.5
`batch, in_c, h, w, p`). `validate_conv2d_symbolic_requirements` is the
new owner of the conv2d-specific validation surface; the shared cascade
suppression infrastructure (`failed_let_names`) prevents duplicate
"concrete tensor argument metadata" diagnostics when a downstream
conv2d's input depends on an earlier failed conv2d let-binding, including
through `relu` / `tanh` / `sigmoid` / `gelu` / `add` / `max_elem`
shape-passthrough wrappers.

Locked by `crates/chelis-types/tests/issue_186_conv2d_validator.rs`,
which exercises the issue text's exact repro plus the borrow-form variant,
borrowed-`mean` / `layer_norm` sibling-sweep cases, the formula-evaluation
spatial-rejection cases, symbolic-batch acceptance, and the pass-through
cascade-dedup invariant.

### Fixed - `shrink` / `stride` / `pad` not callable from Surf (#187, PR #214)

The windowing builtins `shrink` and `stride` were registered as
single-arg `tensor_unop` in `crates/chelis-types/src/builtins.rs` even
though the RISC lowering (`crates/chelis-ir/src/lower.rs::lower_shrink`
/ `lower_stride`) reads the window parameters from `args[1..]`. The
bare 1-arg form type-checked but the host runtime answered
`unsupported builtin \`shrink\` in host runtime`; the parameterized
form that the lowering actually needed was type-rejected with
`function arity mismatch: expected 1 args`. `pad` had the same
antipattern. Hydronnx H3 emitter could not express MaxPool /
AvgPool / GlobalMaxPool / GlobalAvgPool because of this gap.

The fix introduces dedicated parameterized inference handlers
(`infer_shrink_app`, `infer_stride_app`, `infer_pad_app`) that mirror
the lowering's expected signature: `shrink(&tensor, pair-list-of-int)`,
`stride(&tensor, int, int, ...)` (one positional step per axis), and
`pad(&tensor, pair-list-of-int, fill)`. Host-runtime arms in
`crates/chelis-compiler-api/src/runtime.rs::eval_builtin` delegate to
the existing IR evaluator. The parser-side `extract_pair_list` walks
the Cons chain instead of `list.elements` so cast-wrapped int literals
(`cast(N, int32)`) participate in infer-time bounds and stride checks
the same way reshape's dim list does, and malformed inner pairs
(`[[0, 1, 2]]` triples, `[[0]]` singletons, `Nil` empty inner lists)
are rejected at infer time with axis-tagged diagnostics. `pad`'s fill
arg is now unified against the tensor's precision instead of being
silently discarded.

Cast-aware extraction was extended across the movement-op family
(`shrink`, `stride`, `pad`, `permute`, `expand`) so neg-of-cast and
arbitrarily-nested-cast bound expressions resolve at infer time.

Locked by `crates/chelis-types/tests/issue_187_shrink_stride_sig.rs`,
`crates/chelis-compiler-api/tests/issue_187_shrink_stride_host_runtime.rs`,
and four red-team rounds of adversarial probes covering the cast,
neg-of-cast, double-cast, empty-list, malformed-pair, and per-axis-stride
surfaces.

The broken `shrink(&qkv)` call in
`examples/illustrative/mha_slice_combined_qkv.ch` was rewritten to use
the parameterized form with concrete dims. Three CLI tests
(`build_hip_rejects_pad_lowering_without_panic`,
`target_metal_rejects_pad`, `target_metal_rejects_shrink`) that fed the
no-longer-accepted bare 1-arg form were updated.

### Fixed - `chelis lint` panics on em dash inside Python docstring (#209, PR #210)

`crates/chelis-lint/src/rules/no_em_dash_in_public_strings.rs::dash_spacing`
took a `line: &str` plus a `dash: usize` byte offset INTO THAT LINE and
returned `DashSpacing { after_end: after_start + usize::from(after) }`
where `after_start = dash + '—'.len_utf8()`, also line-relative. Callers
in the same file (`fix()`, `clause_replacement`, `spaced_dash_replacement`)
then treated `spacing.after_end` as a SOURCE-absolute byte offset and
sliced `source[next_start..]` or built `Replacement { end: next_start + ... }`
with it. On line 1 (`line_start == 0`) the bug was masked. For any later
line, `after_end` indexed a byte that has nothing to do with the dash's
source position; when that byte happened to land inside an earlier
multi-byte char (commonly another em dash inside an excluded docstring),
the slice panicked. Hydronnx added `chelis lint --check .` to its full
workspace gate; the command panicked on existing docs / scripts before
producing actionable output.

The fix threads `line_start` into `dash_spacing` and returns
source-absolute offsets, matching the source-absolute calling convention
that `dash` already used. The `DashSpacing.after_end` field now carries
a doc comment naming the previous bug.

The sibling sweep audited `crates/chelis-lint/src/rules/redundant_linearity_call.rs`,
the only other rule that crosses line and source byte coordinates;
it composes `line_start_offset(...) + col - 1` correctly and required
no change.

Locked by `crates/chelis-lint/tests/issue_209_em_dash_utf8_boundary.rs`
with 16 regression fixtures plus an end-to-end CLI assertion that
`chelis lint --check` does not panic on any of the multi-byte / multi-line
docstring shapes the rule walks.

### Fixed - Runtime-dim `reshape` does not preserve declared symbolic shape (#206, PR #213)

`crates/chelis-types/src/infer.rs::infer_reshape_app::list_literal_dims`
recognized only concrete-int dim list elements (looking through
`cast(N, int{32,64})`) and fell back to `vec![Dim::Wildcard; rank]` for
anything else, including the common runtime-batch pattern
`cast(shape(x, axis), int64)`. The neighbouring `expand` arm "works"
only because `check_expand_signature` does not assert a precise output
dim list when the size arg is non-literal; reshape actively emitted a
rank-matching all-`Wildcard` dim list, collapsing the body type to
`tensor[Wildcard, ...]` and producing a spurious `TypeMismatch` against
the declared sig.

The fix adds a strict-syntactic recognizer (`reshape_output_dims` plus
helpers `collect_shape_list_elements`, `extract_shape_axis_of`,
`peel_cast`, `is_target_ty`, `is_shape_app`) that recognizes
`cast(shape(x, cast(lit_axis, int32)), int64)` as a same-tensor
dim-reference and injects the input's resolved dim at the named axis
into the output dim list. The recognizer rejects through-different-tensor
references, non-literal axes, and arithmetic-wrapped shape calls. The
earlier `list_literal_dims` / `cons_chain_int_dims` helpers are
subsumed.

Sibling sweep confirmed `view` and `broadcast_to` are not builtins; only
`broadcast_pair` exists and takes two tensors. `expand`'s sig-driven
result type was not affected.

Locked by `crates/chelis-types/tests/issue_206_runtime_dim_reshape.rs`
with the issue's exact reproducer (`flatten_batch`) plus axis variations,
reorder and duplicate-axis cases, and rejection of through-different-tensor
references.

### Added - Host-runtime dispatch for 14 builtins; `BUILTIN_NAMES` invariant lock (#185, PR #217)

`BUILTIN_NAMES` in `crates/chelis-types/src/builtins.rs` listed 12
builtins that `chelis check` type-accepted but
`crates/chelis-compiler-api/src/runtime.rs::eval_builtin` did not
dispatch, so `chelis test` / `chelis eval` failed with
`unsupported builtin \`X\` in host runtime`. Sibling sweep found 6
more in the same gap. Hydronnx H3 emitter could not run Conv2D,
MaxPool / AvgPool / GlobalMaxPool / GlobalAvgPool, reductions, or
activations through the host runtime against an ONNX-Runtime reference.

This release ships host-runtime arms for:

| Group | Builtins |
|---|---|
| A — Unary RISC (spec §2.2) | `abs`, `cos`, `tan`, `floor`, `ceil`, `atan` |
| B — Reductions (spec §2.2) | `max_reduce` |
| C — Binary Tier 2 (spec §3.4) | `max_elem`, `min_elem` |
| E — Composed Tier 2 (spec §3.4 / §4.4 / §4.5) | `mean`, `layer_norm`, `conv2d` |
| F — Tensor-bool (spec §3.2) | `and`, `or`, `not` (tensor arms; scalar arms already existed) |

`pad` (Tier 1 §2.4) shipped in PR #214 alongside the shrink / stride
sig fix. The IR evaluator (`crates/chelis-ir/src/eval.rs`) is the
canonical numerical oracle per the evaluator-vs-backend agreement
discipline; each new arm delegates to it rather than re-deriving op
semantics.

`normalize` is registered as a builtin but marked unstable in
`builtins.rs:55-56`; it is intentionally allowlisted in the new invariant
test with reason `pending-spec-stability` and tracked separately.
`scatter_replace` is allowlisted with reason `lowered-before-eval` (the
Surf call always lowers to `RiscOp::Scatter`).

`crates/chelis-compiler-api/tests/builtin_dispatch_invariant.rs` locks
the schema: every name in `BUILTIN_NAMES` must have either an
`eval_builtin` arm or a documented allowlist entry, and every allowlist
entry must reference a name in `BUILTIN_NAMES`. The invariant catches
missing arms, double entries, and stale allowlist entries; the policy
header documents the closed reason vocabulary
(`type-only`, `target=<backend>-only`, `lowered-before-eval`,
`pending-spec-stability`).

Locked by per-group test files in
`crates/chelis-compiler-api/tests/issue_185_host_runtime_*.rs`.

### Fixed - `BlasMatmul` had no gradient rule; `grad` rejected any matmul-bearing body (#199 Part 1, PR #211)

`grad` (`crates/chelis-ir/src/grad.rs::compute_adjoints`) returned `None`
for `RiscOp::BlasMatmul` because the specialise pass introduces
`BlasMatmul` whenever a pure `matmul` lowering matches the tier2
`Sum(Mul(Expand(A), Expand(B)))` template, and `compute_adjoints` only
had an arm for plain `RiscOp::Matmul`. Any function body containing a
matmul could not be differentiated, surfacing as
`AdError::NotSupported { op: "<unknown>", reason: ... }`.

The fix adds a `BlasMatmul` arm to `compute_adjoints` that emits the
standard reverse-mode adjoint `dA = dY @ B^T`, `dB = A^T @ dY` via
`Permute` (last-two-axis swap, rank-N safe) plus a fresh `BlasMatmul`
using the spec §5.7.1 default accumulator. Finite-difference numerical
agreement verified across 2x2x2, 3x4x2, 1x3x1, batched 2x2x2x2, and
chain-rule-via-`relu` 2x3 + 3x2 configurations within `abs_tol=1e-3`
(f64) and `abs_tol=1e-2` (f32). The sibling sweep enumerated every
remaining `=> None` arm in `compute_adjoints`; the rest are intentional
non-differentiability (`Floor`, `Ceil`, `Argmax`, `Argmin`,
`Scatter`, `ScatterAdd`, `Stride`, `Drop`, `FusedElem`) and are not
realistic-model blockers.

Locked by `crates/chelis-ir/tests/issue_199_grad_blas_matmul_to_tensor.rs`
with five tests covering analytic match for the 2x2x2 example, batched
shapes, and the relu-composed chain rule.

**Part 2 is deferred.** Allowing `to_tensor` literal weight constants
inside differentiable function bodies (the rest of issue #199's text)
requires either a `Dim::Wildcard`-stripping rule that survives every
downstream consumer or a type-system change to bind `Dim::Name(_) ↔
Dim::Lit(_)`. PR #211 went through R1 → R4 of red-team rounds, each
finding a new consumer (reductions, elementwise, concat, named-dim
sigs); the consumer-by-consumer pattern signalled architectural
redesign rather than continued patching. Follow-up issues:
[#218](https://github.com/Chelis-Lang/chelis/issues/218) (to_tensor in
differentiable bodies, with the R-round failure modes as design
requirements), [#219](https://github.com/Chelis-Lang/chelis/issues/219)
(Name-vs-Var unification asymmetry),
[#220](https://github.com/Chelis-Lang/chelis/issues/220)
(`extract_dim_list` parser bug).

### Added - End-to-end acceptance lock for Hydronnx H3.x shapes (PR #221)

`crates/chelis-e2e/tests/issue_185_hydronnx_h3_shapes.rs` exercises the
three H3.x ONNX-emitter shapes against pen-and-paper references plus
hand-computed conv2d output pixels:

- `hydronnx_h3_maxpool_2x2_stride2_no_padding_matches_ir_eval` —
  shrink + stride + concat + max_reduce composition (the H2
  decomposition pattern)
- `hydronnx_h3_layer_norm_batch2_hidden4_matches_ir_eval` —
  `layer_norm(x, gamma, beta)` against the spec §4.4 formula
- `hydronnx_h3_conv2d_1x3x8x8_kernel_8x3x3x3_matches_ir_eval` —
  conv2d against a pure-Rust 7-loop NCHW/OIHW reference plus an
  explicit hand-computed lock on `out[0, 0, 0, 0]`

Forward-pass only; the `grad` surface is covered by the Part 1
fixtures and (when #218 lands) by Part 2 follow-up work. Verified
against 15 adversarial probes (ties, all-negative, zero-variance,
negative gamma / beta, zero / identity / negative kernel,
batch variations, padding>0).

## [0.7.12] — 2026-05-23

Cut to surface the post-`v0.7.11` `main` work — in particular two
downstream-blocking lint fixes — to consumers (hydronnx v0.1.0 release
is gated on this).

### Fixed - `module-pascal-components` lint demoted to advisory

`crates/chelis-lint/src/rules/module_pascal_components.rs` was
demoted from a blocking error to an advisory warning. The heuristic
("module component looks like a multi-word compound but has no
internal capital") is genuinely useful for catching `tabularmlp`-style
typos in user code, but its blocking posture broke every legitimate
single-word module name not on the hardcoded `KNOWN_SINGLE_WORDS`
allowlist (the downstream `Hydronnx` shell hit this on every emitted
module). The advisory demotion means the rule still surfaces the
finding in `chelis check` output but does not fail the build.

### Fixed - `Hydronnx` re-allowlisted in `module-pascal-components`

`crates/chelis-lint/src/rules/module_pascal_components.rs` re-adds
`Hydronnx` to `KNOWN_SINGLE_WORDS` so the lint never even fires the
advisory warning for it (the shell-ecosystem prefix is a known
single-word compound name, per spec §6.3 "Shell Ecosystem"). This is
belt-and-braces with the advisory demotion above: even if a future
change re-promotes the rule to blocking, the prefix stays clean.

### Other work since `v0.7.11`

Everything previously under `[Unreleased]` shipped in this release.

### Fixed - C backend `emit_cast` corruption on bf16/f16 boundaries (WS-Cleanup-Fixups)

`crates/chelis-backend-c/src/emit.rs::emit_cast` previously emitted the
C language cast `((dst_et*)t->data)[i] = (dst_et)((src_et*)t->data)[idx]`
for every cross-precision arm, including the four arms that touch the
reduced-float dtypes (`Bf16` and `F16`). Those dtypes store as
`uint16_t`, so the cast `(uint16_t)1.5f` integer-truncates the float to
`1` and writes bit pattern `0x0001` instead of `bf16(1.5)=0x3FC0` /
`f16(1.5)=0x3E00`. The widening direction was symmetrically broken:
`(float)(uint16_t)0x3FC0` is `16320.0f`, not `1.5f`. Three RT-Cleanup
BLOCKER tests (cast_f32_to_bf16, cast_bf16_to_f32, cast_f32_to_f16) were
gated on this fix.

The fix routes every cast that touches `Bf16` or `F16` through the
existing WS-1 runtime helpers (`chelis_f32_to_bf16` /
`chelis_bf16_to_f32` / `chelis_f32_to_f16` / `chelis_f16_to_f32`).
Cross-narrow-float casts (`Bf16 <-> F16`) chain through `f32` as the
intermediate. Casts between reduced floats and `intN` / `f64` route
through `f32` for the reduced-float leg and use the C primitive cast for
the other. Pairs not involving reduced floats keep the existing C cast
semantics.

Locked by un-ignoring the three RT-Cleanup BLOCKER tests in
`crates/chelis-backend-c/tests/rt_cleanup_redteam.rs`
(`cast_f32_to_bf16_preserves_value_per_ieee_754`,
`cast_bf16_to_f32_preserves_value_per_ieee_754`,
`cast_f32_to_f16_preserves_value_per_ieee_754`) plus a new
`crates/chelis-backend-c/tests/dtype_matrix_bf16_f16_extended.rs` with
bit-pattern + round-trip tests for all six reduced-float cast
directions and a sweep over non-round operand values (0.5, 2.5, -1.75,
100.0, 0.125).

### Fixed - Metal `host_const_fill_body` invalid C++ literal for F32 NaN/Inf (WS-Cleanup-Fixups)

`crates/chelis-backend-metal/src/dtype.rs::host_const_fill_body` formatted
the F32 fill value with `{value:?}f`. Rust formats `f64::NAN` as the
token `NaN`, so the emitted code carried `p[i] = NaNf;` which is not a
valid C/C++ float literal. The infinity path emitted `inff` with the
same shape. Latent today because no Phase 0 corpus exercises Metal
Const with NaN/Inf, but locked as a SPEC-DIVERGENCE by RT-Cleanup.

The fix discriminates `f64::is_nan()` / `f64::is_infinite()` on the F32
arm and emits the C99 `(float)NAN` / `(float)INFINITY` /
`-(float)INFINITY` macros, which are well-formed C++ tokens. Finite
values keep the existing `{value:?}f` shape (byte-identical to the
pre-fix emission). The F16/Bf16 arms already use the bit-pattern
literal path (`0x{bits:04X}u`) and so were unaffected; a regression
test pins that behavior.

Locked by un-ignoring two RT-Cleanup SPEC-DIVERGENCE tests plus three
new regression tests in
`crates/chelis-backend-metal/tests/rt_cleanup_redteam.rs`:
`rt_metal_emit_const_f32_negative_infinity_emits_negated_infinity_macro`,
`rt_metal_emit_const_f32_finite_value_keeps_typed_float_literal_form`,
`rt_metal_emit_const_f16_bf16_nan_inf_emit_well_formed_integer_literal`.

### Added - C backend bf16/f16 elementwise + cast coverage (WS-Cleanup-Fixups)

`crates/chelis-backend-c/tests/dtype_matrix_bf16_f16_extended.rs` closes
the RT-Cleanup coverage gap for ops the original WS-1 matrix did not
exercise directly. Per-dtype evaluator-agreement tests at `BF16_TOL =
1e-2` / `F16_TOL = 1e-3` for `Div`, `Recip`, `Sqrt`, `Sin`, `Cos`,
`Tan`, `Atan`, `Floor`, `Ceil`, plus bit-pattern + round-trip locks for
all six reduced-float cast directions. `MinReduce` and `ProdReduce` on
bf16/f16 are pinned with `#[should_panic]` tests against the
pre-existing `emit_reduce_simple` f32-hardcoded guard
(`crates/chelis-backend-c/src/emit.rs:3540`); widening that path to
reduced floats is follow-on work outside the WS-Cleanup-Fixups scope.

### Fixed - Metal `emit_const` host-fill for f16/bf16 (WS-2)

`crates/chelis-backend-metal/src/emit.rs::emit_const` previously emitted
host Objective-C++ code using the MSL-only kernel types `half` and
`bfloat` for both the typed-pointer cast and the value cast inside the
CPU fill loop. Those types are not visible to host `clang++ -fobjc-arc`,
so any Const-rooted f16 or bf16 program produced a `.mm` that failed to
compile. The CLI gate (`reject_unsupported_metal_ops`) admits all 8
active Metal dtypes post-PR #112, so this was a live bug, not latent.

The fix factors the host-fill block through a new
`dtype::host_const_fill_body` helper that routes through host-safe
types per dtype: typed `float*` for f32, `uint16_t*` plus IEEE-754
bit-pattern literal (computed at codegen time via
`half::f16::from_f64(...).to_bits()` and the bf16 analogue) for f16 /
bf16, matching `intN_t*` with an explicit integral cast for the
integer family, and C++ `bool*` for bool. The pre-existing
`sizeof(msl_ty)` was already routed through `host_sizeof_expr` in PR
#112; the WS-2 change finishes the migration on the value-write side.

Locked by two new tests:

- `crates/chelis-backend-metal/tests/codegen_structure.rs::ws2_emit_const_f16_bf16_use_uint16_bit_pattern_not_msl_kernel_types`
  — Linux structural assertion that the emitted source uses
  `uint16_t*` plus the pinned bit-pattern literal and never `(half*)`,
  `(bfloat*)`, `sizeof(half)`, or `sizeof(bfloat)` for f16/bf16
  Const fills. Sweeps eight pinned (precision, value, bits) tuples
  covering 2.5, 1.5, -1.0, and 0.0.
- `crates/chelis-backend-metal/tests/gpu_correctness.rs::metal_const_f16_bf16_compiles_under_objc_arc_and_produces_exact_bits`
  — macOS-only (`#[cfg(target_os = "macos")]`, `#[ignore]`) compile +
  run gate that builds Const-rooted f16/bf16 DAGs, links them via
  `xcrun clang++ -fobjc-arc`, runs the binary, and asserts the
  runtime buffer holds the exact pinned bit patterns.

Plus `crates/chelis-backend-metal/tests/codegen_structure.rs::ws2_emit_const_f32_integer_bool_paths_unchanged`
which pins the byte-identical shape of f32 / int32 / bool Const
emission so a future refactor cannot silently route them through the
bit-pattern path.

### Added - C backend admits bf16 and f16

The C backend now admits `bf16` and `f16` at every active op surface
(elementwise, reductions, Const, Load, Store, matmul) per
`spec/04-type-system.md` §1.1.3. Storage stays two bytes (`uint16_t`);
arithmetic always converts to `f32` via new runtime helpers
(`chelis_bf16_to_f32` / `chelis_f16_to_f32` and inverses). Matmul
routes through `convert-then-cblas_sgemm` with f32 scratch buffers,
matching the §5.7.1 accumulator promise. Reductions on bf16/f16
operands use an f32 accumulator and produce an f32 result, with a
locked test that distinguishes the f32-accumulator path from a naive
bf16-direct accumulation. Brings all three first-party backends
(C, HIP, Metal) to true 9/9 active-dtype parity modulo Metal's f64
hardware exclusion. The `sparse_elem_type` silent default-arm
(`_ => "float"`) and the `emit_const` silent f32 truncation default
arm are gone; both are replaced with explicit per-dtype arms plus
panic-on-unsupported.

### Changed - descriptive test names in backend crates

Renamed scaffolding-named integration test files in `chelis-backend-c`,
`chelis-backend-hip`, and `chelis-backend-metal` so each is named for
what it verifies, not the wave/workstream that created it. Pure rename:
zero behavior change, every test still runs, every regression-lock
invariant preserved. The `rt1_`/`rt2_`/`rt_`/`s4_`/`s6_`/`ws_a3_`/
`red_team_w5_`/`redteam_` prefixes were dev-process noise.

- `chelis-backend-c`: `rt1_adversarial` -> `dtype_boundary_adversarial`,
  `rt2_adversarial` -> `dtype_matrix_adversarial`, `redteam_adversarial`
  -> `simd_math_codegen_adversarial`, `redteam_c_fused_compile` ->
  `fused_compile`, `redteam_exec_compile` -> `exec_compile`,
  `red_team_w5_f64_matmul_miscompile` -> `f64_matmul_miscompile`,
  `s4_span_comments` -> `span_comments`, `s6_host_span_comments` ->
  `host_span_comments`.
- `chelis-backend-hip`: `redteam_adversarial` -> `codegen_adversarial`,
  `red_team_w5_strided_batched` -> `strided_batched_dispatch`,
  `s4_span_comments` -> `span_comments`, `ws_a3_bf16_f16_matmul` ->
  `bf16_f16_matmul`.
- `chelis-backend-metal`: `redteam_adversarial` -> `codegen_adversarial`,
  `rt_metal_adversarial` -> `dtype_expansion_adversarial`,
  `s4_span_comments` -> `span_comments`.

References updated in `docs/phase_oracles.md`, `spec/08-backends.md`,
`spec/design/chelis_phase1_plan.md`,
`spec/design/chelis_metal_backend_plan.md`,
`spec/design/phase1c_memory_planning.md`,
`spec/design/phase1d_flattening.md`, the renamed files' own doc-comment
cross-references, and the `scripts/test_timing_baseline.json` binary
keys.
### Changed - descriptive test-file names (ir/types/api/e2e/effects)

Renamed scaffolding-named integration test files in `chelis-ir`,
`chelis-types`, `chelis-compiler-api`, `chelis-e2e`, and `chelis-effects`
so each file is named for what it verifies rather than the
phase/wave/workstream that created it (dropped `phase_d/e/f/g/i_`,
`rt1/rt2_`, `rt_g_`, `red_team_w5/w7_`, `red_team_0_7_8`, `ws_a0/a3/b2_`,
`s3_`, `phase1f_` scaffolding stamps). Pure rename: every test still
runs, every regression-lock invariant is preserved, only names and
references changed. The three same-named `red_team_0_7_8.rs` files now
have distinct names reflecting their distinct layers
(`implicit_copy_fanout_shape_a_adversarial` in `chelis-ir`,
`linearity_alias_destructure_adversarial` in `chelis-types`,
`runtime_dtype_coupling_adversarial` in `chelis-e2e`). Updated all
`mod`/doc-comment references, `docs/manual_gates.md`,
`docs/phase_oracles.md`, the Phase 1f oracle command in the active
specs, and `scripts/test_timing_baseline.json` keys.

### Changed - descriptive test names in `chelis-cli` (dropped phase/ws/rt scaffolding)

Renamed 41 `chelis-cli` integration-test files whose names encoded the
development phase or workstream that created them (`phase3*`,
`phase_a_*`, `phase_h_*`, `phase_k_*`, `wsa5/6/7/8_*`, `wsc_*`,
`rt3/rt3a/rt4_adversarial`, `red_team_0_7_*`, `red_team_w*`) to names
that describe what each suite verifies (for example `phase3j_pre_std`
to `std_nn_build_acceptance`, `wsa8_monomorphization_build` to
`monomorphization_build`, `rt4_adversarial` to
`numeric_dtype_adversarial`). Pure rename: every test still runs and
every regression-lock invariant is preserved; `cargo nextest list`
reports the same 735 `chelis-cli` tests across 75 binaries as before.
Updated all references in `.config/nextest.toml` `binary_id` filters,
`crates/chelis-cli/Cargo.toml`, source doc comments,
`docs/manual_gates.md`, `docs/phase_oracles.md`, the active Phase 3
specs, and `scripts/test_timing_baseline.json`.

### Changed - `production_stdlib_typechecks` back on the per-PR gate

Moved the `chelis-cli::production_stdlib_typechecks` suite (19 tests, one
`chelis check` per unique stdlib file) off the nightly heavy-e2e profile
and back onto the per-PR `ci`/`default` nextest profiles. It was placed
in the heavy set before the cross-process typecheck cache landed; with
the cache the checks are warm-cache fast, so they belong on the per-PR
gate for the coverage.

Also reviewed the three dedupe candidates flagged in
`docs/investigations/integration_test_suite_trim.md`. All three were
verified and kept with documented reasoning: the 19
`production_stdlib_typechecks` tests (per-file attribution; the cache
removed the cost the dedupe targeted), the three `phase3j_pre_std`
RNG-seed tests (each pins a distinct assertion that cannot be folded
without loss), and the two `wsa8_monomorphization_build` stdlib build
tests (`linear.ch` standalone-polymorphic vs `attention.ch`
cross-function-polymorphic are distinct monomorphization shapes).
### Fixed - atomic reef package-cache writes (concurrent-build race)

`load_registry_package` extracted a package archive into the shared
`$CHELIS_REEF_HOME/cache/<archive_sha256>/` directory in place: it
created the directory, then unpacked `reef.toml` and its siblings file
by file. The `if !cache_root.exists()` guard went false the instant the
directory was created, so a concurrent `chelis reef build` against the
same reef-home could observe a half-written `reef.toml` and fail with
`TOML parse error at line 1, column 1`. This surfaced as an intermittent
failure of the `phaseA_item8_two_concurrent_builds_serialize` CLI test.

- `crates/chelis-reef/src/lib.rs`: new `extract_archive_atomic` helper
  unpacks into a unique sibling `.extract-*.tmp` staging directory and
  `fs::rename`s it into place. Same-directory rename is atomic, so a
  concurrent reader sees either no cache directory or the fully
  populated one, never a torn file. A lost rename race (another process
  published first) is treated as success because the cache key is the
  archive's own content hash, so the trees are byte-identical.
- Unit coverage: `extract_archive_atomic_publishes_and_leaves_no_staging_dir`
  and `extract_archive_atomic_concurrent_writers_never_tear_reef_toml`
  (16 threads extracting and reading back the same cache directory).

### Added - adversarial coverage for the post-#130 compiled-context cache

Replaced the reverted pre-#130 red-team file (`#128`, reverted in `#131`
because it was written against the old cache API and asserted the
path-collision bug as observed behavior) with adversarial coverage that
compiles against the as-shipped cache and pins the post-#130 correct
behavior.

- `crates/chelis-compiler-api/tests/redteam_typecheck_cache.rs`: 15
  tests on the per-PR `ci` profile covering distinct-roots-do-not-
  collide (negative parity for the original bug), `load_if_fresh`
  rejecting a foreign-root identity as a clean miss, identity
  canonicalization equivalence, identity-fingerprint sensitivity to
  every component and its boundary length-prefix, torn-write /
  truncation / hostile-byte rejection on the recompute fall-through,
  the format-version-4 magic and envelope-version rejection of stale
  files, `stdlib_cache_key` folding `COMPILER_VERSION` and the actual
  decl bytes, and the `CacheIdentity` bincode round trip.

## [0.7.10] — 2026-05-14

### Fixed - negative axes and rank-0 standalone parameters in IR lowering

`chelis eval` / `chelis test` panicked during IR lowering of any
package that links bundled chelis-std: `softmax axis requires a
statically known axis in IR lowering`. Two distinct pre-existing bugs
(both present on the v0.7.9 release tag) shared that one
downstream-blocker symptom.

Bug B - negative axes. `extract_axis` in IR lowering did `*n as usize`,
so a negative axis literal like `-1` became `usize::MAX`; `tier2`
indexed past the operand rank and `require_dim` panicked. The type
checker was inconsistent: `gather`/`scatter` normalized negative axes,
the reductions rejected them outright (`requires non-negative axis`),
and `softmax` never validated its axis at all. Negative axes are a
supported, uniform convention - `-1` is the last axis - so the spec,
the checker, and lowering now all agree on it.

Bug A - rank-0 standalone parameters. A top-level def whose parameter
types come from a separate `sig` declaration desugars to bare, untyped
`fn` params. Standalone library lowering bound them to a rank-0
`default_type()`, so any shape-sensitive op on such a param hit the
same `require_dim` panic even with a non-negative axis. This is the
exact shape of `Std.Loss.CrossEntropy.loss`'s `softmax(logits, 1)`,
which poisoned the linked chelis-std context for every downstream
`chelis test`.

- `crates/chelis-ir/src/lower.rs`: `extract_axis` is now
  `extract_axis_raw` (returns the raw `i64`); a new rank-aware
  `normalize_axis` helper maps a negative axis to `rank + axis` and
  turns a still-out-of-range axis into a clean `LowerDiagnostic`
  instead of a `require_dim` panic. Wired into all 7 axis-taking
  lowering sites (softmax, mean, sum, max_reduce, the
  min/prod/argmax/argmin reduction family, gather, scatter_replace),
  each fetching the operand rank from the operand node (falling back
  to the ascribed `app` type when the operand node is rank-0).
- `crates/chelis-types/src/infer.rs`: the type checker normalizes
  negative axes the same way so the IR only ever sees non-negative
  axes. `check_reduction_signature` no longer rejects negative axes;
  `softmax` now validates and normalizes its axis; shared
  `resolve_builtin_axis` / `resolve_axis_pair_member` helpers carry the
  normalization for gather, scatter, scatter_replace, cumsum, sort,
  split, trace, and diagonal. For Bug A, a new `annotate_params_node`
  stamps each def's declared `sig` parameter type expressions - copied
  verbatim, so `&` borrow wrappers survive - onto the `(params ...)`
  node, plumbing the type the checker already inferred to where
  `lower_fn` reads it. Defs with no `sig` keep bare params so
  `infer_signature_metadata`'s read-only/borrow inference is
  unaffected.
- `spec/05-risc-primitives.md`: the **Axis** paragraph now states the
  negative-axis-indexes-from-the-end convention explicitly, resolving
  an internal contradiction with the formula examples that already use
  `axis=-1`.
- Coverage: `crates/chelis-ir/tests/negative_axis_normalization.rs`
  (negative + positive axis parity per op, out-of-range as a clean
  error not a panic, and the rank-0 standalone-param regression) and
  `crates/chelis-cli/tests/downstream_chelis_std_axis_oracle.rs` (a
  minimal downstream package importing chelis-std runs `chelis test`
  clean). Both on the per-PR `ci` profile.
### Fixed - gate.py CI-parity parser catches chelis invocations

The `scripts/test_gate.py` CI-parity lock greps
`.github/workflows/ci.yml` and fails if a gate job hand-inlines a
command that `scripts/gate.py` does not produce. RT-2 found the
parser's command filter only matched `cargo `-prefixed invocations, so
a bare `chelis ...` command inlined into the `lint-and-unit` or
`integration` job slipped past the lock undetected, contradicting the
"every `cargo`/`chelis` invocation" claim in the `test_gate.py`
docstring and `docs/investigations/test_toolchain_guards_design.md`.

- `scripts/test_gate.py`: added `_is_gate_relevant_command`, which
  matches both `cargo ` and `chelis ` prefixes; `_parse_ci_gate_invocations`
  now uses it. The `cargo run -p chelis-cli --bin chelis -- ...` form
  was already caught by the `cargo ` prefix; the bare `chelis ...` form
  is the case that was missed.
- `scripts/test_gate_parity_adversarial.py`: converted the
  `test_known_gap_bare_chelis_command_is_not_caught` known-gap test
  into `test_bare_chelis_command_is_caught`, a positive assertion that
  the lock now fails on a hand-inlined bare `chelis` command, and added
  `test_cargo_run_chelis_cli_command_is_caught` to pin the other
  `chelis` invocation shape.

### Fixed - package-identity and compiler-version in compiled-context cache keys

The Phase K `CompiledContext` disk cache keyed its cache file name only
on `(package_name, package_version, source_hash)` and re-verified only
`source_hash` in `load_if_fresh`, a content check that never checked
package identity. After PR #127 routed the no-`CHELIS_REEF_HOME` case
through the XDG compiled cache, two distinct on-disk packages that
shared name+version+byte-identical source collided on one cache file:
the second silently loaded the first's `CompiledContext`, including its
`package_root`. The `phase3t_*` integration tests reuse deterministic
package names without setting `CHELIS_REEF_HOME`, so a cold workspace
test run wrote `.ctx` files into the real `~/.cache/chelis/compiled/`
and every subsequent warm run failed. `chelis eval --file`'s
`run_eval_in_context` had no `package_root` guard, so the same
collision was a silent wrong-result risk there.

- `crates/chelis-compiler-api/src/context.rs`: added a `CacheIdentity`
  (canonical `package_root` + `COMPILER_VERSION`) to `CompiledContext`
  and the on-disk `CacheEnvelope`. `cache_file_name` / `cache_path_for`
  now fold an identity fingerprint into the file name so two distinct
  checkouts land on separate cache files. `load_if_fresh` recomputes
  the identity from the live package and the running binary and treats
  a mismatch as a clean miss, not a stale hit. The cache format version
  and magic bumped to `4`, so pre-existing `V3` entries are rejected as
  `UnsupportedVersion`.
- `crates/chelis-compiler-api/src/stdlib_cache.rs`: folded
  `COMPILER_VERSION` into `stdlib_cache_key`. `STDLIB_CACHE_FORMAT_VERSION`
  only guards the on-disk struct shape, so without this a chelis binary
  built from different compiler source but the same bundled chelis-std
  could read an older binary's cached sub-context.
- Regression coverage in `phase_i_disk_cache.rs` and
  `stdlib_cache.rs` tests: the warm-run package-identity collision and
  the compiler-version skew are both pinned, on the per-PR `ci` profile.

### Changed - split heavyweight e2e off the per-PR integration gate

Moved the heavyweight end-to-end tests off the per-PR `Integration
Tests` job and onto a new nightly workflow, so per-PR CI wall-clock is
not gated by tests that run a real `chelis build` + gcc + run, a real
`reef build` / `reef install`, or a full-stdlib `chelis check`. Each of
those re-typechecks or rebuilds the whole chelis-std transitive graph
and costs 9-28s wall, and nextest cannot parallelize within a single
test, so they set a hard multi-minute wall-clock floor on the per-PR
gate. They are high-level end-to-end signal, not inner-loop coverage.

- `.config/nextest.toml`: rewrote the profile structure. The `default`
  and `ci` profiles now carry a `default-filter` that excludes the
  named heavy e2e suite; a new `nightly` profile selects exactly that
  set and runs with `retries = 2` so a transient gcc/CPU-contention
  hiccup does not fail the nightly run. The heavy set is named
  explicitly (per-binary `binary_id()` and per-test `test()` entries)
  so adding or removing a heavy test is a reviewable diff.
- `.github/workflows/heavy-e2e.yml`: new workflow running
  `cargo nextest run --workspace --profile nightly` on a nightly
  `schedule:` and on `workflow_dispatch:` only. It does not run per-PR
  and does not run on push-to-main.
- The previous `gcc-compile-and-run` concurrency-cap test group is
  removed: capping `max-threads` traded per-PR wall-clock for
  flake-stability, and once the heavy tests are off the per-PR gate the
  per-PR contention it addressed is moot. The nightly job's
  `retries = 2` absorbs any residual transient gcc contention there
  instead.
- `crates/chelis-reef/src/lib.rs`: `#[ignore]`-gated
  `prepare_reef_graph_amortizes_work_across_multiple_files`. Its
  assertion compares wall-clock elapsed time against a 3x multiplier of
  a sub-millisecond baseline, which is inherently contention-sensitive.
  Per CLAUDE.md's separate-perf-from-correctness rule it is now a
  documented manual perf gate; the contention-independent correctness
  check lives in `prepare_reef_graph_split_matches_single_shot_semantics`,
  which stays in the default pass.
- `docs/manual_gates.md`: registered the newly ignored reef perf gate
  with its manual command and expected success condition.
- `docs/investigations/integration_test_suite_trim.md`: recorded the
  heavy e2e split, the per-test keep/move categorization, and the
  dedupe candidates flagged for follow-up review.
### Added - cross-process chelis-std typecheck cache

`chelis check` and `chelis build` no longer re-typecheck the entire
chelis-std import graph from scratch on every invocation. The
typechecked + lowered chelis-std library sub-context is now content-
addressed and cached on disk, shared across every process and every
stdlib-importing fixture.

- New `StdLibContext` sub-context cache in `chelis-compiler-api`,
  keyed on the struct-format version, the bundled-stdlib version +
  archive + shell hashes, and a hash of the actual linked chelis-std
  decls. The decl hash keeps the key honest when a `chelis-std`
  checkout is itself the root package being checked.
- Three-layer build: cached chelis-std `StdLibContext` (Layer 1), the
  non-chelis-std library decls checked `_with_context` against it
  (Layer 2), and the entry checked against Layer 2 (Layer 3). The
  monolithic typecheck path stays byte-identical and is reachable via
  `CHELIS_STDLIB_CACHE_DISABLE=1`.
- Mandatory XDG cache-dir fallback (`$XDG_CACHE_HOME/chelis/` then
  `~/.cache/chelis/`) when `CHELIS_REEF_HOME` is unset, applied to both
  the new typecheck cache and the existing `CompiledContext` cache, so
  test workers that do not set `CHELIS_REEF_HOME` get cache reuse.
- Atomic temp-file + `fs::rename` writes with a magic header, version
  envelope, and payload SHA-256; corrupt or torn cache files fall
  through to a full recompute rather than aborting.

Acceptance oracle: `crates/chelis-cli/tests/stdlib_typecheck_cache_oracle.rs`
and `stdlib_typecheck_cache_concurrency.rs` pin cold-vs-warm and
monolithic-vs-layered byte-identical output, the stale-key negative,
and concurrency / corruption resilience.

### Changed - backend integration test suite parsimony pass

Trimmed redundant gcc/codegen invocations across the backend-crate
integration test cluster without losing coverage.

- `crates/chelis-backend-c/tests/simd_reductions.rs`: collapsed the
  30-test 5-op-by-6-size matrix into 5 tests (one per reduction op).
  Each test now emits one C program that sweeps every size internally,
  cutting gcc compile-and-run invocations from 30 to 5 while still
  checking scalar-vs-SIMD agreement at every AVX2 lane-boundary size.
- `crates/chelis-backend-hip/tests/perf_f1_strided_batched_default.rs`:
  removed; its two Perf-F1 structural acceptance cases were moved into
  `red_team_w5_strided_batched.rs` (the keep-by-default strided-batched
  lock file), since the W5 adversarial cases did not subsume the rank-3
  literal-stride dispatch and broadcasted-lhs helper-loop fallback.
- `crates/chelis-backend-metal/tests/codegen_structure.rs`: removed the
  three `wsm1_{f32,f16,bf16}_matmul_routes_to_*` tests, which duplicated
  the per-dtype routing assertions in
  `dtype_matrix.rs::matmul_{f32,f16,bf16}_*` (the canonical per-dtype
  home). The non-routing `wsm1` tests (subgraph folding, M/N/K uniform
  packing) stay.
- `crates/chelis-backend-hip/tests/gpu_correctness.rs`: removed
  `ws_a3_hip_admits_bf16_matmul_at_codegen`, a "does not panic" subset
  of `ws_a3_bf16_f16_matmul.rs::bf16_matmul_default_accumulator_emits_bf16_gemm_wrapper`,
  which additionally asserts the emitted bf16 GemmEx wrapper and link
  flag.

### Changed - phase3 integration test parsimony pass

Trimmed redundant tests from the `chelis-cli` `phase3*` integration
test cluster. Deleted
`phase3j_pre_oracle_build_path_repros_uniform_like_seed_succeeds` from
`crates/chelis-cli/tests/phase3j_pre_std.rs`: it asserted the seed=7
`uniform_like` vector, a strict subset of
`phase3j_pre_oracle_build_path_repros_uniform_like_seed_distinct_seeds_differ`,
which builds seed 7 and seed 42 in one program and asserts the
identical seed=7 vector. Deleted
`reef_std_parquet_module_resolves_and_type_checks` and
`reef_std_parquet_write_resolves_and_type_checks` from
`crates/chelis-cli/tests/phase3g_io.rs`: both were strict subsets of
`reef_std_parquet_module_builds_cleanly`, which imports both
`read_parquet` and `write_parquet`, type-checks a `def` for each, and
also runs `chelis build`. No invariant coverage was lost.

### Changed - e2e parsimony pass (ir/types/api/e2e/effects test cluster)

Trimmed redundant integration-test boilerplate in the
chelis-ir / chelis-types / chelis-compiler-api / chelis-e2e cluster
without dropping any pinned invariant.

- Deleted `crates/chelis-types/tests/rt_lin_div_diagnosis.rs`, a
  `println!`-only Phase G' investigation scaffold with zero
  assertions. The Phase G' linearity behavior it probed stays pinned
  by `phase_e_linearity_with_context.rs`.
- Deleted two subset tests in `crates/chelis-e2e/tests/bench_phase1e.rs`
  (`bench_phase1e_missing_pytorch_interpreter_emits_structured_skip_report`,
  `bench_phase1e_hidden_hip_device_emits_structured_skip_report`); both
  skip-reason assertions are already covered by
  `bench_phase1e_linreg_smoke_emits_structured_json`.
- Folded `parity_pair_1/2/3` in `phase_f_with_context.rs` and the three
  `eval_in_context_matches_prepare_eval_*` tests in
  `phase_g_compiled_context.rs` into table-driven tests with identical
  coverage.
- Deleted two looser duplicates in `rt1_adversarial.rs`
  (`cast_scalar_to_u8_rejected_at_check_time`,
  `out_of_i32_range_literal_default_behavior`); the exact-diagnostic
  versions in `ws_a0_rt1_unsigned_rejection.rs` and
  `ws_a0_rt1_int_overflow.rs` assert a strict superset.
  `d1_diagnostic_mentions_i64_suffix_and_cast` is kept because it
  exercises an int64-return-position input no `ws_a0_*` test covers.
- `#[ignore]`-gated the wall-clock perf-ratio test
  `rt_g_compose.rs::g5_cold_path_overhead_at_most_2x_monolithic`,
  separating it from the correctness pass per the CLAUDE.md
  perf-vs-correctness rule, and documented it in `docs/manual_gates.md`.
### Changed - e2e parsimony pass over the wsa/wsc/rt3a integration test cluster

Trimmed redundant coverage in the `chelis-cli` `wsa*` / `wsc*` /
`rt3a*` integration test cluster without losing any pinned invariant.
Three cross-cluster triplicate tests (the WS-C-blocker reproducer and
two integer-matmul-rejection copies) were removed because the
invariant is already pinned by keep-by-default regression locks in
`rt3a_adversarial.rs`, `rt4_adversarial.rs`, and
`wsa5_precision_polymorphism.rs`. The WS-A6 and WS-A7 dtype-matrix
re-tests were consolidated into their owning files
(`wsa6_def_annotation_desugar.rs`, `wsa7_bareref_return_inference.rs`),
which now carry the full arithmetic-dtype matrix. Near-identical tests
in `wsc_v3_stdlib_finish.rs` and `wsc_stdlib_generalization.rs` were
collapsed into table-driven tests with identical coverage, and two
literal-duplicate fixtures in `rt3a_adversarial.rs` were merged into
one test that keeps the union of both assertions.
### Added - test toolchain footgun guards

Three standing guards that turn recurring CI footguns into
enforcement. See `docs/investigations/test_toolchain_guards_design.md`.

- Test-timing budget: a `ci` nextest profile (`.config/nextest.toml`)
  writes per-test JUnit timing XML; `scripts/test_timing_check.py`
  flags tests that regressed past their committed budget or that are
  new and over the absolute ceiling. Thresholds are config
  (`scripts/test_timing_config.json`,
  `scripts/test_timing_baseline.json`); the baseline is hand-curated
  and regenerated explicitly via
  `python3 scripts/test_timing_check.py --update-baseline`. CI runs it
  as an informational, non-failing step.
- Em-dash visibility: `chelis lint --check` now buckets output by
  severity so blocking errors print last under a delimited header plus
  a summary line instead of being buried in advisory-warning noise.
  Investigation confirmed the `no-em-dash-in-public-strings` (§8.6)
  rule already catches em dashes in raw strings, `format!` arguments,
  and multi-line strings; the recurring failure was visibility, not a
  parser gap.
- CI-gate parity: `scripts/gate.py` is the single source of truth for
  the per-PR developer gate; `.github/workflows/ci.yml` calls it per
  stage and `scripts/test_gate.py` asserts the workflow inlines no
  gate command the script does not produce. `AGENTS.md` now points at
  `python3 scripts/gate.py` and is corrected to run `cargo nextest`
  and include `chelis lint --check .`.

## [0.7.9] — 2026-05-14

### Fixed - chelis check advisory warnings now match chelis lint --check

`chelis check` (the workflow `chelis reef build` invokes on `.ch`
files) flooded with `redundant-linearity-call` and
`prefer-pipe-operator` advisory warnings that `chelis lint --check`
already suppressed. The two emit paths had diverged: `cmd_lint`
applied `should_suppress_unfixable_violation` before printing kept
violations, but `emit_advisory_lint_warnings_for_file` (the path
`chelis check` invokes) applied only the path-glob exception filter.
For rules opted into `check_mirrors_fix=true`, a non-actionable
warning whose autofix the typed-pipeline gate rejects therefore still
fired through `chelis check`. Fixed by threading
`should_suppress_unfixable_violation` into the advisory-emit path so
both code paths apply the same gate. A genuinely-redundant `copy()`
on an owned tensor still warns (the autofix is safe). Closes
`Lint-CheckMirrorsFixAdvisoryEmitLeak-F1`. Regression tests at
`crates/chelis-cli/tests/red_team_0_7_9.rs` (`lp_leak_a`, `lp_leak_b`,
`lp_leak_fix_*` positive control).

### Fixed - lint workspace-root detection uses a real Cargo workspace probe

`detect_lint_workspace_root` was implemented as
`canonicalize(current_working_directory)` with no workspace probe, so
the workspace-rooted exception matching shipped in the previous fix
held only when `chelis lint` was invoked from the workspace root.
Running `chelis lint --check .` from a subdirectory, or `chelis lint
--check /abs/workspace` from an unrelated directory, re-surfaced the
false-positive `surf-def-arrow-form` errors. Fixed by detecting the
workspace root with `cargo locate-project --workspace`, Cargo's own
canonical workspace-locating probe, run with its working directory
set to the lint target's directory rather than the process CWD. When
the targets are not inside any Cargo workspace, no workspace-rooted
exception glob can apply, so violations pass through unfiltered
(linting a loose file outside a workspace remains supported). The
same detection is now shared by `cmd_lint`, the advisory-emit path,
and the build-time style gate. Closes
`Lint-WorkspaceRootCwdAssumption-F1`. Regression tests at
`crates/chelis-cli/tests/red_team_0_7_9.rs` (`le_leak_a`,
`le_leak_fix_sibling_path_invocation_matches_workspace_root_invocation`).

### Fixed - type checker rejects divergent declared dimension parameters

`chelis check` previously accepted a function whose declared return type
and body type differed in dimension *identity* but matched in dimension
*rank*. A definition like
`def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = y`
type-checked with `score=1` and then evaluated with a runtime shape
mismatch (HIGH severity silent miscompilation, red-team finding
SR-LEAK-A). Two independent leak paths are closed:

- `types_structurally_equal` (the Shape A relaxed-retry guard) compared
  tensor dimensions by rank only. It now compares dimension *identity*:
  two dims match iff same concrete name, same literal, or the same dim
  variable. `Dim::Wildcard` still matches anything. Distinct dim
  variables (`n` vs `m`) no longer satisfy the structural guard.
- A plain owned-tensor body bypassed the relaxed-retry entirely: the
  post-body signature unification collapsed two distinct declared dim
  parameters via free dim-variable unification. A new post-body rigidity
  check, `check_declared_dvars_rigid`, flags this `Var->Var` collapse
  (and the previously-handled `Var->Lit` pin) as a `DimensionMismatch`.
  Declared dimension parameters are rigid within the def body.

Closes `TypeCheck-FreeDimVarUnification-F1` (red-team finding
SR-LEAK-A, `docs/investigations/terminal_redteam_0_7_9.md`). Diagnosis:
`docs/investigations/typecheck_dim_identity_diagnosis.md`. Regression
tests at `crates/chelis-cli/tests/red_team_0_7_9.rs` (the SR-LEAK-A
fixtures, divergent-dim negatives plus same-dim positive controls for
both paths). Spec completion: `spec/04-type-system.md` §4.4 now states
the dim-parameter rigidity rule the fix enforces.

### Fixed - redundant-linearity-call false-positive on copy(borrow)

The `redundant-linearity-call` advisory warning previously fired on
`copy()` calls where stripping the call would leave the program with
a borrow-to-owned type mismatch the implicit-copy inserter cannot
bridge. The CLI driver's typed-pipeline gate correctly suppressed the
`[fix]` marker, but the warning itself still fired, leaving 137+
false positives in Nautilus `src/linalg.ch` against 0.7.8. Fixed by
opting the rule into `check_mirrors_fix=true` so the CLI driver's
`should_suppress_unfixable_violation` helper suppresses the warning
when the typed-pipeline gate rejects the proposed strip. Closes
`Lint-RedundantLinearityCopyOnBorrowWarn-F1`. Regression tests at
`crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F10,
F11 negative control).

### Fixed - redundant-linearity-call false-positive on 2-arg list primitives in pipe form

The pipe form `xs |> drop(n)` puts a literal single-arg `drop(n)` in
the source text, but semantically it is the 2-arg list-drop with the
first argument piped in. The rule's syntactic
`has_single_top_level_argument()` accepted the pipe-form shape and
fired a redundant-linearity-call warning, which the user could not
satisfy without breaking the program (Coral reported the pattern
against 0.7.7 in `src/internal/hamt.ch` and `src/internal/window.ch`
and worked around it by writing the 2-arg form directly). Closed
transparently by the same `check_mirrors_fix` opt-in above: the
typed-pipeline gate rejects the strip `drop(n) -> n` and the warning
is suppressed. Closes `Lint-PreferPipeRedundantLinearityPair-F1`.
Regression tests at
`crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F12,
F13 negative control).

### Fixed - lint exception path matching anchored at workspace root

`chelis lint --check crates docs examples packages` produced 6
false-positive `surf-def-arrow-form` errors against
`crates/chelis-surf/tests/fixtures/*.ch` that `chelis lint --check .`
correctly excepted. CI invokes the `.` form so the gate was not
broken, but developers linting sub-trees saw spurious errors. Root
cause: `apply_exceptions` at
`crates/chelis-lint/src/exceptions.rs::is_excepted`
strip-prefixed the violation path against the walk-target root rather
than a shared workspace root, so workspace-rooted exception patterns
silently failed to match under sub-directory walks.

Fixed by adding a `workspace_root` parameter to `apply_exceptions`
and `is_excepted`, distinct from the walk root, and anchoring
prefix-stripping against it. The CLI detects the workspace root by
canonicalizing the current working directory at the boundary,
reusing the canonicalize-at-CLI-boundary pattern PR #93 established
for walk targets. No walk-up filesystem search is introduced (per
`feedback_no_walkup_filesystem_detection.md`).

Updates four call sites:
`cmd_lint`, `apply_lint_fixes`,
`emit_advisory_lint_warnings_for_file`, and
`style_gate::run_lint_for_single_file`. Closes
`Lint-ExceptionPathRoot-F1`. Regression tests at
`crates/chelis-cli/tests/lint_path_walk_consistency.rs`
(`lint_cli_exception_pattern_matches_under_subtree_and_cwd_walks`
and `lint_cli_exception_pattern_matches_under_multi_subtree_walk`)
plus a unit test in
`crates/chelis-lint/src/exceptions.rs::tests::apply_exceptions_anchors_on_workspace_root_not_walk_target`.

### Fixed - implicit-copy Shape A covers let/if/match tail-position returns

PR #91's W4-A relaxed-retry in `shape_a_relaxed_return`
(`crates/chelis-types/src/infer.rs`) only accepted a bare
`(fn (params...) (var x))` body, so

```surf
def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = { y = x; y }
def g[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] = if c then x else x
def h[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] =
  match c with { | Left => x | Right => x }
```

all surfaced `def 'f' body doesn't match declared signature` even
though the tail expression of every desugared body is the same
bare-var ref the v3 fix already accepts. Added
`descend_to_tail_var` next to `shape_a_relaxed_return`; the walker
descends through `(let bind body)`, `(if cond then_e else_e)`, and
`(match scrutinee arm ...)` and returns `Some(name)` only when every
sibling branch resolves to the same bare-var name. The structural
type-equality check and relaxed-type construction are unchanged, so
the broader gate stays as conservative as PR #91's. Closes
`Linearity-ShapeABroadReturn-F1`. Regression fixtures at
`crates/chelis-ir/tests/implicit_copy_shape_a_broader_return.rs`.

## [0.7.8] — 2026-05-13

### Fixed - host-eval scalar zero-arg fn-call silent miscompilation (#80)

`def go -> f32 = 7.5; result = go()` previously evaluated `result`
as `0.0` via `chelis eval --file` instead of `7.5`. Root cause was
an off-by-one arity guard in IR lowering: `LowerCtx::lower_app` at
`crates/chelis-ir/src/lower.rs:2712` rejected zero-arg
`(app {meta} (var fn-name))` forms (3 elements: tag + meta + fn)
via `if elems.len() < 4`, emitting `RiscOp::Const { value: 0.0 }`
before any callable resolution. Fixed by relaxing the guard to
`if elems.len() < 3`; the downstream arms already handle empty
arg slices correctly. Closes `HostEval-ScalarFn-F1`. Regression
tests at `crates/chelis-cli/tests/host_eval_scalar_fn_call.rs`
cover f32/f64/i64/bool zero-arg return types plus a one-arg
negative control.

### Changed - linearity checker uses typed `ConsumeKind` discrimination (#83)

`ConsumeSite` now carries an `enum ConsumeKind { Aliasing,
Structural }` field. The previous string-prefix check
(`descriptor.starts_with("binding ")`) at
`crates/chelis-types/src/linearity.rs::read_or_error` is replaced
by `matches!(site.kind, ConsumeKind::Aliasing)`. All eight
consume-site producers set `kind` explicitly. Closes
`Linearity-F1`.

### Fixed - alias-consume linearity bypass (#83)

`let y = x; let z = realize(y); add(x, z)` previously passed
`chelis check` silently because `y`'s linearity state was tracked
per-bound-name, not per-underlying-value. Fixed by adding a
`LinearScope.aliases: HashMap<String, Vec<Option<String>>>`
parallel to `bindings` and `types`. `Structural` consumes
forward through `resolve_alias_chain` to the underlying source
name's scope entry; multi-level chains (`let z = y; let y = x;
consume(z)`) walk to `x`. Closes `Linearity-AliasedConsume-F1`.
Regression tests at
`crates/chelis-types/tests/linearity_aliased_consume.rs`.

### Fixed - tuple-destructure linearity false-negative now errors (#83 + #90)

`let (a, b) = pair; realize(a); realize(a)` previously passed
silently because `chelis-surf` desugar synthesized `__chelis_tmp_N`
intermediaries without type metadata, making
`expr_is_owned_linear` return false and skipping the entire
destructure chain. Fixed in PR #83 via a local
`tuple_get_element_type` helper that indexes into the underlying
tuple var's `t-tuple` scope-type at linearity-check time. PR #90
swept the production corpus (zero warnings surfaced), then
removed the temporary `LinearityInfo::warnings` cascade channel
and routed destructured-component use-after-consume directly
through `errors`. Closes `Linearity-F2`. Regression tests at
`crates/chelis-types/tests/linearity_typed_consumekind.rs` and
`linearity_aliased_consume.rs`.

### Changed - C runtime tensor data is dtype-aware (#84 + #86 + #87 + #88)

`chelis_tensor.data` is now `*mut u8`, accessed through a new
`TensorElement` trait with checked `data_ptr`, unchecked
`data_ptr_unchecked` (debug_assert), and a default `fill` lifting
the cast-then-loop pattern from `chelis_fill_i64`. Trait impls
for `f32, f64, i32, i64` (`bool` and `i32` route through `f32`
internally because the runtime stores them f32-encoded today —
tracked separately as `CRuntime-BoolStorage-F1` and
`CRuntime-I32Storage-F1`). `CHELIS_*` constants promoted to
`pub const`.

Migration shipped in four PRs: PR #84 introduced the trait +
struct change + first anchor migrations; PR #86 migrated 31
runtime call sites across 14 ops; PR #87 migrated 6 host_emit
code-generation sites; PR #88 added 22 multi-op composition
fixtures and the workstream-wide sibling sweep audit.
**110 dtype-coupling fixtures lock the migration**:
`crates/chelis-e2e/tests/dtype_op_matrix.rs` (77),
`crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs` (11),
`crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`
(5), plus the four 0.7.6 surface-fix regression locks.
Sibling-sweep audit: 11 intentional `*mut f32` references remain
in `crates/chelis-runtime/` (each enumerated with justification);
0 in `crates/chelis-backend-c/`, HIP, Metal, IR. Closes
`CRuntime-F32Coupling`. Audit note:
`docs/investigations/c_runtime_dtype_coupling_workstream_audit.md`.

### Fixed - implicit-copy inserter handles return-position borrow-to-owned and grad/vmap fan-out (#91)

Two fan-out shapes that previously failed:

- **Shape A** (return-position borrow-to-owned): `def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x`
  previously failed with a type mismatch. Fixed at
  `crates/chelis-types/src/infer.rs::check_top_level` via a
  relaxed-retry path in the def-body unify that recognizes
  body-as-bare-var with `Ref(T)` inferred and `T` declared.
  Limited to the bare `(var x)` body shape; broader return-position
  coercions (`let`/`if`/`match` tails) are tracked as a follow-on.
- **Shape B** (grad/vmap-callee fan-out): `dw = grad(f, wrt=w)(args); db = grad(f, wrt=b)(args)`
  previously failed with `UseAfterConsume` on the fanned-out args.
  Fixed at
  `crates/chelis-types/src/linearity.rs::arg_is_borrowed` via a
  new `callee_is_observational_higher_order` helper that treats
  every arg of a `(grad ...)` or `(vmap ...)` callee as borrowed
  rather than consumed. Covers piped grad/vmap stages too
  (pipe-stage path routes through the same `arg_is_borrowed`).

Closes Item 1 v3 of the implicit-copy inserter rollout (after
PR #29 v1 and the v2 fix-up). Six fixtures at
`crates/chelis-ir/tests/implicit_copy_fanout_v3.rs`.

### Changed - lint precision: math/ML prefixes, ecosystem allowlist, kebab-case docs, docstring em-dash (#92)

Four lint-rule precision fixes:

- **§7.1.1** `prefix-namespace` accepts a curated math/ML
  well-known prefix list (`exp_`, `log_`, `sin_`, `cos_`, `tan_`,
  `sqrt_`, `lin_`, `std_`, `var_`, `min_`, `max_`, `sum_`,
  `mean_`, `relu_`, `gelu_`, `silu_`, `tanh_`, `sigmoid_`) in
  addition to module-domain-derived prefixes. Module-level
  allowlist annotations are also supported.
- **§6.3** `module-pascal-components` `KNOWN_SINGLE_WORDS`
  extended by 18 entries (4 from the hello-chelis report —
  `Linearity`, `Hypothesis`, `Integration`, `Optimize` — plus a
  math/ML sweep adding 14 more). `Vocabulary-F2` structural
  closure (auto-generation from spec §2.6) remains pending.
- **§8.3** `doc-filename-convention` accepts kebab-case in
  mdBook-rooted narrative-doc trees (detected by walking
  ancestors for `book.toml`) and accepts hyphens when the
  filename stem matches a workspace Cargo `[package].name`
  (e.g., `c-earchin.md` matches package `c-earchin`).
- **§8.6** `no-em-dash-in-public-strings` excludes Python
  docstrings (first non-whitespace triple-quoted string on its
  line). Mid-line triple-quoted strings (`print("""…""")`)
  remain in scope.

Spec text updated at `spec/01-nomenclature.md` §7.1.1, §8.3, §8.5,
§8.6.

### Fixed - lint CLI produces identical output for CWD walk and explicit-path walk (#93)

`chelis lint --check src/ tests/ verify/ scripts/ docs/`
previously returned a different error count than
`chelis lint --check .` because
`doc-filename-convention::classify_doc` did substring matching on
the walker's path output. `walkdir::WalkDir::new(root)` yields
paths prefixed with the root passed in, so `.` produced
`./docs/file.md` (substring `/docs/` matched) while `docs/`
produced `docs/file.md` (no leading slash, no match). Fixed by
canonicalizing user-supplied target paths to absolute paths at
the CLI boundary in `cmd_lint`. Companion fix to
`crates/chelis-lint/src/walker.rs::is_skip_dir` ensures the
worktree-scoped skip filter never matches the walk-root entry
itself.

Sibling-sweep finding: `chelis-lint/src/exceptions.rs::is_excepted`
has parallel path-spelling sensitivity in the opposite direction
(`crates/chelis-surf/tests/fixtures/*.ch` patterns fail when the
CLI walks a sub-directory). Filed as `Lint-ExceptionPathRoot-F1`.

### Added - `redundant-linearity-call` autofix covers implicit-copy v3 shapes (#95)

The implicit-copy fan-out v3 fix (Shape A borrow-to-owned at return
position and Shape B grad/vmap fan-out across observational
higher-order calls) closes a coverage gap for the
`redundant-linearity-call` autofix. The Path 1B safety gate
(typed-pipeline-accepts) now accepts strip candidates for these
shapes, so `chelis lint --check` emits the `[fix]` marker and
`chelis lint --fix` rewrites them. No rule logic changed; the
unlock comes entirely from the upstream typecheck and linearity
fixes. See
`docs/investigations/redundant_linearity_autofix_recoverage_diagnosis.md`
for the diagnosis.

### Fixed - `numel(to_tensor([]))` returns 0 for empty input (#100)

Three clamp sites inflated a zero-element shape product back to one:
`eval_builtin "numel"` did `shape.iter().product::<usize>().max(1)`,
and both `chelis_alloc_tensor` and `chelis_alloc_view` had
`if size == 0 { size = 1 }`. Shape inference was correct
(`to_tensor([])` produces rank-1 shape `[0]`); the bug was in
`numel` reporting only. Dropping the three clamps lets the
empty-product identity (`[].iter().product() == 1`) handle the
scalar case naturally while rank-1 zero-element tensors correctly
report `numel = 0`. Allocator safety preserved by the existing
`bytes.max(1)` on `posix_memalign`. Closes
`Runtime-EmptyTensorNumel-F1`, the upstream root cause of the
Coral-reported `filter`/`head`/`tail`/`slice` crashes on empty
results. Regression tests at
`crates/chelis-cli/tests/eval_empty_tensor_numel.rs` and
`crates/chelis-cli/tests/cbackend_empty_tensor_numel.rs` (eval-vs-C
agreement).

## [0.7.7] — 2026-05-12

### Added - pipe-stage callable surface

`chelis check`, `chelis eval`, and `chelis build` now accept callable
shapes as pipe stages that previously required an explicit lambda or
rewrote-to-application form:

- `x |> grad(f)` and `xs |> vmap(grad(f))` lower through the existing
  application path. Equivalent to `grad(f)(x)` and `vmap(grad(f))(xs)`.
- `x |> f` where `f` is a function-valued parameter of a callable.
- `x |> tensor_to_scalar` and other non-elementwise unary primitives
  (the in-IR lowering now mirrors the host lane's beta-reduction path).
- `x |> realize` as a bare-keyword pipe stage (parser synthesizes a
  lambda; the existing realize inference accepts the resulting form).

### Added - implicit-copy fan-out covers var-RHS let-bindings

The linearity checker now permits implicit copy insertion for
cross-statement var-RHS let-aliasing such as `let alias = x; mul(x, alias)`.
The DAG-level Copy node is inserted automatically when fan-out across
non-borrow consume sites is detected; explicit `copy(x)` is no longer
required in this shape.

### Added - lint allowlist now matches canonical sources

- `deep-user-symbol-charset` accepts `t-ref` (the canonical compound
  tag for read-only borrow types in §1.4 / §2.5). `chelis deep` output
  no longer self-conflicts with `chelis lint --check`.
- `module-pascal-components` accepts `Capstone` as a top-level
  ecosystem module prefix; `spec/01-nomenclature.md` §2.6 documents it
  alongside the other ecosystem packages.

### Added - `chelis eval --file` warns when there is nothing to evaluate

Programs whose `--file` contains only `def` declarations no longer
silently return success with no output. A stderr warning is emitted
(`warning: input contains only def declarations; nothing to evaluate`)
and the process exits 0. Scripted consumers are unaffected.

### Fixed - lint auto-fix re-enabled with typed-pipeline proof

`chelis lint --fix` now applies `redundant-linearity-call` and
`prefer-pipe-operator` rewrites again. The fixer drives proposed
rewrites through the typed pipeline and only writes the result when
parsing, type-checking, and linearity all accept it. The 0.7.6
limitation that disabled these auto-fixes is closed.

### Internal

- `inlining_names` recursion guard narrowed: legitimate nested
  fn-typed parameter applications (`f(f(x))`) no longer trip the
  inlining guard's false-positive rejection. True self-recursion
  still terminates with the documented fallback.
- Dead `t-dims` parsing in `chelis-ir::lower` removed (vestige of the
  pre-flattened `t-tensor` form).

## [0.7.6] - 2026-05-10

### Fixed - conservative lint auto-fix rollout

`chelis lint --fix` no longer rewrites `redundant-linearity-call` or
`prefer-pipe-operator` warnings. Those rules remain visible as
warnings, but their source rewrites are disabled until the fixer can
prove that removing explicit linearity calls or converting nested calls
to pipes preserves type, ownership, and call-argument behavior.

## [0.7.5] — 2026-05-10

### Fixed - context lowering for shell test runs

Context-based evaluation now uses the same library-aware lowering map
when lowering new code that the compiler API uses when counting new-code
roots. This fixes `chelis test` failures in downstream shells where a
test wrapper called a host-only library helper returning a scalar value.

## [0.7.4] — 2026-05-10

### Added - lint auto-fix and cleanup rules

`chelis lint` now supports `--fix`, `--rules`, and `--list`. The lint
engine records rule severity, applies non-overlapping source fixes
in-place, and distinguishes `chelis-lint: allow` from
`chelis-lint: keep`: allow suppresses diagnostics and fixes, while
keep preserves the source but can still warn.

The first fixable cleanup rules are `redundant-linearity-call` and
`prefer-pipe-operator`. Public string literals are now checked by the
blocking `no-em-dash-in-public-strings` rule; the initial rollout
cleaned existing source strings while leaving Markdown-prose
enforcement queued for a later doc-corpus pass.

## [0.7.3] — 2026-05-10

### Fixed — lowering diagnostics and release test coverage

Chelis lowering now exposes structured diagnostics for valid language forms that
are not supported by IR evaluation instead of letting internal lowering panics
escape through user-facing commands. Diagnostics include source offsets or span
IDs when available and point users at `chelis build --target c` when the C host
backend is the supported path.

`chelis test` now has regression coverage for pipe stages that must fall back to
the host runtime, and the release workflow runs the same pipe-stage fixture
against the built release binary before publishing assets.

## [0.7.2] — 2026-05-10

### Fixed — bridge provenance diagnostics

`chelis prove` now resolves c-earchin `.spans.json` manifests for Deep bridge
properties. Human failure/error diagnostics show the originating EARS file,
line, column, requirement ID, and requirement text; JSON output includes the
same data under `source.requirement`.

## [0.7.1] — 2026-05-10

### Added — first-class Surf properties and `chelis prove`

Chelis now supports Level 2 executable properties with canonical Surf syntax:
`@property NAME forall(params...) where ...: expr`. Properties desugar to
ordinary Deep `defsig` plus `def` forms and carry canonical property metadata:
`chelis_role`, `property_source_kind`, `property_quantifiers`, and
`property_preconditions`.

The new `chelis prove` subcommand discovers Surf properties and Deep bridge
witnesses, runs deterministic type-directed sampling, filters false
preconditions without calling the predicate, and reports stable human or NDJSON
output. V1 supports scalar binders and fixed-shape numeric tensors; unsupported
selected properties exit `2`, failures exit `1`, and setup/input errors exit
`3`.

Deep bridge compatibility accepts legacy `c_earchin_role:
"property_witness"` metadata while c-earchin moves to the canonical Chelis
property metadata contract.

### Added — style gate on `chelis build` / `chelis check` / `chelis validate` / `chelis eval --file`

The four CLI ingestion paths now enforce `chelis fmt --check` and the
full `chelis lint` rule set on the user-supplied source file before
the front-end pipeline runs. Non-canonical formatting and any lint
violation fail the command with a one-line-per-issue diagnostic and
an actionable hint (`run \`chelis fmt --inplace <file>\` to fix`).

The new `--allow-style-violations` flag bypasses the gate with a
stderr warning. CI must not pass it; it exists for emergency builds
and one-off migrations. The opaque environment variable
`CHELIS_STYLE_GATE_DISABLE=1` does the same and is reserved for
integration-test harnesses that synthesize ad-hoc Surf to exercise
type/effect/linearity behavior independently of style.

`chelis eval EXPR` (the inline-expression form) is unchanged — the
expression has no on-disk source to check.

### Added — four new lint rules to reflect §3 and §10.1 of the style guide

- `surf-type-pascal-case` (§3.1): `type Name` declarations must be PascalCase.
- `surf-value-snake-case` (§3.2): `def name` declarations must be snake_case (no leading underscores, no uppercase letters).
- `surf-test-name-prefix` (§10.1): functions carrying the `Test` effect must be named `test_*` or `example_*`.
- `surf-def-arrow-form` (§3.5, new spec section): `def name(params) -> T = expr` is the canonical signature form; the `def name(params) : T = expr` colon variant is flagged.

The total `chelis lint` rule set is now 13. `spec/01-nomenclature.md`
gains a new §3.5 documenting the def-arrow-form preference.

## [0.6.1] — 2026-05-06

Bootstrap-list patch. Updates DEFAULT_BOOTSTRAP_LIST in
crates/chelis-reef/src/lib.rs to point at the post-rename shell
tags (nautilus v0.6.0, coral v0.6.0, shoals v0.3.0, octant v0.4.1).
Required reef bundle rebuild against compiler =0.6.1 (from =0.6.0)
and propagated test-fixture pin updates.

No API or runtime change beyond the bootstrap-list and pin bumps.

## [0.6.0] — 2026-05-06

Ecosystem-wide naming-convention sweep. Codifies the cross-shell
identifier conventions in `spec/01-nomenclature.md`, ships a new
`chelis lint` subcommand backed by the `chelis-lint` crate that
enforces them, and bumps `chelis-std` from 0.1.0 to 0.2.0 alongside
breaking module renames.

### Added — `chelis lint` subcommand

New `chelis lint [paths...]` and `chelis lint --check` (gating mode,
exits nonzero on any violation). Backed by the new `chelis-lint`
workspace crate, which implements 9 rules covering `.ch`, `.dp`,
`.rs`, `.py`, `.sh`, `.md`, `.snap`, `Cargo.toml`/`reef.toml`, and
`.github/workflows/*.yml` surfaces. Each rule cites a section of
`spec/01-nomenclature.md`. Exception entries carry a mandatory
`cross_ref: SectionRef` field per §12 — the schema rejects free-form
prose so post-hoc justifications can't accrete in the lint config.

CI gate (`.github/workflows/ci.yml`) runs `chelis lint --check`. The
post-sweep state of chelis main reports zero violations.

### Changed (breaking) — `chelis-std` runtime renamed `Std.IO` → `Std.Io`

The `Std.IO`, `Std.IO.Csv`, `Std.IO.Json`, `Std.IO.Parquet`, and
`Std.IO.Safetensors` modules are renamed to `Std.Io.*` per §6.2 of
the recorded style guide (Title-case compound, never ALL-CAPS
abbreviation). Downstream `import Std.IO ...` statements must update
to `import Std.Io ...`. The `chelis-std` reef package version bumps
from 0.1.0 to 0.2.0 to reflect this. The bundled artifacts in
`crates/chelis-std-bundle/dist/` are regenerated.

### Changed — `Hellotensor` → `HelloTensor` in `examples/`

The `module Hellotensor` declaration in the canonical hello-world
example becomes `module HelloTensor` per §6.3 (PascalCase per
component, including each word inside a compound). The on-disk
filename `examples/hello_tensor.ch` is unchanged.

### Changed — `phaseA_*.rs` → `phase_a_*.rs` integration tests

Five `crates/chelis-cli/tests/phaseA_*.rs` files renamed to
`phase_a_*` per §9.1 (lowercase phase-letter form). Function names
inside the tests retain their `phaseA_*` style — only the filenames
moved.

### Changed — `.github/scripts/smoke_macos_accelerate.sh` ported to `.py`

Per §2.9 (no shell scripts; Python only). The macOS smoke job in
CI invokes `python3 .github/scripts/smoke_macos_accelerate.py`.

### Spec — `spec/01-nomenclature.md` recorded as canonical source

The empty 8-section glossary becomes a 13-section style guide
covering hard language constraints (§1), filesystem and manifest
naming (§2), Surf/Rust/Python identifier conventions (§§3-5),
module conventions (§6), function naming patterns (§7) including
§7.1.1 model/algorithm sub-namespaces (`bs_/mc_/gbm_/fd_/lm_/cg_/
airy_/beta_/chi_/det_/eig_/inv_/erf_`) and §7.2 type-suffix policy
with the parser/converter idiom carve-out, documentation
conventions (§8) including §8.5 mdBook source-tree exception with
`SUMMARY.md`/`README.md` tool-required exemptions, project-cutting
conventions (§9), test naming (§10), resolved escalations from the
May 2026 cleanup (§11), and the lint enforcement surface (§12).

The §11.1 Surf-vs-Deep hyphen asymmetry is recorded as intentional
structure (Deep's compound tag vocabulary uses hyphens like `t-fn`,
`pat-ctor`, `d-name`; user-defined Deep symbols inherit Surf's
no-hyphen rule via desugaring). The lint enforces a narrow guard
that any non-tag Deep symbol must satisfy the Surf identifier
charset.

The §11.2 Rust kebab-package / snake-lib hyphen→underscore
shoreline is documented in §2.2 as a Rust-language-norm crossing.

### Fixed — `scripts/regenerate_chelis_std_bundle.py`

The script invoked `cargo build -p chelis`, but the workspace
package is `chelis-cli` (the binary is named `chelis`). Fixed.

### Migration

Downstream shell-package authors:

- Update `import Std.IO ...` to `import Std.Io ...` (and any
  `Std.IO.<sub>` imports likewise).
- Bump the `chelis-std` pin in your `reef.toml` from `0.1.0` to
  `0.2.0` and the `compiler` pin from `=0.5.0` to `=0.6.0`.
- The shell repos `nautilus`, `coral`, `octant`, `shoals` are
  republishing in lockstep with this release; update their pins to
  the new tags.
