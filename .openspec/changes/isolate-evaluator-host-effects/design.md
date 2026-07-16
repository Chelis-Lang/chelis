## Context

The checked language currently marks file, process, `print`, and `debug` builtins with `IO`; the active roadmap later separates `Filesystem` and `Network`. `EvalContext` directly executes filesystem and subprocess effects through `std::fs`, path queries, and `std::process::Command`, while `print` and `debug` are already captured into a returned transcript rather than writing a terminal. Direct host execution prevents deterministic embedding and central capability enforcement. The evaluator policy registry therefore keys coverage to host-capable builtin classifications and their declared language effects, not to the assumption that `IO` remains the only relevant effect variant.

This change concerns evaluator implementation boundaries. It does not change which Chelis expressions carry `IO` or which builtins are available. It follows the decision, request, identity, secret, and failure laws in `.openspec/FCIS_ARCHITECTURE.md`.

## Goals / Non-Goals

**Goals:**

- Make every external host effect a typed request with a correlated structured observation.
- Preserve deterministic evaluator-local output events such as transcripts as returned data.
- Keep evaluation, continuation, random, and limit state explicit and isolated per run.
- Provide production, deterministic, denying, and selective-policy adapters.
- Make relative paths, subprocess cwd/environment, timeouts, output limits, mapped-snapshot limits, and deterministic evaluator fuel explicit with resolved defaults.
- Prevent nested functions, callbacks, transforms, imported code, or repeated continuations from bypassing policy.

**Non-Goals:**

- Changing Chelis effect inference or introducing user-defined effect handlers.
- Adding network builtins.
- Adding compiled-backend support for eval-only `process_run`.
- Making asynchronous evaluation part of the public language surface.
- Treating deterministic returned transcripts as terminal I/O.

## Decisions

### 1. Evaluation is a resumable state machine

The semantic core exposes an invocation-owned evaluator machine whose step operation returns progress, accepted completion, semantic rejection, protocol failure, host execution failure, or a suspension containing one `HostEffectRequest` and continuation state. Local mutation inside the machine is allowed, but all state that can influence later steps belongs to that machine.

A suspension is non-cloneable, is consumed by value, and has exactly one outstanding request. Request identities are deterministic invocation-local sequence numbers starting at zero. Resumption requires a response carrying the matching request identity and response kind and validates it before changing continuation state. Duplicate, stale, replayed, malformed, oversized, or mismatched responses are protocol errors and cannot partially mutate a live machine. Transcript replay starts a fresh machine; it never reuses a consumed suspension.

Alternative considered: pass a `HostEffectHandler` trait directly into recursive `eval_expr`. This would execute effects behind an injected interface and obscure the FCIS boundary, so the semantic machine yields data instead.

### 2. Requests and observations form a closed typed vocabulary

The initial external request vocabulary covers text/line/byte read, write, existence, directory listing, mapped-file open, and subprocess execution. Requests carry bounded normalized runtime arguments, deterministic request identity, builtin/source identity, span context, and non-secret policy references required for diagnostics. Resolved secret environment values, renderer redaction data, OS handles, and ephemeral absolute paths are not persisted in semantic transcripts. A live read or subprocess observation may transiently carry sensitive payload required by evaluation; such an observation is marked non-persistable before the core consumes it and is excluded from normalized transcripts, identities, diagnostics, and logs.

Responses are typed per request. Unknown request kinds, malformed observations, correlation mismatches, and response-kind mismatches are internal protocol errors rather than unchecked conversions.

Mapped-file open returns an evaluator-owned immutable byte snapshot. `mmap_read` and `mmap_len` remain pure operations over that snapshot.

### 3. Evaluator-local IO events are returned data

`print` and `debug` remain language-level `IO`, but their evaluator implementation appends deterministic transcript events to invocation-owned state and returns those events in the evaluation outcome. They do not issue a filesystem/process request and do not write a terminal.

The effect-coverage gate therefore requires every builtin carrying a current or future host-capable language effect to have exactly one declared evaluator policy: external request, evaluator-local captured event, or explicit unsupported-target policy. It jointly checks the builtin's declared effect (`IO` today and `Filesystem`/`Network` when introduced) rather than treating the present effect enum as permanent. It does not incorrectly require `print` and `debug` to become host terminal effects.

### 4. Semantic policy, host execution context, and rendering are separate

`EvaluationPolicy` is a semantic input containing allowed operation classes, lexical path policy identifiers, subprocess authorization, evaluator fuel, mapped-snapshot bounds, and requested subprocess timeout/output bounds. Those requested bounds are copied into typed requests because changing them can change the observation and final evaluation. `HostExecutionContext` is adapter input containing the resolved relative-path base, directory capability handles, subprocess working directory, resolved environment values authorized by the policy, and implementation safety ceilings. An adapter ceiling may be equal to or stricter than a requested bound, but a stricter executor-only ceiling reports `HostFailure::ExecutorLimit`; it must not masquerade as the semantic timeout/output-limit observation selected by `EvaluationPolicy`. `EvaluationRenderPolicy` contains secret-redaction and display configuration and is report metadata, not machine state.

The v1 defaults are resolved before machine construction:

- `DEFAULT_EVAL_FUEL = 10_000_000` semantic steps;
- `DEFAULT_MAX_MAPPED_FILE_BYTES = 268_435_456` (256 MiB); callers may select a smaller bound, and mapped open fails before allocating or persisting a larger snapshot;
- trusted CLI eval/test explicitly selects unrestricted filesystem access, follows normal host symlinks without a containment root, resolves relative paths and subprocess cwd from the invocation cwd, snapshots the inherited subprocess environment, uses a 300-second timeout, limits stdout and stderr to 64 MiB each, and uses the 256-MiB mapped-snapshot bound;
- Tide supplies deny-all filesystem and subprocess policy unless deployment configuration provides explicit contained-directory or executable capabilities;
- Python `eval_json` supplies deny-all external policy; `eval_json_with_policy` accepts explicit capabilities; `eval_json_unrestricted` selects the trusted-local production policy and is named accordingly;
- in-memory and replay adapters receive no process cwd or environment.

The semantic machine and policy mapper never inspect process cwd or environment. Requested timeout and output bounds are semantic policy/request fields; executor safety ceilings, elapsed report duration, and redaction configuration are not. Equal machine inputs plus equal validated observations produce equal decisions regardless of the adapter context that obtained those observations.

### 5. Handlers and policies live outside the semantic core

A production OS adapter preserves established trusted CLI behavior. An in-memory adapter supports deterministic tests and embeddings. A deny-all adapter rejects every external request, and a policy wrapper can authorize operation kinds, paths, and subprocess capabilities independently.

Path policy defines lexical normalization, symlink traversal, containment, and nonexistent-write parent handling. Contained production policies use directory-capability-relative operations so authorization and access use the same parent handle; they do not authorize by canonicalizing one path and reopening another. Platforms that cannot provide the required race-resistant containment fail closed for contained policies. Nonexistent writes resolve and authorize an existing parent capability before creating the final component.

A convenience `evaluate_with_handler` driver may loop over machine requests, but it belongs in an adapter module and cannot be imported by the designated semantic core.

### 6. Embedding defaults and the prerequisite Tide guard are secure and explicit

- The independent `deny-tide-evaluator-host-effects` change owns the fail-closed deny-external dispatch mode and `.venv/bin/python scripts/fcis_gate.py tide-evaluator-denial` oracle. This protocol migration requires that evidence, retains its nested denial corpus, and replaces the temporary mode with deny-all request handling only after equivalent coverage passes.
- Trusted CLI eval/test entry points select the production adapter and the v1 trusted-local defaults explicitly.
- Python exposes `eval_json_with_policy`; ordinary `eval_json` is deny-by-default, and the only unrestricted entry point is clearly named `eval_json_unrestricted`.
- Direct Rust convenience entry points either require a policy or carry `unrestricted` in their name.

These defaults and API names are part of this change and are not deferred implementation questions.

### 7. Errors remain structured across the boundary

Adapters return typed host observations distinguishing not-found, permission/OS failure, policy denial, timeout, output-limit exhaustion, mapped-snapshot-limit exhaustion, malformed protocol data, and unsupported capability. The evaluator maps them to established user-facing contracts where compatibility applies. `file_exists` maps only not-found to `false`; permission and other OS failures remain structured failures rather than silent absence. Structured sensitive fields are marked before the core consumes a live observation and before rendering. Sensitive payload may exist only in the transient live response; it never enters persistent requests/observations, normalized transcripts, diagnostics, identities, or logs.

### 8. Exact boundary, effect coverage, and bypass prevention are executable

The designated semantic core is a planned dependency-minimal `crates/chelis-eval-core` crate containing the evaluator machine, protocol, runtime values, pure host-operation semantics, invariants, named-axis logic, and transforms. Host adapters remain in `chelis-compiler-api::runtime::adapters::{driver,os,memory,deny,policy}` during the compatibility window; `compiler.rs`, CLI, Python, and Tide are caller adapters. `chelis-eval-core` cannot depend on `chelis-compiler-api`, adapter crates, filesystem/process libraries, or capability-bearing callbacks/traits. The existing compiler API re-exports compatibility entry points without weakening the dependency boundary.

A consistency gate spans builtin registration, all current and planned host-capable language effect declarations, evaluator policy classification, request mapping, transcript-event mapping, and compiled-target policy. Adding or reclassifying an `IO`, `Filesystem`, `Network`, or later host-capable builtin without exactly one evaluator policy fails.

The architecture gate uses the `chelis-eval-core` Cargo dependency allowlist as its primary enforcement and through contract mechanics, the accepted `establish-dylint-tooling` prerequisite's pinned `chelis-fcis-boundaries` library as its resolved Rust layer. Manifest-derived Dylint policy checks the declared library/test/feature lanes and rejects direct, aliased, re-exported, qualified, function-item, callback, trait, declarative-macro-expanded, and adapter-module references for filesystem, process, environment, network, clock, terminal, entropy, thread-scheduling, unsafe-FFI, and mutable-global capabilities. It also rejects public callback/function-pointer/unsealed-trait or capability-handle inputs that could execute host effects inside the core. Unresolved configured entries, zero matched production-core items, or an omitted declared lane fail closed.

Allowed-positive and violating-negative Dylint UI detector fixtures classify production functions compiled in a test lane separately from true `cfg(test)` owners and prove each claimed syntax form before use. Each domain registration is `NoFix`; it may reference a machine-applicable fix only after a separately accepted Dylint-tooling change supports that fix ID and `FCIS-EVIDENCE-011` validates disposable exact-rewrite/no-machine-fix, compile/lint, evaluator-denial/parity, idempotence, and checkout-immutability plans. The gate records active-cfg, procedural-macro/build-script implementation, arbitrary-dynamic-dispatch, precompiled-dependency, and future-syntax blind spots rather than claiming completeness. Behavioral fixtures remain authoritative for nested functions, callbacks, transforms, tests, imports, duplicate responses, and denying-policy bypass attempts.

### 9. Observable compatibility corrections are explicit

Production compatibility is exact except for these approved corrections:

- directory listing results are sorted by bytewise UTF-8 entry name before becoming a runtime list;
- `file_exists` returns `false` only for not-found and surfaces other host failures;
- `mmap_read` preserves current behavior: an offset greater than snapshot length is rejected, while a length extending past the end is clamped; neither path issues another request;
- subprocesses are bounded by the selected explicit timeout and per-stream output limits;
- mapped-file snapshots larger than the selected explicit bound are rejected before evaluator-owned allocation; trusted CLI v1 uses 256 MiB;
- Tide and ordinary Python evaluation become deny-by-default as specified above.

These changes receive dedicated public-surface tests and active-spec updates rather than being hidden inside a parity normalization.

### 10. Replay is explicit, but persistent evaluation caching is deferred

A fresh evaluator machine may consume an explicitly supplied normalized, non-secret observation sequence for deterministic tests and audit. Replay validates request identity, response kind, bounds, integrity, and policy through the same path as live execution and performs no host action. Secret-bearing or non-persistable live observations may be consumed ephemerally but cannot enter a replay fixture or persistent transcript.

This change introduces no `EvaluationQueryKey`, `EvaluationOutcomeDigest`, persistent transcript store, or final-result cache. Machine inputs and decisions remain structurally comparable where their values permit it, but no stable cross-version hash or reuse authorization is implied. Adding persistent evaluation caching requires a separate proposal covering value confidentiality, observation freshness, replay safety, and compatibility.

### 11. Builtin policy, protocol bounds, and sensitive data are mechanically closed

One typed host-capable builtin registry owns builtin identity, declared language effect, evaluator classification (`ExternalRequest`, `CapturedEvent`, or `Unsupported`), request/response kind, persistability, and compiled-target support. Builtin registration, effect inference, evaluator dispatch, captured-event mapping, backend rejection, docs, and acceptance matrices are generated projections or tripwire-checked consumers. Independent string lists are not authoritative.

Request and observation enums are versioned and exhaustive. The suspension has private continuation fields, does not implement `Clone`, `Copy`, or serialization, and consumes `self` on resume. Sensitive live payload uses a wrapper with no persistent `Serialize` or revealing `Debug` implementation; conversion to a normalized transcript is fallible and returns `NonPersistableObservation` rather than hashing or redacting the payload into persistence.

The FCIS manifest pins exact v1 maxima for path bytes, file read/write bytes, directory entries and aggregate name bytes, request count, observation bytes, transcript bytes, subprocess argv entries/bytes, mapped snapshot bytes, and total evaluator-owned protocol memory. Those values are semantic `EvaluationPolicy` defaults where they can affect observations; stricter executor-only safety ceilings retain the separate host-failure path. Protocol implementation is blocked while any required bound remains unspecified.

The manifest also owns the exact projection table from policy denial, expected not-found/permission/timeout/output-limit observations, malformed/correlation failure, executor ceiling/failure, and semantic runtime rejection to the public result variants. Before implementation, stable requirement/scenario/fixture IDs, this table, the builtin registry, and the failing `evaluator-effects --slice protocol` runner are registered.

## Risks / Trade-offs

- **Risk: A state machine complicates recursive evaluation.** Mitigation: introduce a trampoline/continuation representation behind compatibility drivers and migrate one builtin family at a time.
- **Risk: Byte snapshots change mapped-file behavior under concurrent mutation.** Mitigation: document snapshot-at-open behavior and test it explicitly.
- **Risk: Explicit path policy exposes historical ambiguity.** Mitigation: trusted CLI behavior is pinned above; contained policies use directory capabilities and fail closed when race-resistant containment is unavailable.
- **Risk: Subprocess limits are a visible correctness change.** Mitigation: the v1 timeout/output defaults and diagnostics are explicit and tested rather than treated as unchanged behavior.
- **Risk: Secure Tide defaults change an unsafe implicit capability.** Mitigation: document the migration and provide explicit deployment configuration; no silent unrestricted fallback is retained.

## Migration Plan

1. Require the independently completed `deny-tide-evaluator-host-effects` oracle and retain its pure/captured-event and nested denial corpus as regression inputs.
2. Add request protocol, transcript classification, path/context, continuation-linearity, mapped-size, default-policy, and handler-policy tests before changing evaluator dispatch.
3. Introduce request, response, split context/policy, error, non-cloneable suspension, deterministic identity, and machine types with a production compatibility driver; pass `--slice protocol`.
4. Migrate read-only filesystem builtins, writes, mapped files, and subprocess execution in independently reviewable `--slice files`, `--slice mapped-files`, and `--slice subprocess` slices.
5. Preserve `print` and `debug` as deterministic captured events and lock the complete host-capable effect policy table, including future effect-taxonomy fixtures.
6. Route CLI, Python, and Tide through their explicit policies and defaults; pass `--slice surfaces`.
7. Add architecture and closed-vocabulary gates and remove direct host APIs from semantic modules.

Rollback is builtin-family scoped through compatibility adapters; rollback must not restore a bypass around the selected policy. Focused slice commands are supporting evidence and never replace the final acceptance oracle.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py evaluator-effects
```

The runner must cover the closed host-capable effect policy vocabulary across current and planned effect variants, manifest-configured pinned Dylint plans, positive/negative detector fixtures, and registered disposable fix/no-fix/evaluator-parity/idempotence fixtures for every declared evaluator-core lane, deterministic request identities and single-use continuation/replay rules, all standard adapters, directory-capability path policy, bounded mapped snapshots, canonical directory/existence behavior, the pinned fuel/mapped/requested-subprocess defaults, executor-ceiling separation, nested mediation, transient-sensitive-payload non-persistence, immediate/final Tide denial, no-persistent-cache surface checks, and CLI/compiler API/Python/Tide parity in one recorded run. Success means exit status 0, an empty error list, zero observed host actions in every denying, replay, or dry in-memory fixture, no persisted sensitive payload, and no `EvaluationQueryKey`/`EvaluationOutcomeDigest` contract introduced.

## Resolved Compatibility Decisions

- The machine API remains crate-internal until every external builtin family and continuation-law fixture passes; only adapter entry points are public during migration.
- No ambiguous unrestricted Python entry point is retained: `eval_json` is deny-by-default and trusted-local access is explicitly named `eval_json_unrestricted`.
