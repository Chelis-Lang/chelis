## Context

The checked language marks file, process, `print`, and `debug` builtins with `IO`. `EvalContext` currently executes filesystem and subprocess effects directly through `std::fs`, path queries, and `std::process::Command`; `print` and `debug` are already captured into a returned transcript rather than writing a terminal. Direct host execution prevents deterministic embedding and central capability enforcement.

This change concerns evaluator implementation boundaries. It does not change which Chelis expressions carry `IO` or which builtins are available. It follows the decision, request, identity, secret, and failure laws in `.openspec/FCIS_ARCHITECTURE.md`.

## Goals / Non-Goals

**Goals:**

- Make every external host effect a typed request with a correlated structured observation.
- Preserve deterministic evaluator-local output events such as transcripts as returned data.
- Keep evaluation, continuation, random, and limit state explicit and isolated per run.
- Provide production, deterministic, denying, and selective-policy adapters.
- Make relative paths, subprocess cwd/environment, timeouts, output limits, and deterministic evaluator fuel explicit with resolved defaults.
- Prevent nested functions, callbacks, transforms, imported code, or repeated continuations from bypassing policy.

**Non-Goals:**

- Changing Chelis effect inference or introducing user-defined effect handlers.
- Adding network builtins.
- Adding compiled-backend support for eval-only `process_run`.
- Making asynchronous evaluation part of the public language surface.
- Treating deterministic returned transcripts as terminal I/O.

## Decisions

### 1. Evaluation is a resumable state machine

The semantic core exposes an invocation-owned evaluator machine whose step operation returns progress, accepted completion, semantic rejection, protocol/host failure, or a suspension containing one `HostEffectRequest` and continuation state. Local mutation inside the machine is allowed, but all state that can influence later steps belongs to that machine.

A suspension is non-cloneable, is consumed by value, and has exactly one outstanding request. Request identities are deterministic invocation-local sequence numbers starting at zero. Resumption requires a response carrying the matching request identity and response kind and validates it before changing continuation state. Duplicate, stale, replayed, malformed, oversized, or mismatched responses are protocol errors and cannot partially mutate a live machine. Transcript replay starts a fresh machine; it never reuses a consumed suspension.

Alternative considered: pass a `HostEffectHandler` trait directly into recursive `eval_expr`. This would execute effects behind an injected interface and obscure the FCIS boundary, so the semantic machine yields data instead.

### 2. Requests and observations form a closed typed vocabulary

The initial external request vocabulary covers text/line/byte read, write, existence, directory listing, mapped-file open, and subprocess execution. Requests carry bounded normalized runtime arguments, deterministic request identity, builtin/source identity, span context, and non-secret policy references required for diagnostics. Resolved secret environment values, renderer redaction data, OS handles, and ephemeral absolute paths are not persisted in semantic transcripts.

Responses are typed per request. Unknown request kinds, malformed observations, correlation mismatches, and response-kind mismatches are internal protocol errors rather than unchecked conversions.

Mapped-file open returns an evaluator-owned immutable byte snapshot. `mmap_read` and `mmap_len` remain pure operations over that snapshot.

### 3. Evaluator-local IO events are returned data

`print` and `debug` remain language-level `IO`, but their evaluator implementation appends deterministic transcript events to invocation-owned state and returns those events in the evaluation outcome. They do not issue a filesystem/process request and do not write a terminal.

The effect-coverage gate therefore requires every `IO` builtin to have exactly one declared evaluator policy: external request, evaluator-local captured event, or explicit unsupported-target policy. It does not incorrectly require `print` and `debug` to become host terminal effects.

### 4. Semantic policy, host execution context, and rendering are separate

`EvaluationPolicy` is a semantic input containing allowed operation classes, lexical path policy identifiers, subprocess authorization, and `EvaluationLimits`. `HostExecutionContext` is adapter input containing the resolved relative-path base, directory capability handles, subprocess working directory, resolved environment inheritance/allowlist behavior, timeout enforcement, and output bounds. `EvaluationRenderPolicy` contains secret-redaction and display configuration and is report metadata, not machine or cache identity.

The v1 defaults are resolved before machine construction:

- `DEFAULT_EVAL_FUEL = 10_000_000` semantic steps;
- trusted CLI eval/test explicitly selects unrestricted filesystem access, follows normal host symlinks without a containment root, resolves relative paths and subprocess cwd from the invocation cwd, snapshots the inherited subprocess environment, uses a 300-second timeout, and limits stdout and stderr to 64 MiB each;
- Tide supplies deny-all filesystem and subprocess policy unless deployment configuration provides explicit contained-directory or executable capabilities;
- Python `eval_json` supplies deny-all external policy; `eval_json_with_policy` accepts explicit capabilities; `eval_json_unrestricted` selects the trusted-local production policy and is named accordingly;
- in-memory and replay adapters receive no process cwd or environment.

The semantic machine and policy mapper never inspect process cwd or environment. The explicit timeout and output bounds affect host observations; elapsed report duration and redaction configuration do not affect machine identity.

### 5. Handlers and policies live outside the semantic core

A production OS adapter preserves established trusted CLI behavior. An in-memory adapter supports deterministic tests and embeddings. A deny-all adapter rejects every external request, and a policy wrapper can authorize operation kinds, paths, and subprocess capabilities independently.

Path policy defines lexical normalization, symlink traversal, containment, and nonexistent-write parent handling. Contained production policies use directory-capability-relative operations so authorization and access use the same parent handle; they do not authorize by canonicalizing one path and reopening another. Platforms that cannot provide the required race-resistant containment fail closed for contained policies. Nonexistent writes resolve and authorize an existing parent capability before creating the final component.

A convenience `evaluate_with_handler` driver may loop over machine requests, but it belongs in an adapter module and cannot be imported by the designated semantic core.

### 6. Embedding defaults and the immediate Tide guard are secure and explicit

- Before the state-machine migration, Tide adds a fail-closed checked-effect guard that rejects any evaluation capable of reaching external `IO`; positive pure-eval and negative file/process tests land first. The request-protocol migration later replaces that temporary guard with deny-all handling.
- Trusted CLI eval/test entry points select the production adapter and the v1 trusted-local defaults explicitly.
- Python exposes `eval_json_with_policy`; ordinary `eval_json` is deny-by-default, and the only unrestricted entry point is clearly named `eval_json_unrestricted`.
- Direct Rust convenience entry points either require a policy or carry `unrestricted` in their name.

These defaults and API names are part of this change and are not deferred implementation questions.

### 7. Errors remain structured across the boundary

Adapters return typed host observations distinguishing not-found, permission/OS failure, policy denial, timeout, output-limit exhaustion, malformed protocol data, and unsupported capability. The evaluator maps them to established user-facing contracts where compatibility applies. `file_exists` maps only not-found to `false`; permission and other OS failures remain structured failures rather than silent absence. Structured sensitive fields are marked before rendering, and render adapters redact them; secret bytes never enter persistent requests, observations, transcripts, diagnostics, or identities.

### 8. Exact boundary, effect coverage, and bypass prevention are executable

The designated semantic core is `crates/chelis-compiler-api/src/runtime/eval.rs` after host calls are removed, plus planned `runtime/protocol.rs`, `runtime/machine.rs`, `runtime/value.rs`, and the existing pure `runtime/{host_ops,invariant,named_axis,transforms}.rs` modules they import. Host adapters live only under planned `runtime/adapters/{driver,os,memory,deny,policy}.rs`. `compiler.rs`, CLI, Python, and Tide are caller adapters. Designated modules cannot import `runtime::adapters`; the manifest records the exact transitive production set and applies a separate `cfg(test)` policy rather than exempting whole files.

A consistency gate spans builtin registration, language effect declarations, evaluator policy classification, request mapping, transcript-event mapping, and compiled-target policy. Adding an `IO` builtin without exactly one evaluator policy fails.

Architecture tests reject direct, aliased, re-exported, qualified, callback-hidden, and trait-hidden filesystem/process/environment access from semantic evaluator modules. Behavioral fixtures cover nested functions, callbacks, transforms, tests, imports, duplicate responses, and denying-policy bypass attempts.

### 9. Observable compatibility corrections are explicit

Production compatibility is exact except for these approved corrections:

- directory listing results are sorted by bytewise UTF-8 entry name before becoming a runtime list;
- `file_exists` returns `false` only for not-found and surfaces other host failures;
- `mmap_read` preserves current behavior: an offset greater than snapshot length is rejected, while a length extending past the end is clamped; neither path issues another request;
- subprocesses are bounded by the selected explicit timeout and per-stream output limits;
- Tide and ordinary Python evaluation become deny-by-default as specified above.

These changes receive dedicated public-surface tests and active-spec updates rather than being hidden inside a parity normalization.

### 10. Evaluation identity and cacheability are explicit

`EvaluationQueryKey` covers checked-program identity, selected roots, input runtime values, deterministic random state, semantic policy/version, and limits. It excludes elapsed time, renderer/redaction configuration, process-local handles, and secret bytes. A host-effectful production evaluation is not eligible for final-outcome cache lookup from that query key alone because external observations can change; pure, in-memory, or explicitly replayed evaluations may use it.

`EvaluationOutcomeDigest` adds the normalized observation/transcript classifications and final machine outcome only when the active policy marks every observation persistable and replay-safe. Secret-bearing or non-persistable observations are consumed ephemerally and disable persistent transcript/digest caching rather than hashing or storing secret content. This does not prevent deterministic replay fixtures from using explicitly non-secret observations.

## Risks / Trade-offs

- **Risk: A state machine complicates recursive evaluation.** Mitigation: introduce a trampoline/continuation representation behind compatibility drivers and migrate one builtin family at a time.
- **Risk: Byte snapshots change mapped-file behavior under concurrent mutation.** Mitigation: document snapshot-at-open behavior and test it explicitly.
- **Risk: Explicit path policy exposes historical ambiguity.** Mitigation: trusted CLI behavior is pinned above; contained policies use directory capabilities and fail closed when race-resistant containment is unavailable.
- **Risk: Subprocess limits are a visible correctness change.** Mitigation: the v1 timeout/output defaults and diagnostics are explicit and tested rather than treated as unchanged behavior.
- **Risk: Secure Tide defaults change an unsafe implicit capability.** Mitigation: document the migration and provide explicit deployment configuration; no silent unrestricted fallback is retained.

## Migration Plan

1. Add pure Tide positive and file/process negative tests, then land the immediate fail-closed Tide IO guard.
2. Add request protocol, transcript classification, path/context, continuation-linearity, default-policy, and handler-policy tests before changing evaluator dispatch.
3. Introduce request, response, split context/policy, error, non-cloneable suspension, deterministic identity, and machine types with a production compatibility driver.
4. Migrate read-only filesystem builtins, writes, mapped files, and subprocess execution in independently reviewable slices.
5. Preserve `print` and `debug` as deterministic captured events and lock the complete `IO` policy table.
6. Route CLI, Python, and Tide through their explicit policies and defaults.
7. Add architecture and closed-vocabulary gates and remove direct host APIs from semantic modules.

Rollback is builtin-family scoped through compatibility adapters; rollback must not restore a bypass around the selected policy.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py evaluator-effects
```

The runner must cover the closed `IO` policy vocabulary, deterministic request identities and single-use continuation/replay rules, all standard adapters, directory-capability path policy, mapped snapshots, canonical directory/existence behavior, the pinned fuel/subprocess defaults, nested mediation, secret exclusion, immediate/final Tide denial, and CLI/compiler API/Python/Tide parity in one recorded run. Success means exit status 0, an empty error list, and zero observed host actions in every denying, replay, or dry in-memory fixture.

## Resolved Compatibility Decisions

- The machine API remains crate-internal until every external builtin family and continuation-law fixture passes; only adapter entry points are public during migration.
- No ambiguous unrestricted Python entry point is retained: `eval_json` is deny-by-default and trusted-local access is explicitly named `eval_json_unrestricted`.
