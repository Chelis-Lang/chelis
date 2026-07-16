## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, Tide prerequisite edge, exact core/adapter boundaries, stable requirement/scenario IDs, explicit polarity, and complete fixture-to-slice-to-final-oracle traces in the FCIS contract manifest
- [ ] 0.2 Pin exact v1 path, file read/write, directory entry/name, request count, observation, transcript, subprocess argv, mapped-snapshot, and total protocol-memory bounds; absent bounds block protocol implementation
- [ ] 0.3 Register one typed host-capable builtin/effect/evaluator-policy/request-event/target owner and tripwire every existing string/list projection
- [ ] 0.4 Register the exact policy-denial/expected-observation/semantic-rejection/protocol-failure/host-failure projection table and sensitive non-persistence type constraints
- [ ] 0.5 Make `.venv/bin/python scripts/fcis_gate.py evaluator-effects --slice protocol` exist and fail on current direct-host, missing-policy, replay, malformed, and sensitive-persistence fixtures
- [ ] 0.6 Pass the owning FCIS contract registration/traceability slice before semantic protocol implementation

## 1. Require The Security Prerequisite And Define Protocol Tests

- [ ] 1.1 Require recorded green evidence from `.venv/bin/python scripts/fcis_gate.py tide-evaluator-denial` in the independent `deny-tide-evaluator-host-effects` change and import its pure/captured-event and nested denial corpus as regression inputs
- [ ] 1.2 Add external request-protocol stubs for every filesystem, directory, mapped-file-open, and subprocess builtin
- [ ] 1.3 Add an exhaustive host-capable effect policy-table fixture covering current `IO`, planned `Filesystem`/`Network`, external requests, captured `print`/`debug` events, unsupported policies, missing mappings, duplicate mappings, and effect reclassification
- [ ] 1.4 Add positive and negative continuation tests for deterministic sequence identities, matching, mismatched, malformed, oversized, stale, replayed, duplicate, and fresh-machine transcript replay
- [ ] 1.5 Add nested function, callback, transform, imported-definition, and test-body fixtures proving adapters cannot be bypassed
- [ ] 1.6 Add production, in-memory, deny-all, and selective-policy tests with allowed and denied counterparts
- [ ] 1.7 Add relative-base, directory-capability containment, symlink escape, unsupported-platform fail-closed, nonexistent-write parent, and host-change path fixtures
- [ ] 1.8 Add mapped-file snapshot, 256-MiB default/custom limit, pre-allocation oversize rejection, offset rejection, length clamping, and host-mutation-after-open tests
- [ ] 1.9 Add subprocess argv, resolved cwd/environment, policy-requested 300-second/64-MiB defaults, custom requested limits, stricter executor-ceiling host failure, timeout, output-limit, denial, failure, and no-shell-interpolation tests
- [ ] 1.10 Add deterministic 10,000,000-step default-fuel positive and negative tests across different execution speeds
- [ ] 1.11 Add directory-order and `file_exists` not-found/permission/error compatibility tests
- [ ] 1.12 Add transient-sensitive-observation, persistent secret-exclusion, explicit non-secret replay, no-persistent-cache/digest surface, diagnostics, redaction, and elapsed-duration tests
- [ ] 1.13 Commit the stubs and verify intended negative cases fail before evaluator changes
- [ ] 1.14 Verify the prerequisite guard remains active until deny-all request handling passes equivalent coverage; do not reimplement or re-claim the prerequisite security correction

## 2. Build The Semantic Protocol

- [ ] 2.1 Create dependency-minimal `chelis-eval-core` and define the closed request/response, deterministic invocation-local identity, persistability/sensitivity classification, host observation, protocol failure, host failure, policy denial, source context, semantic `EvaluationPolicy`, adapter `HostExecutionContext`, and non-semantic `EvaluationRenderPolicy` types
- [ ] 2.2 Define invocation-owned machine, non-cloneable continuation, normalized transcript, random, and deterministic-limit state
- [ ] 2.3 Make suspension consumed-by-value and validate correlation, kind, bounds, integrity, stale, replayed, and malformed responses before state transition
- [ ] 2.4 Preserve `print` and `debug` as deterministic evaluator-local transcript events
- [ ] 2.5 Add an outer `evaluate_with_handler` driver without importing host capabilities into semantic machine code
- [ ] 2.6 Implement/extend and pass `.venv/bin/python scripts/fcis_gate.py evaluator-effects --slice protocol`

## 3. Implement Standard Adapters And Policies

- [ ] 3.1 Implement the production OS adapter with specified path, encoding, directory, write, and error-rendering behavior
- [ ] 3.2 Implement the deterministic in-memory adapter with explicit initial/final state and base directory
- [ ] 3.3 Implement deny-all and selective policies for operation kinds, directory-capability-contained paths, and subprocess capabilities
- [ ] 3.4 Implement lexical normalization and race-resistant directory-capability path access for reads and writes, failing closed where required containment is unavailable
- [ ] 3.5 Implement subprocess execution with argv, resolved cwd/environment, policy-requested timeout/output limits, distinct stricter executor ceilings, and child cleanup
- [ ] 3.6 Mark sensitive live payloads before core consumption, keep them out of persistent protocol data/logs/identities, and add structured not-found, permission, redaction, and error-class tests across all adapters/renderers

## 4. Migrate Evaluator Builtins In Reviewable Slices

- [ ] 4.1 Migrate file and byte read builtins to machine suspension
- [ ] 4.2 Migrate file write and existence builtins to machine suspension
- [ ] 4.3 Migrate directory listing to bytewise UTF-8 name ordering and migrate `file_exists` to false-only-for-not-found behavior
- [ ] 4.4 Migrate mapped-file open to a size-bounded immutable byte response with pre-allocation rejection and retain pure mapped length/read
- [ ] 4.5 Migrate `process_run` to the bounded subprocess protocol while preserving eval/test-only target policy
- [ ] 4.6 Remove direct filesystem, path-query, process, cwd, and environment access from semantic evaluator modules
- [ ] 4.7 Implement/extend and pass the `--slice files`, `--slice mapped-files`, and `--slice subprocess` commands as their builtin families land

## 5. Migrate Evaluation Callers And Defaults

- [ ] 5.1 Route trusted CLI eval/test through an explicitly selected production policy
- [ ] 5.2 Route compiler API convenience entry points through explicit drivers or clearly named compatibility adapters
- [ ] 5.3 Make Python `eval_json` deny-by-default, add `eval_json_with_policy`, and clearly name/document `eval_json_unrestricted`
- [ ] 5.4 Replace the prerequisite Tide deny-external dispatch guard with deny-all request handling by default only after equivalent nested denial coverage passes, then add explicit directory/executable capability configuration
- [ ] 5.5 Preserve established production success/failure fixtures except separately specified timeout, output-limit, and security-default changes
- [ ] 5.6 Add caller-specific positive and negative migration tests
- [ ] 5.7 Implement/extend and pass `.venv/bin/python scripts/fcis_gate.py evaluator-effects --slice surfaces`

## 6. Close The Vocabulary And Architecture

- [ ] 6.1 Add the exhaustive consistency gate spanning builtin registration, current/planned host-capable effect declarations, evaluator policy, protocol mapping, captured events, and target policy
- [ ] 6.2 Enforce the `chelis-eval-core` dependency allowlist, check in the compatibility-adapter manifest and threat model, and add specifically claimed direct, alias, re-export, qualified, callback/macro/trait, and `cfg(test)` fixtures without claiming arbitrary macro/dynamic-dispatch completeness
- [ ] 6.3 Make every missing mapping, duplicate mapping, continuation replay, and adapter-bypass fixture fail for the specified reason

## 7. Documentation And Acceptance

- [ ] 7.1 Update runtime, embedding, Python, Tide, and CLI docs with protocol, continuation, split context/policy/rendering, requested-versus-executor limits, v1 defaults, path containment, subprocess, snapshots, explicit replay/no-persistent-cache scope, approved behavior corrections, and policy semantics
- [ ] 7.2 Update examples with deterministic, deny-all, selective-policy, and captured-transcript positive/negative cases
- [ ] 7.3 Consolidate the already landed focused commands and implement the final `.venv/bin/python scripts/fcis_gate.py evaluator-effects` acceptance runner
- [ ] 7.4 Run each focused command before landing its slice; do not claim change completion until the final oracle exits 0 with an empty error list, zero denied-fixture host actions, bounded mapped snapshots, requested/executor limit separation, no persisted sensitive payload, and no persistent evaluation cache/digest surface
- [ ] 7.5 Run the repository local gate and record any CI-owned workspace evidence required for completion
