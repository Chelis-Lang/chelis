## 1. Define Contract Tests And Immediate Security Guard First

- [ ] 1.1 Add Tide pure-eval positive tests and file/process negative tests, then add the temporary fail-closed checked-IO guard before the protocol migration
- [ ] 1.2 Add external request-protocol stubs for every filesystem, directory, mapped-file-open, and subprocess builtin
- [ ] 1.3 Add an exhaustive `IO` policy-table fixture covering external requests, captured `print`/`debug` events, unsupported policies, missing mappings, and duplicate mappings
- [ ] 1.4 Add positive and negative continuation tests for deterministic sequence identities, matching, mismatched, malformed, oversized, stale, replayed, duplicate, and fresh-machine transcript replay
- [ ] 1.5 Add nested function, callback, transform, imported-definition, and test-body fixtures proving adapters cannot be bypassed
- [ ] 1.6 Add production, in-memory, deny-all, and selective-policy tests with allowed and denied counterparts
- [ ] 1.7 Add relative-base, directory-capability containment, symlink escape, unsupported-platform fail-closed, nonexistent-write parent, and host-change path fixtures
- [ ] 1.8 Add mapped-file snapshot, offset rejection, length clamping, and host-mutation-after-open tests
- [ ] 1.9 Add subprocess argv, cwd, environment, 300-second/64-MiB default, custom-limit, timeout, output-limit, denial, failure, and no-shell-interpolation tests
- [ ] 1.10 Add deterministic 10,000,000-step default-fuel positive and negative tests across different execution speeds
- [ ] 1.11 Add directory-order and `file_exists` not-found/permission/error compatibility tests
- [ ] 1.12 Add secret-exclusion and query/decision/report identity tests for requests, transcripts, diagnostics, redaction, and elapsed duration
- [ ] 1.13 Commit the stubs and verify intended negative cases fail before evaluator changes

## 2. Build The Semantic Protocol

- [ ] 2.1 Define the closed request/response, deterministic invocation-local identity, host observation, policy denial, source context, `EvaluationPolicy`, `HostExecutionContext`, and non-semantic `EvaluationRenderPolicy` types
- [ ] 2.2 Define invocation-owned machine, non-cloneable continuation, normalized transcript, random, and deterministic-limit state
- [ ] 2.3 Make suspension consumed-by-value and validate correlation, kind, bounds, integrity, stale, replayed, and malformed responses before state transition
- [ ] 2.4 Preserve `print` and `debug` as deterministic evaluator-local transcript events
- [ ] 2.5 Add an outer `evaluate_with_handler` driver without importing host capabilities into semantic machine code

## 3. Implement Standard Adapters And Policies

- [ ] 3.1 Implement the production OS adapter with specified path, encoding, directory, write, and error-rendering behavior
- [ ] 3.2 Implement the deterministic in-memory adapter with explicit initial/final state and base directory
- [ ] 3.3 Implement deny-all and selective policies for operation kinds, directory-capability-contained paths, and subprocess capabilities
- [ ] 3.4 Implement lexical normalization and race-resistant directory-capability path access for reads and writes, failing closed where required containment is unavailable
- [ ] 3.5 Implement subprocess execution with argv, explicit cwd/environment, timeout, output limits, and child cleanup
- [ ] 3.6 Keep secrets out of persistent protocol data and add structured not-found, permission, redaction, and error-class tests across all adapters/renderers

## 4. Migrate Evaluator Builtins In Reviewable Slices

- [ ] 4.1 Migrate file and byte read builtins to machine suspension
- [ ] 4.2 Migrate file write and existence builtins to machine suspension
- [ ] 4.3 Migrate directory listing to bytewise UTF-8 name ordering and migrate `file_exists` to false-only-for-not-found behavior
- [ ] 4.4 Migrate mapped-file open to an immutable byte response and retain pure mapped length/read
- [ ] 4.5 Migrate `process_run` to the bounded subprocess protocol while preserving eval/test-only target policy
- [ ] 4.6 Remove direct filesystem, path-query, process, cwd, and environment access from semantic evaluator modules

## 5. Migrate Evaluation Callers And Defaults

- [ ] 5.1 Route trusted CLI eval/test through an explicitly selected production policy
- [ ] 5.2 Route compiler API convenience entry points through explicit drivers or clearly named compatibility adapters
- [ ] 5.3 Make Python `eval_json` deny-by-default, add `eval_json_with_policy`, and clearly name/document `eval_json_unrestricted`
- [ ] 5.4 Replace Tide's temporary checked-IO guard with deny-all request handling by default and add explicit directory/executable capability configuration
- [ ] 5.5 Preserve established production success/failure fixtures except separately specified timeout, output-limit, and security-default changes
- [ ] 5.6 Add caller-specific positive and negative migration tests

## 6. Close The Vocabulary And Architecture

- [ ] 6.1 Add the exhaustive consistency gate spanning builtin registration, `IO`, evaluator policy, protocol mapping, captured events, and target policy
- [ ] 6.2 Check in the exact semantic/adapters manifest and add architecture fixtures rejecting direct, aliased, re-exported, qualified, callback-hidden, and trait-hidden filesystem/process/environment access with intentional `cfg(test)` handling
- [ ] 6.3 Make every missing mapping, duplicate mapping, continuation replay, and adapter-bypass fixture fail for the specified reason

## 7. Documentation And Acceptance

- [ ] 7.1 Update runtime, embedding, Python, Tide, and CLI docs with protocol, continuation, split context/policy/rendering, v1 defaults, path containment, subprocess, snapshot, transcript, approved behavior corrections, and policy semantics
- [ ] 7.2 Update examples with deterministic, deny-all, selective-policy, and captured-transcript positive/negative cases
- [ ] 7.3 Implement `.venv/bin/python scripts/fcis_gate.py evaluator-effects` as the named acceptance runner
- [ ] 7.4 Run `.venv/bin/python scripts/fcis_gate.py evaluator-effects` and require exit 0, an empty error list, and zero denied-fixture host actions
- [ ] 7.5 Run the repository local gate and record any CI-owned workspace evidence required for completion
