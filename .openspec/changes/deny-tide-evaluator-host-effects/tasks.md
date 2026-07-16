## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, prerequisite edge, exact core/Tide adapter boundary, stable requirement/scenario IDs, explicit polarity, and scenario-to-fixture-to-oracle traces in the FCIS contract manifest
- [ ] 0.2 Register the typed host-capable builtin/effect/denial owner and tripwires against evaluator dispatch, nested call forms, and the future request protocol
- [ ] 0.3 Make `.venv/bin/python scripts/fcis_gate.py tide-evaluator-denial` exist and fail on the current host-action fixtures; missing or skipped evidence is not an acceptable precondition
- [ ] 0.4 Pass the owning FCIS contract registration/traceability slice before guard implementation

## 1. Lock Denial Behavior With Tests

- [ ] 1.1 Add Tide pure-expression and captured `print`/`debug` positive fixtures
- [ ] 1.2 Add direct negative fixtures for every file, directory, mapped-open, and subprocess builtin
- [ ] 1.3 Add nested function, callback, transform, imported-definition, and test-body denial fixtures
- [ ] 1.4 Instrument fixtures to assert zero filesystem reads/writes and zero child launches rather than relying only on returned diagnostics
- [ ] 1.5 Commit the stubs and verify at least the direct and nested external cases fail before implementation

## 2. Implement The Temporary Guard

- [ ] 2.1 Add invocation-owned deny-external evaluator input without process-global or environment state
- [ ] 2.2 Check the mode at the common external-builtin dispatch boundary before every host call
- [ ] 2.3 Propagate the mode through every evaluator call form covered by the negative corpus
- [ ] 2.4 Route all unconfigured Tide evaluation entry points through deny-external mode
- [ ] 2.5 Preserve pure values and deterministic captured events

## 3. Documentation And Acceptance

- [ ] 3.1 Document Tide's deny-external default and the temporary nature of the dispatch guard
- [ ] 3.2 Implement `.venv/bin/python scripts/fcis_gate.py tide-evaluator-denial`
- [ ] 3.3 Run the oracle and require exit 0, an empty error list, zero denied-fixture host actions, and unchanged captured events
- [ ] 3.4 Record this change as a prerequisite of `isolate-evaluator-host-effects`; remove the temporary mode only after deny-all request handling passes equivalent coverage
