## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, exact production/test-support boundary, stable requirement/scenario IDs, explicit polarity, and scenario-to-fixture-to-oracle traces in the FCIS contract manifest
- [ ] 0.2 Register production dependency/public-API negative fixtures proving scripted outcomes are dev-only and no production feature/constructor can expose them
- [ ] 0.3 Make `.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal` exist and fail on the current forced-result branch/source-absence fixtures; missing evidence is never skipped
- [ ] 0.4 Pass the owning FCIS contract registration/traceability slice before deleting or replacing production behavior

## 1. Lock The Security Contract With Tests

- [ ] 1.1 Add hostile-environment tests for former `proved`, `disproved`, `timeout`, `unknown`, invalid, and oversized values plus an absent-variable control
- [ ] 1.2 Add explicit raw scripted fixtures for proved, disproved, timeout, unknown, and error outcomes
- [ ] 1.3 Add a negative isolation test proving normal production builds expose no scripted-result constructor or enableable production trust path
- [ ] 1.4 Add production-source checks for the named variable and equivalent result-substitution branches
- [ ] 1.5 Commit the stubs and verify the hostile `proved` case and source-absence check fail before implementation

## 2. Remove Ambient Result Substitution

- [ ] 2.1 Migrate affected unit, integration, worker, and CLI tests to the explicit raw fixture
- [ ] 2.2 Delete the production environment read, parser, and forced-result branch
- [ ] 2.3 Preserve normal mapping for explicit/real proved, disproved, timeout, unknown, and error outcomes
- [ ] 2.4 Verify outer adapter discovery such as `CHELIS_BEACON_BIN` is not incorrectly classified as result substitution

## 3. Documentation And Acceptance

- [ ] 3.1 Remove obsolete environment-fixture documentation and document explicit scripted test support
- [ ] 3.2 Implement `.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal`
- [ ] 3.3 Run the oracle in default and SMT-enabled configurations and require exit 0, an empty error list, no production forced-result branch, and no hostile environment value changing a green verdict
- [ ] 3.4 Record this change as a prerequisite in `make-proof-verdict-inputs-explicit` without treating the larger protocol oracle as completion evidence for this correction
