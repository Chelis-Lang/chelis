## 1. Planning tree

- [x] 1.1 Replace the stock configuration with Chelis project context and artifact rules.
- [x] 1.2 Record the proposal, design, and requirement delta for the initial adoption.

## 2. Advisory validation

- [x] 2.1 Add `scripts/check_openspec.py` with separate finding and operational exit codes.
- [x] 2.2 Use the pinned central action in advisory mode.
- [x] 2.3 Add positive and negative tests for the workflow and checker contracts.
- [x] 2.4 Keep the workflow outside the developer gate in `scripts/test_gate.py`.

## 3. Validation

- [x] 3.1 Run `python3 -m unittest scripts.test_openspec_validation scripts.test_ci_detect_docs_only scripts.test_gate`.
- [x] 3.2 Run `openspec validate --all --strict --no-interactive` with OpenSpec 1.6.0.
- [x] 3.3 Validate that the checker has no lifecycle, citation, planning-order, branch-scope, or `spec/**` policy.
