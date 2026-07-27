## 1. Planning tree

- [x] 1.1 Add the canonical `openspec/` root with `config.yaml` and the `openspec-validation` baseline capability.
- [x] 1.2 Record the proposal, design, and requirement delta for the initial adoption.

## 2. Advisory validation

- [x] 2.1 Add `.github/workflows/openspec-validate.yml` running `openspec validate --all --strict` under `contents: read`, non-blocking.
- [x] 2.2 Keep the workflow out of the developer gate scope (`scripts/test_gate.py` non-gate list).

## 3. Validation

- [x] 3.1 Run `openspec validate --all --strict --no-interactive` and confirm it passes.
- [x] 3.2 Confirm no governance coupling remains: `spec/**` is not treated as non-docs, and no merge-bound checker or citation gate is present.
