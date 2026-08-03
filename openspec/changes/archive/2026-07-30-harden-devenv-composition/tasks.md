**Authoritative completion oracle:** `devenv test --no-tui` must exit with status 0.

## 1. Add Contract Tests

- [x] 1.1 Add a positive test for `require_version: true`.
- [x] 1.2 Add a negative test for an absent or false CLI version requirement.
- [x] 1.3 Change the composition test to parse imports from `devenv.nix`.
- [x] 1.4 Add negative tests for a missing root import and duplicate YAML imports.
- [x] 1.5 Run the focused tests and record the expected failures.

## 2. Change the Devenv Composition

- [x] 2.1 Add `require_version: true` to `devenv.yaml`.
- [x] 2.2 Move the five local imports into `devenv.nix`.
- [x] 2.3 Remove the local imports from `devenv.yaml`.
- [x] 2.4 Run the focused contract tests.

## 3. Synchronize the Contract

- [x] 3.1 Update the active `cross-platform-devenv` specification.
- [x] 3.2 Update the contributor documentation.
- [x] 3.3 Format all changed Nix files.

## 4. Run Validation

- [x] 4.1 Run the complete Python script suite.
- [x] 4.2 Run strict OpenSpec validation for this change.
- [x] 4.3 Run `devenv test --no-tui` as the completion oracle.

## 5. Run Adversarial Review

- [x] 5.1 Run an independent review of the changed files.
- [x] 5.2 Correct each verified finding.
- [x] 5.3 Re-run the focused tests and the completion oracle after corrections.
