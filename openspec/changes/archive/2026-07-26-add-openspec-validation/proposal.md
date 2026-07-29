## Why

Main has a stock OpenSpec planning tree from #839. The tree has template configuration and no hosted structural validation.

`spec/design/spec_provenance.md` describes a future required regime. This initial adoption adds useful structure without that policy.

## What Changes

- Replace the stock configuration with Chelis project context and artifact rules.
- Add the `openspec-validation` capability.
- Add an advisory workflow that uses the pinned `Chelis-Lang/ci` OpenSpec action.
- Add a consumer checker that runs structural validation only.
- Keep schema findings advisory and keep operational failures nonzero.
- Do not require OpenSpec lifecycles or govern `spec/**`.

## Capabilities

### New Capabilities

- `openspec-validation`: Structural validation of the `openspec/` tree, with no governed-change enforcement or `spec/**` coupling.

### Modified Capabilities

None.

## Impact

- Changes `openspec/config.yaml`.
- Adds `.github/workflows/openspec-validate.yml`.
- Adds `scripts/check_openspec.py` and its contract tests.
- Changes no language, compiler, CLI, backend, package, or generated-code behavior.
- Leaves `spec/design/spec_provenance.md` Phase 0 inactive.
