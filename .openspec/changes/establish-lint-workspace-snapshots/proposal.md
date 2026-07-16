## Why

`chelis-lint` contains deterministic rule logic, but its public context exposes a live root and several cross-file rules reopen or rescan the repository. A single lint invocation can therefore observe multiple workspace states, silently skip unreadable files, and make built-in rule behavior depend on hidden filesystem access.

## What Changes

- Define an immutable, path-sorted workspace snapshot containing normalized logical paths, selected source bytes/text, surface classification inputs, and metadata declared by active built-in rules.
- Move traversal, ignore rules, symlink/containment handling, decoding, and collection failures into explicit snapshot-builder adapters.
- Make built-in rules and shared indexes consume only supplied snapshot data and immutable rule configuration.
- Build cross-file catalogs once per snapshot rather than once per checked entry.
- Introduce a versioned pure rule interface. Keep the legacy root-exposing `Rule` API only in a clearly named legacy adapter outside the designated pure core for one documented minor-version window.
- Return collection diagnostics and lint violations as canonically ordered data without direct terminal output.
- Preserve established rule codes, severities, exceptions, clean examples, and negative-corpus ordering. Newly surfaced unreadable/invalid/escaping-entry collection diagnostics are an explicit compatibility correction with pinned CLI severity and exit behavior.

## Capabilities

### New Capabilities

- `workspace-snapshot-analysis`: Defines immutable, reusable inputs for repository-wide lint analysis.

### Modified Capabilities

None.

## Impact

The change affects `chelis-lint`, CLI lint/style-gate adapters, external Rust rule authors, and lint integration tests. It is intentionally independent from Reef workflow planning and has its own acceptance oracle. The pure interface is additive during the compatibility window; arbitrary legacy rules are not falsely counted as part of the pure-core guarantee.
