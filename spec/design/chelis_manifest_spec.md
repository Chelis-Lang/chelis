# `chelis manifest` Specification

## Purpose

`chelis manifest` produces a machine-readable reproducibility certificate for a Chelis program. The certificate documents every random operation, the seed its key derives from, and the static guarantee that the program is reproducible given the same inputs and seeds.

The compile-time guarantee already exists: every random primitive takes an explicit, affine `key`, and a key is produced only by `key_from_seed`, a key derivation, or an entry argument (`randomness_explicit_keys.md`), so no draw depends on hidden state. `chelis manifest` is the user-facing artifact that documents this guarantee in a form auditable by humans, model validation teams, and regulators.

## What's Already Real

- The `key` dtype and key-first random primitives: a function that draws takes a `key` parameter.
- Compiler rejection of a second use of a key: keys are affine.
- `key_from_seed(seed)`: the one way a seed becomes a key.

What's missing: the CLI tool that emits the manifest artifact.

## CLI Surface

```
chelis manifest src/pricer.ch              # produce manifest for a file
chelis manifest                            # manifest for current package
chelis manifest --check                    # verify every random op's key traces to a seed or entry argument; exit 1 if not
chelis manifest --format json              # default
chelis manifest --format human             # human-readable text format
chelis manifest --output dist/manifest.json
```

## Manifest Schema (JSON)

```json
{
  "schema_version": "1.0",
  "package": {
    "name": "shoals",
    "version": "0.1.0",
    "compiler_version": "0.2.7"
  },
  "source_hash": "sha256:abc123...",
  "compilation_timestamp": "2026-04-25T12:00:00Z",
  "random_operations": [
    {
      "function": "Shoals.MonteCarlo.simulate_paths",
      "source_location": "src/monte_carlo.ch:42",
      "seed_source": "explicit (parameter `seed: i64`)",
      "key_root_at": "src/monte_carlo.ch:38 (key_from_seed(seed))",
      "operation_type": "normal_sample",
      "shape": "tensor[n_paths, n_steps, f32]"
    },
    {
      "function": "Shoals.MonteCarlo.bootstrap_sample",
      "source_location": "src/risk.ch:127",
      "seed_source": "explicit (parameter `bootstrap_seed: i64`)",
      "key_root_at": "src/risk.ch:124 (key_from_seed(bootstrap_seed))",
      "operation_type": "uniform_sample",
      "shape": "tensor[n_samples, f32]"
    }
  ],
  "guarantees": {
    "all_random_ops_seeded": true,
    "no_implicit_random": true,
    "no_io_in_pure_functions": true,
    "linearity_verified": true,
    "dimension_safety_verified": true
  },
  "structural_properties": [
    {
      "function": "Shoals.BlackScholes.call_price",
      "effects": [],
      "differentiable": true,
      "linear": true
    }
  ]
}
```

## Human-Readable Format

```
Reproducibility Manifest for shoals 0.1.0
Compiled with chelis 0.2.7
Source hash: sha256:abc123...
Generated: 2026-04-25T12:00:00Z

Random Operations: 2

  1. Shoals.MonteCarlo.simulate_paths
     Location: src/monte_carlo.ch:42
     Operation: normal_sample over tensor[n_paths, n_steps, f32]
     Seed: explicit parameter `seed: i64`
     Key root: src/monte_carlo.ch:38 (key_from_seed(seed))

  2. Shoals.MonteCarlo.bootstrap_sample
     Location: src/risk.ch:127
     Operation: uniform_sample over tensor[n_samples, f32]
     Seed: explicit parameter `bootstrap_seed: i64`
     Key root: src/risk.ch:124 (key_from_seed(bootstrap_seed))

Guarantees (compiler-verified):
  ✓ Every random operation's key traces to an explicit seed
  ✓ No implicit randomness reaches any production code path
  ✓ No I/O effects in functions declared pure
  ✓ Linearity verified (no use-after-consume on any tensor)
  ✓ Dimension safety verified (all tensor operations have consistent shapes)

This program is reproducible: same source, same inputs, same seeds will
produce identical outputs on the same platform with the same compiler version.
```

## Cross-Platform Reproducibility

The compile-time guarantee covers seeded randomness. Bit-exact reproducibility across platforms (different CPU architectures, different SIMD widths, GPU vs CPU) requires additional discipline:

- **Within a platform** (same CPU/GPU, same compiler version, same SIMD level): bit-exact reproducibility holds for code that doesn't use non-deterministic reductions.
- **Across platforms**: bit-exact reproducibility requires explicit cross-platform-deterministic reduction modes (currently a future feature).

The manifest documents this distinction. The `guarantees` section currently asserts within-platform reproducibility. A future `cross_platform_deterministic` flag will be added when the cross-platform mode ships.

## CI Integration

```yaml
jobs:
  reproducibility:
    steps:
      - run: chelis manifest --check
      - run: chelis manifest --output dist/manifest.json
      - uses: actions/upload-artifact@v4
        with:
          name: reproducibility-manifest
          path: dist/manifest.json
```

The `--check` flag fails the build if any random operation's key does not trace to a seed or an entry argument (this duplicates what `chelis build` already guarantees, but provides a clear single-purpose gate for compliance teams).

## Implementation Components

- **CLI subcommand**: `chelis manifest` with the flags specified above
- **Compiler integration**: extract random operation locations from the checked program, follow each key through its derivations and the call graph to its `key_from_seed` root or entry argument
- **Schema validation**: ensure the JSON output conforms to schema version 1.0
- **Human-readable formatter**: structured text output as specified

## Acceptance Criteria

1. `chelis manifest src/file.ch` produces a valid JSON manifest matching the schema.
2. `chelis manifest --check` exits 0 if every random op's key traces to a seed or an entry argument, 1 otherwise.
3. `chelis manifest --format human` produces the human-readable format specified above.
4. The manifest correctly identifies every random primitive call in the program.
5. The manifest correctly identifies the seed source (explicit parameter, literal seed, environment variable, etc.) of each random op's key.
6. End-to-end demo: a Shoals Monte Carlo example where the manifest is generated, a customer can read it, and re-running with the same seeds produces identical numerical output.
