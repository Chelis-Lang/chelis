# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added - lint auto-fix and cleanup rules

`chelis lint` now supports `--fix`, `--rules`, and `--list`. The lint
engine records rule severity, applies non-overlapping source fixes
in-place, and distinguishes `chelis-lint: allow` from
`chelis-lint: keep`: allow suppresses diagnostics and fixes, while
keep preserves the source but can still warn.

The first fixable cleanup rules are `redundant-linearity-call` and
`prefer-pipe-operator`. Public string literals are now checked by the
blocking `no-em-dash-in-public-strings` rule; the initial rollout
cleaned existing source strings while leaving Markdown-prose
enforcement queued for a later doc-corpus pass.

## [0.7.3] — 2026-05-10

### Fixed — lowering diagnostics and release test coverage

Chelis lowering now exposes structured diagnostics for valid language forms that
are not supported by IR evaluation instead of letting internal lowering panics
escape through user-facing commands. Diagnostics include source offsets or span
IDs when available and point users at `chelis build --target c` when the C host
backend is the supported path.

`chelis test` now has regression coverage for pipe stages that must fall back to
the host runtime, and the release workflow runs the same pipe-stage fixture
against the built release binary before publishing assets.

## [0.7.2] — 2026-05-10

### Fixed — bridge provenance diagnostics

`chelis prove` now resolves c-earchin `.spans.json` manifests for Deep bridge
properties. Human failure/error diagnostics show the originating EARS file,
line, column, requirement ID, and requirement text; JSON output includes the
same data under `source.requirement`.

## [0.7.1] — 2026-05-10

### Added — first-class Surf properties and `chelis prove`

Chelis now supports Level 2 executable properties with canonical Surf syntax:
`@property NAME forall(params...) where ...: expr`. Properties desugar to
ordinary Deep `defsig` plus `def` forms and carry canonical property metadata:
`chelis_role`, `property_source_kind`, `property_quantifiers`, and
`property_preconditions`.

The new `chelis prove` subcommand discovers Surf properties and Deep bridge
witnesses, runs deterministic type-directed sampling, filters false
preconditions without calling the predicate, and reports stable human or NDJSON
output. V1 supports scalar binders and fixed-shape numeric tensors; unsupported
selected properties exit `2`, failures exit `1`, and setup/input errors exit
`3`.

Deep bridge compatibility accepts legacy `c_earchin_role:
"property_witness"` metadata while c-earchin moves to the canonical Chelis
property metadata contract.

### Added — style gate on `chelis build` / `chelis check` / `chelis validate` / `chelis eval --file`

The four CLI ingestion paths now enforce `chelis fmt --check` and the
full `chelis lint` rule set on the user-supplied source file before
the front-end pipeline runs. Non-canonical formatting and any lint
violation fail the command with a one-line-per-issue diagnostic and
an actionable hint (`run \`chelis fmt --inplace <file>\` to fix`).

The new `--allow-style-violations` flag bypasses the gate with a
stderr warning. CI must not pass it; it exists for emergency builds
and one-off migrations. The opaque environment variable
`CHELIS_STYLE_GATE_DISABLE=1` does the same and is reserved for
integration-test harnesses that synthesize ad-hoc Surf to exercise
type/effect/linearity behavior independently of style.

`chelis eval EXPR` (the inline-expression form) is unchanged — the
expression has no on-disk source to check.

### Added — four new lint rules to reflect §3 and §10.1 of the style guide

- `surf-type-pascal-case` (§3.1): `type Name` declarations must be PascalCase.
- `surf-value-snake-case` (§3.2): `def name` declarations must be snake_case (no leading underscores, no uppercase letters).
- `surf-test-name-prefix` (§10.1): functions carrying the `Test` effect must be named `test_*` or `example_*`.
- `surf-def-arrow-form` (§3.5, new spec section): `def name(params) -> T = expr` is the canonical signature form; the `def name(params) : T = expr` colon variant is flagged.

The total `chelis lint` rule set is now 13. `spec/01-nomenclature.md`
gains a new §3.5 documenting the def-arrow-form preference.

## [0.6.1] — 2026-05-06

Bootstrap-list patch. Updates DEFAULT_BOOTSTRAP_LIST in
crates/chelis-reef/src/lib.rs to point at the post-rename shell
tags (nautilus v0.6.0, coral v0.6.0, shoals v0.3.0, octant v0.4.1).
Required reef bundle rebuild against compiler =0.6.1 (from =0.6.0)
and propagated test-fixture pin updates.

No API or runtime change beyond the bootstrap-list and pin bumps.

## [0.6.0] — 2026-05-06

Ecosystem-wide naming-convention sweep. Codifies the cross-shell
identifier conventions in `spec/01-nomenclature.md`, ships a new
`chelis lint` subcommand backed by the `chelis-lint` crate that
enforces them, and bumps `chelis-std` from 0.1.0 to 0.2.0 alongside
breaking module renames.

### Added — `chelis lint` subcommand

New `chelis lint [paths...]` and `chelis lint --check` (gating mode,
exits nonzero on any violation). Backed by the new `chelis-lint`
workspace crate, which implements 9 rules covering `.ch`, `.dp`,
`.rs`, `.py`, `.sh`, `.md`, `.snap`, `Cargo.toml`/`reef.toml`, and
`.github/workflows/*.yml` surfaces. Each rule cites a section of
`spec/01-nomenclature.md`. Exception entries carry a mandatory
`cross_ref: SectionRef` field per §12 — the schema rejects free-form
prose so post-hoc justifications can't accrete in the lint config.

CI gate (`.github/workflows/ci.yml`) runs `chelis lint --check`. The
post-sweep state of chelis main reports zero violations.

### Changed (breaking) — `chelis-std` runtime renamed `Std.IO` → `Std.Io`

The `Std.IO`, `Std.IO.Csv`, `Std.IO.Json`, `Std.IO.Parquet`, and
`Std.IO.Safetensors` modules are renamed to `Std.Io.*` per §6.2 of
the recorded style guide (Title-case compound, never ALL-CAPS
abbreviation). Downstream `import Std.IO ...` statements must update
to `import Std.Io ...`. The `chelis-std` reef package version bumps
from 0.1.0 to 0.2.0 to reflect this. The bundled artifacts in
`crates/chelis-std-bundle/dist/` are regenerated.

### Changed — `Hellotensor` → `HelloTensor` in `examples/`

The `module Hellotensor` declaration in the canonical hello-world
example becomes `module HelloTensor` per §6.3 (PascalCase per
component, including each word inside a compound). The on-disk
filename `examples/hello_tensor.ch` is unchanged.

### Changed — `phaseA_*.rs` → `phase_a_*.rs` integration tests

Five `crates/chelis-cli/tests/phaseA_*.rs` files renamed to
`phase_a_*` per §9.1 (lowercase phase-letter form). Function names
inside the tests retain their `phaseA_*` style — only the filenames
moved.

### Changed — `.github/scripts/smoke_macos_accelerate.sh` ported to `.py`

Per §2.9 (no shell scripts; Python only). The macOS smoke job in
CI invokes `python3 .github/scripts/smoke_macos_accelerate.py`.

### Spec — `spec/01-nomenclature.md` recorded as canonical source

The empty 8-section glossary becomes a 13-section style guide
covering hard language constraints (§1), filesystem and manifest
naming (§2), Surf/Rust/Python identifier conventions (§§3-5),
module conventions (§6), function naming patterns (§7) including
§7.1.1 model/algorithm sub-namespaces (`bs_/mc_/gbm_/fd_/lm_/cg_/
airy_/beta_/chi_/det_/eig_/inv_/erf_`) and §7.2 type-suffix policy
with the parser/converter idiom carve-out, documentation
conventions (§8) including §8.5 mdBook source-tree exception with
`SUMMARY.md`/`README.md` tool-required exemptions, project-cutting
conventions (§9), test naming (§10), resolved escalations from the
May 2026 cleanup (§11), and the lint enforcement surface (§12).

The §11.1 Surf-vs-Deep hyphen asymmetry is recorded as intentional
structure (Deep's compound tag vocabulary uses hyphens like `t-fn`,
`pat-ctor`, `d-name`; user-defined Deep symbols inherit Surf's
no-hyphen rule via desugaring). The lint enforces a narrow guard
that any non-tag Deep symbol must satisfy the Surf identifier
charset.

The §11.2 Rust kebab-package / snake-lib hyphen→underscore
shoreline is documented in §2.2 as a Rust-language-norm crossing.

### Fixed — `scripts/regenerate_chelis_std_bundle.py`

The script invoked `cargo build -p chelis`, but the workspace
package is `chelis-cli` (the binary is named `chelis`). Fixed.

### Migration

Downstream shell-package authors:

- Update `import Std.IO ...` to `import Std.Io ...` (and any
  `Std.IO.<sub>` imports likewise).
- Bump the `chelis-std` pin in your `reef.toml` from `0.1.0` to
  `0.2.0` and the `compiler` pin from `=0.5.0` to `=0.6.0`.
- The shell repos `nautilus`, `coral`, `octant`, `shoals` are
  republishing in lockstep with this release; update their pins to
  the new tags.
