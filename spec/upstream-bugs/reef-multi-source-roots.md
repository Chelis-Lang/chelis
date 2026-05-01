# reef-multi-source-roots: chelis-reef hardcodes `src/` as the only source root

**Status:** RESOLVED
**Filed:** 2026-04-30
**Owning phase:** chelis-reef
**Discovered by:** Phase 3l Shoals fix-up #2

## Resolved

Resolved 2026-04-30 in chelis-reef commit `6b58030 feat(reef):
multi-source-roots — additional_sources in reef.toml; bump v0.4.1`.
Shoals migrated to canonical layout in commit `553021a`. Octant
continues to track for downstream customer outputs (no Octant-side change
required; the layout is the customer's choice per
`spec/design/phase3n_octant.md` Section 3a).

**Why this matters.** This bug blocks the trust stack convention as documented in
`chelis_reference_implementations_spec.md` and `chelis_canonical_reference.md`. The
convention requires shells to ship `properties/` and `references/` at the repo root
because they are spec artifacts the customer reviews — distinct from `src/` which
contains implementation. Placing them under `src/` (as Shoals v0.1.0-alpha does as a
workaround) blurs that distinction and weakens the customer-facing pitch ("you
review properties, not generated code").

The required fix is not a general-purpose source-roots configuration knob. The
specific shape is: chelis-reef must resolve files in top-level `properties/` and
`references/` as importable Chelis modules, using the same module resolution it
uses for `src/`, with the same `module Shell.Properties.Foo` / `module
Shell.References.Foo` naming convention. A future agent reading this bug should
implement that exact behavior, not invent a more general configuration mechanism
that misses the specific shape the convention requires.

## Summary

Domain shells per `chelis_reference_implementations_spec.md` and
`chelis_canonical_reference.md` ship `properties/` and `references/`
as repo-root peers of `src/` and `tests/`. Reef's resolver must
recognize and type-check files in those directories using the same
module resolution it uses for `src/`. Currently chelis-reef v0.4.0
hardcodes `src/` as the only source root, so any `.ch` file outside
`src/` is rejected as not being a source file under the package root.

## Code references

The hardcoding lives in `crates/chelis-reef/src/lib.rs`:

- line 159: `source_root.join("src").join(&module.file_rel)` — the
  source-resolution path is built by unconditionally joining the
  package root with the literal `"src"` segment.
- line 1311: `validate_module_path` derives module names from the
  path within `src/`; there is no plumbing for an alternate source
  directory.
- `ManifestPackage` at lines 29-35: no `additional_sources` (or
  equivalent) field. The struct only carries `name`, `version`,
  `compiler`, and `module_prefix`.
- `ReefManifest`: no `[paths]` or `[layout]` table is parsed, so
  even if a package author wanted to opt into additional roots,
  there is no manifest surface to declare them.

## Minimal repro

In any reef package, create a top-level `properties/foo.ch`:

```chelis
module Pkg.Properties.Foo
def trivial() -> bool = true
```

Then run:

```
chelis check properties/foo.ch
```

Observed:

```
error: properties/foo.ch is not a source file under <root>/src
```

Expected: type-check succeeds and the file is treated as part of the
package under module path `Pkg.Properties.Foo`, identically to how
`src/properties/foo.ch` would be treated today.

## Required fix

Either:

1. Extend `ManifestPackage` with a way to declare additional source
   roots, e.g.:

   ```toml
   [package]
   name = "shoals"
   version = "0.1.0"
   compiler = "=0.4.0"
   module_prefix = "Shoals"
   additional_sources = ["properties", "references"]
   ```

   The resolver would then walk each declared root the same way it
   walks `src/`, and `validate_module_path` would accept any of them
   as the implicit prefix root.

2. Special-case `properties/` and `references/` in chelis-reef the
   same way `tests/` is special-cased in chelis-cli. This aligns
   with the trust-stack convention because `properties/` and
   `references/` are spec artifacts, not generic source directories,
   and the convention is already documented in
   `chelis_canonical_reference.md` and
   `chelis_reference_implementations_spec.md`.

Option 2 is the lower-friction path for the trust stack, since it
matches the way the customer-facing pitch already describes those
directories.

## Trust stack relevance

The customer-facing pitch describes `properties/` and `references/`
as top-level artifacts the customer reviews — they are evidence that
the implementation matches the textbook, not implementation detail.
The current workaround (placing them under `src/`) weakens that
framing because the directories appear as "implementation files" in
the customer's mental model rather than spec-side artifacts.

Treat as CProof-relevant priority, not cosmetic.

## Status of workarounds

Shoals v0.1.0-alpha keeps `properties/` and `references/` under
`src/` (i.e. `src/properties/...`, `src/references/...`). The
deviation is documented in Shoals's `README.md` under "Known
limitations in v0.1.0-alpha" and points back to this file.

Octant Phase 3n will face the same decision when its shell packages
ship; the same workaround applies until this is fixed. The Octant
phase plan (`spec/design/phase3n_octant.md`) carries a note pinning
shell packages to the `src/properties/`, `src/references/` layout so
the eventual migration to the canonical layout is mechanical and
consistent across shells.

## Resolution

When reef gains multi-root support (option 1) or the special-cased
property/reference roots (option 2):

- Shoals moves `src/properties/` → `properties/` and
  `src/references/` → `references/` in v0.1.x or v0.2.0.
- Octant's shell-generation templates emit the canonical layout
  directly.
- The "Layout deviation" subsection of Shoals's `README.md` is
  removed.
- The Octant phase plan's toolchain-dependencies note is removed.

Status update 2026-04-30: orchestrator approved fix plan at
`/home/jeff/.claude/plans/you-re-going-to-do-synchronous-salamander.md`.
Implementation in flight; this bug will be marked resolved when the
chelis-reef change ships in v0.4.1.
