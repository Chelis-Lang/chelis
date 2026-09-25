# Effect Taxonomy Expansion

**Status:** Planning. Source-of-truth design for the trust-stack expansion items
that extend the Chelis `Effect` enum and add tooling on top of it. Companion to
`chelis_trust_stack.md` (which describes the verification stack overall) and
`reef_distribution.md` (which the install-time effect manifest item depends on).

---

## Context

The Chelis type system already tracks effects on functions. A red-team review of
the actual taxonomy found that the architectural foundation is correct but the
set of variants is narrower than the broader trust story needs.

This document captures the bounded expansion of the taxonomy and the tooling
that makes effects auditable and enforceable from the command line. Signing of
artifacts is a separate concern tracked but not included here (it is demand-
driven and not bounded enough to design speculatively).

---

## Current taxonomy

The shipped `Effect` enum in `crates/chelis-types/src/types.rs` has
exactly four variants (randomness is not an effect: random primitives take an
explicit `key`, [05-RNG-1]):

- `Accum` — internal design hook for backward-pass accumulation. Not yet
  user-facing as a checked effect; reserved.
- `Io` — host-side print and debug. Narrow today; covers stdout/stderr-style
  output only, not network or filesystem.
- `Test` — the in-language test runner's effect. Functions defined as
  `! { Test }` may use test-only assertions.
- `Resource(String)` — device resource boundaries (`gpu:0`, `cpu`). Validated
  at build boundaries by `chelis build --target c` (rejects GPU resource
  regions) and `chelis build --target hip` (rejects CPU-only regions).

The set is correct for what it tracks: each variant has a well-defined
compile-time guarantee and at least one production code path that exercises
it. The set is narrow because two categories that matter for trust stories
beyond reproducibility and determinism are absent: network access and
filesystem access. Subprocess execution is a third category that does not
fire today because Chelis programs cannot spawn subprocesses (no FFI, no
`exec` primitive, no string-to-code path); it can be added when FFI is
designed in a later phase.

The expansion below covers `Network` and `Filesystem` only. Subprocess
tracking is intentionally deferred to the FFI design conversation.

---

## Items

### Item 1 — Add `Network` and `Filesystem` variants

**Driver.** The trust stack story needs the effect taxonomy to cover the
categories that matter for the broader supply-chain pitch — "this package
touches the network" and "this package reads files" are the questions
operators ask, and the type system can answer them only if the variants
exist.

**Scope.**

- Extend the `Effect` enum in `crates/chelis-types/src/types.rs` with two new
  variants: `Network` and `Filesystem`. No subdivision (no
  `NetworkRead` / `NetworkWrite`, no `FilesystemRead` / `FilesystemWrite`)
  in this round; finer granularity can land later if a real driver appears.
- Effect inference and propagation in `chelis-effects` handles the new
  variants identically to existing ones. Transitive inference, cross-module
  propagation, and the `validate_declared_vs_inferred` check all flow
  through unchanged because the propagation logic is variant-agnostic.
- Annotate every chelis-std function that touches network or filesystem
  with the new effects. This is mechanical but exhaustive: each
  file-reading function gains `! { Filesystem }`; each network-touching
  function gains `! { Network }`. The annotation pass runs across `Std.Io`
  (filesystem-touching functions), any `Std.Net` or future network-touching
  modules, and any helper that transitively reaches one.
- Annotate the four shells (Coral, Nautilus, Shoals, Octant) so their
  public-API signatures correctly declare the new effects where they
  apply. This is a downstream consumer task per shell; expect ~half a day
  per shell after the chelis-std annotation lands.

**Acceptance oracle.**

- The `Effect` enum has the two new variants and the `Display` / `Debug` /
  serde implementations cover them.
- A small test program calls a `Network`-effecting function from an
  `! {}` context; `chelis check` rejects it with a clear error.
- A second test program calls a `Filesystem`-effecting function from a
  `! { Filesystem }` context; `chelis check` accepts it.
- Every `Std.Io` function in chelis-std that opens a file declares
  `! { Filesystem }`; verified by a corpus check that walks `Std.Io`
  exports and asserts presence of the effect on each.

**Effort.** Roughly one focused week. Most of the cost is the exhaustive
annotation pass across chelis-std and the shells; the enum extension and
inference plumbing is a few hours.

---

### Item 2 — Effect aggregation CLI: `chelis audit --effects`

**Driver.** Per-function effect tracking exists; the data is stored in the
`.chb` shell artifact at `crates/chelis-reef/src/lib.rs:1752-1778` (each
exported symbol carries an `effects: Vec<String>` field). Consumers cannot
audit a package as a whole because there is no command that surfaces this.

**Scope.**

- New CLI subcommand: `chelis audit --effects <package.chb>`. Reads the
  shell artifact, walks every exported symbol, computes the union of
  declared effects across the public API, and prints both a
  human-readable summary and a structured JSON form for programmatic
  consumption.
- The same data exposed via Tide: a new HTTP endpoint and a new MCP tool
  named `chelis_audit` (mirrors the existing `chelis_check` /
  `chelis_compile` / `chelis_eval` etc. tools at
  `crates/chelis-tide/src/mcp.rs`). Agent-driven workflows can request
  the effect aggregation alongside the rest of the compiler API.
- Output format: human-readable lists "this package's public API uses
  the following effects: {Filesystem, Io}", with optional
  `--per-symbol` flag that lists each export and its declared effect
  row. JSON form is a flat object keyed by symbol name with effect
  arrays as values plus an aggregated `union` field at the top.

**Acceptance oracle.**

- `chelis audit --effects nautilus.chb` outputs the union of effects
  declared by Nautilus's public API.
- A round-trip test: build a fixture package with known effects on its
  exports, run `chelis audit --effects` against it, assert the output
  matches a checked-in golden.
- The MCP tool `chelis_audit` returns the same JSON shape when invoked
  through the MCP harness.

**Effort.** Roughly one to two days. The data layout is already in the
artifact; the work is exposing it through the CLI and JSON formats and
wiring the MCP tool entry alongside existing ones.

---

### Item 3 — Capability-based execution flag: `chelis run --refuse`

**Driver.** Operators want to mandate "this binary, regardless of what its
source claims, can only run with these effects allowed." Today there is no
enforcement mechanism — effect checking is compile-time only, and a
deployer has no way to reject a binary at execution time.

**Scope.**

- New CLI flag on `chelis run`: `--refuse Network,Filesystem` (and the
  symmetric `--allow-only Io`). Reads the binary's effect
  manifest (the union computed at compile time, stored alongside the
  binary), compares it against the deployer's allow/refuse list, and
  refuses to execute if the binary declares effects beyond the allowlist.
- This is signature-based pre-flight refusal, not runtime sandboxing.
  The compiled binary is still trusted to be honest about itself; the
  flag enforces what the manifest claims, not what the binary actually
  does at run time. True syscall interception (intercepting actual
  network calls or file opens) is a separate post-roadmap concern that
  requires OS-level integration.
- Clear error messages: "binary declares effects {Network,
  Filesystem}; --refuse list excludes Network; refusing to execute".

**Acceptance oracle.**

- A binary compiled with `! { Network }` runs successfully under
  `chelis run` (no flag).
- The same binary fails with a clear error under
  `chelis run --refuse Network`.
- Test cases for every effect variant cover both the accept path and
  the refuse path.

**Effort.** Three to five days after Item 1 lands. Manifest reading and
effect comparison are straightforward; the time is spent on the
allow/refuse list parsing, the run-command wiring, and the error-message
catalog.

**Dependency.** Item 1 must land first (the new variants need to exist
before they are meaningful in the allowlist). Item 2 helps but is not
strictly required; the run-command can read the manifest directly without
going through the audit CLI.

---

### Item 4 — Reef effect manifests at install time

**Driver.** When installing a dependency, users should see the effects it
requires before agreeing to install. "This package requires Network and
Filesystem — proceed?" is a different posture than "trust whatever you
install."

**Scope.**

- The reef install path (any of `--from-monorepo`, `--from-github`,
  `--from-lockfile`, `--bootstrap`) computes effect aggregation during
  install using the same logic Item 2's audit command exposes. The
  effect manifest is stored alongside the artifact in the local
  registry; subsequent `chelis audit --effects` invocations can read
  it without re-walking the package.
- New flag `chelis reef install --print-effects` prints the
  aggregated effects before placing the artifact. In interactive mode,
  prompts for confirmation. In non-interactive mode (the default for
  CI), proceeds without prompting but the print is still emitted for
  log capture.
- New flag `chelis reef install --refuse Network,Filesystem` rejects
  the install if the package's effects exceed the allowlist (parallel
  to Item 3's runtime flag at the install boundary).

**Acceptance oracle.**

- Installing a package with declared effects under `--print-effects`
  prints the effect manifest before placement.
- Installing a package whose effects include `Network` under
  `--refuse Network` fails with a clear error and does not place the
  artifact.
- The lockfile records the effect aggregation so a subsequent install
  on a different machine sees the same manifest without re-walking.

**Effort.** Two to three days after Items 1 and 2 land. Mostly wiring
existing components together at the install boundary.

**Dependency.** Item 1 (the variants need to exist), Item 2 (the
aggregation logic, factored out of the CLI handler so it can be called
from inside reef), and the reef remote-fetch machinery in
`reef_distribution.md` (so the install path has multiple sources to
plug into).

---

## Dependencies summary

```
Item 1  (Network/Filesystem variants)
   |
   +------> Item 2  (chelis audit --effects)
   |           |
   |           v
   +------> Item 3  (chelis run --refuse)         (depends on Item 1; Item 2 helpful)
   |
   +------> Item 4  (reef install effect manifest) (depends on Items 1, 2, and reef remote fetch)
```

Item 1 is the load-bearing prerequisite. Items 2, 3, 4 layer on top
and can land in any order after Item 1 — though Item 2 lands first in
practice because Items 3 and 4 reuse the aggregation logic.

---

## Reused machinery

- `chelis-effects` propagation logic: variant-agnostic; new variants
  flow through unchanged.
- `infer_program_effects`: same.
- The per-export effect rows already in `.chb` artifacts at
  `crates/chelis-reef/src/lib.rs:1752-1778`: source of data for Items
  2 and 4.
- The MCP tool dispatch table in `crates/chelis-tide/src/mcp.rs`:
  Item 2 adds an entry alongside existing tools.
- The reef install code path: Item 4 hooks at the validation boundary
  shared with `--from-monorepo` and `--from-github`.

---

## Out of scope

- **Signing and authenticity.** Content addressing via SHA256 is
  shipped today; cryptographic signing of artifacts (publisher key,
  signature verification on install) is a separate concern. Tracked
  but demand-driven; not designed speculatively.
- **FFI security model.** No FFI exists today. When FFI is designed
  in a later phase, security becomes a design question for that
  phase. The taxonomy expansion here does not anticipate FFI shape.
- **Runtime syscall interception.** OS-level integration; post-
  roadmap. Item 3 is signature-based pre-flight refusal, not real
  sandboxing.
- **Bit-reproducible-build verification end-to-end.** The C emitter
  has not been audited for non-determinism (timestamps, embedded
  paths, etc.). Inputs are pinned via the manifest today; verifying
  that the same inputs produce bit-identical outputs is a separate
  workstream.
- **Finer-grained network/filesystem subdivision.** A future round
  may split `Network` into `NetworkRead` / `NetworkWrite` (or by
  protocol) and `Filesystem` similarly. Not in this round; would
  ride on a real driver showing the current granularity is too
  coarse for some operator's policy.
