# `module-pascal-components` allowlist gap — diagnosis

Tracking note for Item 4 of the 0.7.6 toolchain hygiene workstream. Locates
where the `module-pascal-components` lint flags ecosystem names that the
spec accepts, identifies the canonical chelis-ecosystem name set the
allowlist should cover, and records a structural-fix recommendation for
the orchestrator.

This note is not a §5 entry. It is the input to the orchestrator's decision
on whether to file one.

## Files inspected

- `crates/chelis-lint/src/rules/module_pascal_components.rs`
  - `KNOWN_SINGLE_WORDS` allowlist (L61-170)
  - `KNOWN_PASCAL_COMPOUNDS` allowlist (L178-204)
  - `component_violation()` predicate (L228-276)
  - In-module tests (L278-end)
- `spec/01-nomenclature.md` §6.1 ("Module ladder structure") and §6.3
  ("Module-name PascalCase per component")
- `spec/design/chelis_canonical_reference.md` §"Shell Ecosystem"
- `docs/archive/snapshots/ecosystem_naming_snapshot.md` (May 2026
  ecosystem-naming survey; descriptive prior art)
- `Cargo.toml` workspace `members` list

## What the rule does, line by line

`component_violation(s: &str) -> Option<&'static str>` in
`module_pascal_components.rs:228`:

1. **L231-233** — reject anything that does not start with an ASCII
   uppercase letter.
2. **L239-241** — if the component is in `KNOWN_SINGLE_WORDS`, accept
   unconditionally. The allowlist wins over later checks (the comment at
   L234-238 explains the precedence, citing the `Orderbook`-vs-`OrderBook`
   collision case).
3. **L246-248** — if the component matches the "first-cap-only" flattening
   of a name in `KNOWN_PASCAL_COMPOUNDS`, reject. Example: `Linalg` is the
   flattened form of `LinAlg`, so `Linalg` is flagged.
4. **L254-257** — if the component has any internal uppercase letter,
   accept. (Internal caps signal proper word breaks per §6.3.)
5. **L258-274** — otherwise count the longest run of consecutive lowercase
   letters after the leading cap. If the maximum lowercase run is `>= 7`,
   flag the component as "looks like a multi-word compound but has no
   internal capital". Otherwise accept.

The predicate is correct per spec §6.1/§6.3. The 7-char threshold is a
heuristic — it catches `Examplerootfind` (12 chars) but lets `Pricing`
(6 chars) through. Single English words longer than 7 chars need to be
on the `KNOWN_SINGLE_WORDS` allowlist to bypass step 5.

## Empirical probe against the user's report

Re-implemented `component_violation` outside the workspace and ran it
against each name in the user's report plus the canonical ecosystem
roster. Results:

| Name       | Length after leading cap | Result today |
|------------|--------------------------|--------------|
| `Capstone` | 7 lowercase              | **flagged** (long-run >= 7) |
| `Coral`    | 4 lowercase              | accepted (allowlist L66) |
| `Nautilus` | 7 lowercase              | accepted (allowlist L65) |
| `Chelis`   | 5                        | accepted (allowlist L63) |
| `Std`      | 2                        | accepted (allowlist L69) |
| `CEarchin` | has internal cap         | accepted (step 4) |
| `Shoals`   | 5                        | accepted (allowlist L67) |
| `Octant`   | 5                        | accepted (allowlist L68) |
| `School`   | 5                        | accepted (short run) |
| `Darwin`   | 5                        | accepted (short run) |
| `Hull`     | 3                        | accepted (short run) |
| `Hydrostatic` | 5                        | accepted (short run) |

So only `Capstone` is actually flagged today among the names the
user's report names. The detection logic at L228-276 is **correct**:
it neither under-rejects nor over-rejects relative to spec §6.3. The
gap is purely in the `KNOWN_SINGLE_WORDS` allowlist — a missing entry,
not a regex bug.

This rules out the escalation path the dispatch brief flagged
("if the rule's case-detection actually does flag valid Pascal names —
that would be a regex bug, not an allowlist gap"). The fix scope stays
at L61-170; the detection logic at L228-276 is not touched.

## The canonical chelis-ecosystem name set

Cross-checked three sources:

1. **`spec/01-nomenclature.md` §2.6 "Reef package manifests"** lists
   `module_prefix` values for shipped reef packages: `Std`, `Nautilus`,
   `Coral`, `Shoals`, `Octant`, `CEarchin`.
2. **`spec/design/chelis_canonical_reference.md` §"Shell Ecosystem"**
   lists active and stub shells:
   - Runtime: `Chelis`, `Std`
   - Active shells: `Nautilus`, `Coral`, `Shoals`, `Octant`
   - Post-Phase-3 stubs: `School`, `Darwin`, `Hull`, `Hydrostatic`
   - Special-case external: `CEarchin`
3. **`Cargo.toml` workspace `members`** lists crate dirs prefixed
   `chelis-*`. All map to the `Chelis` module prefix; no other ecosystem
   prefixes appear.

`Capstone` does **not** appear in any of those three sources. It is a
user-reported name that is not part of the documented shell ecosystem.
The dispatch brief treats it as an ecosystem name to be added — this
note records that distinction so the orchestrator can decide whether
`Capstone` belongs in the spec's shell table as well as in the lint
allowlist.

The current allowlist (L61-170) already covers:

- All five reef-manifest `module_prefix` values (`Std`, `Nautilus`,
  `Coral`, `Shoals`, `Octant`) plus the runtime `Chelis` and the
  special-case `CEarchin`.
- All four post-Phase-3 stub shells (`School`, `Darwin`, `Hull`,
  `Hydrostatic`) — via the short-run path, not the allowlist, but accepted
  nonetheless.

The single allowlist gap relative to the user's report is `Capstone`.

## Fix scope (for commit 3)

Add `Capstone` to `KNOWN_SINGLE_WORDS` in `module_pascal_components.rs`.
Do **not** modify the detection logic at L228-276. Flip the
`accepts_capstone_ecosystem_name` fixture from `#[ignore]` to running.

## Structural-fix recommendation

The allowlist is a hand-maintained Rust constant. Every new ecosystem
shell or every new long single-word module name that trips the 7-char
threshold requires an editor pass over `module_pascal_components.rs`.
That drift is the underlying problem; `Capstone` is a symptom.

Two ways to make `KNOWN_SINGLE_WORDS` track the canonical vocabulary
automatically:

1. **Centralize**: extract the ecosystem-name set into a single Rust
   constant in (e.g.) `chelis-validate` or a new `chelis-ecosystem`
   crate, and have `chelis-lint`, `chelis-reef`, and any other consumer
   import it. One source of truth in code, but still hand-maintained.
2. **Generate from spec**: add a canonical ecosystem-name table to
   `spec/01-nomenclature.md` §6 (or extend §2.6) listing every
   `module_prefix` that the shell taxonomy admits, then generate the
   Rust constant at build time from that table. Source of truth lives
   in the spec; drift is detectable by `chelis lint` itself.

Either path eliminates the per-shell hand edit. **This is a
recommendation for the orchestrator, not a §5 entry.** The orchestrator
decides whether to file the recommendation as a §5 entry in
`docs/gap_synthesis.md`.

## What this PR does **not** do

- Does not change the rule's detection logic at L228-276 (the regex /
  long-run heuristic is correct per spec §6.3).
- Does not implement the structural fix (centralized constant or
  spec-generated allowlist) — recorded as recommendation only.
- Does not file a §5 entry — that is the orchestrator's call.
- Does not edit `spec/01-nomenclature.md` — the spec is canonical;
  the rule's allowlist is what's wrong. The user-reported name
  `Capstone` is not in the spec's shell table; whether to add it there
  is also an orchestrator decision.
