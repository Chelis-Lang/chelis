# Proposal: add-bounded-monomorphization

## Why

A recursive generic host call is rejected at C lowering with the branded diagnostic
`recursive generic host call ... requires bounded monomorphized symbols (chelis#941; [05-UNS-1])`
even though the program is well typed — chelis#941 is closed and chelis#1158 is the live
successor. The blocking gap is twofold: the compiler's only specialization mechanism for
ordinary generics is call-site inlining, which cannot represent a recursive call graph;
and the language rule that makes per-instantiation specialization finite (recursive calls
must reuse the caller's own instantiation — no polymorphic recursion) was never authored
in `spec/04-type-system.md`, so today's rejection has no deciding spec atom to cite per
[05-UNS-5]. This blocks every downstream program whose host code walks a recursive
generic structure, most concretely coral's invoked native Frame lane
(`frame__drop_nan` → `hamt__from_pairs_rec`).

## What Changes

- **New normative language rule (spec/04 §3.1, new atoms):** a top-level generic
  function may recurse, directly or mutually, but every recursive call in the
  recursive group MUST be at the caller's own instantiation. A recursive call at a
  different instantiation (polymorphic recursion) is a **type error**, reported by
  the checker (the earliest competent stage per [05-UNS-2]) and citing the new atom
  as its deciding authority per [05-UNS-5]. This makes the checked-program
  instantiation set finite by construction and applies uniformly to every lane
  (eval, C, HIP, Metal).
- **Bounded memoized monomorphization in host lowering:** the C backend host lane
  compiles a recursive generic call by emitting one specialized definition per
  distinct checked type application, memoized so each `(function, instantiation)`
  pair is generated exactly once and recursive edges lower to ordinary calls to the
  (possibly in-progress) specialized symbol. No eager unbounded AST expansion; no
  generated reference to an omitted generic definition. Non-recursive generics may
  keep the existing inlining path; its recursion-detected fallback becomes
  specialization instead of rejection.
- **Branded rejection retired:** the chelis#941-branded lowering error is removed for
  uniform recursive generics (they now compile). Until implementation lands, the
  in-flight citation moves from closed chelis#941 to open chelis#1158 so no shipped
  diagnostic cites a closed issue.
- **BREAKING (conditional, checker semantics):** if any lane currently accepts a
  polymorphic-recursive program (e.g. `eval`), that program becomes a type error.
  Design task 1 verifies current checker behavior; if the checker already rejects
  non-uniform recursive instantiation as a consequence of monomorphic recursive
  binding assumptions, this bullet is a formalization, not a behavior change.
- Regression coverage for direct and mutual recursion (positive: compile, link, run
  with exact outputs; negative: polymorphic recursion rejects with the atom-citing
  diagnostic), per the acceptance criteria carried in chelis#1158.

Non-goals:

- No opt-in boxed/uniform representation for generics; nested-datatype-style
  polymorphic recursion stays rejected, deliberately.
- No change to tensor rank/precision polymorphism lowering (Tier-2/Tier-3 paths in
  `host.rs` keep their own specialization rules).
- No change to the eval interpreter's execution strategy; it already executes
  recursive generics without specialization and only inherits the new type-level
  rejection.
- No numeric-surface change: specialization introduces no new public ADT variant,
  wire field, or exported C declaration carrying numeric values outside the tagged
  carrier; specialized symbols are ordinary lowered defs under the existing census
  rules.

## Capabilities

### New Capabilities

- `generic-monomorphization`: the compile-and-run contract for generic host
  functions on native build targets — per-instantiation specialized symbols,
  memoized, recursion preserved as calls, no references to omitted generic
  definitions, and the deliberate atom-citing rejection of non-uniform recursive
  instantiation.

### Modified Capabilities

- `type-system`: add the uniform recursive instantiation requirement — recursive
  generic groups are typed with every in-group recursive call at the caller's own
  instantiation; polymorphic recursion is a checker-reported type error citing its
  deciding spec/04 atom.

## Impact

- `spec/04-type-system.md` §3.1 — new normative atoms (uniform recursive
  instantiation; checker-stage rejection). Numbered spec remains the deciding
  authority; the capability delta mirrors it.
- `crates/chelis-types/src/infer/` — uniformity check on recursive generic groups
  (direct and mutual), new `ErrorWitness`-conformant diagnostic.
- `crates/chelis-ir/src/host.rs` — replace the inline-or-reject path
  (`INLINING_STACK` bail → chelis#941 rejection) with worklist + memo keyed by
  checked type application; `lower_host_program`'s generic-definition skip gains
  specialized-definition emission; deterministic symbol mangling.
- `crates/chelis-backend-c` (and HIP/Metal host lanes to the extent they share host
  lowering) — emission of specialized definitions; link-clean output.
- `crates/chelis-cli/tests/` — `issue_935_nullary_generic_adt.rs` branded-message
  assertions flip; new corpus for direct/mutual recursion (build + run) and
  polymorphic-recursion rejection.
- `examples/` — an executable recursive-generic example on the Phase 0 path.
- `spec/design/loud_unsupported.md` — the chelis#941 boundary entry retires to a
  delivered-behavior record.
- Docs and trackers: `CHANGELOG.md`; chelis#1158 closes on the green acceptance
  oracle; coral's 6 citation sites re-cite or retire on its next conform bump.
