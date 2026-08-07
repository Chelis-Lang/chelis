# Design: add-bounded-monomorphization

## Context

Ordinary generic host functions have no standalone C symbol: `lower_host_program`
skips their definitions and `crates/chelis-ir/src/host.rs` specializes each checked
call site by inlining (`inline_top_level_host_call`), substituting the body with the
call's arguments. Inlining is a tree-shaped operation, so recursion cannot survive
it: the `INLINING_STACK` cycle guard detects the self-call, inlining bails, and the
call falls through to a branded `[05-UNS-1]` rejection whose citation names the
closed chelis#941 (live successor: chelis#1158). Two gaps block the fix:

1. **No deciding language rule.** `spec/04-type-system.md` says nothing about
   recursive generic instantiation. Per-instantiation specialization terminates only
   when the instantiation set is finite, and nothing in the normative tier
   guarantees that today. Per [05-UNS-5], a *deliberate* rejection must cite a spec
   atom; none exists to cite.
2. **No specialization mechanism that can represent a cycle.** Inlining copies
   bodies into call sites; a recursive call graph needs named symbols whose edges
   are calls.

Downstream, coral's invoked native Frame lane (`frame__drop_nan` →
`hamt__from_pairs_rec`) is blocked on exactly this, and its conform audits trip on
the closed-issue citation.

## Goals / Non-Goals

**Goals:**

- Author the normative uniform-recursive-instantiation rule in spec/04 §3.1 and
  enforce it in the checker (lane-uniform, check-time, atom-citing).
- Compile uniform recursive generic host calls on native targets via bounded,
  memoized, per-instantiation specialized symbols with recursive edges preserved as
  calls.
- Retire the chelis#941-branded rejection for the newly supported cases; keep the
  fail-closed [05-UNS] boundary for anything that still cannot lower, citing open
  chelis#1158 while not-yet-implemented residue exists.

**Non-Goals:**

- No boxed/uniform runtime representation and no runtime type metadata; polymorphic
  recursion stays rejected, deliberately.
- No migration of *non-recursive* generic lowering off the existing inlining path.
- No change to the Tier-2 precision or Tier-3 rank specialization paths in
  `host.rs`; they keep their own rules.
- No eval-lane execution changes; eval inherits only the new check-time rejection.

## Decisions

### D1: Monomorphization, not uniform representation or runtime metadata

Chelis emits plain C the user compiles manually; there is no runtime to carry
witness tables (Swift) or a JIT to specialize lazily (.NET), and a Lean-style boxed
`object*` representation would route numeric values behind an untagged pointer
funnel — the exact channel shape the numeric-surface discipline exists to prevent —
while adding an RC runtime the project deliberately does not have. Per-instantiation
specialization matches the language's static ethos (fixed dtypes, named dims, no
implicit anything) and the existing precision/rank specialization precedents.
**Alternatives considered:** boxed uniform representation (rejected: runtime
machinery plus numeric-surface conflict); Swift-style value witness tables
(rejected: requires runtime metadata); keep inlining and special-case recursion with
a depth cap (rejected: still unbounded code growth and no named symbol for the
recursive edge to target).

### D2: Boundedness by static rule (MLton/SML), not by instantiation-depth limit (Rust)

Rust caps monomorphization with `recursion_limit` because its trait system makes
exact detection impractical; the failure surfaces at instantiation time, far from
the source cause, and acceptance depends on a numeric knob. Chelis can instead adopt
the Standard ML property: recursive calls inside a binding group are typed at the
caller's own instantiation, so every accepted program has a finite instantiation set
*by construction*. Checking uniformity of a given program is decidable (it is
polymorphic-recursion *inference* that is undecidable), lands at the checker — the
earliest competent stage per [05-UNS-2] — with a source span, and specs cleanly as a
timeless language rule. Lowering may then assume boundedness and never re-litigates
it (a checker/lowering disagreement is a defect in the checker, mirroring the
[05-UNS-4] gate/emitter rule).

### D3: Spec-first atom placement

The rule is a language decision, so it is authored in `spec/04-type-system.md` §3.1
(Algorithm W: constraint generation, unification, generalization) as new `[04-INF-N]`
atoms in the same change set, before implementation. The `type-system` capability
delta mirrors it; the numbered chapter remains controlling (no transfer). Atom
numbers are allocated against current `main` at authoring time, not reserved here.

### D4: Lowering architecture — worklist + memo at program level

Replace the recursion-detected fallback (inline bail → chelis#941 rejection) with a
specialization request: a memo table keyed by `(def name, canonical checked type
application)` and a worklist processed at `lower_host_program` level, emitting one
specialized definition per key. The memo entry is inserted *before* the body is
lowered, so a recursive edge encountered mid-emission resolves to the in-progress
symbol (this is what terminates mutual recursion: the group's entries exist by the
time cross-calls are lowered). Symbol names are mangled deterministically from the
def name plus the canonical type identity (reusing the existing checker-type
canonicalization, e.g. the `canonicalize_representation_erased_adt_args` family) so
repeated builds and distinct instantiations are collision-free and stable.
Non-recursive generics keep the existing inlining path unchanged; only the path that
today returns the branded error changes behavior. **Alternative considered:**
specializing every generic call (dropping inlining entirely) — rejected as churn
beyond the blocking defect; revisit separately if inlining costs show up.

### D5: Checker enforcement shape

In `chelis-types::infer`: compute recursive binding groups (SCCs of the top-level
def call graph), and for each in-group call require the callee's type application to
equal the caller's own instantiation of the group parameters. A violation produces
an `ErrorWitness`-conformant diagnostic naming the function, both instantiations,
and the deciding atom. A verification task first probes current behavior: Algorithm
W's monomorphic recursive-binding assumption may already reject these programs, in
which case the check formalizes existing behavior (not BREAKING) and the work is
mostly diagnostic quality; if authored signatures currently let a
polymorphic-recursive program through any lane, the CHANGELOG carries a BREAKING
note with the reproducer.

### D6: Citation hygiene ships first

The shipped diagnostic must not cite a closed issue. The first task group flips
`chelis#941` → `chelis#1158` at the `host.rs` rejection site and the
`issue_935_nullary_generic_adt.rs` assertions. It is deliberately independent so it
can land even if the rest of the change stalls; the loud-unsupported boundary stays
fully intact until D4 proves out.

### D7: Acceptance oracle

One authoritative oracle: `cargo nextest run -p chelis-cli --test
recursive_generic_monomorphization --no-fail-fast` — a new suite covering (positive)
direct and mutual recursion at one and at two instantiations through `chelis build`
plus native compile-link-run with exact-output comparison against the eval lane, and
(negative) polymorphic recursion rejected at check time with the atom-citing
diagnostic in both `eval` and `build`, plus the no-closed-issue-citation assertion.
The chelis#941 minimized `Box[a]`/`loop` reproducer is a member of the suite.
Supporting evidence (not the oracle): coral's blocked probes flip on its next
conform bump.

## Risks / Trade-offs

- [Checker currently accepts polymorphic recursion somewhere] → D5's verification
  probe runs before implementation; if acceptance exists, BREAKING note plus
  reproducer-derived negative fixtures; the corpus (`examples/`, stdlib) is scanned
  for affected programs (expected: none, since such programs cannot lower today).
- [Symbol-mangling collision or instability] → mangle from canonical type identity
  only; golden test asserts exact emitted symbol set for a two-instantiation
  program, run twice for determinism.
- [Code-size growth with instantiation count] → accepted monomorphization
  trade-off; memoization guarantees one definition per instantiation; out of scope
  to optimize further.
- [Divergence between C and HIP/Metal host lanes] → specialization lives in shared
  `chelis-ir` host lowering; a task verifies both non-C targets consume the same
  path or records the gap as explicit [05-UNS] residue citing chelis#1158.
- [Regression risk to Tier-2/Tier-3 precision/rank specialization] → those paths
  are untouched by construction (D4 changes only the ordinary-generic fallback);
  their existing suites are the regression oracle.
- [In-progress memo entries leaking state across builds] → memo lifetime is one
  `lower_host_program` invocation, matching the existing per-invocation cache
  discipline (`HOST_LOWERING_CACHE_ACTIVE`); a test asserts repeated invocations
  rebuild specializations.
- [Fail-closed erosion] → any call the specializer cannot serve still routes to the
  branded [05-UNS] rejection; negative fixtures assert the failure is loud and no C
  artifact is written.

## Migration Plan

1. Citation flip (D6) — independent, land first.
2. Spec/04 atoms + capability deltas + failing tests (spec-first: negative
   polymorphic-recursion fixtures and positive compile-run stubs red).
3. Checker uniformity enforcement (D5) — negative half of the oracle green.
4. Lowering specialization (D4) — positive half green; branded rejection retired for
   supported cases.
5. Docs, executable example, `spec/design/loud_unsupported.md` boundary retirement,
   CHANGELOG; red team per protocol; chelis#1158 closes on the green oracle.

Rollback: each phase is independently revertible; reverting D4 restores the branded
rejection (citing the still-open chelis#1158), never a silent path.

## Open Questions

- Exact `[04-INF-N]` numbers — allocate against `main` at authoring time.
- Does any non-C host lane bypass `chelis-ir` host lowering for these calls? (Task
  verifies; expected: no.)
- Should non-recursive generics later migrate from inlining to the same
  specialization machinery for code-size and compile-time reasons? Out of scope
  here; file separately if D4's machinery proves cheaper than inlining.
