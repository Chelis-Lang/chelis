# Faithful Observation: one dtype-true formatter for every exit, both lanes

**Status:** Phases 0-2 LANDED. Phase 0 (the round-trip harness and the
exit census) landed 2026-07-17 (PR #752, tightened by PR #774). Phase 1
(the formatter, eval adoption, and the eval-side §B2.1 migration) landed
2026-07-20: `format_element` lives at `chelis-types::observation`, every
eval exit routes through it, §C1 and the number grammar are FROZEN (B1),
and the contract is ratified as `spec/05-risc-primitives.md` §8 atoms
[05-OBS-1..5] (the chelis#775 scalar-root decision is [05-OBS-4]). One
annexed value-layer exception, surfaced by PR #792's red team: int64
scalar roots above 2^53 render the f64-collapsed stored value at the
labeled root ([#684]'s rank-0 realization, repaired by [#729]; the
exception and its ignored red cell are recorded at spec/05 §8).
Phase 2 (the generated C side and the C-side §B2.1 migration) landed
2026-07-24: `chelis_format_shortest` in the runtime, the print helper
generated from an exhaustive `Prim` match, `to_list`'s F16/BF16 arms,
and §C2.3 cross-lane byte equality locked for identical stored bits -
[#716]/[#723]/[#726]-C/[#748]/[#749] fixed by un-ignoring their red
cells (close the issues on the PR #863 merge). Four recorded
boundaries, each issue-linked ([#864] and [#865] with ignored red
cells in the harness): eval TENSOR float elements still render at the stored
f64 width (the deliberate §8.1 width note, [#729]'s metadata repair),
so non-dyadic narrow-float tensor cells stay width-divergent across
lanes until then; eval's LABELED ROOT of a cast-constructed f64 tensor
narrows through the stale F32 tag ([#864], the [#717] family - an
[05-OBS-1] violation inside eval, surfaced by PR #863's red team); the
compiled lane's untagged f64 value box renders narrower float elements
(f32 as well as f16/bf16) at f64-image width through `to_list` and
list/tuple boxing - faithful parse-back, not own-width shortest
([#865], the [#729]/[#686] capacity family); and unit-valued
single-print-root labeling diverges ([#862], a root-labeling discovery
outside the [05-OBS] atoms). Phase 3 remains. Tracking issue: [#732].
**Owning specs:** `spec/05-risc-primitives.md` (its §8 carries this
plan's ratified contract as current blockquote authorities [05-OBS-1..5]; the
per-op tolerance table lands into the same section at Phase 3, while
chelis#733 later migrates authority form and revisions through the pinned Buoy
shell-side integration), `spec/04-type-system.md` (dtype value-set
definitions, shared with `spec/design/dtype_semantics.md` §C1), and the audit
record in
`docs/investigations/numeric_audit_next_sweeps.md` (sweeps 1, 3) /
`docs/investigations/numeric_audit_structural_prevention.md` (item 6).
**Class fixed:** [#728] (the observation channel is not dtype-faithful),
concretely [#716] (f16/bf16 buffers printed as f32 garbage), [#723] (exact int64
printed through double), the bool `2.0`-vs-`true` exit split, and the
eval-vs-C float-formatting divergence that blocks [#687].
**Ownership note:** `spec/design/dtype_semantics.md` §C4 states these rules
as the interface its phases rely on; THIS document is the authoritative
elaboration and the delivery plan. The rules are identical by construction -
an edit to either updates both in the same change set (their B1 protocol).
§I1 pins the boundary and the both-orderings story.

## Summary

Every printed numeric value in both lanes funnels through one f64-shaped
exit: the emitted C print helper narrows every dtype into a single
`double value` (explicit-but-lossy for int64, wrong-pointer default for
f16/bf16), and eval formats through its own f64-width path with different
digit rules than the C side's `%.1f`/`%.16g` split. Measured consequences:

- an **exact** compiled int64 sum of 2^53 + 1 prints as `...992.0` - the
  correct lane manufactures evidence against itself ([#723]; provable only
  via `to_list`, which has its own per-dtype gaps);
- **correct** f16/bf16 kernel results print as garbage
  (`0.0004898309707641602` = two right answers read as one f32, [#716]);
- one bool tensor reads `2.0` from `print` and `true` from `to_list`
  ([#726]'s observation half);
- the same correct f32 value prints `1.4142135381698608` from eval and
  `1.414213538169861` from C - so byte-exact lane comparison ([#687]'s
  oracle) is impossible even where both lanes are RIGHT.

Three audit probes initially mis-scored a lane because of this channel -
inside an audit designed around exact strings. Anything validating the
[#680]/[#684]/[#695]/[#727] fixes against printed output inherits the same hazard,
which is why this class has its own meta and its own plan.

**The central design fact:** observation-faithfulness ("print exactly what
is stored, at its dtype") is separable from semantics-correctness ("store
the right thing", [#729]'s job). The C lane's int64 sum is already
exact and merely printed wrong; the f16 kernels are already correct and
merely read wrong. **This plan is therefore independently landable before
[#729]** - it fixes real bugs on its own ([#716], [#723]) and hands [#729] the
instrument its phases are validated with. §I1 pins how the two compose in
either landing order.

**Scope verdict:** no rewrite, no storage change, no semantics change. One
new shared component (the formatter and its C generator), a per-dtype
completion of the runtime's read paths (`to_list` and the print helper),
one coordinated expectations migration (§B2.1's carve-out - the ONLY plan
in this family allowed to touch control strings, exactly once), and a
round-trip invariant that makes regressions mechanical to catch.

## Non-goals

- **Not** value semantics. What gets STORED - eval's missing f16 rounding
  ([#717]), the storage decision ([#684]), width traps ([#718]) - is [#729]. This
  plan prints stored bits faithfully even when the stored bits are wrong;
  that is a feature (it makes [#729]'s bugs visible instead of laundered).
- **Not** ingress. `to_tensor`'s narrow-float runtime abort and the host
  literal paths are [#714]/[#729] territory. This plan owns EXITS: `print`,
  `to_list`, diagnostics rendering of values, and the wire's *rendering*
  (the wire's representational capacity is [#729]'s storage decision - §I1).
- **Not** the `<value>` placeholder (an unclassifiable-value substitution:
  `loud_unsupported.md` census row 4, becomes a diagnostic there).
- **Not** kernel accuracy. Where lanes legitimately compute different
  VALUES (libm/SLEEF/vForce, [#719]), this plan documents the per-op bound
  (§C4) and requires `sqrt` correctly rounded per [#719]; it does not
  replace kernels. Formatting itself never has tolerance.

## Vocabulary

- **Exit** - any surface where a stored value becomes text or leaves the
  process: `print`, `to_list`, diagnostics, the wire schema's rendering.
- **Faithful** - the emitted text round-trips to exactly the stored bits
  at the value's own dtype width. The class's definition of "not lying".
- **Rendering vs capacity** - `format(bits)` vs "can the channel carry
  the bits at all". Rendering is this plan; capacity is storage ([#729]).
  Example: the CURRENT wire schema (`Vec<f64>`) cannot carry exact int64
  above 2^53 no matter how it renders - a capacity limit, [#686]/[#729].
- **The migration** - the one coordinated update of printed-string
  expectations across the test suite when the contract lands (§B2.1).

---

# Part I - the normative contracts (§C1-§C4)

## C1. The formatting contract (normative; = dtype_semantics.md §C4 rules)

One function, one output, every exit, both lanes:

```rust
/// THE printed form of one element. Eval calls it directly; the C print
/// helper is GENERATED from it (§C3). No other formatting path for
/// tensor/scalar payloads may exist in any lane.
pub fn format_element(prim: Prim, value: ElementRef) -> String;
```

(Landed signature note, Phase 1: `ElementRef` is a `Copy` enum carrying
the element at its dtype's own width, mirroring the runtime's
`ScalarBits`; the pre-implementation sketch's lifetime was dropped - no
borrow is needed for scalar payloads. A `prim`/variant mismatch panics
loudly, the same dtype/bits invariant `ScalarPayload` enforces.)

Per-dtype rules:

1. **Integers print as integers.** `data=[750]`, never `750.0`. int64
   prints all digits exactly (i64 formatting, never through double).
2. **Floats print shortest-round-trip AT THEIR OWN WIDTH**: the shortest
   string that parses back to exactly the stored f64 / f32 / f16 / bf16
   value. This single rule replaces C's `%.1f`/`%.16g` split and eval's
   f64-width formatting, and is what makes byte-equal lane comparison
   possible. (Phase 1 width note: eval TENSOR float elements still
   render at the stored f64 width - the eval tensor store is f64-backed
   and its precision tag is [#717]-unreliable, so narrowing at render
   time would launder stored bits, which §C2.1 forbids. Scalar exits
   render at own width now; own-width tensor digits arrive when [#729]
   repairs the metadata. Recorded normatively at spec/05 §8.1.)
3. **The number grammar is Rust `{:?}` (`Debug`) float formatting,
   normatively**: shortest round-trip digits, `inf`/`-inf`/`NaN`
   spellings, lowercase `e` with unpadded exponent (`1e-7`, not `1e-07`),
   `-0.0` preserved and printed with its sign. (`Display` is NOT this
   grammar: it never emits e-notation - `1e20` prints as
   `100000000000000000000` under `Display`; eval's current output is
   already `{:?}`-shaped, e.g. `9.999999980506448e19`.) The generated C
   normalizes to this grammar (§C3.2) - platform printf variance (`nan`,
   two-digit exponents) must not leak. **Pinned at Phase 1**: the
   decimal/e-notation thresholds are the normative constants
   `DECIMAL_LOWER_BOUND = 1e-4` / `DECIMAL_UPPER_BOUND = 1e16`
   (`chelis-types::observation`, rustc-locked by unit tests), decided on
   the RENDERED magnitude - the value the chosen shortest digits denote
   - which is rustc's actual `{:?}` behavior at straddling-ulp
   boundaries (PR #792 red-team F2); the full ratified grammar,
   including the f16/bf16 shortest-at-width rule and its tie-break, is
   spec/05 §8.1.
4. **bool prints `true`/`false`** at every exit, including inside tensor
   `data=[...]`.
5. **Containers**: tensor rendering stays `tensor(shape=[..],
   data=[..])`; `to_list` stays `[..]`; both use `format_element` for
   every element, so the exits can no longer disagree ([#726]'s split).
   Truncation (decided 2026-07-17): every exit in both lanes truncates
   tensor element rendering at 32 with the marker `, ...` - one rule,
   no exceptions (open question 4 has the rationale and the migration
   note). Full-element fidelity is `to_list`'s and the wire's job.
   **Scalar roots and rank-0 tensors (decided 2026-07-20, [#775]'s
   acceptance; ratified as [05-OBS-4])**: a scalar-typed value renders
   as the BARE scalar at every exit in both lanes, including as a
   top-level labeled root (`root = 0.1`); a rank-0 tensor renders as
   its single element, bare - `tensor(shape=[], data=[..])` is not an
   exit form. Rationale: `print` of a scalar already rendered bare in
   both lanes and the compiled lane's labeled roots did too, so the
   bare form is the only choice consistent with §C2.2's intra-lane
   exit agreement; eval's rank-0 realization of scalar bindings is an
   interpreter storage artifact and must not leak.
6. **Diagnostics that embed values** (traps, mismatch messages) use
   `format_element` for the embedded value - a diagnostic must not
   launder what it reports.

## C2. The agreement contract

For every dtype and every storable value:

1. `parse(format_element(prim, bits))` == `bits` exactly, at the dtype's
   width (**the round-trip invariant** - the mechanical detector, §C4.1).
2. `print`, `to_list`, and diagnostics agree with each other and with the
   stored bits, within one lane.
3. eval and compiled C emit **byte-identical** text for identical stored
   bits. (Identical *values* across lanes is [#729]/[#687] business; this
   contract is conditional on the bits.)
4. The wire schema renders bits faithfully within its representational
   capacity; capacity limits are [#729]'s storage decision and are recorded
   there (§I1). Concretely today: JSON numbers carry f64 exactly; exact
   int64 above 2^53 waits for the [#729] schema change.

## C3. The single-source architecture

1. **One Rust implementation.** `format_element` lives beside the dtype
   definitions ([#729] open question 4 is DECIDED 2026-07-17: a
   `chelis-types` module - the pre-[#729] fallback is the final answer,
   so this placement is permanent, not provisional). Exhaustive over `Prim` - no
   `_` arm (`loud_unsupported.md` §C4's lint patrols it and its
   generator).
2. **The C side is generated, not written.** A Rust function emits the
   print helper's per-dtype C code by matching `Prim` exhaustively;
   `host_emit.rs`'s hand-written switch (the `(double)` funnel at
   `:495-512`) is deleted. The generator emits a
   `default: abort-with-dtype-id` arm so an unknown dtype id at runtime
   is loud (per `loud_unsupported.md` §C1), and per-dtype bodies:
   integer `printf` at width; float paths through the runtime's
   shortest-round-trip routine (next item); f16/bf16 decoded via the
   same conversion helpers the WS-1 kernels already use, then formatted
   at THEIR width.
3. **The compiled lane's formatting routine** (§C3.3; rewritten at
   Phase 2 landing to state the SHIPPED architecture - the B1 row's
   frozen requirement is grammar identity to Rust `{:?}`, and this text
   previously described a pre-implementation sketch): the runtime gains
   `chelis_format_shortest(double v, int width_kind, char* buf)`,
   exported from the Rust runtime staticlib every compiled binary links.
   `v` is the exact double image of the stored float (every supported
   width widens losslessly), `width_kind` is the value's `RuntimeDType`
   id, and an unknown or non-float id aborts loudly with the raw id. The
   wide widths (f32/f64) format via `{:?}` itself - the normative
   grammar's own definition and the exact code path `format_element`
   takes, so grammar identity holds by construction and no platform
   printf variance exists to normalize. The half widths (f16/bf16, which
   Rust cannot format natively) use the ratified §8.1 escalation: digit
   counts 1..=5, the correctly rounded scientific form plus its two
   decimal-grid neighbors at each count, first round-tripping count
   wins, same-length ties break to the numerically closest then the
   even mantissa. Byte equality against `format_element` is locked by
   test - exhaustively over all 65536 bit patterns per half format,
   table- and sweep-driven for f32/f64. (The originally sketched
   `%.{p}g`/`strtod` escalation-plus-normalization existed to
   approximate exactly this from C; implemented faithfully it can still
   diverge from `{:?}` at same-length ties - the executed example is
   f32 1916442.25, where `{:?}` renders `1916442.3` and the loop's
   nearest-candidate rule renders `1916442.2` - so the loop would
   VIOLATE the frozen grammar and is not an admissible implementation.)
4. **`to_list` completes its per-dtype reads**: the runtime's `to_list`
   gains F16/BF16 arms (reading the 2-byte buffers via the existing
   conversion helpers) instead of the current `runtime_fail!`, and its
   int64 path stays exact (it already is - the audit's proof instrument).
   NOTE the boundary: to_list VALUES leaving as list elements is an exit
   (ours); constructing tensors is ingress ([#729]'s).
5. **The DECODE half of an exit is in scope** (added 2026-07-28, after
   PR #863's review surfaced a live violation this architecture did not
   cover). "Print exactly what is stored" is two steps -
   `bits <- read(bytes, dtype)` then `text <- format(bits)` - and §C1-§C3
   above govern only the second. `format_element(prim, ElementRef)` and
   `chelis_format_shortest(value, width_kind, buf)` both receive an
   ALREADY-DECODED element, so a canonical formatter cannot detect a
   wrong-width read: the element arrives correct-looking and is rendered
   faithfully. That is exactly how the compiled lane's nested-value
   renderer came to decode NATIVE two's-complement int32 through an f32
   view - rendering `5i32` as `7.006492321624085e-45` and `-2147483648`
   as a plausible `0` - while `to_list` and the generated print helper,
   reading the same buffer natively, returned the right integers.
   Phase 2 made formatting canonical and left decoding as hand-written
   arms: **one formatter, N decoders.**
   The rule: an exit's decode must go through the dtype's own typed
   accessor, never a view chosen at the call site. **Width is not
   representation** - `Ieee754Binary32` and `TwosComplement32` are both
   four bytes and are not interchangeable, so a width-keyed check is
   blind to precisely this defect ([#894] makes `Repr` the ABI primitive
   with width DERIVED from it). Ownership: the general mechanism is
   [#893] (seal `chelis_tensor.data`, then type the forced accessor - in
   that order), not this plan and not [#729], whose §C3 storage decision
   is the same discipline one layer up. This plan owns the rule AT ITS
   EXITS and enforces it with the Phase 2 oracle's declared decode
   table, which fails on any arm whose pointer view drifts.

## C4. The tolerance table and the oracle handshake

1. **The round-trip invariant** (§C2.1) lands as a property harness over
   the probe corpus and the matrix tests' outputs - red today on [#723] and
   [#716]'s cells, green after Phase 2, and permanent thereafter.
2. **The per-op value-tolerance table** is a MACHINE artifact first
   (2026-07 review integration): its authoritative form is `const` Rust
   beside the capability table (the same no-second-parser resolution as
   that doc's open question 1), consumed directly by the [#687] oracle
   and by [#754]'s shell-facing gate; the `spec/05-risc-primitives.md`
   §8 rendering is generated from or tripwire-checked against it, so
   prose and data cannot drift. Content: for each transcendental, the
   documented cross-lane bound (default: 1 ulp at the computed width;
   `sqrt`: 0 - correctly rounded per IEEE; [#719] is FIXED (PR #760,
   merged 2026-07-17), so the row may be authored when Phase 3 arrives;
   add/sub/mul/div/comparisons: 0). The [#687] oracle
   consults ONLY this table when values differ; formatting differences
   are never tolerated (they are bugs here).
3. **The oracle handshake**: with §C1-§C3 landed, [#687]'s exact-string
   comparison becomes implementable as: byte-equal or (value-parse +
   table-bounded for the listed ops). `parity.rs`'s silent float
   fallback and `eval_agreement.rs`'s f64 tolerance are replaced by
   exactly that rule - delivered in Phase 3 jointly with [#687]. ONE
   COMPARATOR, TWO SURFACES: [#754]'s shell-invokable cross-lane gate
   is the external consumer of this same rule and MUST wrap the
   identical Rust comparison implementation the internal oracle suite
   uses - a second hand-rolled comparison would recreate the
   per-consumer scatter this plan set exists to kill.

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C1 rules + number grammar | Phase 1 (FROZEN 2026-07-20; ratified as spec/05 §8/§8.1) | this doc + dtype_semantics.md §C4 + the migration corpus, one change set |
| §C2 agreement contract | Phase 1 (intra-lane; FROZEN for eval 2026-07-20), Phase 2 (cross-lane byte equality) | same protocol |
| §C3.3 C formatting routine behavior | Phase 2 | this doc; must stay grammar-identical to Rust `{:?}` (§C1.3 - an earlier revision of this row said `Display`, which §C1.3 explicitly rules out) |
| §C4.2 tolerance table | Phase 3 | spec/05 edit + [#687] corpus, one change set |

## B2. Invariants that hold across every boundary

1. **The one-time migration carve-out.** The sibling plans' "controls
   never move" rule has exactly one sanctioned exception, here: printed
   STRING expectations across the existing suite move ONCE, in the
   Phase 1/2 adoption PRs, mechanically, with the round-trip invariant
   proving that only the rendering changed (same bits, new text). The
   migration is a dedicated change set per lane - never mixed with a
   value-semantics change, so a diff in a migration PR that alters a
   parsed VALUE is by definition a bug.
2. **Bits before text.** Any test updated by the migration asserts (or
   is accompanied by) the value at the bit level where exactness
   matters, so future formatting work can never again mask a value
   change (the [#711] bit-pattern lesson, generalized).
3. **Red-to-green only by un-ignoring** for the class's `#[ignore]`d
   tests ([#716]'s print row, [#723]'s row, the bool-exit row).
4. **No third formatter.** Any new exit added to either lane must route
   through `format_element` / the generated helper; a hand-rolled
   `printf`/`format!` of a tensor element in the numeric crates is a
   review-blocking finding (and the tripwire greps for the old
   `%.16g`/`%.1f` pair).
5. **Discoveries fork** (shared rule): new unfaithful exits found
   mid-phase are filed, added to the census in the tracking issue, and
   scheduled - not silently absorbed.
6. **No untyped decode at an exit** (§C3.5's operational form). An exit
   reads its bytes through the dtype's typed accessor; a raw cast that
   picks a pointer view at the call site is the same review-blocking
   finding as a third formatter, and for the same reason - it makes the
   rendered text a faithful report of the wrong bits. A dtype whose
   storage genuinely has no typed accessor (the halves) decodes its bits
   explicitly, with the reason stated at the arm. Any exception is
   DECLARED with the issue that retires it, in the Phase 2 oracle's
   decode table - bool's f32 encoding is the only one today, and it
   retires with [#894].
7. **A public exit owes exit coverage, or does not exist.** A
   `#[no_mangle]` render entry point with no emitter is not harmless
   dead code: it is an exit the census never has to account for, so a
   defect in it is invisible to a harness that drives only reachable
   programs. `chelis_print_f32` was exactly that, and carried the §C3.5
   misdecode for as long as it existed. Removed at Phase 2; the oracle
   fails if it returns in either the Rust source or the published header.

## B3. How to pick up a phase

1. Read Part I, your phase, the previous phase's frozen-at-exit list,
   and §I1 if your work touches storage or ingress (it decides whether
   the work is yours or [#729]'s).
2. Run the round-trip harness and your oracle suite first; the red set
   is the work-list.
3. The probe corpus (`docs/investigations/probes/`) holds the byte-decode
   evidence for [#716]/[#723] if you need to re-derive what "faithful" must
   produce for those cells.
4. Gate with `scripts/gate.py --local`; macOS Smoke is the workspace
   oracle.

---

# Part III - the phases

## Phase 0 - the round-trip harness and the exit census (small, land-first)

**You inherit:** the audit's evidence (byte-decodes for [#716]/[#723], the
formatting-divergence rows) and the matrix test files.

**You deliver:**

1. **The round-trip harness**: for each dtype, a table of boundary bits
   (max/min, first-unrepresentable neighbors, subnormals, -0.0, inf,
   NaN, and the audit's specific values), driven through every exit in
   both lanes, asserting §C2.1 - `#[ignore]`d red on the known cells,
   green elsewhere.
2. **The exit census**: every code path in either lane that turns a
   numeric payload into text (the C helper, eval's printers, `to_list`
   both lanes, diagnostics, wire rendering), committed as a table in the
   tracking issue with its current per-dtype behavior verified by
   execution.
3. The `%.16g`/`%.1f` token tripwire row added to `loud_unsupported.md`
   Phase 0's tripwire (one grep pattern; coordinate, do not duplicate).

**Frozen at your exit:** the harness's value table (append-only) and the
census baseline.

**Explicitly not yours:** any production change.

**Oracle:** the harness red on exactly [#716]/[#723]/bool-exit cells, green
on the rest; every census row execution-verified.

## Phase 1 - the formatter and eval adoption (+ the eval-side migration)

**You inherit:** the harness (your instrument) and the census (your
work-list's eval half).

**You deliver:**

1. `format_element` per §C1, exhaustive over `Prim`, with unit tests per
   rule and per dtype (positive and negative: the grammar tests include
   the exponent/specials edge set).
2. **Eval adoption**: every eval-lane exit in the census routes through
   it (print, to_list rendering, diagnostics embedding).
3. **The eval-side migration** (§B2.1): the coordinated
   expectations update across the suite for eval-printed strings
   (integers lose `.0`, bool tensors print true/false, float digits move
   to own-width shortest round-trip). Bit-level companions added per
   §B2.2 where exactness matters.

**Frozen at your exit:** §C1 (all rules, the grammar) and §C2's
intra-lane agreement for eval. Eval is now the REFERENCE RENDERER: the
generated C in Phase 2 is validated byte-against eval's output.

**Explicitly not yours:** anything C-side; the tolerance table.

**Oracle:** the round-trip harness green for every eval exit; the
migrated suite green; `chelis lint` and the tripwire green (no residual
eval-side hand formatting).

## Phase 2 - the generated C side (+ the C-side migration)

**You inherit:** the frozen contract, eval as reference renderer, and
the census's C half.

**You deliver:**

1. `chelis_format_shortest` in the runtime (§C3.3) with its grammar
   normalization, unit-tested against the Rust formatter over the
   harness's value table (byte equality is the test).
2. **The generated print helper** (§C3.2) replacing `host_emit.rs`'s
   hand-written switch; per-dtype integer printf at width; f16/bf16
   decode-then-format; the loud default arm.
3. **`to_list` per-dtype completion** (§C3.4) in the compiled lane.
4. **The C-side migration** of compiled-lane string expectations,
   validated by byte-diffing against eval on the shared corpus (§C2.3
   comes true here for agreeing bits).

**Frozen at your exit:** §C2.3 cross-lane byte equality (conditional on
bits); the generated-helper architecture (no hand-written dtype switch
may return).

**Explicitly not yours:** making the LANES' bits agree where they differ
today - that divergence is [#729]'s subject matter and stays visible
(faithfully!) in the [#687] corpus until fixed.

**Oracle:** one command -
`.venv/bin/python scripts/faithful_observation_phase2_oracle.py`,
accepted at exit 0 with the final line `PHASE 2 ORACLE: PASS`. It is the
executable form of what this phase used to state as prose, plus the leg
that prose could not carry. Its obligations:
`c_int64_tensor_print_is_exact_above_2p53` ([#723]) and
`c_print_of_f16_tensor_prints_f16_values` / `c_to_list_of_f16_tensor_works`
([#716]) present, un-ignored, and green; the
`c_dag_kernels_compute_correct_f16_bits_despite_print` byte-decode lock
retired per its own instructions (replaced by the direct print row); the
round-trip harness green on every exit in both lanes **except the
enumerated known-red cells**, which are NOT skipped silently - the
oracle holds a ledger of every `#[ignore]`d harness cell with its owning
issue, requires the harness's ignore inventory to EQUAL that ledger (an
undeclared skip is a narrowed corpus; a stale row overstates what the
suite covers), runs each
cell, and fails if one is red for an undeclared reason **or has gone
green**. A green known-red cell means its upstream [#729]-family repair
landed: un-ignore the cell on its original assertion and delete the
ledger row in that change set. The oracle also pins the harness's two
documented corpus-exclusion lists ([#751] C ingress, [#717] eval
`to_list`) against silent widening, runs the `chelis_format_shortest`
byte locks, and requires the §B2.4 format-narrowing tripwire's
PRODUCTION allowlist to stay empty.

Scope, stated rather than assumed: the ignore-inventory equality covers
the observation harness, this plan's own instrument. The sibling matrix
files carry `#[ignore]`d cells owned by [#682]/[#714]/[#717]/[#724]/[#729];
for those the oracle asserts only that the three rows named above are
un-ignored and green.

**Default CI does NOT run this oracle** (same standing as [#730]'s Phase
2 oracle): it is a manual phase gate, invoked at phase acceptance and at
any change to the observation surface, and it needs a host C toolchain
because most obligations build, link, and run generated C. What CI does
carry continuously is the harness's green set, the matrix oracle rows,
the `chelis_format_shortest` byte locks, and the §B2.4 tripwire - every
suite the oracle runs, minus the known-red re-execution and the
structural ledger scan, which are exactly the legs that need the ledger
to mean anything. Run it before claiming this phase, not once per PR.

**Delivered** (2026-07-24), with five recorded notes (note 5 added
2026-07-28, with the oracle it describes). (1) The
`chelis_format_shortest` routine lives in the Rust runtime library, so
the wide widths use `{:?}` formatting directly - the normative grammar's
own definition and the exact code path `format_element` takes - while
f16/bf16 use the ratified escalation search; the C-side sketch's
printf/strtod loop and normalization pass exist to approximate exactly
this from C, and byte equality is still locked by test (exhaustive per
half format). (2) `to_list` completion required naming a boxed-only
list-element ABI state (`ReducedFloatBoxed`) in the C backend: PR #799's
typed boundary had begun rejecting `list[f16]` wholesale at build (the
census recorded the older runtime abort); the named state keeps every
scalar-materialization path loudly rejected per [#714] while letting the
heap list print. The state is recorded in `loud_unsupported.md` §C6.3
(the boundary's owning doc; their interlock-edits-are-bidirectional
rule) and its Phase 2 oracle gained an added-variant `HostAbiType`
mutation leg, so a new ABI variant is a compile-error work-list at
every exhaustive consumer. (3) §C2.3's byte-identity lock runs where stored bits
AND rendered widths agree; the eval tensor width note (spec/05 §8.1)
keeps non-dyadic narrow-float tensor cells width-divergent until [#729],
and the [#862] unit-root labeling discovery is filed, not absorbed.
(4) PR #863's fresh-context red team (round 1) surfaced two further
width-annex gaps, filed per §B2.5 and annexed at spec/05 §8 with
ignored red cells: [#864] (eval's labeled-root render of
`cast(<tensor>, f64)` results narrows through the stale F32 tag - an
eval-lane [05-OBS-1] violation the harness's `via_cast=false` F64 table
structurally never constructed) and [#865] (the compiled lane's
untagged f64 value box renders f32 - not only f16/bf16 - elements at
f64-image width through `to_list`/boxing; faithful but not own-width
shortest). Both are [#729]-family value/capacity repairs; rendering is
not the fix site for either.
(5) The phase's oracle became a script rather than a prose conjunction,
after PR #863's exact-head red team (F3) observed that the default
harness run reported "30 passed, 3 skipped" while nothing asserted what
the three skips were, that they still failed for their stated reasons,
or that none had gone green. Three annexed cells is a defensible
boundary; three cells nobody re-executes is not, and the difference is
not visible from a green suite. The ledger makes the boundary
executable, and its unexpectedly-green leg turns each cell into
[#729]'s exit-criteria instrument: the day the tag repair or the box
width lands, this oracle fails until the cell is un-ignored. The
accompanying status texts here and at spec/05 §8 were narrowed in the
same change set to say "conformant except the enumerated annexed cells"
rather than leading with an unqualified conformance claim.

## Phase 3 - the tolerance table and the [#687] handshake

**You inherit:** two lanes that render identically; value divergences now
visible as exactly themselves.

**You deliver:**

1. The per-op tolerance table in `spec/05-risc-primitives.md` (§C4.2),
   authored from the audit's measurements (the vvsqrtf/SLEEF rows). The
   fix-precedes-the-row gate on `sqrt = 0` is SATISFIED: [#719] was
   fixed by PR #760 (merged 2026-07-17; contiguous f32 sqrt now takes
   the correctly-rounded scalar path, layout-independent, and the
   scalar loop measured ~1.6x FASTER than vvsqrtf). The row may be
   authored; the table is never authored with a known-false row.
2. Jointly with [#687]: `parity.rs` and `eval_agreement.rs` replaced by /
   rebuilt on the byte-equal-or-table-bounded rule (§C4.3). (The silent
   float-parse fallback itself is deleted earlier, by [#729] Phase 0 -
   this phase replaces the comparison rule it left behind; see the
   corpus-diet note in that deliverable.)
3. The rejected-cells corpus (from `loud_unsupported.md` Phase 0) and
   the value corpus unified under the same comparison rule so
   diagnostics and values are oracle-checked identically.

**Frozen at your exit:** the table (B1); the oracle rule.

**Explicitly not yours:** closing [#687]'s remaining scope if any lanes
still disagree on VALUES - those are [#729]-tracked cells, now perfectly
visible.

**Oracle:** the [#687] oracle suite running in CI on the full corpus:
byte-exact everywhere except table-listed ops within bounds; any
remaining value divergence appears as a named, issue-linked ignore -
never as tolerance.

---

# Part IV - bookkeeping

## I1. The interlock with [#729] (dtype semantics)

- **Ownership**: this plan owns EXITS (rendering); [#729] owns VALUES and
  CAPACITY (what is stored, in what buffer, across which wire type).
  `dtype_semantics.md` §C4 and this §C1 are the same rules by
  construction; edits go to both in one change set.
- **Landing order - this plan first (expected)**: [#729] Phases 1-3 then
  inherit the formatter and validate against it; their "eval is the
  reference lane" claim strengthens to "reference bits AND reference
  bytes". [#723]/[#716] are fixed without waiting.
- **Landing order - [#729] first**: its Phase 1 delivers `format_element`'s
  Rust side per its §C4 and THIS doc's Phase 1 collapses into an
  adoption/migration pass; its Phase 3 delivers §C3.2's generation and
  this doc's Phase 2 collapses likewise. Either way the contracts here
  govern the result; the tracking issues cross-check the boxes.
- **Capacity limits**: exact int64 across the wire and the Python
  boundary wait for [#729]'s storage decision ([#686]/[#685]); until then the
  wire renders faithfully within f64 capacity and the limitation is
  documented at the schema, not papered over in rendering.
- **The dtype-carrying payload** (raised as F1/F2 by PR #863's
  exact-head review, which asked why Phase 2 leaves dtype-unfaithful
  states *representable* rather than merely unreached): that ask is
  [#729]'s, and it is the same ownership line this section already
  draws - a closed `{dtype, bits}` (or per-width) scalar payload
  carried through the runtime's `chelis_value` box, eval's tensor
  store, the containers, the roots, and the wire schema is a STORAGE
  and CAPACITY change, not a rendering one. Three separations remain
  open on that side and each already has its named cell: the runtime
  box's single `CHELIS_VALUE_FLOAT64` tag and `f64_` slot ([#865]),
  eval's separable `Vec<f64>` plus `precision: Prim` fields ([#864],
  [#717]), and the wire's `Vec<f64>` tensor data plus lone `Float64`
  scalar variant ([#686]).
  **Phase 2's FORMATTING is forward-compatible with that payload by
  construction; its DECODING was not, and that distinction matters**
  (this bullet was corrected 2026-07-28 - an earlier revision claimed
  compatibility for the renderers as a whole, which overstated it).
  On the format side the claim holds: `format_element(prim, ElementRef)`
  and `chelis_format_shortest(value, width_kind, buf)` are both already
  keyed by dtype, so the payload's arrival replaces two arguments with
  one at the CALL sites and changes no rule in §C1, no byte of the
  grammar, and no rendered output.
  But both take an ALREADY-DECODED element, so neither says anything
  about whether the bytes were read at the right representation - and
  the int32 misdecode PR #863's review found lived entirely upstream of
  them (§C3.5). Two consequences worth stating plainly: the parameter
  spelled `width_kind` carries a `RuntimeDType` id and is therefore
  correct today, but its NAME encodes the width-thinking [#894]
  disproves and should not be read as license; and `chelis_format_shortest`'s
  `(double, int)` pair remains a seam the payload closes - the pair is
  caller-supplied, and the routine can reject an invalid dtype id but
  cannot prove the value is the exact widening of one stored at that
  width. Hardening it (a tagged struct or per-width entry points, plus
  `(buf, capacity)` and an explicit result) is the natural joint moment
  with [#729]'s payload work and [#893]'s seal, recorded here rather
  than absorbed - this plan does not own the storage side of it.
- **With [#730]**: the `<value>` placeholder and the print helper's abort
  default are its census rows; the shared tripwire carries this plan's
  `%.16g`/`%.1f` pattern. No delivery overlap.

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | detection; the round-trip invariant exists |
| 1 | eval-side exit splits (bool `1.0`-vs-`true`); the reference renderer exists |
| 2 | [#716], [#723]; cross-lane byte equality on agreeing bits; the hand-written-switch class |
| 3 | [#687] unblocked and largely closed; [#719] gets its normative row |

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | exact number grammar edge set | DECIDED 2026-07-17 (mechanics; exact constants pinned by P1's tests): `{:?}`'s e-notation thresholds are captured empirically and recorded as NORMATIVE CONSTANTS in §C1.3, so a rustc formatting change breaks our tests loudly instead of silently shifting the grammar the generated C must match. f16/bf16 shortest-digit = the shortest string whose parse-back (strtod to f64, then round to the half width - safe by the same excess-precision argument as [04-NUM-1]'s single-rounding rule) yields the stored bits, verified EXHAUSTIVELY over all 65536 bit patterns per format (a required P1 deliverable - the narrow widths are fully enumerable, so no boundary-case debate survives). DELIVERED at P1 (2026-07-20): constants pinned and rustc-locked, both exhaustive half-format tests landed in `chelis-types::observation` | §C1.3 + spec/05 §8.1 + the formatter's unit tests |
| 2 | precision-escalation loop vs vendored Ryū for the C routine | DECIDED at Phase 2: neither - the routine lives in the Rust runtime staticlib, so f32/f64 use `{:?}` directly (grammar identity by construction) and f16/bf16 use the ratified §8.1 escalation; a faithful printf-loop emulation diverges from `{:?}` at same-length ties (f32 1916442.25 -> loop `1916442.2` vs `{:?}` `1916442.3`), so the loop was never grammar-admissible for the wide widths | §C3.3 |
| 3 | whether the wire schema renders numbers as JSON numbers or strings for int64 once [#729]'s storage lands | with [#729] Phase 1 | schema.rs + both docs' §I1 |
| 4 | truncation story | DECIDED 2026-07-17: ONE rule at every exit in both lanes - truncate tensor element rendering at 32 with the marker `, ...` (C's existing form). P0's census proved "keep as-is" was incoherent (three stories: eval transcript unlimited, eval root `+ ...`@32, C `, ...`@32). Eval-transcript's unlimited printing is REMOVED in the migration (the one place §B2.1's carve-out changes how MUCH is printed, flagged with §B2.2 bit-level companions); print-based cross-lane comparison beyond 32 never worked (the C lane already capped), and full-element fidelity is `to_list`'s and the wire's job, never print's. The threshold is one documented constant; configurability deferred until a real need | §C1.5 |

## The one-sentence summary for a reviewer

One formatter, written once in Rust, generated into C, exhaustive over
dtypes, with a parse-back-identity invariant standing over every exit in
both lanes - so a stored value can no longer be misreported, the two lanes
become byte-comparable, and every remaining numeric disagreement is
guaranteed to be a real value bug wearing its own name.

[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#685]: https://github.com/Chelis-Lang/chelis/issues/685
[#686]: https://github.com/Chelis-Lang/chelis/issues/686
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#711]: https://github.com/Chelis-Lang/chelis/issues/711
[#714]: https://github.com/Chelis-Lang/chelis/issues/714
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#719]: https://github.com/Chelis-Lang/chelis/issues/719
[#723]: https://github.com/Chelis-Lang/chelis/issues/723
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#748]: https://github.com/Chelis-Lang/chelis/issues/748
[#749]: https://github.com/Chelis-Lang/chelis/issues/749
[#751]: https://github.com/Chelis-Lang/chelis/issues/751
[#775]: https://github.com/Chelis-Lang/chelis/issues/775
[#754]: https://github.com/Chelis-Lang/chelis/issues/754
[#865]: https://github.com/Chelis-Lang/chelis/issues/865
[#864]: https://github.com/Chelis-Lang/chelis/issues/864
[#862]: https://github.com/Chelis-Lang/chelis/issues/862
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#894]: https://github.com/Chelis-Lang/chelis/issues/894
