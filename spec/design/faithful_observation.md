# Faithful Observation: one dtype-true formatter for every exit, both lanes

**Status:** Design proposal, pre-implementation. Tracking issue: [#732].
**Owning specs:** `spec/05-risc-primitives.md` (its §8 carries this
plan's decided contract as provisional atoms [05-OBS-1..3], seeded ahead
of Phase 1; the per-op tolerance table lands into the same section at
Phase 3), `spec/04-type-system.md` (dtype value-set definitions, shared
with `spec/design/dtype_semantics.md` §C1), and the audit record in
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
pub fn format_element(prim: Prim, value: ElementRef<'_>) -> String;
```

Per-dtype rules:

1. **Integers print as integers.** `data=[750]`, never `750.0`. int64
   prints all digits exactly (i64 formatting, never through double).
2. **Floats print shortest-round-trip AT THEIR OWN WIDTH**: the shortest
   string that parses back to exactly the stored f64 / f32 / f16 / bf16
   value. This single rule replaces C's `%.1f`/`%.16g` split and eval's
   f64-width formatting, and is what makes byte-equal lane comparison
   possible.
3. **The number grammar is Rust `{:?}` (`Debug`) float formatting,
   normatively**: shortest round-trip digits, `inf`/`-inf`/`NaN`
   spellings, lowercase `e` with unpadded exponent (`1e-7`, not `1e-07`),
   `-0.0` preserved and printed with its sign. (`Display` is NOT this
   grammar: it never emits e-notation - `1e20` prints as
   `100000000000000000000` under `Display`; eval's current output is
   already `{:?}`-shaped, e.g. `9.999999980506448e19`.) The generated C
   normalizes to this grammar (§C3.2) - platform printf variance (`nan`,
   two-digit exponents) must not leak.
4. **bool prints `true`/`false`** at every exit, including inside tensor
   `data=[...]`.
5. **Containers**: tensor rendering stays `tensor(shape=[..],
   data=[..])`; `to_list` stays `[..]`; both use `format_element` for
   every element, so the exits can no longer disagree ([#726]'s split).
   Truncation (decided 2026-07-17): every exit in both lanes truncates
   tensor element rendering at 32 with the marker `, ...` - one rule,
   no exceptions (open question 4 has the rationale and the migration
   note). Full-element fidelity is `to_list`'s and the wire's job.
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
3. **Shortest-round-trip in C** without vendoring a big formatter: the
   runtime gains `chelis_format_shortest(double v, int width_kind, char*
   buf)` implemented as the precision-escalation loop - try
   `%.{p}g` for p = 1..17 (f64) / 1..9 (f32) / 1..5 (f16/bf16 via their
   exact double value), `strtod` back, stop at the first exact
   round-trip - then normalize specials and exponent digits to §C1.3's
   grammar. Simple, portable, provably shortest-in-digits; printing is
   not a hot path. (Vendoring Ryū is the recorded alternative if the
   loop's cost ever matters - open question 2.)
4. **`to_list` completes its per-dtype reads**: the runtime's `to_list`
   gains F16/BF16 arms (reading the 2-byte buffers via the existing
   conversion helpers) instead of the current `runtime_fail!`, and its
   int64 path stays exact (it already is - the audit's proof instrument).
   NOTE the boundary: to_list VALUES leaving as list elements is an exit
   (ours); constructing tensors is ingress ([#729]'s).

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
   `sqrt`: 0 - correctly rounded per IEEE, its row landing only after
   [#719]'s fix; add/sub/mul/div/comparisons: 0). The [#687] oracle
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
| §C1 rules + number grammar | Phase 1 | this doc + dtype_semantics.md §C4 + the migration corpus, one change set |
| §C2 agreement contract | Phase 1 (intra-lane), Phase 2 (cross-lane byte equality) | same protocol |
| §C3.3 C formatting routine behavior | Phase 2 | this doc; must stay grammar-identical to Rust `Display` |
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

**Oracle:** `c_int64_tensor_print_is_exact_above_2p53` ([#723]) and
`c_print_of_f16_tensor_prints_f16_values` ([#716]) green and un-ignored;
the round-trip harness green on every exit in both lanes; the
`c_dag_kernels_compute_correct_f16_bits_despite_print` byte-decode lock
retired per its own instructions (replaced by the direct print row).

## Phase 3 - the tolerance table and the [#687] handshake

**You inherit:** two lanes that render identically; value divergences now
visible as exactly themselves.

**You deliver:**

1. The per-op tolerance table in `spec/05-risc-primitives.md` (§C4.2),
   authored from the audit's measurements (the vvsqrtf/SLEEF rows). The
   `sqrt = 0` row lands only AFTER [#719]'s fix is merged: the fix
   precedes the row ([#719]'s owner line and the roadmap ledger agree);
   the table is never authored with a known-false row.
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
| 1 | exact number grammar edge set | DECIDED 2026-07-17 (mechanics; exact constants pinned by P1's tests): `{:?}`'s e-notation thresholds are captured empirically and recorded as NORMATIVE CONSTANTS in §C1.3, so a rustc formatting change breaks our tests loudly instead of silently shifting the grammar the generated C must match. f16/bf16 shortest-digit = the shortest string whose parse-back (strtod to f64, then round to the half width - safe by the same excess-precision argument as [04-NUM-1]'s single-rounding rule) yields the stored bits, verified EXHAUSTIVELY over all 65536 bit patterns per format (a required P1 deliverable - the narrow widths are fully enumerable, so no boundary-case debate survives) | §C1.3 + the formatter's unit tests |
| 2 | precision-escalation loop vs vendored Ryū for the C routine | Phase 2 (loop is the default; revisit only on measured cost) | §C3.3 |
| 3 | whether the wire schema renders numbers as JSON numbers or strings for int64 once [#729]'s storage lands | with [#729] Phase 1 | schema.rs + both docs' §I1 |
| 4 | truncation story | DECIDED 2026-07-17: ONE rule at every exit in both lanes - truncate tensor element rendering at 32 with the marker `, ...` (C's existing form). P0's census proved "keep as-is" was incoherent (three stories: eval transcript unlimited, eval root `+ ...`@32, C `, ...`@32). Eval-transcript's unlimited printing is REMOVED in the migration (the one place §B2.1's carve-out changes how MUCH is printed, flagged with §B2.2 bit-level companions); print-based cross-lane comparison beyond 32 never worked (the C lane already capped), and full-element fidelity is `to_list`'s and the wire's job, never print's. The threshold is one documented constant; configurability deferred until a real need | §C1.5 |

## The one-sentence summary for a reviewer

One formatter, written once in Rust, generated into C, exhaustive over
dtypes, with a parse-back-identity invariant standing over every exit in
both lanes - so a stored value can no longer be misreported, the two lanes
become byte-comparable, and every remaining numeric disagreement is
guaranteed to be a real value bug wearing its own name.

[#680]: https://github.com/Chelis-Lang/chelis/issues/680
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
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#754]: https://github.com/Chelis-Lang/chelis/issues/754
