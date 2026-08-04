# Faithful Observation: one dtype-true formatter for every exit, both lanes

**Status:** Phases 0-2 LANDED; Phase 3 is DELIVERED in this revision.
Phase 0 (the round-trip harness and the
exit census) landed 2026-07-17 (PR #752, tightened by PR #774). Phase 1
(the formatter, eval adoption, and the eval-side §B2.1 migration) landed
2026-07-20: `format_element` lives at `chelis-types::observation`, every
eval exit routes through it, §C1 and the number grammar are FROZEN (B1),
and the contract is ratified as `spec/05-risc-primitives.md` §8 atoms
[05-OBS-1..6] (the chelis#775 scalar-root decision is [05-OBS-4]). Phase 1
carried one annexed value-layer exception, surfaced by PR #792's red team:
int64 scalar roots above 2^53 rendered the f64-collapsed stored value at
the labeled root ([#684]'s rank-0 realization). [#729]'s per-dtype storage
repaired it. The cell is un-ignored on its original assertion and its
known-red row is gone ([#1078]); spec/05 §8 no longer records the
exception, which the [#1043] timeless rewrite removed along with the other
dated status prose.
Phase 2 (the generated C side and the C-side §B2.1 migration) landed
2026-07-24: `chelis_format_shortest` in the runtime, the print helper
generated from an exhaustive `Prim` match, `to_list`'s F16/BF16 arms,
and §C2.3 cross-lane byte equality locked for identical stored bits -
[#716]/[#723]/[#726]-C/[#748]/[#749] fixed by un-ignoring their red
cells; all five issues were closed 2026-08-04 against that shipped
behavior rather than on the PR #863 merge itself. [#729] Phase 3 retires the
remaining annexed value boundaries: non-f64 boxed scalars preserve their
stored dtype through the tagged rank-0 tensor carrier ([#865]), suffixed
literal leaves finalize at their checked width before the C lane widens them
([#1110]), and exact-bit literal emission returns [#751]'s four former C
ingress exclusions to the always-run corpus. Exact f16/bf16 host scalars also
return `to_string` to own-width parity ([#734]) through private generated C
helpers rather than a new public numeric ABI. The known-red and declared-
exclusion ledgers are therefore empty. The former single-root
label discrepancy [#862] is now authored by [05-OBS-6] and its prefix
has landed; complete manifest-backed root availability and artifact
acceptance remain under [#912]/[#1023], not this formatter class. The
cast-constructed f64 root cell [#864], surfaced by PR #863, is repaired
and un-ignored after a post-Phase-3 re-diagnosis: the root tag was already
F64, but the static `to_tensor` DAG shortcut widened lexical decimals
without first materializing each typed leaf and cast chain at its declared
width. The shipped repair carries NO precision-tagged leaf: static literal
extraction keeps [#856]'s exact `RawScalar` and finalizes a FLOAT leaf at
its own declared width in the `lit` arm, so an explicitly f32 scalar widened
into an f64 tensor preserves the same stored value as the host path. The
`{value: f64, Prim}` leaf carrier that an earlier draft proposed is NOT what
landed, and deliberately so: an f64 value field would have reintroduced the
above-2^53 integer loss that [#856]'s exact i64 lane exists to prevent.
Integer leaves therefore stay on the exact lane and only float leaves are
finalized. Phase 3
authors the [05-OBS-3] table, moves both value harnesses and the rejected-cell
corpus onto `chelis_types::agreement`, and supplies one executable acceptance
oracle. Its implementation is complete here; merge/CI acceptance is recorded
on the carrying PR rather than claimed by this source revision. Tracking
issue: [#732].
**Owning specs:** `spec/05-risc-primitives.md` (its §8 carries this
plan's ratified contract as current blockquote authorities [05-OBS-1..6]; the
per-op tolerance table is authored in the same section by Phase 3, while
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
   possible. (Width note, resolved: the Phase 1 interim rendered eval
   TENSOR float elements at the stored f64 width because the pre-[#729]
   tensor store was f64-backed with a [#717]-unreliable tag; [#729]
   Phase 1's per-dtype storage removes that state, and own-width tensor
   digits are implemented with it in the current [#729] stack - the deferred
   half of the §B2.1 migration. Recorded normatively at spec/05 §8.1.)
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

### C1.1 Root envelope addendum ([05-OBS-6])

Each emitted root is rendered as `name = value` at every exit in both lanes;
when an exit emits multiple roots, their order is manifest entry order. The
`value` half obeys §C1 unchanged; the prefix is an envelope around that
payload, not a new numeric formatter or container shape. The root set, its
order, and whether `build` owes a runnable artifact belong to [#912]. A lane
that cannot produce an owed root uses [#730]'s typed failure channel to emit
[05-UNS-1] with the root, lane, and reason.

This addendum records the numbered-spec decision without absorbing the #912
class into #732. In particular, a labelled line does not prove that the
manifest is complete or that production observation consumed it. Those
acceptance obligations remain explicit in [#1023].

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
   `int chelis_format_shortest(double v, int dtype, char* buf,
   size_t cap)`,
   exported from the Rust runtime staticlib every compiled binary links.
   `v` is the exact double image of the stored float (every supported
   width widens losslessly), `dtype` is the value's `RuntimeDType` id
   (its STORAGE width is what the rendering round-trips at - spec/04
   [04-NUM-8] declares storage and arithmetic width separately and they
   differ for f16/bf16, which compute at f32; the parameter names a dtype
   rather than a width because "width" denotes two properties of one), `cap` is the caller's buffer capacity, and the return is the byte
   count written excluding the NUL. **Every contract violation aborts
   loudly** rather than truncating or returning a sentinel: an unknown or
   non-float id aborts with the raw id, a null buffer aborts, and a `cap`
   too small for the rendering plus its NUL aborts naming both numbers. A
   silent short write would be a [#703]-class substitution at the byte
   level - a truncated rendering parses back as a DIFFERENT value, which
   is precisely the unfaithful exit this plan exists to kill, so the
   return value is a length and never an error channel.
   (`cap` was added at PR #863's R2 review, which demonstrated that the
   original `(double, int, char*)` signature could not express - let alone
   check - the buffer contract its own documentation stated, and wrote
   past a four-byte logical buffer. Landing it before the 0.18 tag is
   deliberate: adding a parameter afterwards is a breaking change for
   every shell that links the runtime, and 0.18 is the mechanical cut.)
   The
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
   `chelis_format_shortest(value, dtype, buf, cap)` both receive an
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
   blind to precisely this defect ([#964] defines `Repr` as the ABI
   representation primitive and derives width from it). Ownership: the general
   mechanism is
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
   at `chelis_types::agreement::OP_TOLERANCES`, beside the future
   capability table (the same no-second-parser resolution as that doc's
   open question 1), consumed directly by the [#687] oracle
   and by [#754]'s shell-facing gate; the `spec/05-risc-primitives.md`
   §8 rendering is generated from or tripwire-checked against it, so
   prose and data cannot drift. Content: for each transcendental, the
   documented cross-lane bound (default: 1 ulp at the computed width;
   `sqrt`: 0 - correctly rounded per IEEE; [#719] is FIXED (PR #760,
   merged 2026-07-17), so the row is authored by Phase 3;
   add/sub/mul/div/comparisons: 0 by absence). A row is eligible only
   after both lanes establish [04-NUM-8] arithmetic-width conformance;
   in particular, [#897]'s current eval float path may not use the table
   to launder a mismatch. For f16/bf16, a rendered mismatch also requires
   both pre-final f32 bit patterns and proof that they round to the two
   observed stored values; the finalized strings alone cannot establish
   an f32 ULP distance. The [#687] oracle
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
| §C3.3 C formatting routine behavior AND its C ABI signature | Phase 2 (the signature settled at the R2 review, before the 0.18 tag) | this doc; must stay grammar-identical to Rust `{:?}` (§C1.3 - an earlier revision of this row said `Display`, which §C1.3 explicitly rules out). The signature is frozen for the same reason the render is: after 0.18 ships, changing it breaks every shell that links the runtime |
| §C1.1 root envelope | [05-OBS-6] authored 2026-07-31; full implementation acceptance pending [#1023] | spec/05 [05-OBS-6] + this doc + dtype_semantics.md §C4 + the release roadmap + the root-boundary corpus, one change set |
| §C4 tolerance table (authored into spec/05 §8 per [05-OBS-3]) | Phase 3 | spec/05 edit + [#687] corpus, one change set |

## B2. Invariants that hold across every boundary

1. **The bounded migration carve-outs.** The sibling plans' "controls
   never move" rule has two separately authored string-only migrations.
   First, payload expectations moved once per lane in the Phase 1/2
   adoption PRs, with the round-trip invariant proving that only rendering
   changed. Second, [05-OBS-6]'s root `name = ` prefix shipped once in both
   lanes in v0.18.1; it may not change payload digits or value shape. Each
   migration is isolated from value-semantics work, so a diff that alters a
   parsed VALUE is by definition a bug. That shipped prefix is not proof of
   full [05-OBS-6] conformance: [#1023]'s root-boundary corpus and manifested
   production path still owe the complete root set, order, and unavailable-root
   behavior.
2. **Bits before text.** Any test updated by the migration asserts (or
   is accompanied by) the value at the bit level where exactness
   matters, so future formatting work can never again mask a value
   change (the [#711] bit-pattern lesson, generalized).
3. **Red-to-green only by un-ignoring** for the class's `#[ignore]`d
   tests ([#716]'s print row, [#723]'s row, the bool-exit row).
4. **No third formatter.** Any new exit added to either lane must route
   through `format_element` / the generated helper; a hand-rolled
   `printf`/`format!` of a tensor element in the numeric crates is a
   review-blocking finding. The rule spans BOTH lanes and every Rust
   exit surface, and its instruments (stated per §B2.8) are the three
   hosted classes in `loud_unsupported_tripwire.rs`, each a two-sided
   exact-count ratchet: `c-format-narrowing` (C printf tokens
   `%.16g`/`%.1f`, all crate src; production allowlist empty since
   Phase 2), `rust-format-narrowing` (interpolations whose format spec
   pins a decimal precision - `.N`, argument-supplied `.*`, named
   `.prec$`, in any combination with fill/alignment/sign/width, so
   `{value:8.2}` and `{v:0>8.2}` count alongside `{x:.N}` - or selects
   the exponential trait grammar (`{v:e}`/`{v:E}` render 0.5 as `5e-1`,
   a float grammar the normative form never produces; round-4 F5); all
   crate src, comment lines excluded), and `rust-debug-numeric-format`
   (interpolations whose spec selects the Debug trait, i.e. ends in `?`
   - `{v:?}`, `{x:#?}`, `{value:8?}`, `{v:x?}` - over the declared
   `OBSERVATION_EXIT_SURFACES`: directory prefixes deliberately, so a
   NEW file inside a declared surface is covered from its first line at
   baseline zero; BOTH identifier positions - the capture argument and
   the named dynamic precision `.ident$` - are parsed with Rust's
   Unicode XID rules, so `{值:.2}` and `{v:.精度$}` both count
   (round-4 F1: upgrading only the capture position left the
   dynamic-precision position evadable); creating an exit surface
   anywhere else obliges adding its prefix in the same change set).
   Interlock with the host doc (bidirectional, recorded in its §C5):
   `loud_unsupported.md`'s own §B2.8 requires DERIVED universes for its
   own tripwire classes; a hosted class's scope is instead declared by
   the owning plan's §B2 rule with residue declared per this doc's
   §B2.8, and `OBSERVATION_EXIT_SURFACES` is revisited when [#730]
   Phase 4's §C7.1 derived-universe rewrite lands.
   Both Rust classes scan LOGICAL
   lines - string continuations (`\` at end of line) are joined first,
   so a format spec split across physical lines is the single spec the
   compiler sees (round-2 F3's executed evasion, closed). The
   sanctioned implementations -
   the only sites where the grammar may be spelled directly - are
   `chelis-types/src/observation.rs` (`format_element`) and
   `chelis-runtime/src/format_shortest.rs` (`chelis_format_shortest`);
   their baseline rows say so. Declared residue the token instruments
   cannot see, each with its owner per §B2.8: derived-`Debug`
   containers embedding floats (`{other:?}` on a `#[derive(Debug)]`
   value; today's diagnostic carriers are annotated in the tripwire
   baseline, retired by [#729]'s dtype-carrying payload plus the review
   rule), bare `{}` Display / `.to_string()` of a numeric payload (the
   review rule; too common to token-scan), ALTERNATE INTEGER/POINTER
   grammars (`{v:x}`/`{v:X}`/`{v:o}`/`{v:b}`/`{v:p}` - legitimately
   common for addresses, ids, and bitmasks, so a token scan would drown
   in benign hits; the review rule), an exp selector behind a non-ASCII
   fill or dynamic width (the grammar-only-prefix rule that keeps prose
   brace-groups from counting also skips those; review rule), exits
   born outside the
   declared surfaces (the review rule), and MACRO-COMPOSED format
   strings (`concat!`/`format_args!` indirection assembles a spec that
   never appears whole in source; the scanner joins string
   continuations but does not expand macros - the review rule). The
   Rust lane was added by the 2026-07-30 detector-scope review, after
   the rule's only instrument - C tokens - let PR #891's Rust-side
   `format_f64_json` reach review with no mechanical signal.
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
8. **Detector-scope parity** (added 2026-07-30). A §B2 prohibition
   exists only with a named instrument, and the instrument's covered
   scope is stated at the rule beside the rule's own scope; whatever
   the instrument cannot see is declared residue with a named owner (an
   issue that retires it, or the review rule). An undeclared gap
   between what a rule claims and what its detector covers is itself a
   violation of this item - that gap is how `format_f64_json` shipped
   with no mechanical signal while §B2.4 said "either lane".
   Enforcement, with its own limits stated (PR #962 red-team F5, then
   round-2 M2): the Phase 2 oracle carries `B2_RULE_INSTRUMENTS`, one
   row per item of this section, checked three ways - the doc's §B2
   item list must equal the manifest (a new rule lands only with a
   deliberate instrument decision, which may be the literal review-rule
   entry with a substantive justification), every named instrument must
   exist, and every callable or suite instrument must have produced both
   a RUNTIME invocation receipt and a centrally consumed-result receipt
   by the end of the run (the `@instrument` decorator records entry;
   `consume_findings` refuses any list that is not THE active verdict
   sink and records consumption only after extending it, and the sink's
   own context - not the leg's caller - raises from that same list;
   suites record both on success). The first
   cut checked invocation by scanning the oracle's own source, which
   round-2 M2 refuted with an `if False:` branch - only a receipt
   written by the running code counts. The exact-head review then showed
   that entry alone is insufficient: a manifest detector returned a
   non-empty violation which its caller discarded; round-4 F2 showed a
   scratch list could impersonate the sink, so sink identity is now
   asserted and the raise is helper-owned. Stated as the mechanism's
   limit rather than hidden: a receipt proves detector -> active sink ->
   helper-owned raise; it cannot prove the oracle's own code was not
   edited to tamper with the sink between extend and exit - an oracle
   cannot police modifications to itself, and that residue is the
   review rule. A detector named in
   the manifest but absent from result consumption now fails the final
   check. The oracle
   further requires every tripwire pattern whose `doc()` cites this
   document to carry an oracle coverage row (its per-class permitted
   baseline paths), closing the review's second finding one layer up.
   What the manifest deliberately does NOT prove: that an instrument's
   checks are non-vacuous. That burden sits with the mutation tests in
   `test_faithful_observation_phase2_oracle.py` - one shipped-clean
   test plus one mutation test per failure mode per instrument - and a
   new instrument owes its mutation tests in the same change set.
9. **Three-legged boundaries** (added 2026-07-30). Every artifact that
   DECLARES current breakage - the `#[ignore]` ledger, the
   corpus-exclusion lists, the tripwire baselines - carries three legs,
   or may not exist: (1) inventory equality between the artifact and
   its declared ledger; (2) continuous re-execution of the declared
   behavior; (3) fail-on-unexpected-green, with a wrong-reason
   discriminator wherever the declared failure has a fingerprint. A
   boundary with fewer legs decays silently: the pre-2026-07-30
   exclusion check compared lists for equality only, so a stale list
   overstating breakage stayed green forever. Standing conformance:
   `KNOWN_RED_CELLS` has all three legs from the oracle's authoring
   (ledger equality, per-cell re-runs, the gone-green failure with its
   `fragment` discriminator); the tripwire baseline carries the three
   legs at NET-COUNT granularity - the row list, the every-CI scan, and
   the DECREASE branch as its gone-green leg - which is honestly weaker
   than per-occurrence identity: a same-file edit that removes one
   benign token and adds one violating token preserves the count and
   fires neither branch (PR #962 red-team F4; `loud_unsupported.md`
   §C4.5 has always named count relocation among the inventory's
   evasions, and the review rule owns the swap case). Before [#729] Phase 3,
   the exclusion lists gained legs 2-3 via TWO mechanisms with different
   trust models (PR #962 round-2 M1 forced the split): the harness's
   exclusion probes were the CONTINUOUS CI leg - each carried exactly the
   unconditional `#[test]` attribute
   (checked structurally; cfg-gated or cfg_attr-ignored probes failed
   the scan, and unrecognized attribute shapes failed closed), iterated the
   exclusion const itself, and printed an ORDERED visited receipt the
   oracle compared against the probe's declared sequence with
   multiplicity (shrunken, duplicated, or reordered receipts failed).
   Receipts are probe-authored text, so a probe could forge them -
   which is why the INDEPENDENT leg is
   `run_exclusion_ground_truth`: the oracle itself writes the
   per-label programs, runs eval and the C emitter, then compiles and runs
   every executable C exclusion and compares the rendered bits; it
   re-derives each exclusion's fingerprint from its own observations,
   trusting no probe output. The fingerprints included the chelis#717
   F32-tag narrowing and the chelis#751 bare-integer-literal EMISSION (the lexical scanner
   ignores C line/block comments and string/character literals, so a
   stale expected token in non-code cannot earn the exclusion; the
   native-stage result is always driven, and exact rendered bits fire
   the shrink protocol even if an active-code fingerprint remains;
   fingerprint presence discriminates the reason for a still-broken
   result but never suppresses behavioral verification); the dropped
   -0.0 sign. Every gone-green failure named `DECLARED_EXCLUSIONS` and the
   shrink protocol. The mechanism remains executable and mutation-tested,
   but `DECLARED_EXCLUSIONS` is now empty and the retired probes no longer
   narrow the ordinary harness. Scope notes: corpus
   FLOORS (the §C2.3 byte-identity floor) assert coverage, not
   breakage - they owe legs 1 and 2 only, since unexpected green is
   meaningless for a floor; a declared boundary whose behavior cannot
   be driven is converted into a probe of the rejection (the C-lane
   probes are the worked example) or deleted.

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
`eval_f64_cast_tensor_root_renders_stored_width` ([#864]),
`c_int64_tensor_print_is_exact_above_2p53` ([#723]) and
`c_print_of_f16_tensor_prints_f16_values` / `c_to_list_of_f16_tensor_works`
([#716]) present, green, and carrying exactly one unconditional `#[test]`
attribute. A `cfg`, `cfg_attr`, `ignore`, or unrecognized attribute shape
around any repaired row is a structural failure before the suites run. The
`c_dag_kernels_compute_correct_f16_bits_despite_print` byte-decode lock
retired per its own instructions (replaced by the direct print row); the
round-trip harness green on every exit in both lanes, with an empty
known-red ledger. The oracle still requires the harness's ignore inventory
to EQUAL that ledger (an undeclared skip is a narrowed corpus; a stale row
overstates what the suite covers), re-runs every declared cell, and fails if
one is red for an undeclared reason **or has gone green**. That mechanism is
why [#864], [#684], [#865], and [#1110] could leave only by being un-ignored
on their original assertions. `DECLARED_EXCLUSIONS` is also empty after
[#729] Phase 3 returned [#751]'s exact-bit constant rows to the corpus. The
§B2.9 generic machinery and its mutations remain: a future exclusion must
provide inventory equality, an unconditional ordered-receipt probe, and an
independent oracle-owned re-execution with a wrong-reason discriminator in
the same change set. The oracle further runs the
`chelis_format_shortest` byte locks, requires every no-third-formatter
tripwire class's baseline paths to stay inside its per-class permitted
set (`FORMAT_CLASS_TABLE`: the C class's PRODUCTION allowlist stays
empty; the Rust classes' sets are the frozen annotated non-exit
carriers), requires every tripwire pattern whose `doc()` cites this
document to be a `FORMAT_CLASS_TABLE` key (§B2.8's doc-citation
parity), checks this document's §B2 item list against its
`B2_RULE_INSTRUMENTS` manifest (§B2.8: a rule lands only with a
deliberate instrument decision), and finishes by verifying every
manifest instrument's RUNTIME invocation and centrally consumed-result
receipts (round-2 M2 plus exact-head F2: an instrument that never
executed, or whose returned violation was discarded, fails the final
check).

Scope, stated rather than assumed: the ignore-inventory equality covers the
observation harness, this plan's own instrument. Phase 4 capability cells and
the staged manual Decimal fixtures remain outside this oracle; no Phase 3
value cell is hidden behind `#[ignore]`.

**Continuous execution:** THIS Phase 2 command is a MANUAL gate, and the split
matters. `scripts/faithful_observation_phase2_oracle.py` is invoked by no CI
workflow, by `scripts/gate.py`, and by no other oracle: re-measured 2026-08-04,
`dtype_phase3_oracle.py`'s legs are `dtype_phase2_oracle.py`,
`faithful_observation_phase3_oracle.py`, and the compiled-C cargo suites, and
none of them reaches this script. The ledger equality, `FORMAT_CLASS_TABLE`,
doc-citation parity, `B2_RULE_INSTRUMENTS` manifest, and runtime
consumed-result receipts therefore run only when this command is invoked by
hand. The **Phase 3** oracle is the opposite case and runs continuously: it is
a nested leg of `.venv/bin/python scripts/dtype_phase3_oracle.py`, which the
Linux `Dtype Phase 0-3 Oracle` job runs on every non-docs-only PR (that job and
macOS Smoke both skip when `changes.outputs.docs_only` is true), and
`scripts/test_nextest_profile_partition.py` fails if the nesting is removed.
The Phase 2 corpus's constituent Rust cells also run under macOS Smoke's
`cargo nextest run --workspace` on those same PRs, so a red CELL is caught
continuously while a narrowed corpus or a tampered instrument is not. Wiring
this Phase 2 command into the same nesting is open work; this plan owns its
contents and pass criteria either way.

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
every exhaustive consumer. [#729] Phase 3 replaces the boxed-only state with
exact narrow scalar ABI representations while keeping the same exhaustive
boundary. (3) §C2.3's byte-identity lock runs where stored bits
AND rendered widths agree; the eval tensor width note (spec/05 §8.1)
keeps non-dyadic narrow-float tensor cells width-divergent until [#729],
and the [#862] unit-root labeling discovery is filed, not absorbed.
(4) PR #863's fresh-context red team (round 1) surfaced two further
width-annex gaps, filed per §B2.5 and annexed at spec/05 §8 with
ignored red cells: [#864] (eval's labeled-root render of
`cast(<tensor>, f64)` disagreed with the print transcript - an eval-lane
[05-OBS-1] violation the harness's `via_cast=false` F64 table
structurally never constructed) and [#865] (the compiled lane's
untagged f64 value box renders f32 - not only f16/bf16 - elements at
f64-image width through `to_list`/boxing; faithful but not own-width
shortest). Both are [#729]-family value/capacity repairs; rendering is
not the fix site for either. The chelis#864 follow-up diagnosed its exact
cause: the root tag was already F64, while static
`to_tensor` lowering retained lexical f64 decimals in an F32 source node
and widened the wrong stored value. Finalizing an enclosing F32 tensor was
necessary but not sufficient: a scalar `f32 -> f64` cast inside an enclosing
f64 `to_tensor` bypassed that check. Static literal extraction now finalizes
each FLOAT leaf at its own declared width, read from the leaf's type metadata,
while integer leaves stay on [#856]'s exact i64 lane - the leaf carries no
precision tag, because routing every leaf through an f64 value field would
have reintroduced the above-2^53 integer loss that lane prevents. An explicit
cast chain needs no special handling: the authored ladder already finalizes at
each float target. The cell is un-ignored, absent from the known-red
ledger, and present in the separate must-run inventory whose exact
unconditional attribute shape is checked. [#729] Phase 3 likewise retires
[#865] through the tagged rank-0 tensor carrier and [#1110] by representing a
checker-stamped literal-width finalization explicitly in the host expression
before the outer cast. Both original assertions are un-ignored, and
[05-OBS-3] still forbids laundering a future known width violation through
the tolerance table.
(5) The phase's oracle became a script rather than a prose conjunction,
after PR #863's exact-head red team (F3) observed that the default
harness run reported "30 passed, 3 skipped" while nothing asserted what
the three skips were, that they still failed for their stated reasons,
or that none had gone green. Three annexed cells was a defensible
boundary; three cells nobody re-executed was not, and the difference is
not visible from a green suite. The ledger makes the boundary
executable, and its unexpectedly-green leg turns each cell into
[#729]'s exit-criteria instrument: the day a value or box-width repair
lands, this oracle fails until the cell is un-ignored. That transition
has now occurred for [#864] and for [#684]: each cell's ignore and
known-red row left together, and both repaired tests entered the must-run
inventory ([#1078] retired the [#684] row, which the oracle's
unexpectedly-green leg is what flagged as stale). [#865] and [#1110] now
complete the same transition, leaving the ledger empty. The accompanying
planning text records each transition and the empty ledger; spec/05 §8 keeps
only the timeless observation contract and its executable corpus pointer.

## Phase 3 - the tolerance table and the [#687] handshake

**You inherit:** two lanes that render identically; value divergences now
visible as exactly themselves.

**You deliver:**

1. The per-op tolerance table authored into `spec/05-risc-primitives.md`
   §8, which [05-OBS-3] names as its single address (this doc's §C4 and
   `dtype_semantics.md` §C4.5 are pointers to it, not alternate homes),
   authored from the audit's measurements (the vvsqrtf/SLEEF rows). The
   fix-precedes-the-row gate on `sqrt = 0` is SATISFIED: [#719] was
   fixed by PR #760 (merged 2026-07-17; contiguous f32 sqrt now takes
   the correctly-rounded scalar path, layout-independent, and the
   scalar loop measured ~1.6x FASTER than vvsqrtf). The row is now
   authored; the table is never authored with a known-false row.
2. Jointly with [#687]: `parity.rs` and `eval_agreement.rs` replaced by /
   rebuilt on the byte-equal-or-table-bounded rule (§C4.3). (The silent
   float-parse fallback itself is deleted earlier, by [#729] Phase 0 -
   this phase replaces the comparison rule it left behind; see the
   corpus-diet note in that deliverable.) Tolerance selection uses the
   closed `AgreementOp` identity, derived by an exhaustive match on the
   actual result-producing `RiscOp`; a caller cannot label an exact op as a
   transcendental by passing a string.
3. The rejected-cells corpus (from `loud_unsupported.md` Phase 0) and
   the value corpus unified under the same comparison rule so
   diagnostics and values are oracle-checked identically.

**Frozen at your exit:** the table (B1); the oracle rule.

**Explicitly not yours:** closing [#687]'s remaining scope if any lanes
still disagree on VALUES - those are [#729]-tracked cells, now perfectly
visible.

**Oracle:** one command -
`.venv/bin/python scripts/faithful_observation_phase3_oracle.py`, accepted at
exit 0 with the final line `PHASE 3 ORACLE: PASS`. It runs the shared policy
and numbered-spec tripwire, the full current `parity.rs` plus rejected-cell
corpus, and `eval_agreement.rs`; it also freezes each suite's test inventory,
comparator adoption, forbidden legacy f64/epsilon paths, ignore ledger, and
the exact reviewed definition of every required Rust test. The definition
digests are a guard artifact: changing one requires independent evidence for
the replacement behavior, and changing the digest merely to accept a test
edit is not a repair. This makes an emptied parity row, an emptied rejected-cell
driver, or an eval test that emits a forged producer-authored receipt fail
before its suite runs. Eval/C receipts remain runtime-entry and multiplicity
evidence; they are not trusted as evidence of their own free-form detail.
Three digest-locked behavioral canaries perturb the compiled observation before
the shared comparator, drive the shared `assert_expected` helper with a
known-wrong expected value, and present an adjacent f32 result while the
evaluator is marked nonconforming; together with the exact operation-identity
canary, they prove that compiled bytes reach the decision, that the verbatim
expected value reaches it too, that the real IR op selects the tolerance, and
that chelis#897 blocks tolerance rather than relying on source-token presence.
The expected-value canary closes chelis#1104: the source-level checks can only
see that the comparator is NAMED in a suite, so a shared assertion helper
neutered into a no-op used to delete the verbatim leg from every row at once
while every frozen test definition, receipt, and comparator obligation stayed
intact. Each leg of the comparison therefore owes a canary that runs the
shipped helper, and the guard is behavioral: tampering with its digest does not
make it pass.
The sole allowed ignore is
`parity_transformer_block_library_only`, whose exact reason is the
environmental system-CBLAS prerequisite. There are no value-divergence
ignores. This composite command runs CONTINUOUSLY, though not under its own
name: no CI workflow and no `scripts/gate.py` stage invokes it directly, and it
reaches CI as a nested leg of
`.venv/bin/python scripts/dtype_phase3_oracle.py`, which the Linux
`Dtype Phase 0-3 Oracle` job runs on every non-docs-only PR (verified
2026-08-04; that job skips when `changes.outputs.docs_only` is true, as does
macOS Smoke). The nesting is not incidental:
`scripts/test_nextest_profile_partition.py` fails with "Phase 3 no longer
inherits faithful_observation_phase3_oracle.py" if it is removed, and it reads
this oracle's `SUITE_COMMANDS` as the executable manifest of what the dtype
oracle covers. The individual Rust suites also run under workspace
nextest, while `scripts/test_faithful_observation_phase3_oracle.py` runs in
CI's script-unit stage and mutation-checks the oracle's structural guards.
Run the composite command before phase acceptance; byte-exact output is the
default, table-listed operations may differ only within their authored bound,
and every remaining value divergence is a failure rather than tolerance.

---

# Part IV - bookkeeping

## I1. The interlock with [#729] (dtype semantics)

- **Ownership**: this plan owns EXITS (rendering); [#729] owns VALUES and
  CAPACITY (what is stored, in what buffer, across which wire type).
  `dtype_semantics.md` §C4 and this §C1 are the same rules by
  construction; edits go to both in one change set. `Repr` in `chelis-vocab`
  describes each current physical encoding and supplies its byte width. It
  does not select a storage format. `dtype_semantics.md` §C3 owns that
  decision.
- **Landing order - this plan first (expected)**: [#729] Phases 1-3 then
  inherit the formatter and validate against it; eval serves them as the
  reference RENDERER (its Phase 1 exit note - the value authority is
  spec/04 §9, not a lane), so lane validation covers reference bits AND
  reference bytes. [#723]/[#716] are fixed without waiting.
- **Landing order - [#729] first**: its Phase 1 delivers `format_element`'s
  Rust side per its §C4 and THIS doc's Phase 1 collapses into an
  adoption/migration pass; its Phase 3 delivers §C3.2's generation and
  this doc's Phase 2 collapses likewise. Either way the contracts here
  govern the result; the tracking issues cross-check the boxes.
- **Capacity limits**: [#729]'s typed storage decision carries exact int64 and
  per-dtype tensor payloads across the versioned wire and registered Python
  boundary ([#686]/[#685]). Format-native limits such as JSON's non-finite
  number grammar remain explicit serialization decisions, never rendering
  fallbacks.
- **The dtype-carrying payload** (raised as F1/F2 by PR #863's
  exact-head review, which asked why Phase 2 leaves dtype-unfaithful
  states *representable* rather than merely unreached): that ask is
  [#729]'s, and it is the same ownership line this section already
  draws - a closed `{dtype, bits}` (or per-width) scalar payload
  carried through the runtime's `chelis_value` box, eval's tensor
  store, the containers, the roots, and the wire schema is a STORAGE
  and CAPACITY change, not a rendering one. [#729] has now closed the named
  separations without widening the frozen public `chelis_value` layout:
  eval and the versioned execution wire carry typed per-dtype storage, while
  the runtime boxes f16/bf16/f32 through the existing tagged rank-0 tensor
  carrier ([#865]) and reserves `CHELIS_VALUE_FLOAT64` for actual f64 values.
  Static literal extraction keeps exact integer `RawScalar`s and finalizes
  only float leaves, including [#864]/[#1110]'s widening shapes.
  **Phase 2's FORMATTING is forward-compatible with that payload by
  construction; its DECODING was not, and that distinction matters**
  (this bullet was corrected 2026-07-28 - an earlier revision claimed
  compatibility for the renderers as a whole, which overstated it).
  On the format side the claim holds: `format_element(prim, ElementRef)`
  and `chelis_format_shortest(value, dtype, buf, cap)` are both already
  keyed by dtype, so the payload's arrival replaces two arguments with
  one at the CALL sites and changes no rule in §C1, no byte of the
  grammar, and no rendered output.
  But both take an ALREADY-DECODED element, so neither says anything
  about whether the bytes were read at the right representation - and
  the int32 misdecode PR #863's review found lived entirely upstream of
  them (§C3.5). Two consequences worth stating plainly: the C entry
  point's dtype parameter was originally spelled `width_kind`, which
  encoded exactly the width-only model that [#964] replaces - and spec/04
  [04-NUM-8] then gave "width" a SECOND meaning (storage vs arithmetic,
  which differ for f16/bf16), so one parameter name denoted two
  properties of a thing it was not even naming. Renamed to `dtype`
  before the 0.18 tag, matching every other dtype-id parameter in the
  runtime and header; and `chelis_format_shortest`'s
  `(double, int)` pair remains a seam the payload closes - the pair is
  caller-supplied, and the routine can reject an invalid dtype id but
  cannot prove the value is the exact widening of one stored at that
  width. Hardening it (a tagged struct or per-width entry points, plus
  `(buf, capacity)` and an explicit result) is the natural joint moment
  with [#729]'s payload work and [#893]'s seal, recorded here rather
  than absorbed - this plan does not own the storage side of it.
- **With [#730]**: the `<value>` placeholder and the print helper's abort
  default are its census rows; the shared tripwire carries this plan's
  three no-third-formatter classes (§B2.4: `c-format-narrowing`,
  `rust-format-narrowing`, `rust-debug-numeric-format`). No delivery
  overlap.

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
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
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
[#964]: https://github.com/Chelis-Lang/chelis/pull/964
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
[#856]: https://github.com/Chelis-Lang/chelis/issues/856
[#897]: https://github.com/Chelis-Lang/chelis/issues/897
[#1043]: https://github.com/Chelis-Lang/chelis/pull/1043
[#1078]: https://github.com/Chelis-Lang/chelis/issues/1078
[#1110]: https://github.com/Chelis-Lang/chelis/issues/1110
