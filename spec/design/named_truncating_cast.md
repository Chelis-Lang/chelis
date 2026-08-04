# Named truncating cast — the float→int rung of the #759 ladder

Status: **shipped** (2026-08-03). The normative text now lives in
`spec/05-risc-primitives.md` §3.8 as **[05-OP-6]**; §4 below is the drafting record,
not the authority. Read the spec/05 atom for the governing contract — where the two
disagree, the numbered spec wins and this doc has a bug.

Owning tracker: **[#759]** (the named lossy cast ladder). Trigger: **[#1091]** (the
ecosystem break). This doc does not open a new tracker — #759 already owns the class;
per the one-tracker-per-class rule the work hangs off it.

## Why this exists now

`[04-NUM-14]` made the default `cast` a *checked* cast: a fractional or non-finite
float cast to an integer target traps `Domain` (it "SHALL NOT truncate, round,
saturate, or wrap"). That is the correct default and must not change. But it was shipped
(PR #1049, #729 Phase 1) with **no named escape hatch**, so every downstream site that
legitimately wants truncation has nothing to migrate to. The Ecosystem Drift Canary
records the cost: `coral`, `hull`, `school` (×2), and `hello-chelis` all fail with
`numeric trap: domain in cast at int64` (auto-filed drift issues coral#23, hull#14,
school#189, hello-chelis#19).

`spec/design/dtype_semantics.md` §"cast ladder" and `spec/design/capability_table.md`
already name this rung as future work under #759 ("finalize-or-trap rung and left
truncation for a future named lossy form"). The decision (2026-08-03) is to **ship the
named form**, not to paper the shells with `cast(floor(x), int)` at each site — a named
op gives both lanes one auditable semantics, gives the checker and backends a callable to
lower, and gives the capability table a row to author, rather than re-deriving the idiom
per site.

## Non-goals

- Not changing `[04-NUM-14]`: the default `cast` stays checked/trapping.
- Not `bool` (out of scope per `[04-NUM-4]`).
- Not the other ladder rungs (saturating, rounding) beyond noting them as siblings; this
  doc authors the **truncate-toward-zero float→int** rung the shells need. `cast_round`
  (round-half-to-even) and `cast_saturate` are follow-on rungs on the same discipline.

## 4. Drafted atom (LIFTED — now `spec/05-risc-primitives.md` §3.8 [05-OP-6])

The text below was lifted verbatim into spec/05 §3.8 with the placeholder resolved to
`[05-OP-6]` (the next free number in the group after [05-OP-1..5]). It is reproduced
here as the drafting record only; `spec/05` is the authority.

> **[05-OP-N]** `cast_trunc(source, target)` is the explicit truncating narrowing cast
> from a float source dtype to an integer target dtype. For a **finite** source value it
> yields the integer part truncated toward zero (the value with its fractional part
> discarded), finalized at the target width; if that truncated integer is outside the
> target range it traps `overflow` (never wraps or saturates). A **non-finite** source
> (`NaN`, `±inf`) traps `Domain` — truncation of a non-finite value has no integer
> meaning. On any source/target pair that is not float→integer, `cast_trunc` is a type
> error (use `cast` / `[04-NUM-14]`); it never widens, never rounds, and never applies to
> `bool`. Semantics are identical on scalar and tensor surfaces and identical across the
> eval and compiled lanes at the declared widths of `[04-NUM-8]`. `cast_trunc` is
> **non-differentiable**: its adjoint is zero almost everywhere (the map is piecewise
> constant), so it carries the `no_grad` rule — a gradient goal through it is a clean
> error, never a silent zero that masks a modeling bug (same discipline as `argmax`).

Relationship to the default: `cast_trunc(x, T)` agrees with `cast(x, T)` exactly when
`x` is already finite and integral and in range (both yield the same integer); it differs
only by *defining* the fractional case as truncation where `cast` traps `Domain`.

## 5. Numeric-surface registration (mandatory, same change set as impl)

Per CLAUDE.md's Numeric Surface Discipline, this is a **new numeric op** and its landing
change set must, atomically:

- author `[05-OP-N]` in `spec/05-risc-primitives.md` (the text above), not a cross-ref;
- register `cast_trunc`'s exact canonical identity to that atom in the owning family
  registry (the covered-family surface ratchet, `dtype_semantics.md` §C6);
- carry the atom's per-dtype semantics at `[04-NUM-8]` widths, the non-differentiability
  rule, and (n/a) accumulator rule.

A new op that skips registration is a review-blocking finding; this doc is the
registration's source text, not an authorization to skip it.

## 6. Implementation surface (deferred / coordinated)

This is cross-cutting and lands on surfaces currently held by open stacks, so it is **not
part of the collision-free wave**; it is sequenced after (or coordinated with) those:

- checker type rule for `cast_trunc` (`chelis-types` infer) — overlaps #1036/#1037;
- eval kernel (`dtype_semantics.rs` / `chelis-ir/eval`) — #729 surface (#1049/#1054);
- C/HIP/Metal lowering (`chelis-backend-*`) — overlaps #1037;
- spec/05 atom + census registration — overlaps #1037/#1039/#1042/#1043 (all edit spec/05);
- capability-table row (`capability_table.md`) under #729 Phase 4.

Recommended sequencing: land after Robert's #1037/#1042 (rejection-authority /
diagnostic-kind) and #1039/#1043 (observation) settle, so the spec/05 and backend edits
don't collide. Requires @rlronan co-sign on the atom (numeric spec surface).

## 7. Shell migration (consumes the op; per-shell, cites #1091)

Once `cast_trunc` ships, the five broken shells replace each fractional `cast(x, int*)`
with `cast_trunc(x, int*)`, each edit citing `chelis#1091` per the shell narrowing-citation
rule. The canary's per-shell drift issues (coral#23, hull#14, school#189,
hello-chelis#19) close on pass automatically once the migrations land.

## 8. Acceptance

- eval-vs-C byte-identical (f32-bit / exact-int) on a truncation matrix over
  {f32,f64}→{int32,int64}, including boundary values (±0.9, exactly-integral, at/over the
  target range → overflow trap, NaN/±inf → Domain trap). This is the standard cross-lane
  agreement gate.
- negative parity: `cast_trunc` on non-float source, on bool, and gradient-through are
  each a clean typed error.
- the five shells green under the canary; the drift issues auto-close.

## Related

[#759] (owning tracker), [#1091] (ecosystem trigger), [04-NUM-14] (the checked default
this complements), [04-NUM-8] (declared widths), [05-OP-1..5] (the atom format followed
here), `spec/design/dtype_semantics.md` §"cast ladder", `spec/design/capability_table.md`
(the #759 row).

[#759]: https://github.com/Chelis-Lang/chelis/issues/759
[#1091]: https://github.com/Chelis-Lang/chelis/issues/1091
