# Making the numeric bug classes structurally impossible

Written 2026-07-16, after the second sweep (chelis#714-#726). The first
audit's conclusion was "nothing forces completeness, so each fix stops at
the reported symptom" ([#695]). This document is the answer to the next
question: **what would force completeness?**

The class inventory now stands at four metas: [#695] (integers have no
representation), [#703] (unsupported cases substitute values), [#709] (the
checker's silent `Type::Error` analogue), and from this sweep [#727] (no
dtype's semantics are enforced at any single point - the generalization of
#695) and [#728] (the observation channel is not dtype-faithful). Items 1,
2 and 7 below are #727's class fix; item 6 is #728's; items 3-5 and 8 are
#703's and #709's. Not "fix the arithmetic" but
"change the code's shape so the next person cannot reintroduce the bug
without a compile error, a failed tripwire, or a red matrix cell".

The proposals are ordered by leverage. Each names the mechanism, the issues
it makes unwritable, and its cost. The unifying observation across all 40+
findings is that every one of them is a **missing chokepoint**: a decision
(what does this dtype mean here? what happens when this case is
unsupported?) that is made independently at every op x lane x surface site
instead of exactly once. The fixes below are all the same move: create the
chokepoint, then make bypassing it a type error.

## 1. One value-finalizer per dtype, with private constructors

**The mechanism.** Every numeric result in every lane must pass through a
single function that applies the dtype's semantics before the value becomes
observable:

```rust
impl ScalarBits {
    /// The ONLY constructor for op results. Applies dtype semantics:
    /// f16/bf16/f32 rounding, integer width (trap per the #680 contract),
    /// bool domain. Private raw storage; ops cannot skip it because they
    /// cannot construct the type any other way.
    pub fn finalize(prim: Prim, raw: OpResult) -> Result<ScalarBits, NumericTrap>
}
```

with the raw payload fields **private** and no other `pub` constructor.
`TensorValue` gets the identical treatment: the element buffer is not
`pub`, and the only way to produce one from op output is
`TensorValue::finalize(prim, raw_elems)`.

**Why this is structural, not a patch.** Today `tensor_float_unop_f32`
exists and `div` simply does not call it ([#717]); `checked_int_binop`
exists and `add` did not call it ([#680]); the correct narrowing exists in
`convert_cast_data` for f32 and not for f16 ([#717], [#720]). Every one of
those is possible because the value types are constructible without going
through anything. Private constructors turn "forgot to call the helper"
from a silent wrong answer into code that does not compile.

**Makes unwritable:** [#680] (int width skipped), [#684] (storage that
never saw a finalizer), [#717] (per-op narrowing chaos - the f64-destroys,
f32-skips, f16-never cells all become the same one function), [#718]'s
eval half, [#724]'s eval half (187.5 cannot be finalized into an int64),
[#726]'s host halves (2 cannot be finalized into a bool). [#720] falls out
too: `fold_static_cond`'s Cast arm calls `convert_cast_data`, which becomes
a thin wrapper over the same finalizer.

**Cost:** a large but mechanical refactor of `eval.rs` / `host_ops.rs`;
the compiler enumerates every site (that is the point).

## 2. Split the integer and float kernels at the type level

Already decided in [#695]; restated here because mechanism 1 does not
subsume it. `numeric_binop(args, op: impl Fn(f64, f64) -> f64)` accepting
integer operands is how every round of the integer class happened. The
finalizer stops wrong values from *escaping*; the type split stops the
wrong kernel from *running* (and is what makes the overflow-trap contract
implementable - you cannot trap an i64 overflow inside an `Fn(f64,f64)`
kernel).

```rust
fn int_binop(a: i64, b: i64, op: CheckedIntOp) -> Result<i64, NumericTrap>
fn float_binop(a: f64, b: f64, op: FloatOp) -> f64   // then finalize(prim, _)
```

**Makes unwritable:** [#680], [#688] (prove's interpreter env becomes
`HashMap<String, ExactValue>` instead of `f64`), [#718]'s remaining cells.

## 3. Kill the catch-all arm: closed enums, exhaustive matches, and an
##    Err-typed fallback for open sets

Three sub-rules, one per kind of dispatch:

**(a) Closed sets (Prim, RiscOp, the CHELIS_* dtype ids).** No `_ =>` arm
in any numeric crate. Every confirmed dtype-substitution bug is a wildcard:
`elem_kind`'s `_ => ElemKind::F32` ([#689]), `parse_host_type`'s
`_ => Unknown` ([#714]), the print helper's `default:` ([#716]), the
C dtype switch cases that fell through. Deleting the wildcard makes the
next added `Prim` variant a **compile error at every dispatch site** - the
compiler produces the complete work-list that #695 observed nobody ever
assembles by hand. Enforcement: a `chelis-lint` rule (the lint already
parses Rust source for §8.6) that forbids wildcard arms on a named list of
enums outside an allowlisted file, so the gate catches regressions.

**(b) Emitted C switches.** The print helper and its siblings are
*generated* strings, invisible to rustc. Rule: generated switches over
dtype ids must be produced by one Rust function that matches exhaustively
over `Prim` (rule (a) then applies to it) and must emit a
`default: abort-with-dtype-id` arm, never a value-producing default.
[#716] and [#723] both die here, and the next dtype added to the runtime
cannot silently print as f32.

**(c) Open sets (builtin-name strings).** A dispatch keyed by strings
cannot be exhaustive, so its fallback must be **un-writable as a value**:
return type `Result<EmittedExpr, UnsupportedBuiltin>`, where the only
constructor for the error carries the name and span. `host_emit.rs:2300`'s
`format!("/* unsupported builtin {other} */ 0")` ([#682], [#704], [#705],
[#715]) is impossible to write when the arm must produce an
`UnsupportedBuiltin` and the caller must surface it as a diagnostic.
`unwrap_or_default()` on extraction results ([#725]) is the same rule:
extraction returns `Result`, and the `?` operator does the right thing by
default - the wrong thing becomes the thing you have to type extra
characters for.

**Makes unwritable:** [#682], [#689], [#692] (the panic arms become
diagnostics), [#694] (the comment cannot drift from a match the compiler
checks), [#699], [#704], [#705], [#714]-[#716], [#725], and the entire
future supply of #703-class bugs.

## 4. One op x dtype capability table, consumed by everything

The deepest recurrence is *lane skew*: the checker accepts what eval
rejects ([#712], [#715]'s int rows), eval computes what C stubs ([#715]),
C panics where Metal rejects cleanly ([#692] vs Metal), `mean` means three
things ([#724]), `add` on bool exists nowhere on purpose ([#726]). Nobody
decided these cells; they are the residue of five dispatch sites growing
independently.

**The mechanism:** a single `const` table - op x dtype -> `Supported |
Rejected(reason)` - in one crate, from which:

1. the checker's acceptance rules are **derived** (not hand-mirrored);
2. each backend's dispatch skeleton is macro-generated, so a table row
   with no kernel implementation is a **compile error** in that backend,
   and a kernel with no table row is dead code the compiler flags;
3. a generated conformance test executes every `Supported` cell through
   every lane and asserts exact agreement, and every `Rejected` cell
   through every lane and asserts the same diagnostic.

This converts "add a builtin" from *silently-becomes-a-C-stub-by-default*
([#703]'s observation) into: write the table row, and the build breaks
until every lane either implements it or the row says `Rejected`. The
matrix tests this audit wrote by hand (`precision_matrix.rs` and the seven
sweep files) are the manual prototype of item 3; the table makes them
generated and complete instead of curated.

**Makes unwritable:** every checker/eval/backend disagreement as a class -
[#712], [#715]'s three-lane rows, [#724], [#726], [#692]'s
panic-vs-silent-vs-correct split within one op family.

## 5. Tags as an enum: the checker cannot have a hole it cannot see

[#709] exists because `infer.rs` dispatches on tag *strings* and the
wildcard returns `Type::Error` silently. The parser already enforces the
closed 62-tag vocabulary, so the vocabulary IS an enum in fact - make it
one in code (`enum DeepTag`), match exhaustively in `infer.rs` and
`lower.rs`, and rule 3(a) applies: the day someone adds tag 63 to the
parser, every consumer that has not decided what to do with it stops
compiling. The `handle-effect` hole, and [#709]'s point-2 fix (the loud
unknown-tag diagnostic), both become permanent instead of policied.
`lower_handle_effect`'s effect-kind catch-all gets the same treatment with
an `enum EffectKind` (the `.dp` probe proved `effect: teleport` builds and
runs today).

**Makes unwritable:** [#709], [#710]'s silent half, the `.dp`-side
variants of both.

## 6. A dtype-faithful observation channel, shared across lanes

Every printed tensor in both lanes funnels through one f64-shaped exit
today (`double value` in the emitted helper; f64 formatting in eval), which
is how an EXACT compiled int64 sum printed as `...992.0` ([#723]) and
correct f16 kernels printed garbage ([#716]) - the *evidence channel*
corrupts the audit of everything else, and it is why three probes in this
sweep initially mis-scored a lane.

**The mechanism:** one per-dtype formatting function in Rust, used directly
by eval and used to *generate* the C print helper (rule 3(b)), with the
formatting contract (integers print as integers, narrow floats print their
exact narrowed value, transcendental tolerance rows documented per op)
written down once. #687's exact-string oracle becomes implementable the
same day: both lanes print through the same code.

**Makes unwritable:** [#716], [#723], the bool `1.0`/`true` split
([#726]'s observation half), and the eval-vs-C float formatting divergence
that currently blocks byte-exact lane comparison.

## 7. Domain-validity as an executable invariant

Cheap, immediate, and orthogonal to all of the above: a property harness
that walks any printed/returned tensor and asserts **every element is a
member of its declared dtype's value set** - f16 values are f16-bit-exact,
int64 tensors hold integers, bool holds 0/1. It would have caught, with no
knowledge of any specific op: `2049.0` in an f16 tensor ([#717]),
`187.5` in an int64 tensor ([#724]), `2` in a bool tensor ([#726]), `200`
in an int8 ([#718]). Wire it into the cross-lane drivers the audit already
committed, so every future matrix row gets domain-checking for free.
(Rule 1 eventually makes violations unconstructible; this catches them in
the meantime and guards rule 1's own refactor.)

## 8. Tripwires for the patterns that remain expressible

Some hazards cannot be typed away (generated-code content, comment drift).
The repo already uses tripwire tests (conformance MANIFEST, workflow
hand-inlining); add:

- a grep-tripwire over the backend crates for the exact recidivist tokens:
  `*/ 0"` in emitted-string builders, `unwrap_or_default()` /
  `unwrap_or(Prim::` in lowering paths, `as f64` in prove's
  value-flattening modules. Allowlist current-and-audited sites; any new
  site fails the gate with a pointer to [#703].
- [#694]'s enforcement: safety comments that claim "panics" / "never
  reaches" on a dispatch site must sit adjacent to a test exercising that
  claim, or not exist. (Not mechanically checkable in general; the
  tripwire is a review-checklist line in the PR template plus the
  audit-file locks already committed.)

## The formalization map

Every mechanism in this document is now owned by a formal plan; this
investigation doc remains the evidence record behind them:

| this doc | formalized as |
|---|---|
| items 1, 2, 7 (finalizer, kernel split, domain invariant) | [`spec/design/dtype_semantics.md`](../../spec/design/dtype_semantics.md) (chelis#729) |
| items 3, 8 (no catch-alls, Err fallbacks, tripwires) | [`spec/design/loud_unsupported.md`](../../spec/design/loud_unsupported.md) (chelis#730) |
| item 5 (DeepTag / totality) | [`spec/design/checker_totality.md`](../../spec/design/checker_totality.md) (chelis#731) |
| item 6 (shared formatter) | [`spec/design/faithful_observation.md`](../../spec/design/faithful_observation.md) (chelis#732) |
| the spec-silence root beneath all of it | [`spec/design/spec_provenance.md`](../../spec/design/spec_provenance.md) (chelis#733) |
| item 4's table schema | [`spec/design/capability_table.md`](../../spec/design/capability_table.md) |
| cross-plan sequencing, unclaimed issues, deferred evidence | [`spec/design/remediation_roadmap.md`](../../spec/design/remediation_roadmap.md) |

The decided contracts themselves are seeded into the numbered specs as
provisional atoms (spec/04 §9-§10, spec/05 §7-§8), so the active spec is
not silent on anything that has been decided.

## Sequencing against the open fix plan

Unchanged from [#695]: **#687 first** (the oracle, now feasible via
item 6), then #680 as a class fix - which should be implemented AS items
1+2, not as an op-list patch. Items 3(a)/(c) are small, independent, and
retire the largest issue count per line changed; item 4 is the long-term
shape and can start as the generated conformance test before any dispatch
is migrated. Items 7 and 8 are afternoon-sized and guard everything else.

| # | mechanism | kills the class of |
|---|---|---|
| 1 | private ctors + per-dtype finalizer | #680 #684 #717 #718 #720 #724 #726 |
| 2 | int/float kernel type split | #680 #688 #718 |
| 3 | no catch-alls; Err-typed open fallbacks | #682 #689 #692 #694 #699 #704 #705 #714 #715 #716 #725 |
| 4 | one op x dtype table, all lanes derived | #692 #712 #715 #724 #726 |
| 5 | DeepTag / EffectKind enums | #709 #710 |
| 6 | shared dtype-faithful formatter | #716 #723 #687-blockers |
| 7 | domain-validity invariant | cross-cutting detector |
| 8 | tripwires | recidivism guard |

[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#694]: https://github.com/Chelis-Lang/chelis/issues/694
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#704]: https://github.com/Chelis-Lang/chelis/issues/704
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#712]: https://github.com/Chelis-Lang/chelis/issues/712
[#714]: https://github.com/Chelis-Lang/chelis/issues/714
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#720]: https://github.com/Chelis-Lang/chelis/issues/720
[#723]: https://github.com/Chelis-Lang/chelis/issues/723
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
