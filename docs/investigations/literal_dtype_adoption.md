# Contextual literal-dtype adoption: the #3145 attempt and the single-pass design

This record preserves the contextual dtype-adoption work separated from chelis#3145
(chelis#3114) on 2026-10-04, so that option C of chelis#3164 can be judged or revived
from evidence: the rule, what two review rounds found, and the adoption pass designed
to replace it. Hashes are the [archive branch](#where-the-code-is)'s.

## Problem

spec/04 §5.3 gives an unsuffixed integer literal `i32` and a float literal `f32`; a
suffix, a §5.6 adopting position or an explicit `cast` overrides the default. §5.6 names
the closed set of adopting positions:

> 1. the right-hand side of a binding whose declared type is a tensor type [...]
> 2. the corresponding argument position of a call whose callee has a declared
>    signature whose parameter at that position is a tensor type [...]
> 3. the body expression of a function with a declared return type that is a
>    tensor type [...]
> 4. the first argument of a `cast(literal, p)` expression, where `p` is a
>    precision type literal or a dtype-bounded type binder ([04-DTYPE-2])

The argument of `to_tensor` is not one of them, so on main `cast([1.1, 2.2], f64)` is
exact while `cast(to_tensor([1.1, 2.2]), f64)` builds an `f32` tensor and widens it,
giving 1.100000023841858 (bits `3ff19999a0000000`), not 1.1 (`3ff199999999999a`): the
trap of `torch.tensor([1.1]).double()`. `chelis surf` prints the first as the second,
so the round trip silently changes the value (#3080). Under #3145's kind rule a bare
bracket in a cast is a `List`, which leaves `cast(to_tensor(...), f64)` as the only
spelling of a tensor cast.

## The attempted rule

The branch added this paragraph to spec/04 §5.6:

> The bracket-literal argument of a `to_tensor` call that stands in one of these
> positions takes that position's element type, exactly as a bare bracket literal in
> position 1 or 3 does: `cast(to_tensor([1.1, 2.2]), f64)` binds both decimals at `f64`
> and does not round them through the `f32` default. The kind is fixed in positions 2
> and 4: a callee's tensor parameter or a `cast` never converts a bare bracket literal,
> which stays a `List` there. A call through a lexical binding that shadows `to_tensor`
> is an ordinary call and adopts nothing. (A tensor literal's elements adopting a
> dtype-binder cast target is not fully implemented; see chelis#3148.)

spec/02 §P10b widened position 4 to "a scalar numeric literal or a tensor literal",
restated the paragraph above, and, after round 1, added the capture refusal and the
lexical-callee rule:

> A bare bracket literal that its own declaration converts is rejected where a lexical
> binding named `to_tensor` is in scope, because the conversion would resolve to that
> binding. A call through a lexical binding that shadows a top-level function is not a
> call to that function: position 2 reads the declared signature only of the function
> the callee names.

Three more parts rest on text main already has:

- **Pipes.** spec/02 makes `x |> cast(p)` the two-argument cast of `x`, so after round 1
  every pipe spelling adopted as its nested call does, scalars included.
- **The kind filter.** §5.6's matching-kind limit, applied to elements: an integer
  literal adopts an integer or float dtype, a decimal only a float dtype.
- **Binders.** The first head adopted nothing under a dtype binder, though position 4
  names binders (F3); the final head refuses that case loudly (#3148).

## What review found

Bits are `chelis eval --json` output; native C agreed where checked. "Base" is main
without the branch. Round 1 reviewed through ba99e87eb, round 2 through 5ab1a0b78.

### Round 1

**F1 (P1): a name-keyed lookup ignores lexical scope.** The desugarer and the resugarer
found a position-2 callee by bare name, so a local `f` borrowed the signature of the
top-level `f`:

```text
def f(x: tensor[2, f64]) -> tensor[2, f64] = x
def g() -> tensor[2, f64] = {
  f = fn (y) -> cast(y, f64)
  f(to_tensor([1.1, 2.2]))
}
r = g()
```

Base gives `3ff19999a0000000`, the branch gave `3ff199999999999a`, and a parameter named
`f` made a valid program rejected. Repair: a lexical walk in both (db576ceea).

**F2 (P1): pipe and call spellings adopted differently.**

```text
c2 = cast(to_tensor([1.1, 2.2]), f64)
p2 = to_tensor([1.1, 2.2]) |> cast(f64)
```

`c2` was exact and `p2` rounded, and `chelis lint --fix` rewrote `c2` into a pipe,
changing its value (#3151 has since removed that rule). Repair: every pipe spelling
adopts as its nested call (87a5fa3d8, c791a3368), which made the already-valid
`1.1 |> cast(f64)` exact instead of f32-rounded.

**F3 (P1): a binder cast target the spec covered and the code skipped.**

```text
def widen[p: Float](x: tensor[2, p]) -> tensor[2, p] = add(x, cast(to_tensor([1.1, 2.2]), p))
def scalar_b[p: Float](x: p) -> p = add(x, cast(1.1, p))
r1 = widen(to_tensor([0.0f64, 0.0f64]))
r3 = scalar_b(0.0f64)
```

At `p = f64`, `r1` rounded and `r3` was exact. Base behaves the same, but the new text
claimed the class closed. Repair: a loud refusal, followed up as #3148 (632ee8fa2).

**P2: the capture refusal was missing from the numbered spec.** The code refused
`def sample(to_tensor: List[i32] -> tensor[2, i32]) -> tensor[2, i32] = [1, 2]`, but
only the book said so. Repair: §P10b states it (387bca449).

**N1 (P1, found verifying round 1): the resugarer's printing did not mirror the
desugarer's adoption.** Once pipe heads adopted, the printer forced a suffix only for a
direct literal at a primitive target:

```text
macro mh() = neg(1.1)
b = mh() |> cast(f64)
```

`chelis surf` printed `b = -1.1 |> cast(f64)`, which re-desugars as adopting:
`bff19999a0000000` became `bff199999999999a`. Repair: one printer for a cast operand in
both spellings (5ab1a0b78). The call spelling's half predates #3145 and is still live on
main, where `cast(mh(), f64)` prints as `cast(-1.1, f64)` with the same bit change; it is
tracked as [#3165](https://github.com/Chelis-Lang/chelis/issues/3165), and the archived
printer fixes it.

### Round 2

**R2-1 (P1): a macro body adopted in the definition's scope, while §P5b resolves its
free names at the use site.** F1's class again, and on the round trip N1's:

```text
def f(x: tensor[2, f64]) -> tensor[2, f64] = x
macro m() = f(to_tensor([1.1, 2.2]))
via_macro = {
  f = fn (x) -> x
  m()
}
direct = {
  f = fn (x) -> x
  f(to_tensor([1.1, 2.2]))
}
```

`via_macro` gave f64 `3ff199999999999a`, `direct` f32 `3f8ccccd`. Of ten generated
bodies at a rebinding site, a local `to_tensor` among them, nine diverged silently; base
agreed on all ten. Nine also changed value through `chelis deep` and `chelis surf`: the
resugarer saw the names as local but printed `1.1` because the style marker said
unsuffixed. Trusting that marker predates #3145; the branch made Surf produce such Deep.

**R2-2 (P2): three breaking changes to already-valid `cast(to_tensor(...), T)` were
missing from the fragment.** `cast(to_tensor([0.1, 0.3]), f64)` went from f32-rounded to
exact; `cast(to_tensor([1000000.0]), f16)` from `inf` to a compile-time [04-LIT-2]
rejection; `cast(to_tensor([300]), i8)` from a run-time trap to a compile-time rejection.

### The lesson

Deciding a literal's dtype from the syntax around it, in two places that must mirror each
other (the desugarer, which decides, and the resugarer, which must print a spelling that
decides the same way), produced the same two classes three times: a context lookup that
ignores lexical scope (F1), a printer that does not mirror the desugarer (N1), and both
at once (R2-1). F2 was the same shape inside the desugarer, whose pipe and call paths
each carried a copy. Each repair closed its witnesses; the class stayed open because
the rule had more than one owner.

## The single-owner adoption pass

**Where it runs.** One module exposes `adopt_program`, which sets each literal's dtype,
and `analyze`, which tells the printer how to spell it. It runs at the end of every
`desugar_*` entry point and after `chelis_macros::expand_program`. chelis-macros depends
on chelis-deep but not chelis-surf, so the pass lives in `crates/chelis-deep/src/adopt.rs`.
Each run recomputes every marked literal, so it is idempotent; the unfinished code
splits it into a provisional run and a final one.

**What it reads.** Deep only, with every binder visible (`fn` parameters, `bind` pairs,
match patterns, top-level `def`s). P2 is `(app (var f) ..)` where `f` is not locally
bound and its `defsig` declares a `t-tensor` parameter at that index; P1, P3 and P4 read
`t-tensor` declarations and checked casts likewise. A pipe head stands where
`fold_pipes` puts it, and `to_tensor` is the intrinsic only where nothing rebinds it.
It outputs the final `type`, plus `literal_source: integer` for an integer atom at a
float type.

**The marker.** Hand-written `.dp` also passes through `expand_program` and must not be
retyped, so the desugarer marks every unsuffixed numeric literal
`surf_literal_style: "unsuffixed"` at its §5.3 default (main marks only adopted
literals). `-lit` stays `(app neg lit)`; the pass folds the sign in when it adopts and
back out when it no longer does. spec/03 §6.4 would say that the marker means §5.6
decides the dtype on the expanded program and that only round-trip normalization erases
it. The design note has `chelis deep` output gain the marker; the unfinished code's final
run would instead drop it from literals no position adopted, so expanded Deep would keep
main's marking and only Deep printed before expansion would carry it on every unsuffixed
literal. The round-trip law already strips it.

**The resugarer.** `analyze` maps each marked literal, and each `neg` of one, to the
dtype an unsuffixed literal re-derives at its printed position. The printer prints a
literal bare exactly when its `type` equals that dtype. It never reads the marker, so a
wrong marker cannot change a value through `chelis surf`.

**Deleted and kept.** It deletes the desugarer's inline element, pipe and cast adoption
(`desugar_adopting_tensor_literal`, `scalar_*_adopts_*`) and F1's
records, and the resugarer's adoption printers (`DeclaredTensors`, `resugar_adopting_*`,
`resugar_cast_operand`) and its trust in the marker. It keeps the kind conversion, the
capture refusal, the binder-literal visitor and the #3148 refusal.

**Value changes for already-valid programs.** New with the pass: a macro argument adopts
where it expands (`macro c(x) = cast(x, f64)` with `c(1.1)` becomes exact), and a macro
body's free callee resolves at the use site (`macro m() = f(to_tensor([1.1]))` under a
local `f` is f32 again, as on base). Already on the branch: the pipe casts and R2-2's
three changes.

**The spec rule.** One rule in spec/02 §P10b, cited from §P5b and spec/04 §5.6:
adoption is decided on the expanded program, in each literal's use-site scope, so a
macro's body and argument literals adopt where expansion places them.

**Tests.** Each fails first without the pass: the macro matrix (bodies and arguments,
both spellings, sites that rebind and sites that do not), requiring the macro form to
give the direct text's literal types and the round-trip law; fold equivalence,
`analyze(d) == analyze(fold_pipes(d))`; and a pass-driven round trip in which an
unmarked `(lit {f64} 1.1)` prints `1.1f64`.

**Size and risks.** About 3,000 diff lines: `adopt.rs` 1,000 to 1,300, the desugarer
about -600/+80, the resugarer about -700/+150, tests about 600. The estimate is low:
the unfinished `adopt.rs` is already 1,394 lines. Risks: churn from the marker in
exact-Deep-text tests and golden Deep; porting the scalar pipe-cast and binder paths;
the pass's pipe reading drifting from `fold_pipes`; a second linear traversal; and the
capture refusal still not applying through macros, which predates #3145.

## Evidence and tools

The scan and probe scripts are not in the repository.

- **Eval differential.** Extracts Surf programs from both trees (test strings, Markdown
  blocks, tracked `.ch`), desugars each with the base and the head binary, and where
  Deep differs compares checker types and eval kind, dtype and bits. Of about 6,000
  candidates about 300 changed Deep; every value difference where both binaries accept
  was the `List` kind, the amended cast bits or pipe adoption.
- **Printer probe.** 1,044 programs crossing 29 cast-operand forms with eight targets,
  `cast` and `cast_trunc`, both spellings and dtype binders. After the N1 repair none
  changed value or dtype through `chelis surf` and the round-trip law held, apart from
  34 out-of-range programs it refused loudly.
- **Macro matrix.** Round 2's ten bodies; the archive's `issue_3114_macro_adoption.rs`
  widens it to 13 bodies at four sites, plus macro arguments.
- **Mutation checks.** Each mechanism disabled in turn: ten at the first head, twelve for
  round 1's repairs, two for N1. All were caught; one of round 1's survived until it
  gained a killing test.

## Where the code is

Branch `archive/3164-literal-adoption-attempt`, head `0052f656a`: #3145's commits rebased
onto main `b892f20c9`, plus one unfinished commit.

- 4c5d23fbb, bcc2bcaf2: tests, then code, for the kind rule with the first amendment;
  6837346eb, ba99e87eb: a migration and the capture test.
- dab421e41, 53492de35, 7669ef6a6, 4acc26037: round 1's tests and the printer tests.
- db576ceea (F1), 87a5fa3d8 and c791a3368 (F2), 632ee8fa2 (F3), 387bca449 (P2),
  5ab1a0b78 (N1): the repairs; bc175cb59: fragment and book; 34134e2dc, 46936b124:
  test updates after the rebase.
- 0052f656a: unfinished. `adopt.rs` (1,394 lines), `forwarding_stage` shared with the
  pipe fold, the macro matrix and two CLI tests. Nothing calls the pass yet, and this
  record did not build it.

A revival would reuse the four test commits and the unfinished commit's tests,
632ee8fa2's refusal, 387bca449's text and `adopt.rs`. The pass replaces db576ceea,
87a5fa3d8, c791a3368, 5ab1a0b78 and bcc2bcaf2's inline adoption. Two round-2 items are
still open at 0052f656a: the fragment omits R2-2's three changes, and 387bca449's §P10b
position 4 lacks the #3148 parenthetical that §5.6 carries.

## Status

Separated from #3145 on 2026-10-04; #3145 keeps only the kind rule, whose default
[#3122](https://github.com/Chelis-Lang/chelis/issues/3122) decides. Whether literals adopt contextually is pending in #3164
(option C). Pipes are being retired (#3130, after [#3119](https://github.com/Chelis-Lang/chelis/issues/3119)), which would remove
the pass's pipe positions and the fold-equivalence test. Related: [#3080](https://github.com/Chelis-Lang/chelis/issues/3080), the
motivating defect; [#3148](https://github.com/Chelis-Lang/chelis/issues/3148), binder cast targets; [#3152](https://github.com/Chelis-Lang/chelis/issues/3152), a parameter
named `to_tensor` that checks but fails to evaluate.
