# Numeric audit: proposal for the remaining sweeps

A brief for whoever picks this up next. Written 2026-07-16, at the end of the
audit that produced chelis#680-#713 and the three tracking issues
[#695], [#703], [#709].

> **Status: every sweep below has been executed** (results in
> [Sweep outcomes](#sweep-outcomes-2026-07-16) at the bottom; current
> coordination lives in `spec/design/remediation_roadmap.md`, landing via
> the design-set PR #742). The body of
> this file is the audit-time prediction record, kept verbatim as evidence
> of which claim sources were reliable - several predictions were wrong in
> instructive ways, and that is the point of keeping them.

Read [`silent_substitution_audit_backlog.md`](silent_substitution_audit_backlog.md)
first: it records what was settled, what was refuted, and why. This file is only
about what is left.

## The one rule

**Execute everything. Promote nothing to an issue from reading alone.**

This is not a style preference. It is the measured result of the last audit:

| claim source | outcome |
|---|---|
| agent read the code, inferred a bug | **7 refuted** - the code was read correctly, reachability was wrong |
| agent read a *comment*, cleared an op | **1 real bug hidden** (HIP's `sum`) |
| agent read a comment, inferred a bug | **1 refuted** (int32 f32-storage claim) |
| my own probe, no control run | **1 overclaim** (#704's "neural-network layer returns zero") |
| my own probe, bad fixture | **1 false negative** (#711, nearly missed) |

Five source-derived claims were refuted by running the code. The two "worst
findings" an agent ranked were **wrong on both counts**. Meanwhile the one real
extra bug was found in an area an agent had explicitly **cleared**.

So: comments produce **false negatives** (they hide bugs); code-reading produces
**false positives** (it invents bugs that cannot fire). Only execution
distinguishes them.

### Corollaries, each learned the hard way

1. **Probe what was cleared, not just what was flagged.** A confident "this is
   fine" is a *weaker* signal than an uncertain "this looks wrong", because the
   flag gets checked and the clearance does not. HIP's `sum` was cleared by a
   comment and was the only unpredicted bug in the sweep.
2. **A negative probe is only as good as its fixture.** #711 ("no fold occurred")
   was wrong because the probe used `cast(...)`, which creates a non-`Const` node
   and makes the fold decline. Bare suffixed literals reproduce it instantly.
   Before believing "it didn't reproduce", ask what your fixture *prevented*.
3. **Check what your grep actually matched.** A search for `111` in emitted C
   "found" the branch - inside the hash constant `0x94D049BB133111EBULL`. Compare
   bit patterns, not decimal text.
4. **Always run the control.** #704 claimed a neural-network layer returns zero.
   Tensor `relu` is correct; only the scalar overload stubs. One control run
   would have caught it before filing.
5. **Distinguish "the code is there" from "anything reaches it".** The 26
   `RiscOp::Const { value: 0.0 }` sites in `lower.rs` are almost all dead behind
   parser/runtime guards. Reachability is the whole question.

## Core findings to carry forward

### Do not trust the comments. They are wrong in the load-bearing places.

Four comments asserted a safety property the code does not have, and **each one
hid a live bug**:

| artifact | claim | reality | hid |
|---|---|---|---|
| `chelis-backend-hip/src/emit.rs:1253-1255` | non-float dtypes "panic loudly inside `elem_kind` rather than silently downgrading" | `elem_kind`'s default arm is `_ => ElemKind::F32`. It never panics. | [#689] |
| `spec/05-risc-primitives.md:849-855` | "the C/HIP emitters **never see** a `tensor_scan` call" | the emitted C contains `/* unsupported builtin tensor_scan */ 0` verbatim | [#705] |
| `chelis-compiler-api/src/compiler.rs:742-746` | "Without this guard `chelis build` silently emits a C stub ... and the compiled program returns garbage" | the guard is `HOST_ONLY_BUILTINS = &["tensor_scan"]` - one builtin; five others walk past it | [#682] |
| `chelis-backend-hip/src/emit.rs:1121-1126` | "For Add/Mul/**Sum** ... each of those arms resolves the right template inline rather than touching the float-only `elem_kind` shorthand" | true for Add and Mul, **false for Sum** | HIP's `sum` |

The last one is the sharpest: an agent read that comment, cleared `sum` without
probing it, and `sum` turned out to be the only unpredicted bug in the sweep.

Note the failure is **not** "comments are stale prose". Every one of these is
*specific*, *confident*, and describes exactly the property you want to be true -
which is what makes a reviewer stop looking. `spec/05-risc-primitives.md` even
describes the bug's old symptom in the past tense, as a fixed thing.

Tracked as [#694]. **Counter-example worth knowing**:
`reject_with_seed_for_build_target` (`chelis-cli/src/main.rs`) is a no-op whose
comment says *"Today it is a no-op"* and explains why. Honest comments exist. The
problem is you cannot tell which kind you are reading without executing.

### The eval and compiled lanes disagree, in both directions

Do not assume one lane is the reference. Verified divergences run **both ways**:

| program | eval | compiled C |
|---|---|---|
| `add(2^53, 1)` int64 | `9007199254740992` **wrong** | `9007199254740993` |
| `bitand(12, 10)` | `8` | `0` **wrong** |
| `max_elem` int64 tensor at 2^24+1 | `16777217` | `16777216` **wrong** |
| `abs` int64 tensor | `[100,200,300,400]` | `[0,0,0,0]` **wrong** |
| scalar `relu(3.5)` | **errors** | `0` **wrong** |
| scalar `tanh(3.5)` | **errors** | `0.998` **correct** |
| `pad_sequences` id 3e9 | `3000000000` | `2147483647` **wrong** |

Always run both. A bug found in one lane says nothing about the other.

### Nothing in the repo can see this class

Both cross-lane oracles are structurally blind ([#687]):

- `crates/chelis-e2e/tests/eval_agreement.rs` is f64-typed end to end
  (`assert_close(a: f64, b: f64, tol)`), so
  `assert_close(9007199254740992.0, 9007199254740993.0, tol)` **passes**.
- `crates/chelis-cli/tests/parity.rs` compares byte-exactly but **falls back** to
  `parse_tensor_line` (`Vec<f64>`) + a 1e-6 tolerance on mismatch - i.e. exactly
  when a real bug is present.

Both were green through **three** rounds of this bug class. Downstream is worse:
an audit of `Chelis-Lang/school` found its numerical validation is host-eval-only
(`chelis test` is the eval lane; `reef build` only checks that code *compiles*,
never runs it), so no shell repo has a C-backend numerical lane at all.

**#687 should land before any fix.** Fixing the tests last is how the previous
two rounds ended where they did.

### The failure is always "plausible", never loud

This is what makes the class dangerous and why a casual look misses it:

- `add(abs(x), [1,1,1,1])` on int64 compiles to `[1,1,1,1]` - the `add` computed
  `0 + 1 = 1` perfectly correctly on a zeroed operand. No crash, no NaN, no
  absurd magnitude.
- `i64::MAX` round-trips through `Vec<f64>` storage **by luck** (f64 rounds *up*
  to 2^63, the saturating cast clamps back *down*). Two errors cancel. It is not
  evidence of exactness - use `2^53+1`.
- `min_elem` returns the right answer **by luck** (both operands collapse to one
  f64 and `min(x, x) == x`) while its sibling `max_elem` is wrong on the same
  inputs.

Both lucky greens are locked as tests with the coincidence documented. Do not
cite either as evidence.

### One collision that is correct and must stay

`f64 add(2^53, 1) == 2^53` is **correct** (53-bit mantissa, ties-to-even). The
identical numbers at `int64` are a **bug**. Same inputs, opposite verdicts.

A "fix" that corrects the f64 case has broken IEEE-754. The f64 kernel is
mathematically fine for floats (f64 carries >= 2p+2 bits for f32/f16/bf16, so
double rounding is safe) - which is exactly why this survived: the shortcut is
sound for the 99% case and silently wrong only for integers.

## Reusable harnesses (already committed)

Do not rebuild these. `crates/chelis-cli/tests/precision_matrix.rs` and
`crates/chelis-cli/tests/issue_703_silent_placeholders.rs` have working
lane-drivers:

- `eval_lane_str(expr) -> Result<String, String>` - runs `chelis eval --file`,
  returns the printed line **verbatim**. Never parses through `f64`; that is the
  fix for [#687]'s blind oracle and the reason these tests can see the bugs.
- `c_lane_str(expr, ret_ty, name)` - builds to C, links, runs, returns the printed
  value as an exact string.
- `build_and_run_c(program, name) -> Result<(emitted_c, stdout), String>` -
  returns the **emitted source** as well as the output, which is what makes
  GPU-backend and stub findings verifiable without a GPU.
- `check_row(&Row { .. })` - the table-driven matrix driver, with
  `Status::{Locked, ByDesign, Broken(issue)}` and `Lanes::{EvalOnly, Both}`.

**Key technique**: `chelis build --target hip` and `--target metal` emit their
source and print the compile command **without invoking `hipcc`/`xcrun metal`**.
That is how #689 was proven on a machine with neither. Grep the emitted kernel
name for an `_f32` suffix on an int64 program.

## The sweeps, in priority order

### 1. bf16 / f16 across the op surface - NEVER SWEPT

The highest-value gap. Every finding so far is int64/int32/f32/f64; the narrow
floats have had **no coverage at all**.

Concrete reasons to expect something:

- `Prim::F8e4m3` exists in the enum and is documented as deferred/rejected
  (`crates/chelis-types/src/types.rs:38-40`). Is it actually rejected everywhere,
  or does it hit a fallback?
- HIP's `ElemKind` has only `F32`/`F64` (`kernels.rs:75-78`) and `elem_kind`'s
  `_ => F32` arm (`emit.rs:3615`) catches **bf16 and f16 too**, not just integers.
  The audit only probed int64. **Probe bf16/f16 through the same 10 ops** - the
  templates are in #689.
- The C backend's `is_f64` (`emit.rs:1344-1346`) is `Prim::F64` only, and
  `double_math_fn` (`:1352-1369`) remaps float->double with no bf16/f16 arm - the
  same shape as #691.
- Metal's `dtype.rs` panics on `F64` by design. What does it do with bf16/f16?

Boundaries: f16 mantissa 11 bits (first non-representable integer **2049**, max
**65504**); bf16 mantissa 8 bits (first non-representable **257**). These are
*tiny* thresholds - far easier to hit than 2^53, and trivially reachable from
ordinary ML code.

### 2. `grad` through the already-broken ops

`grad`/`vmap` were swept and found to add **no new corruption** - they inherit
[#680]/[#684] (see the backlog doc). But nobody asked the compounding question:

**what is the adjoint of an op that lowers to `Const 0.0`?**

`grad` of `abs(int_tensor)` (#699) - does the backward pass differentiate the
zero placeholder and silently produce a zero gradient? A zero gradient is
*plausible-looking* and would silently stop learning rather than error. Same
question for `floor`/`ceil`/`round`/`cos`/`tan`/`atan`, and for the scalar
activations in #704.

`crates/chelis-ir/src/grad.rs:1897-1908` (`int_scalar_binary`) is the one place
`chelis-ir` explicitly reasons about integer precision, and it still bottoms out
in the f64 `RiscOp::Add`/`Mul`. Start there.

### 3. Complete the dtype x op x lane matrix

`precision_matrix.rs` is table-driven and covers int8/16/32/64 and f32/f64 for
the ops the audit reached. It does **not** cover:

- bf16/f16 (see 1)
- `bitand`/`bitor`/`bitxor`/`shl`/`shr` beyond int64
- reductions beyond the ones #689 probed
- the `Bool` dtype anywhere (note `chelis-runtime/src/lib.rs:144-152` says bool
  storage is "4-byte f32-encoded" - the int32 half of that claim was **refuted**
  by execution, so **probe bool rather than believing either reading**)

Add rows; the driver already exists.

### 4. Metal's op surface

Metal is the **best-behaved** backend (exhaustive matches that panic, clean `Err`
for unimplemented ops, genuine `long` for int64) so expected yield is low. But:

- **#699's Metal symptom was never re-confirmed** after the C-lane root cause
  landed. `abs` on an int64 tensor emits `Const 0` under Metal; that is *assumed*
  to be the same `lower_transcendental` bug. Verify.
- Metal implements only `Abs, Add, Atan, Ceil, Const, Copy, Cos, Drop, Exp,
  Expand, Floor, Load, Log, MaxReduce, MinReduce, Mul, Neg, Pad, Round, Shrink,
  Sin, Sqrt, Store, Sum, Tan`. Probe those with each dtype.

### 5. HIP / Metal *runtime* behavior

Everything GPU in this audit is **emission-inspection only** - no GPU and no
`hipcc` on the audit machine (arm64 macOS). #689 is proven by emitted source, and
the miscompilation is unambiguous (`float*` over a `CHELIS_I64` allocation), but
the runtime symptom is unobserved.

Whoever has the gfx1151 box should run #689's probes end-to-end per
[`docs/local_hip_environment.md`](../local_hip_environment.md) - use
`scripts/hip_test.py`, not bare `cargo test` (see `CLAUDE.md`).

### 6. `chelis prove` Tier C ([#688])

Code-confirmed, never executed. Needs an `@opaque` type with an int64
representation field whose producer can yield magnitudes above 2^53. Today's std
lib has no such type, so this is latent - construct one.

### 7. Low-confidence leftovers

`reduce_window`'s `unwrap_or_default()` (`lower.rs:7522-7537`) and the two
"should never happen" defaults (`named_axis.rs:430`,
`host_emit.rs:4060-4070`'s `assign_partition`). Every sibling claim from the same
sweep was refuted; expect these to be too.

## What was already swept - do not redo

| area | outcome |
|---|---|
| compile-time code deletion | **`lower_if` (`lower.rs:10334`) is the only branch pruner** - #711. `match`'s static arm selection is constructor-tag based with no numeric fold (safe). `fold_static_size` correctly uses `checked_i64`. `dead_code_eliminate` / `prune_to_requested_outputs` / `eliminate_closed_list_noops` are standard DCE. |
| `chelis-cli` parallel gates | swept, **zero new bugs**; bounded at #698 (drifted) + #705 (missing). Two other duplicates are semantically identical; `reject_with_seed_for_build_target` is a no-op but intentional and honestly documented. |
| `.dp` malformed input | 4 of 5 claims **refuted**; the survivor (#710) is low severity. The `Const 0.0` placeholders are never reached. |
| `grad`/`vmap` marshalling | **no new bug**; inherits #680/#684. (But see sweep 2 - the *compounding* question is still open.) |
| tags with no `infer.rs` case | `handle-effect` is the **only** one - #709. All 24 others are handled. |

## Filing standard

Match the existing issues:

- **Lead with the evidence level.** "Confirmed by compiling and running" /
  "code path verified, NOT executed". Every unverified issue in this set says so
  in its first line, and names the probe that would settle it.
- **Include the control.** Every bug should name what *works*, because that is
  what bounds it. #704 without `tensor_relu_is_correct_and_emits_no_stub` was an
  overclaim.
- **Link the meta.** [#695] (integers have no representation) or [#703]
  (unimplemented substitutes a value) or [#709] (the checker's analogue). If it
  fits none, consider that it may be a fourth class.
- **Add the test.** Assert the *correct* behavior, `#[ignore]` it with the issue
  number, the observed wrong value, and the run command. Never invert an
  assertion to match a bug - that documents the bug as intended.
- **Correct by editing the body**, not by piling on comments, so each issue reads
  correctly on its face.

## The thing to look for

Three separate times, the audit found this exact shape:

| correct mechanism | built for | not applied to |
|---|---|---|
| `checked_int_binop` | `div`/`mod`/`floor_div`/`trunc_div` (#387) | `add`/`sub`/`mul`/`neg`/`abs`/comparisons ([#680]) |
| `raise_lowering_error` | `lower_unsupported` | `lower_transcendental` ([#699]) |
| `fold_static_size`'s `checked_i64` | extent folding | `fold_static_cond` ([#711]) |

In each case the right implementation already existed - twice **in the same
file** - and the neighbouring code did not call it. That is the actual disease
behind all three metas, and the most efficient search heuristic available:

> **find a correct implementation, then find its siblings that do not use it.**

[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
[#694]: https://github.com/Chelis-Lang/chelis/issues/694
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#711]: https://github.com/Chelis-Lang/chelis/issues/711

## Sweep outcomes (2026-07-16)

Every sweep above was executed the same day this brief was written. Thirteen
new issues (chelis#714-#726), seven new test files (every broken row
`#[ignore]`d with its issue, every control locked green), and comments with
executed evidence on #682, #688, #692, #695, #699, #709. The one rule held:
several of this brief's own predictions were refuted by running the code.

### Sweep 1: bf16/f16 - the predicted "highest-value gap" over-delivered

Six issues, most of them not the shape predicted:

- **[#714]** f16/bf16 SCALARS have no C-host-lane representation: declared
  narrow types parse to `HostType::Unknown`, arithmetic defaults to
  `int64_t`, and `add(0.5f16, 0.25f16)` compiles to a binary that prints
  `0`. Comparisons pick the wrong branch at threshold 2049; `abs` emits C
  that does not compile. int8/int16 scalars take the same path.
- **[#715]** found *next to* the target: scalar `tan`/`atan`/`floor`/`ceil`/
  `round`/`recip`/`max_elem`/`min_elem` compile to the `/* unsupported
  builtin */ 0` stub at EVERY dtype - plain f32 `floor(1.5)` returns 0 from
  the compiled binary while eval is correct. Same site as #682.
- **[#716]** the C DAG kernels for bf16/f16 are CORRECT (byte-decode proven)
  and every host boundary around them is broken: print reads the 2-byte
  buffers as f32 (garbage), to_list and to_tensor abort at runtime.
- **[#717]** the eval tensor lane's narrowing is per-op chaos: f64 tensors
  destroyed to f32 by every unary float op (`tensor_float_unop_f32`), f32
  add/div/recip never narrowed, f16/bf16 never rounded. Compiled C is right
  in every cell - the reverse of the usual direction.
- **[#718]** integer width semantics are INVERTED between lanes and
  surfaces: scalar C ignores width (eval wraps), tensor eval ignores width
  (C wraps). Corrects #695's "the C backend wraps natively" (comment posted).
- **[#719]** the contiguous f32 sqrt kernel uses Accelerate's vvsqrtf: not
  correctly rounded (IEEE requires sqrt to be), and results change with
  memory layout (the strided path uses sqrtf).
- Clean, locked: HIP REJECTS bf16/f16 compute with a specific diagnostic;
  Metal emits typed `half`/`bfloat` kernels; f8e4m3 is rejected everywhere;
  the eval SCALAR lane rounds f16/bf16 per-op correctly.

### Sweep 2 (grad): the compounding question had a bad answer

- **[#722]** grad of a forward pass containing `abs`/`floor` on an integer
  tensor returns ALL-ZERO gradients in BOTH lanes - #699's placeholder
  poisons eval too, because grad is built over the lowered DAG. The forward
  pass without grad is correct in eval; the same gradient without `abs` is
  correct in both lanes. Both lanes agreeing on the wrong answer is
  invisible to any cross-lane oracle, fixed or not.

### Sweep 3 (matrix completion)

- **[#723]** the C tensor print helper renders int64 through double: an
  EXACT compiled sum of 2^53 + 1 prints as ...992.0. The C sum itself is
  right (provable via to_list, locked) - the printed output manufactures
  false evidence against the correct lane.
- **[#724]** `mean` of an int64 tensor: eval 187.5 (a fractional value
  inside an int64 tensor), C 187.0, checker score 1. Three stages, three
  answers.
- #682's stub confirmed at every width (comment posted); #692's reduce
  panics executed end-to-end at three emit.rs sites (comment posted).
- **Bool is clean** in both lanes (probed, not believed from the lib.rs
  storage comment) - EXCEPT **[#726]**: `add` on bool tensors stores 2 in a
  bool-typed tensor; print says 2.0, to_list says true, and Metal's
  honestly-typed kernel would say 1.

### Sweep 4 (Metal): best-behaved backend, confirmed and locked

Typed long/int/bool kernels, specific f64 rejection, self-naming rank-2
abort stub. #699's Metal symptom root-caused by emission: the `Const 0`
arrives pre-planted from lowering; no separate Metal bug (comment posted).
Metal RUNTIME execution remains unswept (needs a driver harness; this
machine could run it).

### Sweep 5 (#711 adjacents)

- **[#720]** `fold_static_cond`'s Cast arm folds f16/bf16 conditions with
  f32 semantics (via `convert_cast_data`) and DELETES the branch IEEE
  semantics would take, at thresholds 2049/257. Distinct from #711: the
  checked-i64 fix does not touch this arm.
- Bounded: effectful (`fail`) branches keep the def host-lane where the
  compiled int64 comparison is EXACT (locked); int8 conditions do not fold
  but diverge across lanes at runtime through #714/#718's widening.

### Sweep 6 (#709 adjacents) and the .dp path

- The wrapper battery confirms #709's scoping exactly: `with seed` /
  `with device` bodies are the ONLY checker holes (let/if/match/lambda/
  pipe/tuple/list/grad/vmap/jit all catch the same error; the handler
  expressions are checked). Locked as a canary test.
- Escalations (commented on #709): the masked error now provably reaches a
  RUNNABLE binary through the host lane, and `lower_handle_effect`'s
  catch-all is live today via .dp (`effect: teleport` builds and runs).
- **[#721]** found probing the controls: eval cannot ingest the canonical
  Deep of a NULLARY fn (`value is not callable`); check scores it 1 and the
  compiled lane runs it.

### Sweep 7 (#688, prove): executed

The class fires in the default build with no opaque type: prove's fuzz-tier
interpreter collapses int64 offsets 2^53+1 vs 2^53 and reports a TRUE
theorem failed with a spurious counterexample (x = 0). Comment posted; the
opaque produced-value chokepoint itself still needs the smt build.

### Sweep 8 (leftovers): the prediction was wrong

- **[#725]** `reduce_window`'s `unwrap_or_default()` (item 5) is REAL and
  silent - against the "every sibling was refuted, expect these to be too"
  prediction. Non-literal window+strides lower to empty lists and the
  compiled max-pool returns the UNPOOLED input with the wrong shape at
  check score 1; the half-non-literal case panics the emitter instead.
- Item 7 (named_axis `unwrap_or(Prim::F32)`, `assign_partition`): probed as
  far as the input surface reaches; basic partition is correct in both
  lanes, and both fallbacks need an internal desync. Left as inspected-only.

### What genuinely remains

1. **HIP runtime behavior** - unchanged from the brief: emission-only here;
   the gfx1151 box should run #689's probes via `scripts/hip_test.py`.
2. **Metal runtime execution** - emission is locked; running the kernels
   needs a small driver harness (this arm64 macOS machine can compile and
   run Metal).
3. **The #688 opaque chokepoint** - needs `--features smt` plus an @opaque
   int64-field type; the class is proven live in prove's fuzz tier either way.
4. **`with seed` SEMANTICS under compilation** (does seeding actually
   reproduce across lanes?) - noticed but not swept.
5. **Print-formatting divergence between lanes** (eval prints
   `1.4142135381698608`, C prints `1.414213538169861` for the same f32) -
   not a value bug, but the #687 exact-string oracle needs a per-op
   formatting contract before it can compare transcendental outputs.

### Class bookkeeping and prevention

Two further metas were filed once the findings were classified: **[#727]**
(no dtype's semantics are enforced at any single point - the generalization
of #695 that the narrow-float sweep forced) and **[#728]** (the observation
channel is not dtype-faithful - the #716/#723 class, and the missing
prerequisite for #687). The mechanism-by-mechanism plan for making all four
classes structurally unwritable is
[`numeric_audit_structural_prevention.md`](numeric_audit_structural_prevention.md),
and the raw probe corpus behind every finding AND every negative result is
archived under [`probes/`](probes/README.md).

[#714]: https://github.com/Chelis-Lang/chelis/issues/714
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#719]: https://github.com/Chelis-Lang/chelis/issues/719
[#720]: https://github.com/Chelis-Lang/chelis/issues/720
[#721]: https://github.com/Chelis-Lang/chelis/issues/721
[#722]: https://github.com/Chelis-Lang/chelis/issues/722
[#723]: https://github.com/Chelis-Lang/chelis/issues/723
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
