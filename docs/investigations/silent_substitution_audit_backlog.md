# Silent-substitution audit: unverified backlog

Status as of 2026-07-15. Companion to the two tracking issues:

- **[chelis#695]** - integers have no representation in the numeric layer
  (integer arithmetic laundered through `f64`).
- **[chelis#703]** - unsupported cases silently substitute a value instead of
  failing, so wrong programs compile and run.
- **[chelis#709]** - the checker's analogue: a construct with no `infer.rs`
  case types as silent `Type::Error` and disables type checking for its body.

This file records what the audit **did not** finish, so it is not lost. Every
item below is either unverified or unswept. Verified findings live in the
issues, not here.

## Ground rules learned the hard way

Five claims in this audit came from reading code (and code comments) and were
**refuted by executing it**. Do not promote anything below to an issue without
running it.

1. "The C backend corrupts int64 literals via `RiscOp::Const{value: f64}`" -
   false; that is the DAG *tensor* lane, which rejects int64. Scalars use the
   host lane and are exact.
2. "int32 tensors lose precision above 2^24 because the C runtime stores them
   as f32" (per the verbatim comment at `crates/chelis-runtime/src/lib.rs:144-152`) -
   not reproducible; exact in both lanes.
3. "`optimize::constant_fold` bakes wrong integer constants" - true of the
   code, but the pass has zero non-test call sites. Dormant.
4. "A one-line neural-network layer returns zero" (an early draft of #704) -
   false. Tensor `relu` is correct; only the **scalar** overload stubs.
5. "The unknown-Deep-tag catch-all silently returns the last trailing child" -
   false; the parser enforces the 62-tag vocabulary first (item 1 below).

All four are locked as refutation tests in
`crates/chelis-cli/tests/precision_matrix.rs` and
`crates/chelis-cli/tests/issue_703_silent_placeholders.rs`.

## Unverified, ranked by likelihood of being real

### 1. The unknown-Deep-tag catch-all - **REFUTED** (chelis#710)

`crates/chelis-ir/src/lower.rs:4926-4938`. A sweep ranked this its single worst
finding: a misspelled Deep form would be silently accepted and its last trailing
child returned.

**Refuted by execution.** The Deep parser enforces the 62-tag closed vocabulary
before lowering ever runs:

```
$ chelis check unk.dp
"score": 0
"errors": [{"kind":"Other","message":"expected valid Deep tag, found unknown tag
  'totally-bogus-tag'. Not in the 62-tag vocabulary at byte 42","severity":0.5}]
```

The catch-all is dead code behind a real guard. This was the fifth claim in the
audit sourced from reading code and refuted by running it.

### 2. Malformed `.dp` type-checks clean - **CONFIRMED, far milder than claimed** (chelis#710)

The mechanism is real: `infer.rs`'s arity guards return `Type::Error` **without**
pushing a `CheckError`, so `chelis check` reports `"score": 1, "errors": []` on
malformed Deep.

But the sweep claimed fourteen guards were live. Executed, **4 of 6 tested are
correctly rejected**:

| form | `chelis check` | `chelis eval` |
|---|---|---|
| `(def {} orphan)` | **score 1, errors []** | `error: unknown runtime name 'orphan'` |
| `(cast {} (lit ...))` | **score 1, errors []** | `error: cast missing target type` |
| `(let {} (bind {} x))` | score 0 - rejected | - |
| `(fn {} (params {}))` | score 0 - rejected | - |
| `(app {})` | score 0 - rejected | - |
| `(if {})` | score 0 - rejected | - |

And critically: **the `Const 0.0` placeholders are never reached**. Both silent
forms fail loudly at eval. This is a false green from `chelis check`, not a
silent-wrong-answer generator. The predicted "#695 and #703 meeting" does not
happen.

Items 4, 5, 6 and 8 below are all `.dp`-reachability claims from the same sweep
that ranked items 1 and 2 highest. Given that item 1 was refuted and item 2
over-stated by 7x, **discount that sweep's `.dp` reachability estimates
accordingly** and execute before believing any of them.

### 3. HIP's other seven `elem_kind` call sites - **SETTLED, all confirmed + one extra** (chelis#689)

Probed by building to HIP and reading the emitted kernel names (no GPU, no
`hipcc` needed). An `_f32` name on an int64 program is the fallback firing.

| op | emitted kernel | verdict |
|---|---|---|
| `add` | `kernel_add_i64` | correct - typed template (control) |
| `mul` | `kernel_mul_i64` | correct - typed template (control) |
| `neg` | `kernel_neg_f32` | F32 fallback |
| `max_elem` | `kernel_max_elem_f32` | F32 fallback |
| `min_elem` | `kernel_neg_f32` + `kernel_fused_3` | F32 fallback |
| `min_reduce` | `kernel_min_ax0_f32` | F32 fallback |
| `max_reduce` | `kernel_maxred_ax0_f32` | F32 fallback |
| `prod_reduce` | `kernel_prod_ax0_f32` | F32 fallback |
| **`sum`** | **`kernel_sum_ax0_f32`** | F32 fallback - **not predicted** |
| `abs` | `kernel_fill_f32` | #699's zero placeholder reaching HIP |

**8 broken, 2 correct.** `sum` was *not* on the suspect list: `emit.rs:1121-1126`
claims "For Add/Mul/**Sum** ... each of those arms resolves the right template
inline rather than touching the float-only `elem_kind` shorthand", and a
verification agent accepted that claim without probing it. It is true for Add
and Mul, false for Sum. Proof: `sum` on `tensor[2,2,int64]` and on
`tensor[2,2,f32]` emit the **identical** kernel name, so the int64 sum *is* the
f32 kernel.

That also bypasses `RiscOp::Sum`'s `accumulator: Prim` field
(`crates/chelis-ir/src/dag.rs:719-735`), which is documented as implementing
spec §5.7's precision-widening rule and is checked by `verify`.

Lesson: the one op an agent cleared by reading a comment was the one extra bug.

### 4. `Atom::Keyword` -> `Const 0.0` - **REFUTED** (chelis#710)

Guarded at runtime: a bare keyword atom in expression position gives
`error: bare atom is not a runtime expression`. The `Const 0.0` never fires.
`chelis check` does false-green it (score 1), which is #710.

### 5. `reduce_window` `unwrap_or_default()` - **NOT TESTED**

`crates/chelis-ir/src/lower.rs:7522-7537`. Low priority: every sibling claim from
the same sweep was refuted, and the pattern below suggests a runtime guard exists.

### 6. `lower_cast`'s `Prim::parse_name(...).unwrap_or(Prim::F32)` - **REFUTED** (chelis#710)

Guarded at runtime: a typo'd cast target gives
`error: cast target 'bogus_dtype' is not a recognized primitive type` rather than
silently becoming f32. `chelis check` false-greens it (#710).

### 7. "Should never happen" defaults - **NOT TESTED**

`named_axis.rs:430`'s `unwrap_or(Prim::F32)` and `host_emit.rs:4060-4070`'s
`assign_partition`. Both require an internal desync or a type-inference hole to
reach. Lowest priority.

### 8. `fail(non_literal_string)` outside an `if` - **REFUTED**

Works correctly in **both** lanes:

```
def boom(msg: string) -> tensor[2, f32] = fail(msg)
def run() -> tensor[2, f32] = boom(string_concat("dynamic ", "message"))
```
```
chelis eval  : error: dynamic message
compiled C   : exit 1, stderr "dynamic message"
```

The zero placeholder does not fire; `fail` aborts with the dynamic message as
intended.

## The pattern across every `.dp` / lowering-placeholder claim

Items 1, 4, 6 and 8 are refuted; item 2 is confirmed but low-severity. In **every**
probe, the shape was the same:

> **the `Const 0.0` / `unwrap_or` placeholder is never reached** - a parser or
> runtime guard catches the input first - while **`chelis check` false-greens it**.

So the 26 `Const { value: 0.0 }` sites in `lower.rs` are, on this evidence,
almost entirely dead code behind real guards. The genuinely live instances of the
#703 class are the ones found by other routes:

- **#699** `lower_transcendental`'s non-float branch (Surf-reachable, 7 ops x 4 int widths)
- **#682 / #704 / #705** `host_emit.rs:2300`'s `/* unsupported builtin */ 0`
- **#689** HIP's `elem_kind` F32 fallback

and the real residue of the `.dp` work is **#710** (checker false green), which is
low severity because the runtime always catches it.

## Pathways not swept at all

- **HIP / Metal runtime behavior.** Everything GPU is emission-inspection only;
  no GPU and no `hipcc` on the audit machine (arm64 macOS). #689 is proven by
  emitted source, not by running it. `docs/local_hip_environment.md` describes
  the gfx1151 box that could.
- **`chelis prove` Tier C** (#688). Code-confirmed (`ExecutionValue::Int64 { value } => *value as f64`
  at `obligation_engine.rs:1857` and `opaque.rs:1536`), never executed.
- **The `.dp` ingestion path generally.** A whole front end that bypasses Surf's
  structural guarantees. #709 proves the checker behind it is weaker than
  assumed. Items 1, 2 and 4 all live here.
- **`grad` / `vmap` interaction - SWEPT, negative result.** `transforms.rs:619-630`
  really does marshal int64 scalars through `as f64`, but it introduces **no new
  bug**: a `vmap` over an int64 tensor carrying `2^53+1` returns
  `9007199254740992.0`, and the **no-vmap control returns the same**, so the
  corruption is #684's `Vec<f64>` storage, not the marshalling. Likewise a
  `grad`-shaped probe with an exact int64 scalar fails identically **without**
  `grad`, because `sub(2^53+1, 2^53)` already returns `0` (#680). grad/vmap
  inherit #680 and #684 rather than adding a third corruption.
- **Metal's `Const 0`.** #699 explains the C lane's `lower_transcendental` zero;
  Metal's identical symptom was never re-confirmed to share that root cause
  after the diagnosis landed.
- **`chelis-cli`'s parallel pipeline - SWEPT, negative result.** Enumerated every
  `reject_*` / gate fn in both crates and compared. Bounded, and smaller than
  feared:

  | gate | status |
  |---|---|
  | `reject_unsupported_hip_ops` | duplicated, **materially drifted** (CLI permissive) - chelis#698 |
  | `reject_host_only_builtins` | compiler-api **only**, no CLI equivalent - chelis#705 |
  | `reject_unsupported_reduce_window_precision` | duplicated, **semantically identical** (differs only in error construction) |
  | `reject_symbolic_windowed_reduce` | duplicated, **semantically identical** |
  | `reject_with_seed_for_build_target` | a no-op - but **intentional and honestly documented** ("Today it is a no-op", a forward-compatibility hook per Bucket-5 closure). Not a bug; the opposite of #694. |

  CLI-only gates (`reject_unsupported_c_precisions`/`_host`,
  `reject_unsupported_metal_ops`, `reject_unsupported_effect_ops`,
  `reject_eval_only_builtins_host`) have no compiler-api twin, so they cannot drift.

  **Zero new bugs.** The two real instances (#698, #705) were already found by
  accident. The prediction that this was "the highest-yield unswept area" was
  **wrong** - but the sweep is still worth having: the parallel-pipeline hazard is
  now bounded at exactly two live instances plus two benign duplicates, rather
  than an unknown mess.

## Recommended next step

Items 1 and 2 are **settled** (2026-07-15): item 1 refuted, item 2 confirmed but
low-severity and 7x narrower than claimed. Both are recorded in chelis#710.

Item **3** is settled (all 8 confirmed, plus `sum` as an unpredicted extra), and
the **`chelis-cli` parallel-pipeline sweep** is done with a negative result - both
recorded above.

The backlog is **effectively exhausted**. Remaining, both low priority:

1. **Items 5 and 7** - `reduce_window`'s `unwrap_or_default()` and the two
   "should never happen" defaults. Every sibling claim from the same sweep was
   refuted, and each of these needs an internal desync or inference hole to reach.
2. **`chelis prove` Tier C** (#688) - code-confirmed, never executed. Needs an
   `@opaque` type with an int64 representation field, which today's std lib does
   not have.

## Final scoring

| area | prediction | outcome |
|---|---|---|
| HIP `elem_kind` probes | high yield | **paid off** - all 7 confirmed **plus `sum`**, which no one predicted because a comment said it was safe |
| `chelis-cli` gate sweep | "highest-yield unswept area" | **zero new bugs**; hazard bounded at 2 known instances |
| `.dp` placeholder items (1, 2, 4, 6, 8) | 1 and 2 ranked highest by a sweep | **4 of 5 refuted**; the survivor (#710) is low severity |
| `grad` / `vmap` | "concrete reason to look" | **no new bug**; inherits #680/#684 |

Of the claims ranked highest by automated sweeps, **most were wrong**. The one
real extra (`sum`) was found in the area an agent had explicitly *cleared* by
reading a comment. Weight source-derived rankings accordingly, and probe the
things that were cleared.

[chelis#695]: https://github.com/Chelis-Lang/chelis/issues/695
[chelis#703]: https://github.com/Chelis-Lang/chelis/issues/703
[chelis#709]: https://github.com/Chelis-Lang/chelis/issues/709
