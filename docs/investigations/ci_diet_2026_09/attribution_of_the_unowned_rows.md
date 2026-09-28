# chelis#1776: attribution of the six unowned rows

Reviewed head `7e3d01ea6`. Worktree `~/chelis-worktrees/1776-attribution`, detached,
final `git status --porcelain` empty, HEAD restored to `7e3d01ea6`.

## One answer for all six rows

**Every one of them was broken by `22b193cf7`, "Preserve computed reshape claims and
broadcast unit checks through calls" (#1693).** Its parent `062c29c19` is good on every
signal; `22b193cf7` is bad on every signal. That parent is Jeff's CI change, so these
targets left the per-PR path exactly one commit before the commit that broke them.

The subject line names none of this. The squash is 74 files, +6835/-519, and carries
`fix(compiler): use the HIP-specific lane query signature` and a new 676-line
`crates/chelis-ir/src/host/staged.rs` among its commit messages.

## Method

Not bisect-by-test. I built one four-signal probe that reads all rows off a single
`cargo build -p chelis-cli --bin chelis`: the baked `CHELIS_EFFECTIVE_UNIFORM_SEED`
argument, whether `pad_hip.cpp` contains `kernel_pad`, and main-body retain/release
counts for both #1222 sources, plus `distinct_top_level_tensors_are_each_freed`'s source
as a negative control (it must read `(2,4)`, its passing expectation, and it did at
every probe point). Every signal is tri-valued -- value, `BUILD-FAILED`, or
`CARGO-BUILD-FAILED` -- so an uncompilable commit can never read as `seed=0` or
`hip=NO-KERNEL`. No probe point returned an unmeasurable reading.

This is a sampled ladder, not a bisect: a bisect finds one boundary, and each build
here records all four signals at once, localizing every boundary simultaneously. Six
probes over `12c04c66a..0e858f184`:

```
12c04c66a  seed=7  hip=kernel_pad  alias_if=(5,6)  alias_ws=(2,3)   good
010354f91  seed=7  hip=kernel_pad  alias_if=(5,6)  alias_ws=(2,3)   good
8203c32e4  seed=7  hip=kernel_pad  alias_if=(5,6)  alias_ws=(2,3)   good
062c29c19  seed=7  hip=kernel_pad  alias_if=(5,6)  alias_ws=(2,3)   good   <- #1693's parent
22b193cf7  seed=0  hip=NO-KERNEL   alias_if=(3,5)  alias_ws=(3,4)   bad    <- #1693
0e858f184  seed=0  hip=NO-KERNEL   alias_if=(3,5)  alias_ws=(3,4)   bad
7e3d01ea6  seed=0  hip=NO-KERNEL   alias_if=(3,5)  alias_ws=(3,4)   bad    (head)
```

`-S` attributed nothing here and would have misled: it pointed me at
`3dc3f54f6` (#1307), whose `codegen_hip_host_program` device-emits only `Count`-bearing
helpers. Measurement refuted that — #1307's parent already had no `kernel_pad`.

## Row by row

**Row 2, `redteam_baked_seed_deterministic_and_cross_lane` — now PASSES at `7e3d01ea6`.**
Your table measured `e813415d0`. `2c4af13f9` (#1809) deleted its `7ULL` assertion and
replaced the doc comment with "a former macro spelling is not the execution contract",
citing #1799. That was a test update accommodating #1693's behaviour change, filed under
neither. It left row 1's byte-identical assertion in `cli.rs` untouched.

**Row 1, `build_c_with_seed_uniform_like_succeeds` — #1693 (`22b193cf7`), class (A).**
New spelling, from `seeded.c` I generated at head:

```c
508:    uint64_t t2_seed = CHELIS_EFFECTIVE_UNIFORM_SEED(0ULL);
579:    __seed_10 = 7;
581:    chelis_rng_state __rng_seeded_12 = {(uint64_t)__seed_10, 0ULL, 1};
582:    *__chelis_rng = __rng_seeded_12;
```

At `062c29c19` the same line read `CHELIS_EFFECTIVE_UNIFORM_SEED(7ULL);`. The seed is
still routed; it is threaded through the invocation-local RNG state instead of baked at
the op site (`COMPILED_HANDLER_OWNED_SEED = 0` in `lower.rs:17`). Correctness oracle: row
2, which asserts eval-vs-C bit equality and run-to-run determinism on the same seeded
program, passes at head. Repair: this test should match what #1809 already did to its
twin.

**Rows 3 and 4, `build_hip_emits_pad_kernel` / `_shrink_kernel` — #1693, class (B).**
Not a respelling: the HIP emitter still produces `kernel_pad{suffix}` /
`kernel_shrink{suffix}` (`chelis-backend-hip/src/emit.rs:1475,1479`); the node stops
reaching it. At `062c29c19`, `pad_hip.cpp` carried
`extern "C" __global__ void kernel_pad(` with `chelis_prepare_kernel_launch`, and
`shrink_hip.cpp` carried `kernel_shrink`. At head both contain **zero** `__global__` and
zero kernels; the pad is a host loop over

```c
491:    chelis_movement_plan *t1_movement = chelis_tensor_affine_plan(t0, ..., CHELIS_MOVEMENT_PAD);
```

emitted by the **C** backend (`chelis-backend-c/src/emit.rs:6373`). `chelis build --target
hip` on a pad or shrink program now emits no device code at all and runs it on the CPU.
Answers stay correct; the device lane is gone. A `gather` control still emits
`kernel_gather_i64_f32`, so this is specific to the movement ops that moved onto the
staged host-helper path, not a general loss of HIP codegen. Note the direction on #1307: its parent
`0e858f184` already emitted no `kernel_pad`, so it narrowed nothing. Its
`codegen_hip_host_program` added a device lane *back* for `Count`-bearing helpers, and
its comment ("Other tensor helpers keep their C-host disposition") describes a state
#1693 had already created.

**Rows 5 and 6, `issue_1222_root_alias_ownership` — #1693, class (A). There is no
imbalance, and nothing reaches user-runnable code.** I linked and ran both emitted
programs under the runtime's ownership ledger (`release_header` traps "compiled ownership
ledger detected invalid tensor release"): both print correct values and exit 0. Hand
ledger agrees — every allocation nets to exactly zero, no over-release, no leak.

- Row 5 `(3,5)`: `c = if true then a else d` is now constant-folded to a tensor helper
  whose body is `outputs[0] = inputs[0];`, so the two arm clones the test names are
  genuinely absent. **A runtime condition still emits them: I built
  `flag = 3i64 > 2i64; c = if flag then a else d` at head and got `(5,6)`, identical to
  `062c29c19`.** The arm-clone ownership machinery is intact; only the literal-condition
  fold is new. Repair is the pinned pair.
- Row 6 `(3,4)`: the `with seed` scope, which folded away entirely at `062c29c19`, now
  materializes, and its block close emits one extra
  `chelis_tensor_retain(__binding_0_value);` with a matching release. Repair is the
  pinned pair.

## Unattributed

None. All six rows measured to one commit.

All three boundaries coincide at one confirmed parent/child pair, so no interval is
being reported as if it were a first-bad commit.

## Two observations, no action taken

I repaired nothing, per the brief. `22b193cf7` is 74 files and ~6,800 lines in one
squash; six regressions across three unrelated surfaces landed inside it with no red
check anywhere. If #1776 gets a size finding, that is where it goes. Separately, rows 3
and 4 are a silent device-lane regression rather than a stale test, so they do not belong
in a "tests are red" list — they want their own `area:backend` issue.
