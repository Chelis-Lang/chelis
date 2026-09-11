# chelis#1829: the eval slowdown at `22b193cf7` (PR #1693)

## 1. Mechanism

Per applied def name the interpreter asks if that def is a C-lane kernel. The answer comes from
an **unmemoized, exponentially branching probe**:

```
EvalContext::def_kernel  (compiler-api/src/runtime/eval.rs:304)
 -> host_def_kernel -> def_body_decision -> expr_calls_summary_rejecting_top_level_fn
   -> top_level_fn_helper_summary_rejects   (chelis-ir/src/host.rs:10500)
     -> lower_host_function   <-- fully lowers the callee just to read its summary
       -> lower_def_body_kernel -> def_body_decision -> ...   (per call site)
```

Work is a tree expansion over the call graph, not a walk of it: fan-out^depth. A memo
(`HELPER_SUMMARY_REJECTS_CACHE` and six siblings) collapses it to linear, but all seven are gated
on thread-local `HOST_LOWERING_CACHE_ACTIVE`, whose only setter is
`HostLoweringCacheGuard::begin()` at `host.rs:1593`, inside the C-lane whole-program lowering.
The interpreter never arms it.

**What #1693 changed.** `evaluate_host_program_with_library_and_types` now builds
`CheckedProgram::compose(library, program)` and installs it as `EvalContext.program`
(`runtime/mod.rs:570`). Before, that field held new code only, so
`host_def_kernel("parse_value")` found no such def and returned `Ok(None)` at once. After, every
`Std.Io.Json` definition is visible and the uncached exponential probe runs over a ~50-def
mutually recursive parser. #1693 did not author the exponential; it removed the
accidental guard that kept the interpreter away from it, and the composition is a correctness fix.

## 2. Evidence

Debug binaries, `chelis eval --file`, fresh package dir with `reef.lock` per measurement per
commit, own `CHELIS_REEF_HOME`. Main program: `parse_json` of a 27-byte object through
`Std.Io.Json`, nullary `def main()`.

| program | check parent | check culprit | eval parent | eval culprit |
|---|---|---|---|---|
| small.ch (json) | 2.53 s | 2.54 s | **2.10 s** | **TIMEOUT >150 s** |
| row_json_io_e2e | 1.19 s | 1.22 s | 2.28 s | TIMEOUT >90 s |
| row_json_missing_path | 1.46 s | 1.07 s | 3.00 s (rc=1, intended) | TIMEOUT >90 s |
| row_json_bigint | 1.23 s | 1.54 s | 2.71 s | TIMEOUT >90 s |

The front end is flat. `sample <pid> 5` on the stuck culprit is 2925 samples on one stack: 97
`eval_app` frames, then the §1 chain repeating to the sample floor. **All three other #1829 rows
share the mechanism**; I ran each row's program, not its nextest row.

**Cache trap retired.** `~/.cache/chelis` is workstation-wide and `CHELIS_REEF_HOME` does not
isolate it, but it cannot explain these numbers. The parent's 2.10 s came first, cold; the
culprit ran second, when the cache could only be warmer, and still timed out. Re-run last in a
fresh package dir against that warm cache, the pristine culprit still times out past 150 s and
the parent still returns in 2.37 s.

## 3. Reversal experiments

Branch `diag-1829-exp`, local only, deleted after; both worktrees end detached at their exact
commits with clean `git status`.

- **A, `a472079c3`, neutralise the hunk.** `program: Some(program)` for
  `Some(kernel_program.as_ref().unwrap_or(program))`. **eval 1.93 s**, output `main = Some(22)`
  identical. Parent timing restored; the hunk is confirmed.
- **B, `b828279ca`, keep the composition, arm the memo.** Exposed `HostLoweringCacheGuard::begin`,
  held one across `evaluate_host_program_with_library_and_types`. **eval 2.19 s**, same output,
  composition retained. B is decisive: the composition is not the cost, the unmemoized probe is.

## 4. Scaling: superlinear, and pre-existing

Synthetic new-code chain `f_i(x) = add(f_{i+1}(x), f_{i+1}(x))`, no imports. Eval on the
**parent**: 1.87 s at 7 defs, 3.38 s at 9, 19.07 s at 11, TIMEOUT >90 s at 13.
About x4-5 per two levels added; `check` stays ~1.1s. Two controls separate probe from
runtime: `g_i(x) = if gt(x,0) then g_{i+1}(x) else g_{i+1}(x)`, 11 runtime calls and 2^10 probes,
still takes **23.76 s on the parent**; the same 11- and 13-def chains under B take **1.90 s /
2.33 s**. Cost is per distinct def name probed, so it does not scale with document size.

## 5. Among #1693's six issues

#1829 is the sixth (#1746, #1747, #1775, #1776, #1779); the squash carries at least two
independent new mechanisms. #1775 and #1779 name the other in their failure text: the
new 676-line `host/staged.rs` region lowering. **Arming the memo does not touch that path and closes
neither.** They do not interact here: `staged_def_kernel` returns `NotApplicable` at `host.rs:3172` for any
def whose return type is not a tensor, as every `Std.Io.Json` def's is.

## 6. Repair shape

Arm the existing memo on the interpreter path; do not undo the composition. Two files:

- `crates/chelis-ir/src/host.rs`: expose the guard and make `HostLoweringCacheGuard`
  **nesting-aware**. `begin()` clears, and `Drop` clears and sets `ACTIVE=false`
  unconditionally, so an interpreter-held outer guard would be disarmed mid-flight by a nested
  `try_lower_compiled_program_with_lane_overrides`. Refcount it. This is the one real hazard, and
  B did not test it. Deeper, and its own issue: this probe should not run a full lowering to
  answer a summary question.
- `crates/chelis-compiler-api/src/runtime/mod.rs`: hold the guard across
  `evaluate_host_program_with_library_and_types`, for the `kernel_program` lifetime.
  `invariant.rs:826` and `tests.rs:868` build `EvalContext` too.

**Overlap with the #1277 closeout PRs: none at the repair sites.** Nine commits touched `host.rs`
since `22b193cf7` and six touched `runtime/mod.rs`/`eval.rs`, but both sites are byte-unchanged
on current `origin/main`. `lower.rs` and `axis_sources.rs` are untouched.

## 7. Red-first receipt

`HostWorkProfile.helper_summary_builds` counts uncached probe builds, and
`issue_1205_host_lowering_work_is_linear` (`host.rs:16046`) asserts it equals 2, but drives the
C-lane entry where the guard is armed, so it never measured this path.

1. **Class receipt** (fails on parent *and* culprit): a sibling driving `host_def_kernel` per
   def over a synthetic fan-out-2, depth-12 program, asserting
   `helper_summary_builds <= def_count`. Today ~2^12.
2. **Regression receipt** (passes on parent, fails on culprit): the same bounded count from a
   `chelis-compiler-api` test driving `evaluate_host_program_with_library_and_types` with a small
   mutually recursive *library* and a new-code caller. This pins #1693's exposure.

Counters, so neither flakes under load.

## 8. Not validated

- Whether nesting-aware guarding suffices for every re-entry into
  `try_lower_compiled_program_with_lane_overrides` from an eval. The chief risk.
- Whether the memo's `program as *const _ as usize` key survives the interpreter's longer
  lifetime (a freed address could be reused). B ran green but did not probe it.
- Release timings and the nextest rows. All of this is `dev`, and every culprit figure is a
  lower bound: no culprit JSON run was observed completing.
