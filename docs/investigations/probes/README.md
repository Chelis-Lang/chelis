# Numeric-audit probe corpus (2026-07-16 sweep)

The raw probes behind the second-sweep findings (chelis#714-#728) and,
just as deliberately, behind the **negative results** - the probes that
showed no issue are archived here so nobody re-derives a cleared surface
from scratch or, worse, from reading the code. Everything in this
directory was executed against the workspace at the commit range of
PR #696's second round; the CI-runnable distillation (broken rows
`#[ignore]`d with issue numbers, controls green) lives in
`crates/chelis-cli/tests/` - these are the point-in-time artifacts,
**not maintained automation**, and are exempt from the phase-gate and
example-corpus policies (they are neither executable examples nor
gate scripts).

## Layout

| file | what |
|---|---|
| `probe.py` | the lane driver: `probe.py eval|check|c|hip|metal <file>` - eval runs `chelis eval --file`, `c` builds, links via the printed `Compile:` line(s), and runs; `hip`/`metal` are emission-only. All lanes return verbatim strings; nothing parses through f64. Set `CHELIS_BIN` (default `target/debug/chelis`); run from the repo root. |
| `battery.py` | table-driven runner: `python3 battery.py <bat_module>` prints one row per probe with both lanes' verbatim output. The `AGREE/DIVERGE` flag is naive (formatting-sensitive); read the values. |
| `bat_narrow.py`, `bat_narrow2.py`, `bat_narrow3.py` | the bf16/f16 sweep (chelis#714, #716, #717) plus narrow-int scalars/tensors (#718) and f8e4m3 rejection controls |
| `bat_scalar_ops.py` | the full scalar-op surface at f32/f64 that found the eight 0-stub ops (#715) and bounded them with the working-op controls |
| `bat_int_scalar_ops.py` | the stub family at integer dtypes; the mod/floor_div/trunc_div/abs/neg locks; the div/recip rejection controls |
| `bat_f32_tensor_round.py` | which f32 tensor ops skip narrowing in eval (#717: div/recip skip, tan/sqrt narrow) |
| `bat_matrix3.py` | bitwise at every width (#682), reductions across dtypes (#692 panics, #723 print, #724 mean), the Bool battery (clean except #726) |
| `checker_holes.py` | the #709 wrapper battery: which constructs hide an ill-typed body from `chelis check` (answer: only `with seed`/`with device`; the ten other wrappers and the handler expressions all catch it) |
| `fixtures/` | the hand-written probe programs, including every negative/cleared probe (see below). Files named `bat_*.ch` are NOT here - the batteries regenerate them. |

## Negative results worth as much as the findings

These fixtures executed clean or bounded a claim, and are kept precisely
so the clearance is reproducible rather than asserted:

- `part2.ch` - `partition` correct in both lanes (bounds the
  `assign_partition` leftover: its `/* unsupported partition type */` arm
  needs an internal desync to reach; not reachable from the input surface).
- `rw_nonlit.ch` - literal-window `reduce_window` correct in both lanes
  (the control that isolates #725 to the non-literal extraction).
- `fold_fail*.ch` - an effectful (`fail`) branch keeps a def host-lane
  where the compiled int64 comparison at 2^53 is EXACT; bounds #711/#720
  to the DAG-lane pure-const fold path.
- `nullary.ch`/`unary.ch` (+ generated `.dp`) - isolates #721 to nullary
  defs; unary Deep round-trips through eval fine.
- `prove_int64_ctl.ch` - prove's small-offset property passes 8/8; bounds
  #688's spurious counterexample to the f64 mantissa boundary.
- `bool_*.ch` - the Bool dtype battery: scalar logic, tensor round-trip,
  to_list, not, casts, cmplt all correct in both lanes (the storage
  comment at chelis-runtime lib.rs:144-152 was probed, not believed);
  `bool_add*.ch` is the one exception (#726).
- `hip_*.ch` - HIP cleanly REJECTS bf16/f16 compute (the control that
  bounds #689 to integer dtypes); f32 controls emit real kernels.
- `metal*.ch` - Metal emits honestly-typed kernels per dtype, rejects f64
  specifically, and abort-stubs rank-2 loudly; `metal_int64_abs` carries
  the pre-planted `Const 0` that root-causes #699's Metal symptom.
- `tctl_*.ch` / `f64t*.ch` - tensor forms of the #715-stubbed ops are
  correct, and f64 tensor `div` is exact in both lanes (bounds #717 to
  the unary wrapper and the f32 binary gaps).
- `i64_sum_exact*.ch` - the compiled int64 tensor sum is EXACT at 2^53+1;
  only the print lies (#723). The probe that stops #684's fix from being
  mis-scored against the C lane.
- `suffix_i8.ch` - int8 suffixed literals exist and eval wraps at width
  (fixture knowledge for #718/#720 probes).

## Reproduction

```sh
cargo build -p chelis-cli --bin chelis
python3 docs/investigations/probes/probe.py eval  docs/investigations/probes/fixtures/frac_f16.ch
python3 docs/investigations/probes/probe.py c     docs/investigations/probes/fixtures/frac_f16.ch
(cd docs/investigations/probes && python3 battery.py bat_scalar_ops)
```

The batteries write their generated fixtures next to themselves under
`probes/`; outputs are one table row per probe, both lanes verbatim.

The prevention analysis these probes fed is
[`../numeric_audit_structural_prevention.md`](../numeric_audit_structural_prevention.md);
the per-sweep outcomes are at the bottom of
[`../numeric_audit_next_sweeps.md`](../numeric_audit_next_sweeps.md).

## Test twins (added after the corpus landed)

Every battery's broken rows and the meaningful controls now have committed
test twins; the corpus remains the raw evidence and the negative-result
archive, but nothing is probe-only anymore:

| battery | test twin |
|---|---|
| `bat_narrow*.py` scalar rows | `narrow_dtype_matrix.rs` (incl. the full-surface row set `f16_bf16_scalar_op_surface_agrees_across_lanes`) |
| `bat_narrow.py` overflow rows | `int_width_lane_matrix.rs` (incl. `c_scalar_overflow_traps_at_every_width`) |
| `bat_scalar_ops.py` | `scalar_stub_matrix.rs` (f32 + f64 broken rows, f32 + f64 working controls) |
| `bat_int_scalar_ops.py` | `scalar_stub_matrix.rs` int rows |
| `bat_f32_tensor_round.py` | `eval_tensor_narrowing_matrix.rs` (broken rows + the tan/sqrt do-narrow control) |
| `bat_matrix3.py` | `reduction_and_bitwise_matrix.rs` |
| `checker_holes.py` | `issue_709_handle_effect_and_dp_roundtrip.rs` |
| `tostring_*.ch` | `issue_734_tostring_placeholder.rs` |
