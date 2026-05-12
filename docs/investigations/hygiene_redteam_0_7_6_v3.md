# 0.7.6 toolchain hygiene red team -- v3 final report

## Executive summary

Validated against `origin/main` at HEAD `6fd007dff1ba676410e580c050c211c3a0b8bbc9`
(`fix: linearity-F3 PR 2 -- flip module-wrapped violations from warnings to
errors (#68)`); the log includes #64, #65, #67, and #68 as required.

V3 confirms every V2 fix landed correctly on `main` (V2-F2 cast-tensor in
eval, V2-F3 prefer-pipe trigger/emit symmetry, V2-F4 var-RHS aliased
fan-out, CBackend-CastMemcpy #64, CBackend-ReshapeMemcpy #67, Linearity-F3
module-wrapped errors #65/#68) and surfaces **two new findings**: a
MEDIUM-severity linearity false-negative where `y = x; realize(y); add(x, z)`
silently bypasses the use-after-consume check that the direct
`realize(x); add(x, z)` form catches, and a LOW-severity round-trip break
where `chelis surf` (Deep decompiler) lowercases the interior letters of
compound PascalCase module names (`HelloTensor` becomes `Hellotensor`),
producing decompiled source that fails the workstream's own
`module-pascal-components` style gate.

Recommendation: **close the workstream**. Track the two new findings as
§5 follow-ups (Linearity-F4 alias-consume bypass and Decompile-F1 module
PascalCase loss). The two are both pre-existing in origin but were not
previously surfaced; the V3 finding scope is consistent with a workstream
in stable convergence.

## V2 validation table

| Finding / surface | Verification | Status |
|---|---|---|
| V2-F2 cast tensor in eval | `chelis eval --file` on `cast(to_tensor([1.5, 2.5, 3.5]), f64)` returned `tensor(shape=[3], data=[1.5, 2.5, 3.5])` with exit 0. | Verified fixed |
| V2-F3 prefer-pipe trigger/emit symmetry | `chelis lint --check` on `y = add(mul(x, x), x)` exits 0 with no warning; `chelis lint --fix` exits 0 with no rewrite; repeated `--fix` converges. | Verified fixed |
| V2-F4 var-RHS aliased fan-out | Bare top-level `x = ...; y = x; a = mul(y, x); b = mul(x, x); add(a, b)` checks score=1.0 with `errors: []` and evals to correct values. Module-wrapped sibling also passes. | Verified fixed |
| CBackend-CastMemcpy (#64) | `chelis build --target c` on `cast(tensor[n, f32], f64)` emits an element-wise `((double*)t1->data)[i] = (double)((float*)t0->data)[idx];` loop; compiled binary on input `[1.5, -2.7, 3.14159, -0.0]` returned bit-correct f64 values (`bits=0x3ff8000000000000` for 1.5, etc.). Regression suite `cargo test -p chelis-cli --test cbackend_cast_memcpy` 4/4 PASS. | Verified fixed |
| CBackend-ReshapeMemcpy (#67) | Regression suite `cargo test -p chelis-cli --test cbackend_reshape_memcpy` 3/3 PASS (`cbackend_reshape_tensor_f64`, `cbackend_reshape_tensor_int64`, `cbackend_reshape_tensor_f32_control`). | Verified fixed |
| Linearity-F3 (#65, #68) | `module Test; x = to_tensor([...]); y = realize(x); b = add(x, y)` reports `score=0.8` with `{"kind":"UseAfterConsume","message":"variable \`x\` was already consumed by realize at offset 0..."}` in the JSON `errors` array. `cargo test -p chelis-types --test linearity_module_wrapped` 6/6 PASS. | Verified fixed |

## NEW findings

### NEW-F1: Linearity false-negative on aliased consume sites (MEDIUM)

(a) What tested. Compared two semantically identical programs:

```
# Direct -- correctly rejected today
x = to_tensor([1.0, 2.0, 3.0])
z = realize(x)
result = add(x, z)

# Aliased through V2-F4 var-RHS borrow -- silently accepted
x = to_tensor([1.0, 2.0, 3.0])
y = x
z = realize(y)
result = add(x, z)
```

Probed bare-top-level and `module Bypass; def main() -> tensor[n, f32] = { ... }`
shapes. Also probed `y = x; a = realize(y); b = realize(x); add(a, b)`.

(b) Expected. Per `spec/05-risc-primitives.md` line 28
(`Consuming operations such as realize and explicit drop keep owned
parameters`) and `crates/chelis-types/src/linearity.rs::check_realize`,
`realize(y)` consumes its argument. Since `y = x` is a var-RHS binding
that the V2-F4 fix tags with consume-site `binding y at offset N` (a
non-structural consume that `read_or_error` tolerates), the linearity
checker should still recognize that the post-realize state of the
underlying value forbids later borrow-reads of `x` -- otherwise the
alias `y = x` becomes a trivial bypass of every use-after-consume
check.

(c) Actual. The direct form errors with
`UseAfterConsume: variable x was already consumed by realize`. The
aliased form passes with `score=1.0`, `errors: []`, and `chelis eval`
runs to completion. Tested at HEAD `6fd007d` with release binary; the
module-wrapped variant also accepts despite Linearity-F3's new
module-walk coverage. Reproduces both as bare-top-level and inside
`module Bypass; def main() = { ... }`. Mechanism: `consume_var_expr(y, realize_site)`
transitions `y`'s linearity-scope entry to Consumed, but `x`'s entry
stays Live -- linearity tracks names, not the underlying IR value,
while the V2-F4 fix collapses both names onto the same `Load` node at
the IR level via `lower_let`'s cached binding lookup. Runtime is
saved by the DAG-level `insert_copy_nodes_for_consuming_fanout`
pass (`crates/chelis-ir/src/lower.rs::insert_copy_nodes_for_consuming_fanout`),
so eval and the C backend produce correct values, but the safety
check is bypassed for any program shape where a primitive's
linearity-level borrow follows a realize/drop/store/cast through an
alias.

(d) Severity. MEDIUM. Silent acceptance of programs the
spec-of-record use-after-consume invariant says should be rejected.
No miscompilation in the current C backend because the IR-level
fan-out copy compensates, but the linearity contract is the
spec-level guarantee, not the runtime-level "we happen to copy."
A future backend that takes `realize` literally (e.g. moves to
destructive in-place rewrites for HIP fan-out) would surface as
data corruption without warning. Closure: in
`crates/chelis-types/src/linearity.rs`, alias-consume sites need to
forward the structural consume to the source name in the linearity
scope as well, so `read_or_error` on the original sees Consumed.
A typed `ConsumeKind { Aliasing, Structural }` (already the §5
Linearity-F1 follow-up) plus tracking the alias-target name on
each var-RHS binding would close it cleanly.

### NEW-F2: Decompile loses compound PascalCase module names (LOW)

(a) What tested. Sweep over `examples/*.ch` running
`chelis deep <file> > x.dp && chelis surf x.dp > x.rt.ch` and then
`chelis check x.rt.ch`. Verified the specific case with
`module HelloTensor` (canonical) and `module FooBar` (synthetic).
Inspected `crates/chelis-surf/src/desugar.rs::lower_module_path`
(`path.to_ascii_lowercase()`) and
`crates/chelis-surf/src/decompile.rs::capitalize` (capitalizes
the leading character only).

(b) Expected. Per `CLAUDE.md` Contract Invariants
("decompiler output must round-trip through the supported parser
path"), `chelis surf` output should not fail any gate the input
satisfied. The original `module HelloTensor` passes
`chelis check`; the decompiled `module Hellotensor` does not.

(c) Actual. `module HelloTensor` desugars to `(module {} hellotensor ...)`
in Deep (PascalCase compound information is destroyed at desugar
time). The decompiler's `capitalize` only Title-cases the first
character, producing `module Hellotensor`. `chelis check` and
`chelis eval` then reject the decompiled file with
`module-pascal-components (§6.3): module component Hellotensor matches
the lowercase-after-first form of a known PascalCase compound`.
Confirmed against `examples/hello_tensor.ch` -- the round-trip
fails the style gate on the workstream-touched
`module-pascal-components` rule.

(d) Severity. LOW. Decompiled output is not the canonical
serialization path; users who run the round-trip can recover by
manual edit or by `chelis fmt`. But the V2 PR #24 (Vocabulary-F2)
shipped the `module-pascal-components` lint precisely because
canonical PascalCase compounds are spec-mandated -- the decompiler
emits source that fails that lint on its own canonical inputs.
Closure: either (i) preserve module-name casing through Deep IR
(promote the symbol to a metadata-decorated form or change the
desugar contract), or (ii) document the round-trip break and add
the lint allowlist hook to skip decompiled outputs.

## Negative-result probes (nothing surfaced)

Specific surfaces probed where V3 did not produce a finding; each
listed test was actually run and exits cleanly:

- `chelis fmt` idempotency sweep across 14 `examples/illustrative/*.ch`
  files and 10 `examples/*.ch` files: `fmt(fmt(x)) == fmt(x)` for all
  24. `chelis check` after `fmt --inplace` is clean for every executable
  example (zero check fails after format).
- Deep IR round-trip stability: `chelis deep -> chelis surf -> chelis deep`
  produces structurally identical Deep s-expressions modulo span
  offsets.
- C-backend cross-validation: `module SimpleChain; def main(x: tensor[n, f32])
  -> tensor[n, f32] = { y = mul(x, x); z = add(y, x); z }`
  compiled with `chelis build --target c` and run through `gcc -O2 -fopenmp`
  produces `[9.99, -0.21, 3.75, 0.0, 2.0]` on input
  `[2.7, -0.3, 1.5, 0.0, -2.0]`, matching `chelis eval` to f32 precision.
- Cast cross-validation: `cast(tensor[n, f32], f64)` and `cast(tensor[a, b, f32], f64)`
  emit element-wise conversion (not memcpy); a 2-D `cast2d` over
  `[[0.5, 1, 1.5], [2, 2.5, 3]]` produces bit-correct f64 values via
  `gcc`-compiled output. `cast(tensor[n, int32], f32)` correctly maps
  `[0, -1, 127, INT32_MIN, 16777217]` to `[0, -1, 127, -2.1e9, 1.6777216e7]`
  (16777217 truncated to 16777216 is expected f32 precision loss, not a
  bug).
- Multi-level aliasing fan-out: `x = ...; y = x; z = y; w = z;
  add(add(mul(x, w), mul(y, z)), mul(w, x))` checks score=1.0 and evals to
  `[3.0, 12.0, 27.0]` from `[1, 2, 3]`.
- Module-wrapped + autofix: `module Probe; def main() = { x = to_tensor(...);
  y = copy(x); mul(y, x) }` lints down to `y = x` via
  `redundant-linearity-call --fix`, and the rewritten program checks score=1.0
  and runs correctly.
- Edge values in eval: `to_tensor([])` produces `tensor(shape=[0], data=[])`;
  `div([1.0, 0.0], [1.0, 0.0])` produces `[1.0, NaN]`; 36-element tensor
  prints the 32-element prefix plus `... +` truncation marker.
- `cargo test -p chelis-types --test linearity_module_wrapped`,
  `cargo test -p chelis-cli --test cbackend_cast_memcpy`, and
  `cargo test -p chelis-cli --test cbackend_reshape_memcpy` all pass
  (6/6, 4/4, 3/3 respectively).

## Recommended closure call

Close the 0.7.6 toolchain hygiene workstream. V3 surfaces no
high-severity regression from the 20+-PR merge train; the two new
findings (Linearity-F4 alias-consume bypass; Decompile-F1 module
PascalCase loss) belong in §5 of `docs/gap_synthesis.md` alongside
Linearity-F1/F2 and Vocabulary-F1/F2. No V4 pass is recommended.

§5 re-prioritization suggested:

- Linearity-F1 (typed `ConsumeKind` refactor) gains a fresh customer in
  closing NEW-F1 -- both items want the same plumbing
  (alias-vs-structural consume kind tracked explicitly, with
  alias-consume forwarding to the source name's linearity-scope entry).
  Bundle them.
- Vocabulary-F2 (canonical PascalCase ecosystem-name list) and the new
  Decompile-F1 share the root cause `module HelloTensor` -> Deep `hellotensor`
  desugar (`lower_module_path`); the durable fix is to preserve casing
  through Deep rather than relying on hand-maintained allowlists in
  multiple places.
