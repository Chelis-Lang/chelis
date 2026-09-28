# Dropout signed and nonfinite cotangent conformance

`evaluation::tests::source_ad_replays_signed_and_nonfinite_cotangents_with_the_saved_key`
checks typed source AD for `sum(mul(dropout(x, 0.5), weights))`. The closure's
only differentiated parameter is `x`; `weights` is a captured tensor input.
The test covers f16/bf16/f32/f64, rotating `-0`, `-3`, `+0`, `+inf`, `-inf`
and NaN through four coordinates: 24 source executions, with every class at
both a kept and a dropped coordinate. Expected stored words and the seed42
mask are independent of the evaluator's numeric kernel.

Each execution checks the complete forward result, actual replay input,
replay output and returned gradient, including dtype, shape and stored bits.
The actual saved key, local counters, scope stack and inherited seed/counter
are checked; 24 strict-plan next draws verify the next ordinal. Four additional
direct calls at the realized replay site check dropped `+0` and kept `-0`.

[05-OP-37] requires the saved mask and kept-cotangent division; [04-NUM-2/8]
govern storage width, signed zero, infinities and canonical NaN. Source gradient
accumulation additionally follows spec/06 §2.4: the positive-zero base leaf
turns a `-0` contribution into `+0` before replay and again at the final root.
The source expectation includes that arithmetic. A direct `-0` replay input
precedes accumulation and must retain its sign at a kept coordinate.

These are IEEE conformance checks, not a derivative theorem or finite-difference
claim over nonfinite inputs. Existing arbitrary-rate, rounded-unit, invalid-rate,
invalid-dtype and nested-state controls remain separate. This slice does not
establish full E1/E2, Hull, Lean or released cross-platform acceptance. The
separate native checks below cover the sealed fixed-control C body boundary.

A temporary AD mutation inserted `abs(g)` before the generated dropout replay.
The new test rejected the changed negative cotangent; the earlier 46 focused
tests still passed. The mutation was removed before the final passing run.

Focused command (the unit test also runs in ordinary default-feature CI):

```text
cargo nextest run -p chelis-ir --lib --test dropout_fixed_stream_ir -E 'test(evaluation::tests::) | test(load_ingress_rejects_) | binary(dropout_fixed_stream_ir)' --locked --offline --build-jobs 1 --test-threads 1
```

## Native stored-word conformance

`fixed_control_c::native_special_words_reject_mask_sign_and_nan_corruption`
checks primal dropout at rates 0 and 0.5 for all four float widths.
`fixed_control_c::native_source_ad_replays_signed_and_nonfinite_stored_words`
checks the actual Surf-generated gradient of the captured-weight loss above at
rate 0.5. The source is parsed and checked, then its body is lowered to a sealed
evaluation plan. This boundary does not claim full compiler-API/host transport.

The same six special-value classes rotate through four coordinates: 48 primal
and 24 source-AD cases, each executed four times in generated C. Every output
word, dtype and shape is checked, together with unchanged borrowed inputs and a
balanced native ownership ledger. Expected IEEE words and the seed42 ordinal0
mask are independent constants; the evaluator supplies no expected values.
Canonical NaN words are required by [04-NUM-2], not merely the NaN class.
Inputs use canonical NaN; arbitrary NaN payload ingress is not covered.

Sixteen emitted-C corruptions exercise four failure classes at every width:
absolute-value replay cotangents, multiplicative dropped masking, kept-zero sign
erasure, and a noncanonical NaN result. Each must compile and then fail an
executed value assertion. Existing rate-rounding, next-draw and nested-state
tests remain separate. These special-word cases do not establish arbitrary-rate
replay rounding or an IEEE derivative theorem.

## Native replay rounding and mask thresholds

`fixed_control_c::native_source_ad_replay_finalizes_nonbinary_rate_division`
checks the actual Surf-generated captured-weight gradient at stored rate 0.1
for f16/bf16/f32/f64, using the same sealed-body boundary. This is a
representative non-exact decimal rate, not exhaustive coverage of finite rates.
Independent exact-rational round-to-nearest-even calculations implement the
[04-NUM-8] arithmetic/storage stages and [05-OP-37] finalized denominator and
division. The positive input/result stored-word witnesses are:

| Width | Cotangent → kept replay result |
| --- | --- |
| f16 | `3c05 → 3c77`, `3c06 → 3c79` |
| bf16 | `3f81 → 3f90` |
| f32 | `3f800005 → 3f8e38e9`, `3f800007 → 3f8e38ec` |
| f64 | `3ff0000000000005 → 3ff1c71c71c71c77`, `3ff0000000000000 → 3ff1c71c71c71c72` |

Positive and negative witnesses rotate through all four coordinates; seed42,
ordinal0 keeps coordinates 0–2 and drops coordinate3 at this rate. These finite
nonzero cotangents are unchanged by the spec/06 §2.4 positive-zero accumulation.
The plan must contain an actual forward/replay pair sharing the saved draw.
Executed replay mutations replace division with multiplication by a
storage-finalized reciprocal, or use a wrong-width/unfinalized denominator.
The latter skips denominator storage rounding for f16/bf16, evaluates the f32
denominator/division in f64, or narrows the f64 denominator through f32.

`fixed_control_c::native_mask_threshold_uses_arithmetic_width_and_strict_less_than`
uses four independently inverse-SplitMix-derived seeds under [05-RNG-1]. At
ordinal0, coordinate0, their 53-bit units are respectively `0.5 - 2^-53`,
`0.5`, `0.5 + 2^-53`, and `0.5 - 2^-16`:
`7396636047707789066`, `4901139120565445618`, `6117835775437243522`,
`3386422020048024308`. All four output coordinates are pinned independently.
At rate0.5, the first unit rounds to equality in f32 arithmetic but remains
below in f64; the last remains below in f32 but would round to equality if
prematurely narrowed to f16/bf16 storage. Executed mutations distinguish `<`
from `<=`, pre-f32 comparison from required f32 rounding (and erroneous f32
rounding for f64), and premature storage rounding for f16/bf16.

Together these two tests execute 20 positive C artifacts, 32 input cases repeated
four times (128 calls), and 18 emitted-C mutants. Every positive call checks
complete output/input words, dtype and shape; every positive artifact must
balance the native ownership ledger. Each mutant must compile and fail a
runtime assertion, not merely change emitted text. Expectations do not call
the evaluator. These are IEEE operation conformance checks, not an IEEE
derivative theorem, full source/API transport, arbitrary runtime-rate C
support, or completion of E1/E2. HIP/Metal and next-draw/state controls remain
outside this slice.

The existing PR integration selection includes `fixed_control_host_c`, which
requires the `ownership-ledger` feature:

```text
cargo nextest run -p chelis-compiler-api --features ownership-ledger --test fixed_control_host_c --locked --offline --build-jobs 1 --test-threads 1
```

Its native helper links the runtime this test build carries, which the feature
instruments with the ownership ledger; no nested Cargo build runs.
