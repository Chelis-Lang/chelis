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
tests remain separate. These cases do not establish arbitrary-rate replay
rounding or an IEEE derivative theorem.

The existing PR integration selection includes `fixed_control_c`:

```text
cargo nextest run -p chelis-compiler-api --test fixed_control_c --locked --offline --build-jobs 1 --test-threads 1
```

Its native helper builds an ownership-ledger runtime under the worktree's
`target/ownership-ledger-runtime`; set `CARGO_NET_OFFLINE=true` for an offline run
including that nested build.
