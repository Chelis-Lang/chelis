# Dropout invalid-dtype runtime state

LaCaDiLE E1 requires executable validation-failure state evidence. The regression
`evaluation::tests::invalid_dtype_at_dropout_kernel_preserves_frames_keys_and_next_draw`
calls the actual `ExecutionFrame::dropout` boundary with tagged non-float tensors.
It is not well-typed source execution: strict evaluator Load rejects these
tags before dropout. The existing Load-ingress negative remains separate.

The test reuses source lowering and AD to obtain forward/replay/next-forward
sites and real nested controls. Its 20 scenarios cross bool/int8/int16/int32/int64,
empty/four-element tensors, and inherited/nested handlers. Each rejects an invalid
input before each of the three sites (60 failures), checks the complete actual
seed/counter/scope/key snapshot, and then executes that site with valid f32 data.
Every result bit, dtype and shape is checked against existing seed42 expectations.
Saved forward keys and the already-entered prefix survive; a final strict-plan
draw checks the preserved parent ordinal after the frame ends.

Focused command (default-feature lib tests also run in ordinary CI):

```text
cargo nextest run -p chelis-ir --lib --test dropout_fixed_stream_ir -E 'test(evaluation::tests::) | test(load_ingress_rejects_) | binary(dropout_fixed_stream_ir)' --locked --offline --build-jobs 1 --test-threads 1
```

A local implementation mutation inserted `self.enter(node, seed)?` immediately
before the non-float rejection. The regression failed on the first empty bool:
counter 0 became 1 and the absent forward key became `(42, 0)`. The mutation was
removed. No production behavior, numeric oracle, public API or source admission
was changed. This kernel-boundary evidence does not establish full E1 or released
cross-platform acceptance by itself.
