# Evaluator performance oracle

The `eval-performance` suite checks interpreter correctness and resource use.
Run its default cases with:

```sh
cargo nextest run -p chelis-ir --test eval_performance --no-capture
```

The sparse runtime-index case evaluates nonconstant gathered indices at lengths
128 and 4096 for each active signed integer width. It compares exact i64 values
above 2^53 and checks negative, upper-bound, and large-i64 index traps. Evaluation
must allocate fewer than `256 * indices + 131072` bytes in total and increase
peak live heap by fewer than `64 * indices + 131072` bytes. The allocation-traffic
budget rejects copying the whole index tensor for every read. These budgets
include input loads, output storage, graph scheduling, and movement scratch.

Each row prints its wall time, allocation traffic, and peak live heap increase.
The allocator measures requested heap bytes in the test process, including all
threads; it does not measure process RSS or allocator metadata. Each nextest
case runs in its own process. Wall time is informational for the default cases.

Run the full convolution caller separately:

```sh
cargo nextest run -p chelis-ir --test eval_performance --run-ignored only \
  -E 'test(=eval_performance_convolution_index_witness)' --no-capture
```

This case evaluates `[1,128,56,56]` f32 input with a `[1,128,7,7]` kernel,
stride 1, and padding 3. Every result must match the bits of a direct
cross-correlation with the canonical adjacent-pair reduction tree, and DAG
execution must finish within 90 seconds. Run this manual budget on an idle
machine; it does not establish a throughput guarantee. Default PR CI does not
run this ignored case. Its reported heap peak includes residual materialized
convolution coordinates, contraction operands, and reduction scratch.

The suite covers the sparse index-read class of chelis#828. It does not certify
the tracker's other rows: host iteration, routed-DAG reuse, the School batched
corpus, dead-after-last-use chains, contraction memory, movement/reduction
scratch, or streamed convolution coordinates. The whole-model goal in
chelis#3584 requires separate evidence.
