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

Each row prints its wall time and peak live heap increase; sparse and movement
rows also report allocation traffic.
The allocator measures requested heap bytes in the test process, including all
threads; it does not measure process RSS or allocator metadata. Each nextest
case runs in its own process. Wall time is informational for the default cases.

The movement cases compare every exact i64 element above 2^53 for permutation,
rank-preserving expansion, insertion, shrink, stride, and a million-element
runtime gather. A two-gather/permutation window composition checks the layout
used by depthwise convolution. For a movement with `I` input elements and `O`
output elements, peak live heap increase must stay below
`8 * (2 * I + O) + 1048576` bytes. The two input copies are the evaluator's
resolved input and loaded value; the output remains an owned buffer. The large
gather includes both its data and index inputs in `I`. The fixed margin covers
graph and rank-sized coordinate plans, and excludes an output-sized pick array.

The axis cases check sum over a million-element axis, a large output domain,
and all positions in a rank-three tensor. Product, minimum, maximum, argmin,
and argmax also run over the large output domain. Their i64 peak live heap
increase must stay below `16 * I + 48 * O + 1048576` bytes. This permits resolved
and loaded input storage plus typed result construction, and excludes retained
input-index arrays and reduction-leaf arrays. All-values and root-scoped APIs
must agree, while retaining their respective result-map contracts.

The typed-kernel tests compare sums at lengths 0 through 65 against an
independent breadth-first adjacent-pair reference for every active numeric
dtype and admitted accumulator width. Existing exact-reduction tests cover
intermediate overflow, reduced-float finalization, first-NaN selection, ties,
signed zeros, and gradients. Movement copies preserve stored float bits.
The product-reduction arithmetic-order and per-step-finalization gap remains
tracked in chelis#1290; the coordinate-scratch oracle does not certify its repair.

Run the full-size movement/reduction witnesses separately:

```sh
cargo nextest run -p chelis-ir --test eval_performance --run-ignored only \
  -E 'test(/large_window_movement_coordinates|large_axis_reduction_coordinates/)' --no-capture
```

These use the `[1,128,56,56,7,7]` permutation/expansion domain and 19,668,992
reduction elements, including `[3136,6272]` reduction slices. They apply the
same exact-value and heap-budget checks. Default PR CI does not run these
ignored cases.

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
convolution coordinates and contraction operands.

The suite covers sparse index reads and the named movement/reduction scratch
cases of chelis#828. It does not certify
the tracker's other rows: host iteration, routed-DAG reuse, the School batched
corpus, dead-after-last-use chains, contraction memory,
or streamed convolution coordinates. The whole-model goal in
chelis#3584 requires separate evidence.
