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
run this ignored case. Peak live heap increase must remain below 32 MiB,
including source storage, graph scheduling, and output buffers.

Root-scoped evaluation represents private expansion/product nodes as immutable
views and feeds finalized tagged multiplication results into the same typed
adjacent-pair sum kernel. The shared IR contraction planner controls eligibility;
all-values evaluation and explicitly selected intermediate roots retain dense
values. Views own their source values across last-use reclamation. Cartesian
index views retain Add overflow and Gather domain checks at their original node
positions: vector extrema prove the complete index interval, and an interval
that cannot prove bounds falls back to row-major first-invalid-index checking.

Default cases compare stored bits against dense all-values evaluation for all
four float dtypes and admitted accumulator widths, cancellation, nonfinite
values, signed zeros, selected intermediate roots, convolution gradients, and
Cartesian overflow/invalid/empty controls. A many-contraction case reports
planner scaling; the planner uses one graph scan for source Drop positions.

Run the model-sized pointwise pair separately:

```sh
cargo nextest run --release -p chelis-ir --test eval_performance --run-ignored only \
  -E 'test(=eval_performance_pointwise_pair_witness)' --no-capture
```

This evaluates `[3136,128]` by `[128,512]`, GELU, then `[512,128]` in f32.
Every result must match explicit canonical-tree references with an independent
dense scalar GELU graph. Peak live heap increase must remain below 96 MiB and
evaluation must finish within 90 seconds on an idle runner. Default PR CI does
not run this ignored case.

The suite covers sparse index reads, named movement/reduction scratch,
private exact contractions, and Cartesian convolution coordinates of chelis#828.
It does not certify host iteration, routed-DAG reuse, the School batched corpus,
or dead-after-last-use chains. Products with additional readers, shape/claim
dependencies, activation ownership, unequal expansion types, or early source
Drop are not eligible for the shared contraction descriptor. The whole-model
goal in chelis#3584 requires separate evidence.
