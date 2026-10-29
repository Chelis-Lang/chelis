# Symbolic broadcast checking cost

This investigation records a bounded compiler-side contribution to
[chelis#4079](https://github.com/Chelis-Lang/chelis/issues/4079).
The witness is Hydronnx's independently symbolic rank-two Add chain, emitted
from `6db77317b6ffead812058aa8e612feea63a8fa46`. Its production converter
sources are pinned to `ca6e19b68790c174761b1dba0b188c97442b52b8`.

## Attribution

Hydronnx creates the expanding Surf before Chelis parses it. `EmitDim::Broadcast`
stores an operand tree, `dim_as_i64_expr` renders the left extent in both the
condition and the else arm, and `broadcast_to` duplicates extent expressions
and tensor wrappers. Sharing computed extents and materializing tensor
intermediates belongs to
[hydronnx#119](https://github.com/Chelis-Lang/hydronnx/issues/119).

The compiler's effects-only checked reconstruction adds avoidable copies.
`effects_only_rewrite_matches` strips derived effects from complete expression
bodies, then rebuilds those bodies again to collect checked binding-owner
tokens. It does this for both the original and the candidate. The checked
totality owner census also rebuilds bodies whose tokens it only reads.

The reconstruction compares owners at each corresponding metadata boundary
while comparing the expression bodies by reference. The totality census
visits bodies by reference. Structural metadata continues through the shared
owner mapper, including detached binders and annotated containers. Neither
the admitted program, its symbolic dimensions, its authored claims, nor its
effect and ownership rules change.

## Bounded result

The [machine-readable evidence](symbolic_broadcast_check_cost.json) binds the
compiler and probe hashes, source-program hashes, per-phase counters, command
outcomes and native comparisons. Full CLI checking gives these single trials:

| Independent Add nodes | Surf bytes | Baseline kernel max RSS (MiB) | Candidate kernel max RSS (MiB) | Baseline / candidate check |
| ---: | ---: | ---: | ---: | --- |
| 2 | 24,465 | 69.2 | 50.7 | strict pass / strict pass |
| 4 | 128,199 | 193.3 | 136.8 | strict pass / strict pass |
| 6 | 551,649 | 689.7 | 479.3 | strict pass / strict pass |
| 8 | 2,253,963 | 2,050.8 at termination | 1,872.5 | sampled memory limit / strict pass |

A strict pass means score 1, empty error and unresolved-name lists, and zero
untyped nodes. The terminated baseline's peak is an incomplete observation,
not the peak of a completed check.

At depth six, the allocator probe's semantic-completion peak falls from
467,167,759 to 264,938,926 live heap bytes, while allocated bytes fall from
1,740,573,443 to 1,080,251,541. The type-analysis peak falls from 303,103,524
to 203,710,659 bytes because its totality owner census uses the same repair.

The reconstruction-only regression increases an unchanged string body from
1 KiB to 256 KiB. The baseline allocates 1,389,021 bytes for the latter;
the candidate allocates 47,986. At 1 MiB the candidate allocates 48,016 bytes.
The rejection controls remain bounded and still reject the changed body.

The pinned two-Add native graph completes six valid bitwise comparisons with
ONNX Runtime, including both singleton directions and both empty directions.
Both engines reject three incompatible cases: equal-element-count/wrong-axis
inputs, an empty versus nonempty tensor with a mismatched nonzero axis, and
two empty tensors with different other axes. Baseline and candidate emit
byte-identical C for this graph.

## Measurement boundary

The baseline is compiler commit
`e60df675c933d96a5904d40697083cb3f7d263c2` on macOS arm64. The 15 regenerated
independent/static/shared Add programs have the same comment-stripped hashes
as the pinned Hydronnx assessment. Source byte counts differ because provenance
comments contain paths and timestamps.

The Rust allocator probe counts allocated bytes and peak live heap bytes for
parsing, desugaring, macro expansion, type analysis, semantic completion and
raw DAG lowering. It does not count allocations outside Rust's global allocator.
Peak live heap, kernel maximum RSS, and sampled process-tree RSS are different
measurements. Command measurements use the pinned Hydronnx helper's 30-second
CLI or 60-second profile limit, a 2-GiB sampled process-tree ceiling, and a
10-ms sample interval. The ceiling is not an OS hard memory limit.

Wall times are single trials under concurrent machine load; they establish
neither stable throughput nor a performance regression against another release.
Source growth does not establish exponential compiler cost in source size.

## Reproduction

Generate the original fixtures with the commands in the
[pinned assessment](https://github.com/Chelis-Lang/hydronnx/blob/6db77317b6ffead812058aa8e612feea63a8fa46/docs/audits/shape_expression_growth.md).
Pass an explicitly named compiler binary through `HYDRONNX_CHELIS` so the
emitter and native runner use the build being measured.

The checked-in two-Add fixture preserves the original independent input-axis
declarations, conditional singleton handling and per-axis compatibility guards.
To reproduce the deterministic reconstruction allocation oracle:

```sh
CARGO_BUILD_JOBS=1 cargo nextest run --release -p chelis-types \
  --test effect_reconstruction_allocations --test-threads 1 \
  --success-output immediate
```

The oracle checks an unchanged string body at 1 KiB, 256 KiB and 1 MiB, on
accepted and rejected reconstruction paths. Parsing, initial type checking,
and the caller's candidate copy are outside its measurement window. It also
checks the pinned generated model under a bounded allocation budget.

For a phase profile, supply the fixture path to the allocator example:

```sh
CARGO_BUILD_JOBS=1 cargo run --release -p chelis-compiler-api \
  --example profile_frontend_allocations -- \
  crates/chelis-types/tests/fixtures/hydronnx_symbolic_broadcast_2.ch --lower
```

`--split-suffix` measures effects and linearity separately using the same
accepted type-analysis product. This mode does not perform lowering. The
ordinary mode invokes the canonical semantic-completion transition.

## Acceptance limits

The bounded reconstruction oracle and generated fixture cover this copying
mechanism. They do not establish completion of #4079's original model audits,
all compiler allocation bounds, or all C-emission/toolchain costs. In particular,
the large T5, sentiment and LCM components require their own current-head
measurements. The upstream issue remains open for that work and for further
unavoidable-input versus avoidable-work attribution.
