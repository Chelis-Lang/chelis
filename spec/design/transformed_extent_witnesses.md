# Transformed tensor geometry and extent witnesses

Tracking class: [#2515](https://github.com/Chelis-Lang/chelis/issues/2515),
under [#1277](https://github.com/Chelis-Lang/chelis/issues/1277).
Initial gradient leaves: [#1767](https://github.com/Chelis-Lang/chelis/issues/1767),
[#1978](https://github.com/Chelis-Lang/chelis/issues/1978), and
[#2370](https://github.com/Chelis-Lang/chelis/issues/2370).
The already passing but unpinned movement example in
[#2083](https://github.com/Chelis-Lang/chelis/issues/2083) is a regression
control. [`runtime_extents.md`](runtime_extents.md) C2.3–C2.5 owns the shared
value/claim/witness/guard representation; this document owns the transformed
result geometry that representation must survive.

`spec/04-type-system.md`'s gradient target and result rules decide types and
shapes. `spec/06-transformations.md` §5.2–5.4 requires rewrites to preserve
observable traps. The following is an implementation plan, not a new gradient
or movement semantic rule.

## Distinguish three shapes in a transform

An activation has an authored parameter type, a realized actual argument,
and a produced forward result. They may carry differently named axes; equal
widths do not identify them. A gradient has a fourth object: each selected
cotangent, whose geometry comes from its **differentiated actual**, not the
forward result or a scalar zero used to construct its values. A zero
cotangent still has every ordered actual axis, including zero-length axes and
rank zero. The forward activation remains a separate observable dependency
when it carries a claim or trap. Keeping that dependency cannot by itself
give a disconnected cotangent the correct rank.

At the concrete transform call, build one typed mapping from each authored
formal's ordered axes to the corresponding actual's physical axis sources
and scoped witnesses. Preserve both the checked public result type and that
mapping through closure conversion, call composition, `grad`, `vmap`,
`vmap(grad(...))`, inlining, specialization, and cache transport. A copied or
aliased actual forwards its mapping; a new call allocates an independent
scope. A literal shape or printable binder cannot stand in for an absent
actual. For a composed helper, the mapping is instantiated at each call
boundary, then composed by reference, so the inner `model` call cannot
rebind the outer `theta` axis to a rank-zero `template`.

Gradient construction uses the selected actual's mapping for positive and
disconnected cotangents alike. Movement in the forward or backward DAG uses
`output_axis_sources` for each resulting axis: `insert` adds a fresh position,
`permute` applies its exact permutation, and an identity axis forwards its
original source. Do not actualize a shape from an adjacent same-width tensor
or a synthetic name. A gradient-specific node/source verifier runs before
Eval or C publication and checks that every cotangent axis has a live actual
source, every claimed forward guard retains its declaring witness and
producer, and every remapped node/dimension identifier exists. A failed
mapping is a typed lowering failure, never a scalar-shaped successful
gradient, an internal assertion, or a generated C artifact with a missing
declaration.

This representation extends C2.4's authored-signature and forward-activation
transport. It does not change the current unconditional shape dependency;
[#1935](https://github.com/Chelis-Lang/chelis/issues/1935) can condition that
edge only after the obligation-presence proof in C6. Existing #1932
mapped-gradient closure remains required, not a substitute for direct or
composed gradient execution.

## Delivery and acceptance

Create the red positive/negative matrix before changing transform lowering.
Land the actual-to-formal mapping and verifier together; then migrate zero
materialization, generic helper composition, and backward movement consumers.
For each accepted source, compare declared type, actual shape/dtype, every
gradient value, and ordered traps on Eval and a compiled, linked C binary.
Do not count an isolated helper result as proof of composed execution.

| Leaf | Required executable exit | Negative that detects the old class |
|---|---|---|
| #1767 | Constant-loss `grad` at a top-level tensor binding returns a tensor of zeros with the differentiated actual's exact shape; parameter, rank-zero, empty-axis, multiple disconnected selections and cached/helper call forms agree. | A false declared result shape still fails; dropping the actual-axis mapping makes the zero-shape assertion fail. |
| #1978 | The seven-case symbolic ReLU shim matrix produces exact analytic gradients on Eval and native C, including the separately signed wrapper. | A wrong forward claim retains its Domain trap; an incorrect backward axis remap fails verification before output. |
| #2370 | The Nautilus-shaped generic Jacobian helper with a map-built basis composes and returns exact rows on Eval and C; the literal-basis and isolated-helper controls keep their expected values. | Independent `n`/`m` binders and a mismatching actual cannot be unified by equal width or a scalar template; the failure names its owning boundary. |
| #2083 | The agreeing rank-raising `insert` then `permute` gradient receives a permanent Eval/C positive test, reflecting #2144's already observed fix. | A width-three `y` still triggers the original forward `insert` extent failure before a backward geometry check. |

The matrix is an addition to #1277's bounded oracle or an issue-owned target
with explicit inclusion in that oracle's manifest. The #2083 regression can
close against #2144 once both controls execute on merged main; the other
leaves close only on their full matrices. No result here closes all of #1277.

### Disconnected cotangent receipt boundary

`lower_grad_callable_with_values` pairs each selected formal load with its
actual node and ordered tensor type before AD. It validates rank, dtype and
physical axis sources; packing requires that same mapping. A disconnected
cotangent reads every extent from the actual's corresponding axis, including
literal-shaped inputs whose entry claims must still execute. Rank-zero
cotangents retain the actual as an activation dependency. External top-level
inputs obtain their declared type from their authored signature when the
value environment omits their self-reference.

The #1767 matrix runs in `chelis-backend-c::exec_compile` under the
`issue_1767` filter: top-level, parameter, rank-zero, empty, symbolic,
reordered/repeated selections, movement actuals, and live/decoded helper
contexts compare strict Eval with linked C. `chelis-ir::issue_1102_zero_gradient`
pins the original failure and input guard; the lowerer unit
`gradient_axis_mapping_rejects_missing_sources_before_publication` removes
or corrupts sources. `chelis-cli::runtime_extent_slice_b` owns the #1767
false-result-claim pair and #2083's exact agreeing/refuted movement pair.
These tests are included explicitly in `runtime_extent_oracle_targets.json`;
their presence does not change the scope of the other transformed leaves.

### Captured tensor receipt boundary

The #2378 matrix in `chelis-cli::issue_2378_grad_capture_extents` pins tensor
captures after #2628's closure conversion: literal and symbolic captures and
direct arguments agree on Eval and linked C; a disagreeing captured actual
retains its named entry guard. The native test runner executes the original
capture and both direct gradient slots, and a statically wrong capture shape
remains a checker error. This target is included in
`runtime_extent_oracle_targets.json`. These capture receipts do not cover
#2370's map-built basis inside a composed Jacobian helper.

### Runtime scalar list builders

`range(start, end)` uses an exact rank-zero i64 endpoint pair and an Iota
source under [05-OP-54]. Its count is `max(end - start, 0)` in checked integer
arithmetic; the source owns its realized output axis. Lists produced on this
path travel as internal tensor-list values through bindings and captures,
so consuming a list does not replay endpoint expressions. A map preserves
that source axis; its scalar callback captures retain their own gradient
paths while the integer source has zero cotangent.

AD constructs zero leaves by expanding a scalar along the exact primal
axes. A same-shape operation's input cotangent needs an explicit reshape view
when any axis is symbolic; the view reads that input's physical extents and
retains the forward consumer so its checks precede cotangent use. Fully fixed
axes already have checked exact extents and need no view. This carries
anonymous range axes through cotangent accumulation without inventing a
dimension name or treating a matching element count as shape evidence. A branch-owned
broadcast masks inactive cotangent rows before its adjoint reduces them,
while the activation axes are still present.

The bounded callback implementation vectorizes only a scalar DAG with no
observable runtime checks, using the existing capture-aware vectorizer.
[05-OP-55] requires an ordered loop for callbacks that can trap; those
callbacks are rejected before execution until that carrier exists. HIP
runtime sources and ranges nested inside vmap remain outside this slice.
The numbered contracts remain unchanged.

`issue_2419_range_tensor_ad` owns isolated and generic composed basis gradients,
independent runtime widths, captured scalar cotangents and the ordered-callback
rejection. `issue_570_runtime_iota` owns exact integer values, empty ranges,
malformed sources, overflow and false extent claims. These are bounded receipts,
not closure of every list callback or every #2515 leaf.
