# Computed operands at the host/tensor boundary

Investigation for [#3691](https://github.com/Chelis-Lang/chelis/issues/3691)
and [#4019](https://github.com/Chelis-Lang/chelis/issues/4019), recorded on
2026-10-29 against `03c3698856f41d2f230ef3384160f1f76eb06a52`.

## Contract and bounded reproductions

`spec/04-type-system.md` §4.7.2–4.7.6 gives reshape targets ordinary i64
dataflow, preserves their checked list arity, and requires the same admission
across inline expressions, bindings and helper calls. Expand retains its
singleton-axis precondition. `[05-OP-11]` preserves mean's operand dtype and
rejects an empty reduced axis at execution when its extent is dynamic.

The range-expand witness is also pinned in Hydronnx at
[`ea4a78515d767bb714ccf9f957f785f8b8fb3eec`](https://github.com/Chelis-Lang/hydronnx/blob/ea4a78515d767bb714ccf9f957f785f8b8fb3eec/tests_blocked/range_expand.ch.in),
beside its
[literal control](https://github.com/Chelis-Lang/hydronnx/blob/ea4a78515d767bb714ccf9f957f785f8b8fb3eec/tests_blocked/range_expand_control.ch.in).
Its typed `[2, 3]` result contains `[0, 1, 2, 0, 1, 2]` and reduces to `6`.
The mean witness flattens a symbolic input using `[numel(x)]`, then reduces
`[1, 0, 0]` to one third. The f16/f32/f64 evaluator carriers are respectively
`3555`, `3eaaaaab`, and `3fd5555555555555`.

The baseline focused CLI target reaches unsupported host emission for inline
expand and for the symbolic reshape scalar mean at every tested float width.
Direct mean and fixed-size controls execute. This is an admission failure,
not evidence of an incorrect computed result.

## Representation

The host application extractor recognizes an operand whose root stays in the
host lane, but its root-only test misses host work below an otherwise numeric
operation. Dynamic List-to-tensor production below reshape reaches the helper
preflight barrier. A runtime `numel` in a reshape target has a host scalar
producer; ordinary tensor-only extraction cannot supply that value.

Operand preparation uses the existing scoped admission walk, with checked
parameter types and lexical/callee facts, to identify these boundaries before
helper extraction. It evaluates the selected operand once on the host and
passes its typed result to the numeric helper. Earlier working operands also
move when necessary to retain source evaluation order. The helper uses its
ordinary tensor/scalar inputs and runtime extent edges; no emitter arm or
precision conversion is added.

A literal shape list retains its Cons/Nil spine. Its runtime leaves cross the
boundary separately, preserving both checked rank and independently computed
i64 extents. Integer constants and named axes retain their static selector
evidence; only runtime computations need a new boundary. Fresh
binding names are reserved across recursive list preparation.

Physical tensor types come from the produced host value; a logical generic
argument signature cannot reduce their rank. Computed inputs carry their local
producer identity into numeric lowering. An expand precondition on such a
capture remains a LocalExpand witness, rather than becoming an authored Caller
claim merely because partitioning introduced a Load.

## Oracle and limits

`computed_tensor_c_admission` checks score `1`, empty diagnostics and no
untyped nodes before comparing actual generated C binaries with evaluator
observations. It covers inline/bound/literal range-expand, the 48-case
f16/f32/f64 direct/reshape, symbolic/fixed, scalar/tensor, guarded/unguarded
mean matrix, helper/tuple activations, empty domains, runtime product guards,
axis checks, effects, independent dimensions, precision and ownership errors.
Inline and bound linspace/insert operands also feed matmul and return `6.0`
through linked C binaries with the same evaluator observations.
`examples/computed_tensor_consumers.ch` supplies an executable corpus example.

This establishes the tested producer/consumer boundaries. It does not certify
every computed tensor family, close the broader computed-tensor admission
tracker, or establish a compiler resource bound. Test wall times on a contended
developer machine are not throughput measurements.

The focused execution commands are:

```sh
CARGO_BUILD_JOBS=1 cargo nextest run -p chelis-cli --test computed_tensor_c_admission --test issue_2939_host_tensor_calls --test issue_2373_computed_concat_c_parity --test key_pair_host_staging --test-threads 1
CARGO_BUILD_JOBS=1 cargo nextest run -p chelis-ir --test runtime_extent_slice_b_eval_guard --test runtime_extent_checked_transport --test-threads 1
```

The CLI checks pass all 12 new tests and 29 neighboring controls; the two IR
targets pass all 24 tests. The new regression target executes 73 generated C
programs, including failing domain controls and untaken branches for scalar
and tensor results, without skips. The example also passes canonical formatting, lint,
strict check, evaluator execution and linked C execution; both lanes produce
the six integer cells, total `6`, and f32 one third `0.33333334`.
