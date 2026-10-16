`chelis prove` reports a property whose `sqrt` domain obligation is false as
`failed`, with the counterexample the solver located. Previously it reported
`unsupported` and discarded the witness, so `--tier smt-only` said nothing
actionable about a defect the solver had already found, and `--tier auto`
reported it only when sampling happened to hit it.

The witness is a candidate, not a verdict: the domain proof may use only part of
the property's preconditions and decides over real arithmetic. So the witness is
rounded once to each parameter's declared width and evaluated concretely, exactly
as a sampled counterexample is. A witness every precondition admits and at which
the body is false is a definite `failed` with no `real_arithmetic` qualifier; a
witness the evaluation rejects keeps the `unsupported` result, which now names
why. `unsupported` stays reserved for an obligation nothing decided.

An integer binder's witness is bounded by its declared width's representable
range rather than by the fuzz sampler's `[-1000, 1000]` window, so a guard
admitting `n > 4000i32` confirms at 4001 instead of reporting it unrepresentable.
See [#3635](https://github.com/Chelis-Lang/chelis/issues/3635).
