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

The `arith_model` JSON field follows the `real_arithmetic` qualifier it mirrors
rather than the proof tier, so it is absent from a solver-tier result decided in
machine arithmetic. See
[#3635](https://github.com/Chelis-Lang/chelis/issues/3635).
