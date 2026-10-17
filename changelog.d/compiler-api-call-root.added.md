The compiler API can call a selected parameterized definition with typed
argument values. `PreparedEval::call_root` takes each non-tensor parameter as an
`ExecutionValue` and checks it against the parameter's checked type at every
depth before the call evaluates: scalars must carry exactly the declared dtype,
records and variants the declared constructor, and tensors the declared dtype
and literal shape. Nothing is promoted or narrowed, and a value of an
invariant-carrying type is revalidated. Tensor parameters still arrive as
bindings: a live one is required and a dead one need not be bound. The call
runs on the Host lane and always produces its result root. A missing, extra,
tensor-valued or mistyped argument, and a call whose live tensor parameter has
no binding, is rejected. `spec/05-risc-primitives.md` [05-OBS-9] states the
rule. See [#3801](https://github.com/Chelis-Lang/chelis/issues/3801).
