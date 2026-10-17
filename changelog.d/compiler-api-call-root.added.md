The compiler API can call a selected parameterized definition with typed
argument values: `PreparedEval::call_root` takes each non-tensor parameter as an
`ExecutionValue` of exactly the parameter's checked type, with no promotion or
narrowing, and runs the call on the Host lane. Tensor parameters still arrive
as bindings. A missing, extra, tensor-valued or mistyped argument is rejected
before evaluation, and a record argument of an invariant-carrying type is
revalidated. `spec/05-risc-primitives.md` [05-OBS-9] states the rule. See
[#3801](https://github.com/Chelis-Lang/chelis/issues/3801).
