The compiler API can call a selected parameterized definition with typed
argument values. `PreparedEval::call_root` takes each numeric scalar, boolean or
string parameter as an `ExecutionValue` of exactly its declared dtype, with
nothing promoted or narrowed. A tensor parameter's binding must have exactly
the declared dtype and literal extents; a binding of another dtype is rejected,
never cast or rounded. A live tensor parameter is required, and a dead one need
not be bound. A call that supplies an argument runs on the Host lane, a
definition whose parameters are all tensors keeps its own lane, and every call
produces its result root. A missing, extra or mistyped argument, and a call
whose live tensor parameter has no binding, is rejected before evaluation, and
the rejection names the parameter. Parameters of other types (keys, tuples,
lists, dicts, records, variants), and tensor parameters whose precision or
extents are not literal, are not yet admitted (#3804).
`spec/05-risc-primitives.md` [05-OBS-9] states the rule. See
[#3801](https://github.com/Chelis-Lang/chelis/issues/3801).
