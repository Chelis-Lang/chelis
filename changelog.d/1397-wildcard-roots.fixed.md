A top-level definition whose result type carries a runtime extent now runs. A
nullary `def` returning `tensor[*, f32]` was dropped from the root manifest, so
`chelis eval` printed nothing and warned that the input contained only
declarations, and `chelis build` emitted no `int main(`, producing an
object-only program that exited 0. Both lanes now execute it and size the
output from the realized extent. This also unmasks the vectorized case: a
`vmap` over a function whose shrink bound is read from the mapped tensor's own
shape executes with its exact values. A root whose result keeps an unresolved
type, dim or rank *variable* is still refused, because an uninstantiated
variable has no value or ABI. See
[#1397](https://github.com/Chelis-Lang/chelis/issues/1397) and
[#1378](https://github.com/Chelis-Lang/chelis/issues/1378).
