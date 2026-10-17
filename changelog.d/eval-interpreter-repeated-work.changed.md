`chelis eval` no longer repeats per-call work that depends only on a closure's
source. Applying a closure, such as a `fold`, `map` or `scan` callback, no
longer copies its syntax or re-normalizes its parameter and result types, and
evaluating a block no longer copies its bindings. A 1000-element
`Std.Tensor.Construct.linspace` now costs 17.9e9 instructions, down from 78e9.
A `List` argument whose elements hold no tensor is no longer walked element by
element at each application, so a `fold` that grows a `List[i64]` is linear
rather than quadratic. A recursive higher-order function that passes its
function argument back through the same parameter now checks a bounded number
of retained contracts per application instead of one per recursion level, so
it is linear in depth rather than quadratic. Results and diagnostics are
unchanged. See [#3778](https://github.com/Chelis-Lang/chelis/issues/3778),
[#3761](https://github.com/Chelis-Lang/chelis/issues/3761) and
[#3758](https://github.com/Chelis-Lang/chelis/issues/3758).
