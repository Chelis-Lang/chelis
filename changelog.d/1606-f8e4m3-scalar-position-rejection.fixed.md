`f8e4m3` is now rejected in every named type position — scalar parameters
and returns, standalone `sig` declarations, explicit type binders, `&T`,
tuple elements, arrow parameters, and `List[T]` — not only the tensor
element slot and the `cast` target. A rejected signature now retains its
error witness, so body inference and callers do not report the same type
site again. Distinct source sites retain separate diagnostics. Supported
float types are unaffected.
Fixes [#1606](https://github.com/Chelis-Lang/chelis/issues/1606) and
[#1527](https://github.com/Chelis-Lang/chelis/issues/1527).
