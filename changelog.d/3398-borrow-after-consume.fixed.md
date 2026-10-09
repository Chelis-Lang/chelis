A borrow after an ordinary consume of the same binding is now consuming fan-out
that the compiler repairs with an inserted copy, as a later consume already was.
Passing a value to an owned parameter and then to a `&T` parameter, a read-only
primitive such as `sigmoid` or `len`, a closure capture, or a `grad(f)(..)` or
`vmap(f)(..)` call now checks and runs; the earlier consume receives the copy.
A use after `drop`, after a match scrutinee or consuming closure capture, or after
the consume of a destructured component is still rejected. That now holds when the
match scrutinee or capture follows an earlier ordinary consume, and when it sits in
either branch of an `if` or `match`; previously a later consume there was accepted. See
[#3398](https://github.com/Chelis-Lang/chelis/issues/3398).
