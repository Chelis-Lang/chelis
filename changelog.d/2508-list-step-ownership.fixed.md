Compiled C now returns every allocation that `map`, nested `map`, `flat_map`,
`filter`, `partition` and `scan` touch when the list items own heap payloads
(strings, lists, data-type values, tensors). Previously each step copied the
item it had been given into the result and leaked the original, and `filter`
also leaked every item it rejected. A `filter` or `partition` whose callback
body is a bare captured variable, and a `fold` whose unread accumulator is a
`string`, now compile, and a `fold` with an unread data-type accumulator no
longer aborts. See [#2508](https://github.com/Chelis-Lang/chelis/issues/2508),
[#2505](https://github.com/Chelis-Lang/chelis/issues/2505) and
[#2332](https://github.com/Chelis-Lang/chelis/issues/2332).

Releasing a deeply nested value no longer overflows the native stack: a
100,000-link data-type chain is built, passed and released in compiled C.
Recursion over a large list parameter is linear again; every call used to walk
the whole carried value. See
[#2522](https://github.com/Chelis-Lang/chelis/issues/2522).
