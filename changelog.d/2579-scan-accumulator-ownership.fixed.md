Compiled `scan` no longer mutates its seed: a seed the caller keeps (a
parameter, a local read after the scan, or a global) keeps its value, where the
compiled program previously changed it in place (`sd` came back as `sdabc`). A
`scan` whose callback reads its accumulator no longer releases or retains the
variable that already holds the callback's result, which aborted tensor scans
and scans whose callback returns the accumulator through a nested `flat_map`,
and a `scan` with a named callback compiles instead of retaining the output list
as a string. See [#2579](https://github.com/Chelis-Lang/chelis/issues/2579),
[#2580](https://github.com/Chelis-Lang/chelis/issues/2580) and
[#2578](https://github.com/Chelis-Lang/chelis/issues/2578).
