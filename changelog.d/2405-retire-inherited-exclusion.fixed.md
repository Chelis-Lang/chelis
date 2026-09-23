`chelis eval` now runs a fixed-rate `dropout` beneath a recursive function,
inside the taken arm of a runtime `if` or `match`, and in any program that
contains a `match` or recursion elsewhere. It previously failed with "unknown
runtime name `dropout`", including on programs that `chelis build` runs,
because the caller's control flow became an execution exclusion inherited by
every call beneath it. Each definition and each `dropout` call now decides
from its own body, as compiled C does. The draws follow [05-RNG-1] and match
compiled C wherever C runs the program. Control flow that reaches no random
draw is no longer classified at all, and programs without `dropout` produce
the same values as before.

A `dropout` that has no fixed-control plan, which is the case under `vmap`
and under `grad` of a function whose draw sits under runtime control, is now
refused with an `unsupported` error. It previously returned a
mask from an older formula that does not follow [05-RNG-1]
([#2409](https://github.com/Chelis-Lang/chelis/issues/2409),
[#2413](https://github.com/Chelis-Lang/chelis/issues/2413)). Fixes
[#2405](https://github.com/Chelis-Lang/chelis/issues/2405).
