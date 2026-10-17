A warm `chelis build` or `chelis check` in a package with dependencies no longer
reruns the effect and linearity checkers over the whole cached dependency
library. The dependency typecheck cache now stores a lowering of the library
and, on a hit, re-derives and compares it, as the chelis-std cache already did.
A warm build against a 250-definition dependency that imports the standard
library retires about 27% fewer instructions. The cache entry is about 1.8 times
larger, a library that does not lower standalone keeps the checker reruns, and
entries written by earlier compilers are rebuilt once.
See [#3729](https://github.com/Chelis-Lang/chelis/issues/3729).
