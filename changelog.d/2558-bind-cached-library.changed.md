Loading a cached compiled context, standard-library typecheck cache, or
dependency typecheck cache no longer reruns the effect and linearity checkers
over the cached library. The cache entry is accepted on its envelope digest,
format version, build identity, live source hash, the agreement between its
type environment and checked program, and, where it carries one, the agreement
between its program and its lowering. The checker results are those the same
compiler build produced when it wrote the entry. `chelis eval --file`,
`chelis test`, `chelis check` and `chelis build` inside a package spend
correspondingly less time loading the library. See
[#2558](https://github.com/Chelis-Lang/chelis/issues/2558).
