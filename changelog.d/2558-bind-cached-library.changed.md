Loading a cached compiled context, or a standard-library typecheck cache entry
that carries its lowering, no longer reruns the effect and linearity checkers
over the cached library. Such an entry is accepted on its envelope digest,
format version, build identity, its binding to the sources (the live source
hash for a compiled context, the content key for a typecheck cache), the agreement between its
type environment and checked program, and the agreement between its program
and its stored lowering, which the decoder re-derives and compares; the
checker results are those the same compiler build produced when it wrote the
entry. `chelis eval --file`, `chelis test`, `chelis check` and `chelis build`
inside a package spend correspondingly less time loading the library. The
dependency typecheck cache, whose entries carry no lowering, still reruns both
checkers on load, as does a standard-library entry without a lowering. See
[#2558](https://github.com/Chelis-Lang/chelis/issues/2558).
