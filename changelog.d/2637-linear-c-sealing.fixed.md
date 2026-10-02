`chelis build --target c` seals generated C in time linear in the number of
exported definitions. Previously the sealing pass rescanned the whole generated
source once per exported definition, so build time grew quadratically. With a
debug build of the compiler, emitting C for a program of 400 trivial exported
functions took about 300 CPU seconds and now takes about 11. The generated C and
header are unchanged. See [#2637](https://github.com/Chelis-Lang/chelis/issues/2637).
