Compiled C host programs no longer carry result-origin bookkeeping that cannot
matter at run time. A binding whose origin can never be set no longer registers
an arena root, and an origin scope inside which nothing can register a root or a
claim is no longer opened and closed. The generated C for the Hydronnx MiniLM
program is about 20% smaller and clang compiles it with about 31% fewer
instructions; program results and diagnostics are unchanged. See
[#3692](https://github.com/Chelis-Lang/chelis/issues/3692).
