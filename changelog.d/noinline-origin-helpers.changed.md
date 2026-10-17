Compiled C host programs keep their result-origin bookkeeping helpers out of
line. The generated-artifact seal now admits exactly one compiler extension,
`__attribute__((noinline))` on a translation-unit-private `static` function
definition, and the compiler marks its seven origin helpers with it so the C
compiler no longer inlines them at every root, scope and assignment. Clang
compiles the generated C for the Hydronnx MiniLM program with about 26% fewer
instructions, and about 41% fewer together with the elided bookkeeping of
[#3773](https://github.com/Chelis-Lang/chelis/pull/3773); program results and
diagnostics are unchanged. Every other attribute, `__declspec`, `asm` and
`_Pragma` is still rejected. See
[#3692](https://github.com/Chelis-Lang/chelis/issues/3692).
