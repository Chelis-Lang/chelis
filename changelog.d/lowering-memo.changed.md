`chelis check`, `build` and `eval` classify each definition's lane by walking a
called declaration's body once per distinct callable facts its calls supply,
instead of once per call site. A chain of 20 declarations that each call the
previous one twice took about 100 billion instructions to check, whether the
calls forwarded a callable parameter or a precision-polymorphic one; it now
takes about 0.3 billion. `chelis cost` lowers every named entry of a program
through one host-lowering session, and an evaluation that selects several
parameterized host declarations asks their kernel input demand through one
session over the prepared program. On 800 independent definitions,
`chelis cost` took 65 billion instructions instead of 99 billion. See
[#3707](https://github.com/Chelis-Lang/chelis/issues/3707),
[#3727](https://github.com/Chelis-Lang/chelis/issues/3727) and
[#3728](https://github.com/Chelis-Lang/chelis/issues/3728).
