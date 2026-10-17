Compiling against a library no longer costs time quadratic in the library's
size, and checking a long list literal no longer costs time quadratic in its
length. Desugaring copied the whole table of library callables once per library
definition and once per nested scope; it now shares that table. Macro expansion
copied every call's arguments before testing whether the call was a macro; it
now tests first. `chelis test` workers, `chelis prove` fuzz samples, `chelis eval`
inside a package, and layered library builds all compile against a library. On
a 1,000-definition file each `chelis prove` fuzz evaluation costs about a third
of what it did, and `chelis check` of one 8,000-element literal takes about a
twentieth of the time. Results and diagnostics are unchanged. See
[#3739](https://github.com/Chelis-Lang/chelis/issues/3739) and
[#3721](https://github.com/Chelis-Lang/chelis/issues/3721).
