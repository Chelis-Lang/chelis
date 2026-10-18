Checking deep nesting costs time and memory linear in the depth in three more
places. Macro expansion no longer copies the names bound so far at every
`let`, `fn`, `match` arm and pattern, so a block of 16,000 bindings checks in
about a third of the instructions and memory
([#3837](https://github.com/Chelis-Lang/chelis/issues/3837)). The type
checker's occurs check applies the substitution once instead of at every level
of the type it searches, and applying the substitution no longer walks a
subtree it has already found to be free of variables, so nested constructor
applications such as `Some(Some(...))` are no longer cubic in their depth
([#3819](https://github.com/Chelis-Lang/chelis/issues/3819)). `chelis check`
no longer derives the root names of a declaration's nested tuple result,
which only lowering reads and whose dotted paths grow quadratically with the
nesting ([#3820](https://github.com/Chelis-Lang/chelis/issues/3820)).
Diagnostics, checked artifacts and emitted code are unchanged.
