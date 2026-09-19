Evaluating a named-axis reduction no longer re-derives the whole program on
every call. The subexpression lowering context that routing lowers through is
now built once per evaluation context rather than once per routed reduction,
so the pipe fold over every definition, the copy of the definition table, and
the copy of the type environment are each paid once for the program. The
execution-profile classifications on the same path read the program-scoped
definition snapshot instead of copying the program per ask. On a package with
`chelis-std` in scope, a routed reduction costs about a fifth of what it did.
See [#2207](https://github.com/Chelis-Lang/chelis/issues/2207).
