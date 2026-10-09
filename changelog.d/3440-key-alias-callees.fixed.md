Compiled C now calls a typed local key builtin alias that shadows a top-level
definition of the same name; it previously called the definition and returned
its result. A typed local key builtin alias passed by name to `map` or `fold`
now compiles; the generated C previously called the alias's source name, which
C does not declare. See [#3440](https://github.com/Chelis-Lang/chelis/issues/3440).
