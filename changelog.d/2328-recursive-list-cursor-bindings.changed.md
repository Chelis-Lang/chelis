`chelis lint`'s advisory `recursive-list-cursor` now also reports a cursor the
recursive call reaches through a local binding, as in `rest = skip(xs, 1i64)`
followed by `walk(rest, ...)`. Previously it read only a `skip` written in the
argument position itself, which missed every cursor found in the shell
ecosystem. Substitution is one level deep and respects binding order, and a
name a definition binds more than once is still never substituted. A cursor
reaching the argument through a call to another function remains outside the
rule permanently: whether that call returns a suffix of its argument is
interprocedural. `spec/01-nomenclature.md` §12.3 states the new scope. See
[#2328](https://github.com/Chelis-Lang/chelis/issues/2328).
