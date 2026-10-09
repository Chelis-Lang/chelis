A `fail`-guarded function whose body concatenates a runtime-length `List` of
tensors evaluates again. A runtime `if` with a `fail` arm keeps host control
flow outside a transform, so the definition no longer lowers past the guard
into a concat the static tensor DAG cannot carry. The guards still fire with
their messages. See [#3341](https://github.com/Chelis-Lang/chelis/issues/3341).
