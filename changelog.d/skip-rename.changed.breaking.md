The List slice `drop(xs, n)` is now `skip(xs, n)`. `drop` keeps only its
one-argument form, the linearity consume that pairs with `copy()`. The two were
one builtin declaration separated by argument count, so a dropped argument
turned one operation into the other and still type-checked; they are now two
declarations governed by two atoms, `[05-OP-54]` for the slice and a new
`[05-OP-67]` for the consume. `skip` pairs with `take`, which already had the
matching semantics.

Migrate existing sources with `chelis migrate surf --from 0.18 --inplace`. The
rewrite is arity-driven and scope-aware: it renames `drop(xs, n)` and the pipe
stage `xs |> drop(n)`, and leaves `drop(value)`, `xs |> drop`, and any `drop`
shadowed by a parameter, lambda parameter, block binding, or match binder
untouched.

Two further consequences for existing code. `skip` joins the closed builtin
vocabulary, so a top-level `def skip` is now rejected as `BuiltinShadowing`
(spec/04 section 8.6); rename that definition or scope it inside a Reef
package. The standard library's `Std.Index` wrapper `drop_list` is now
`skip_list`, with the same signature and semantics.

`chelis lint` gains the advisory rule `recursive-list-cursor`
(spec/01 section 12.3). It reports a self-recursive definition whose recursive
call passes `skip(p, k)` in the argument position its own parameter `p`
occupies: that walk copies the List once per step and costs time quadratic in
its length. `Std.Tokenizer`'s three JSON entry builders had exactly that shape
and now run on `fold` and `map`.
