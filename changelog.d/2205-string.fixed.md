The C lane no longer clones a string whose last use is `string_concat`. The
last-use scheduler's container-consumer table gains a string row, and a moved
operand routes to a consuming runtime entry point that appends in place only
while the string has a single strong owner, cloning and releasing the input
otherwise. A string still held by a tuple, a list, an option or a callee is
never mutated, which the alias controls lock on both lanes. Building a string
by repeated concatenation is now linear rather than quadratic: a 200,000-step
accumulator falls from 1.40s to 0.05s. `string_slice` and `string_trim` are
deliberately not rows, because each returns a substring of its operand rather
than a grown copy. See
[#2205](https://github.com/Chelis-Lang/chelis/issues/2205).
