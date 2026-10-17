Host lowering for the C lane, and the linearity check, no longer do work at
each `let` binding for every binding around it. The walks that find a body's
free variables keep one table of bound names instead of one per enclosing
binding, renaming a body's top-level reads no longer copies the rest of the
body at each binding, and host type refinement shares its type scope with
each nested binding instead of copying it. Results, diagnostics and emitted C
are unchanged. Host lowering of a definition whose body chains 4,000 bindings
drops from about 12.6 G to 2.4 G instructions, and from about 433 G to 2.7 G
when those bindings read a top-level value. See
[#3797](https://github.com/Chelis-Lang/chelis/issues/3797).
