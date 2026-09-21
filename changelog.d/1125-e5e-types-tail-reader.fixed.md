Stamped checker input now receives the same guarded-match return relaxation as
legacy Deep input. Function, tail, and arm readers use the total Deep carrier
view, and the relaxation now compares binding identity rather than a repeated
source name, so it still returns one caller-owned borrowed parameter. The
accepted set moves in both directions: two different alias spellings of one
parameter now agree across branches, while one spelling reaching two different
parameters, a borrow built inside the function, and a binder naming a
projection out of the scrutinee no longer do. A `match` arm binder still traces
back to the parameter underneath when it denotes the whole scrutinee.
See [#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
