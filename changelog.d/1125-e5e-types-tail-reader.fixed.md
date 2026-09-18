Stamped checker input now receives the same guarded-match return relaxation as
legacy Deep input. Function, tail, and arm readers use the total Deep carrier
view, so an equivalent decoded `Node` no longer creates a false signature
mismatch. See [#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
