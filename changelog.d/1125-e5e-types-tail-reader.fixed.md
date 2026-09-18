Stamped checker input now receives the same guarded-match return relaxation as
legacy Deep input. Function, tail, and arm readers use the total Deep carrier
view, while scoped alias provenance ensures the relaxation still returns one
caller-owned borrowed parameter rather than relying on a repeated source name.
See [#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
