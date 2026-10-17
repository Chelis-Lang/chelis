`chelis check` and lowering no longer copy a lexical scope at every binder.
Type inference shares its environment's name maps between a scope and the
nested scopes it snapshots instead of copying them at every `let`. Lowering
shares its value scope and local callables between a function form and its
caller, and the lowerability walk shares its name sets between a `let` and
its body. Each binder now costs the names it adds, not the scope it enters.
Results, diagnostics and emitted C are unchanged. Checking a 4,096-binding
`let` chain drops from about 35 G to 15 G instructions and from 3.0 GB to
760 MB of memory, and emitting C for a 2,000-binding chain drops from 43 G to
19 G instructions and from 2.4 GB to 450 MB. See
[#3750](https://github.com/Chelis-Lang/chelis/issues/3750),
[#3763](https://github.com/Chelis-Lang/chelis/issues/3763) and
[#3774](https://github.com/Chelis-Lang/chelis/issues/3774).
