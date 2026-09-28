`chelis build --target c` no longer emits C that fails to compile when an ADT
`match` sits in one branch of a conditional and the other branch does not use
the matched value. The emitted release in the other branch named a temporary
declared inside the matching branch (`use of undeclared identifier '__adt_1'`).
See [#2449](https://github.com/Chelis-Lang/chelis/issues/2449).
