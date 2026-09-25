`Result` is no longer a built-in type. The checker admitted `Result[a, b]` as a
native nominal type, but no constructor, runtime layout or specification ever
defined it, so a program could declare a `Result` parameter or field and never
build or read one. `Result[..]` is now an unknown type at check, like any other
undeclared name. A program may declare its own `Result` data type at any arity;
a zero-arity declaration, such as `type Result = | HomeWin | Draw | AwayWin`,
was rejected before because the built-in header demanded two arguments. See
[#2526](https://github.com/Chelis-Lang/chelis/issues/2526).
