The checker now validates literal patterns in unannotated lambdas when their
scrutinee type becomes known and rejects unresolved pattern obligations at the
declaration boundary. Float-pattern diagnostics retain compact exponent
notation. See [#2448](https://github.com/Chelis-Lang/chelis/issues/2448) and
[#2468](https://github.com/Chelis-Lang/chelis/issues/2468).
