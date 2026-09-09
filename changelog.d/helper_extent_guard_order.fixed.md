Tensor helpers check runtime extent claims in declared parameter order. Exported
calls and top-level bindings now report the same first failing parameter as
inlined calls when multiple claims fail. Part of [#1277](https://github.com/Chelis-Lang/chelis/issues/1277).
