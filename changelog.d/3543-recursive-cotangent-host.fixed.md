Compiled C again lowers gradients whose selected parameter is a tuple of
tensors when another argument supplies the tensor helper carrier. This
restores the recursive parameter-cotangent program that `chelis eval` already
accepts. See [#3543](https://github.com/Chelis-Lang/chelis/issues/3543).
