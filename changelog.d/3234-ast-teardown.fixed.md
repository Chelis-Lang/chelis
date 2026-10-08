Direct Surf parser callers can parse, reject, and release deeply nested
reference types and tuple patterns on small worker stacks without a native
stack-overflow abort. See [#3234](https://github.com/Chelis-Lang/chelis/issues/3234).
