Standalone owned `ExecutionValue` values release deeply nested list, tuple,
data-type, and dictionary children without a native stack-overflow abort.
Rust embedding callers must borrow `ExecutionValue` fields or take them through
mutable references instead of moving them out through by-value matches. See
[#2601](https://github.com/Chelis-Lang/chelis/issues/2601).
