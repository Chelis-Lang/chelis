`chelis eval` preserves binding-qualified record and tuple descendant labels
such as `gadt.text`, matching compiled C observation output byte-for-byte.
Previously evaluator text output could drop the originating binding and print
only the field suffix. See
[#1359](https://github.com/Chelis-Lang/chelis/issues/1359) and
[#2193](https://github.com/Chelis-Lang/chelis/pull/2193).
