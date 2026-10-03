# Known issues

## Tensor construction helpers

`Std.Tensor.Construct.stack`, `squeeze`, and `unsqueeze` are exported but reject
ordinary concrete tensor calls during checking. They are not part of the current
executable feature set. [#1416](https://github.com/Chelis-Lang/chelis/issues/1416)
tracks implementation of the decided concrete-rank, static-axis contract.

For a concrete shape, use `reshape` with an explicit dimension list to add or
remove a singleton. `insert(x, axis, 1i64)` adds a singleton at a static position.
For a fixed collection, insert singleton axes and concatenate along that axis.
These recipes do not replace generic List stacking.

The release ledger maintains the remaining issue inventory:
[#1362](https://github.com/Chelis-Lang/chelis/issues/1362).
