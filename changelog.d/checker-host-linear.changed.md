`chelis check` and the C host lane no longer redo module-sized work for every
definition. The checker shares its substitution with the speculative
unifications it may discard, instead of copying it for each annotated result
and each route a binder-suspended call is replayed under. Host lowering
derives its call-graph facts and its index of definition bodies once per
program, instead of once per definition, and the record-projection hoist
rebuilds each node from its parts instead of copying the subtree beneath it.
Results, diagnostics and emitted C are unchanged. Checking 2,000 annotated
definitions drops from about 51 G to 20 G instructions, checking 1,600
definitions that call `range` on binder-typed operands from 37 G to 7 G, and
checking every file of a large numerical library by about 26 percent in
total. Emitting C for a definition whose body chains 2,000 bindings drops
from about 147 G to 24 G instructions. See
[#3784](https://github.com/Chelis-Lang/chelis/issues/3784),
[#2536](https://github.com/Chelis-Lang/chelis/issues/2536),
[#3785](https://github.com/Chelis-Lang/chelis/issues/3785) and
[#3793](https://github.com/Chelis-Lang/chelis/issues/3793).
