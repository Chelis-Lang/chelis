`chelis build` now parses the generated C of a unit once when it seals the
unit. Before this change, the seal parsed the same C twice. The emitted C does
not change. For `examples/mnist.ch`, `chelis build --emit-c` retired 1.42G
instructions instead of 1.68G. See
[#3722](https://github.com/Chelis-Lang/chelis/issues/3722).
