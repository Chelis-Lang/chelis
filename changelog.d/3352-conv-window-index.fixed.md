`chelis build` no longer writes a convolution's whole window-index table into
the generated C. `conv` gathers each output window by a flat index into the
padded input. The compiler used to embed that index as a constant with one
entry per (input channel and kernel tap, output position) pair, about 87 bytes
of C per entry. A ResNet18 forward pass generated 1.26 GB of C, which clang
cannot compile. The index is a sum of a per-row term and a per-column term, so
the compiler now embeds the two short vectors and adds them at run time. Results
are unchanged, and that ResNet18 build now generates about 6.4 MB of C. See
[#3352](https://github.com/Chelis-Lang/chelis/issues/3352).
