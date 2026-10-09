Compiled C code no longer calls into the runtime to compute every element
offset of a permute, expand, insert, pad, shrink, stride, reduction, or
windowed reduction. Each checked plan now proves at construction that every
reachable offset lies inside its target, and the generated loops read the
plan's offset terms once and index with plain arithmetic. Results and traps
are unchanged; an f32 matmul built from `matmul` uses about a thirteenth of the
CPU time it did. The C runtime adds `chelis_movement_term`,
`chelis_movement_base`, `chelis_reduction_term`, and `chelis_window_term` to
observe those terms. See
[#3353](https://github.com/Chelis-Lang/chelis/issues/3353).
