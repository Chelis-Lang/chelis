A compiled `matmul` no longer stores its expanded operands or their product.
The C lane now computes each product inside the contraction's canonical sum
tree, so results keep the exact bits `chelis eval` produces. Peak memory for a
256x2304 by 2304x196 `f32` matmul drops from about 1.4 GB to under 8 MB. Before
this fix, the same matmul built three 256x2304x196 intermediates. See
[#3370](https://github.com/Chelis-Lang/chelis/issues/3370).
