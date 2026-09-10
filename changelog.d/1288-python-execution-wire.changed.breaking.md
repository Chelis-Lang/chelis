Python evaluation uses execution wire v3's exact numeric carriers. Floating-point
values preserve their IEEE bits, including signed zero and NaN payloads; numeric
scalar results retain their NumPy dtype, with `ml_dtypes.bfloat16` for `bf16`.
Tensor results retain the declared dtype and exact element values. Old execution
versions and malformed carriers are rejected. Part of
[#1288](https://github.com/Chelis-Lang/chelis/issues/1288).
