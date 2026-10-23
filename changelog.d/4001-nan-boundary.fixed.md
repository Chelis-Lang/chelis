The C host backend preserves signaling NaN bits when f32 values cross
scalar/tensor boundaries or are unboxed from structured values. Exact-width
decoding avoids the floating widening and narrowing that could quiet NaNs
depending on the C compiler's optimization decisions. Arithmetic NaN
canonicalization is unchanged. Fixes
[#4001](https://github.com/Chelis-Lang/chelis/issues/4001).
