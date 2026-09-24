HIP builds now reject pad and shrink programs that would silently execute through
the C host instead of emitting device kernels. The remaining HIP implementation
is tracked in [#2493](https://github.com/Chelis-Lang/chelis/issues/2493).
