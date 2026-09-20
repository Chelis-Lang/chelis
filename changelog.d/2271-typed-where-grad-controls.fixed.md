The static-control-flow gradient matrix now checks typed IEEE NaN branch
selection and both outcomes of runtime discrete-scalar selection instead of
expecting obsolete pre-`Where` lowering rejections. See
[#2271](https://github.com/Chelis-Lang/chelis/issues/2271).
