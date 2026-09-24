Float literal patterns now compare at the scrutinee's dtype, so a pattern such as `0.1` matches an `f32` value produced from `0.1`. See [#2438](https://github.com/Chelis-Lang/chelis/issues/2438).
