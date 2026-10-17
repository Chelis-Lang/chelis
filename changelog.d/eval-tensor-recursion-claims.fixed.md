`chelis eval --file` no longer uses memory quadratic in recursion depth when
a recursive function carries a tensor. Each call re-declared its result
claim and also inherited an equal copy from every enclosing call, so a call
at depth `d` held `d` claims. Equal claims are now kept once. A bounded
3,000-deep recursion over a `tensor[2, f32]` drops from about 2 GB peak RSS
to 66 MB, the same as the `i64` form. A recursion that never terminates now
stops with `evaluation stack budget exhausted during a function call`, as
the `List` form does, instead of growing the heap until the process is
killed. Results and diagnostics are otherwise unchanged. See
[#3838](https://github.com/Chelis-Lang/chelis/issues/3838).
