`mmap_tensor` is a reserved name, like `to_tensor`: its fourth argument is a
dtype, so no definition, parameter, binding, pattern, or import may bind the
name, and a program that does is rejected with `ReservedName`. See
[#359](https://github.com/Chelis-Lang/chelis/issues/359).
