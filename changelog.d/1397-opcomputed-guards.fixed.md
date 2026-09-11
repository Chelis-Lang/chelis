A declared result dimension over a runtime-bound `shrink` is kept as a claim and
checked at run time. A function declaring `-> tensor[2, f32]` whose body shrinks
to a different extent previously returned that other extent and exited zero on
both the evaluator and the compiled lane; it now raises
`numeric trap: domain in shrink at int64` at the `shrink`, before the result is
allocated, with the claimed and observed extents on an accompanying line. Extents
an operation computes are guarded generally: an equality class may now hold such
an extent as a member, and `pad` with non-zero padding and `stride` with a
non-unit step remain unguarded. The context line accompanying a local extent trap
now names the operation instead of an internal node number, on every lane. See
[#1397](https://github.com/Chelis-Lang/chelis/issues/1397).
