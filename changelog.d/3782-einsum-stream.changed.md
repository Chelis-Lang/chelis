A compiled `einsum` that is not a single matrix-shaped contraction no longer
stores every product of an output before summing them, so its memory no longer
grows with the number of products it reduces: `einsum("i,j->", x, y)` over two
8192-element `i64` vectors peaks at about 3 MB instead of 539 MB. Results are
bit-for-bit unchanged, and an integer overflow reports the same trap line as
before. [05-OP-33] now states that an integer einsum's trap does not depend on
the order in which its multiplies and additions run. See
[#3782](https://github.com/Chelis-Lang/chelis/issues/3782).
