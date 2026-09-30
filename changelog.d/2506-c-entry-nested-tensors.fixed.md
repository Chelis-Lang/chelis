A generated C host entry now validates every tensor nested in a parameter
value before the body reads it. A tensor in a tuple, a record or data-type
field, a list, an option or a dictionary value gets the null, dtype, rank and
literal-extent checks a tensor parameter gets, and a tensor at a fixed tuple
position is also compared with the signature's named extents. Each diagnostic
names the parameter path, such as `p.1`, `r.square`, `c.Ints.0` or `xs[1]`.
Previously a nested tensor was checked only where a tensor kernel received it,
under an internal helper name, and one that reached no kernel was read at the
declared dtype and extents whatever it held. A named extent of a tensor inside
an ADT, option or dictionary is not yet compared. See
[#2506](https://github.com/Chelis-Lang/chelis/issues/2506).
