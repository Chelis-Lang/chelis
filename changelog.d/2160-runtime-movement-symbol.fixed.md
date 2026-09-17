Generated-C native test drivers now call authored definitions through their
canonical `chelis_fn_<utf8-hex>` symbols. Cast, composition, gradient, shape,
movement, and reshape parity oracles retain their numeric assertions and reject
accidental fallback to source spellings. See
[#2160](https://github.com/Chelis-Lang/chelis/issues/2160).
