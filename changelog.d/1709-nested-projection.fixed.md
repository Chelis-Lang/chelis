The Surf formatter preserves the grouping required between adjacent numeric
tuple projections, so formatting `(pairs.0).0` no longer corrupts a valid
program into an unparseable float token. See
[#1709](https://github.com/Chelis-Lang/chelis/issues/1709).
