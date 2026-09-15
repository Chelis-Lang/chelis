The `diagonal` and `trace` axis rules now have expectations that neither runtime
computes. A new CLI suite derives extents and elements from [05-OP-33] and
requires the IR evaluator lane and the compiled C lane to agree with those
expectations separately, over all six ordered rank-3 axis pairs, both rank-2
orders on a non-square matrix, and a signed source. The parity harness header
now states what lane agreement does and does not establish. See
[#1351](https://github.com/Chelis-Lang/chelis/issues/1351).
