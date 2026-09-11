`expand` and `insert` now accept a size that is checked integer arithmetic over
a runtime value, such as `insert(b, 0, mul(shape(x, 0), 2i64))`. The compiled
lanes previously refused to lower such a size while `chelis eval` executed or
crashed on it. The extent is now ordinary integer dataflow that every lane
executes, and a literal or named claim over the axis is checked at execution
and traps `Domain` on mismatch. Arithmetic whose operands carry no shape source
at all, such as `add(k, 1i64)` over a bare scalar parameter, is still rejected
at check with its existing diagnostic.
