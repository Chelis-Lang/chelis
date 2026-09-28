# Registry: Python tensor metadata ([05-OP-45])

This registry is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-45]. Its rows are normative content,
keyed by exact identity; row order and descriptive labels are not identities.

| callable | exact registered PyO3 signature |
|---|---|
| full tensor shape | `chelis_python::NativeTensor::shape(self: &Self) -> Vec<i64>` |
