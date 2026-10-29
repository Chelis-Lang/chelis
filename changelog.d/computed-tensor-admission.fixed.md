Keep host-owned producer operands and computed reshape targets as typed inputs to C tensor helpers, preserving inline range/reshape/expand and symbolic reshape/mean admission, dtypes and runtime guards. Retain checked casts and source-order effects while preparing shape values and registered axis selectors.

Computed concat/softmax examples also execute through the C parity corpus.

Reuse scoped operand facts, persistent host type bindings, demanded helper inputs and source-owned name reservations to avoid repeated operand classification, scope-prefix transport and per-application name scans.
