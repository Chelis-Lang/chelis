Compiled C builds now reject unsupported pure-scalar gradients through
host-lane operations such as `fold` with the documented AD workaround
guidance, before ownership lowering and without emitting artifacts. Supported
scalar gradients, including scalar signatures with [05-OP-50] tensor
intermediates, and tensor-bodied primitive-scalar targets remain available.
