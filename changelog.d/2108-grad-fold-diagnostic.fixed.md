Compiled C builds now reject unsupported pure-scalar gradients through
host-lane operations such as `fold` with the documented AD workaround
guidance, before ownership lowering and without emitting artifacts. Supported
scalar gradients and tensor-bodied primitive-scalar targets remain available.
