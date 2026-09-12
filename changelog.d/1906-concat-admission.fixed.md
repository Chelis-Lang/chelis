Preserve evaluator execution of annotated softmax helpers over tensor concat
when runtime concat extents come directly from inputs, lexical aliases, or
helper forwarding. Static-width concat kernels remain eligible. Forced tensor-DAG
concat now rejects unsupported construction instead of fabricating a scalar input.
Computed dynamic concat and C host/tensor partitioning remain outside this repair.
