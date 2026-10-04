Gather now rejects variables and helper calls as axes during checking rather
than silently assuming axis zero and failing during compiled lowering. Literal
and integer-cast-wrapped axes retain negative-axis normalization; sort retains
runtime i32 axes with evaluator/native bounds checks.
