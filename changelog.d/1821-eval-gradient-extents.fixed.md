Preserve authored named-extent claims when evaluating gradients of tensor, aggregate and primitive scalar targets, including multiple targets and unused zero cotangents. A rejected forward activation reports its required extent failure instead of producing a gradient.

Independent calls retain distinct extent claims, including computed result extents. Their guards keep the original producer attribution and preserve the claimed value until the guard executes in generated C.
