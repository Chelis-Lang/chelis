Generated C validates elementwise input shapes through checked runtime metadata
before allocating or repurposing output storage. Shared elementwise, fused, cast,
and realization loops use exact scalar or identity index projections, including
empty and dynamic-rank domains; scalar fused inputs no longer enter flat-read
optimized paths for multi-element outputs. The runtime exposes two tagged-shape
and tensor-domain validation operations for generated callers.
