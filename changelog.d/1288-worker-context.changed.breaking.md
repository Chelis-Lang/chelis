Compiled-context worker handoffs use the same versioned, integrity-checked
envelope as the disk cache. Workers reject unversioned payloads, incompatible
formats or compiler builds, and corrupted payloads before reconstructing the
checked context. Regenerate previously saved worker handoffs with the current
compiler.
