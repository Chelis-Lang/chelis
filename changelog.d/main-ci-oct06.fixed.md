Surf property fuzzing reuses the checked and lowered library across samples,
guards, and shrinking trials, reducing the cost of properties over standard
library modules. Sample probes still pass the shared compiler checks.
