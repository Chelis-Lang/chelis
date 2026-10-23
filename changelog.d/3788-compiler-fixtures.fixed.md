Compiler-wrapper tests write executable fixtures in a dedicated child process,
preventing concurrent process spawns from retaining writable handles that make
Linux execution fail with `ETXTBSY`.
