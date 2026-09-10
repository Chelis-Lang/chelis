Serialize the wire and Python binding capacity verifiers that intentionally share
their nested rustdoc Cargo target, preventing CI-only lease contention without
weakening the target ownership guard.
