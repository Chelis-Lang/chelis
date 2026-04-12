Illustrative Chelis examples that are useful for syntax and design discussion
but are not part of the executable Phase 0 example corpus.

Files in the top-level `examples/` directory are expected to survive `chelis fmt`
and `chelis check` with Phase 0 semantics. Files in this directory are allowed to
exercise broader language forms that are not yet executable on the Phase 0 path.

Package-backed illustrative examples may appear as subdirectories with their own
`reef.toml` and fixture data when a phase needs a durable acceptance artifact that is
broader than the top-level executable corpus.
