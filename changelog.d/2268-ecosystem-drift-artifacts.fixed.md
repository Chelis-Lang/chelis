Rebuild the ecosystem drift canary's selected release source graph from immutable
peeled commits with HEAD into strictly verified current-format artifacts,
preventing moved tags, predecessor envelopes, release-bootstrap failures, or
auto-fetch from masking source and API drift. Private exact sources are acquired
through one authenticated full clone, and checkout/authentication failures are
reported as infrastructure rather than source drift.
