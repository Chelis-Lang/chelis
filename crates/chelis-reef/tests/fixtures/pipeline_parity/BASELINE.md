# Reef pipeline parity baseline

The capture source is revision `e1065d94fbd7a41f086e0690929a66c2335accc5`.

The capture used the Reef pipeline from that revision before the core extraction.
The parity harness and fixtures were overlaid on that revision for the capture.
The accepted child process set `SOURCE_DATE_EPOCH=315532800` for stable artifact metadata.

The accepted capture recorded these SHA-256 values:

- archive: `1440ad4e85ec08808ebf864b43d344d8cea1dd05b59d7ce68124d3f408089969`
- shell: `581028070482676b021f56e71bd936f6156c6bb4dde95fa02984f5647d4c781f`

The rejected capture recorded the complete type, effect, and linearity error output.
The files under `rejected/` preserve the exact text and order.
