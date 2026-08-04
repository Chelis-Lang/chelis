# Reef pipeline parity baseline

The capture source is revision `261d64260d5a974c06f15245ecd31b671ecc80a4`.

The capture used the Reef pipeline from that revision before the core extraction.
The parity harness and fixtures were overlaid on that revision for the capture.
The accepted child process set `SOURCE_DATE_EPOCH=315532800` for stable artifact metadata.

The accepted capture recorded these SHA-256 values:

- archive: `45ceb75c06ba72f9b1869047fbd7323946f480260e51cdb12c00fd0a7572bac0`
- shell: `f3ce4b979b7cb5af215a4e76556b240dec3c6a4a157863985a32400f5d80b184`

The rejected capture recorded the complete type, effect, and linearity error output.
The files under `rejected/` preserve the exact text and order.
