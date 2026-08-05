# Reef pipeline parity baseline

The original capture source is revision `e1065d94fbd7a41f086e0690929a66c2335accc5`.

That capture used the Reef pipeline from that revision before the core extraction.
The parity harness and fixtures were overlaid on that revision for the capture.
The accepted child process set `SOURCE_DATE_EPOCH=315532800` for stable artifact metadata.

## Re-freeze after merging `main` (7db8c271)

The frozen baselines below were **regenerated** when this branch merged `main` at
`7db8c271`. `main` had evolved independently of the extraction: the Surf grammar now
requires a parameter list on `def ... -> T` and uses newlines (not `;`) as block
separators, and codegen changed the compiled archive bytes. The fixtures were updated to
the current grammar and the baselines were re-captured against the merged compiler.

This weakens the test's original claim. It no longer proves "the extraction preserves the
exact pre-extraction output"; it now proves "the extraction preserves output versus this
re-captured post-merge baseline." The interface metadata (exports, type reprs, schema) was
byte-identical across the re-freeze; only the archive/shell SHA-256 values and the
rejected-error span offsets changed, which is consistent with a grammar/codegen change
rather than a behavior regression.

Follow-up: convert these frozen absolute snapshots to the relative monolithic-vs-layered
equivalence oracle (which needs no re-freeze on upstream change), or retire them now that
the one-time migration is verified. Tracked separately.

## Recorded SHA-256 values (post-merge re-freeze)

The accepted capture recorded these SHA-256 values:

- archive: `c440b1362a9a4f2a256ab0e41b2d220d159c16570eaf32eaf9256acd2c0a6f7e`
- shell: `01310dfed2b16b7d034aa641948aa7b8c1ed354d93e5d9d1df2c28f06f8b2b8f`

The rejected capture recorded the complete type, effect, and linearity error output.
The files under `rejected/` preserve the exact text and order.
