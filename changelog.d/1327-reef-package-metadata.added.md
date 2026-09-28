Reef manifest schema 3 accepts typed package metadata: a description, an SPDX
license or a custom license file, HTTPS repository, documentation, and homepage
URLs, and a README. Declared files use portable paths, no-follow Unix opens,
stable bounded snapshots, and deterministic archive membership. Metadata is
not part of logical package identity or resolution, and its fields are not
serialized into `reef.lock`, `index.json`, or `.chb`. Artifact integrity hashes
still change when declared bytes or the manifest change.
Prepared graph cache version 10 rejects earlier envelopes. See
[#1327](https://github.com/Chelis-Lang/chelis/pull/1327).
