Reef manifest schema 3 accepts typed package metadata: a description, an SPDX
license or a custom license file, HTTPS repository, documentation, and homepage
URLs, and a README. Declared files use portable paths, no-follow Unix opens,
stable bounded snapshots, and deterministic archive membership. Metadata stays
outside package identity, resolution, `reef.lock`, `index.json`, and `.chb`.
Prepared graph cache version 10 rejects earlier envelopes. See
[#1327](https://github.com/Chelis-Lang/chelis/pull/1327).
