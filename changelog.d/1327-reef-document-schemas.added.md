Reef documents now have independent versioned schemas. New manifests write
schema 3 and new lockfiles write schema 1. Legacy files remain readable with an
upgrade warning. `chelis reef upgrade --check|--inplace` provides preflighted
ordered migration. Project writers use `.reef-write.lock` and atomic
single-file replacement. Versioned JSON Schema files under `docs/schemas/reef/`
provide advisory editor validation. See
[#1327](https://github.com/Chelis-Lang/chelis/pull/1327).
