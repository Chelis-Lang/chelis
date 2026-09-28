Reef now parses canonical package identities and prefers valid exact locks.
Package versions use Semantic Versioning without build metadata. Manifest
schema 1 keeps exact dependency semantics; schema 2 activates deterministic
Cargo-style resolver-2 requirements. Noncanonical package names, partial
versions, non-SemVer release tags, and malformed locks, which earlier releases
accepted, now fail closed. See
[#1327](https://github.com/Chelis-Lang/chelis/pull/1327).
