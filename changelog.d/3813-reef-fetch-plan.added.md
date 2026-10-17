`chelis reef fetch-plan` prints, as JSON, the public download URL and the pinned
SHA-256 of each archive, shell, and binary that `reef.lock` locks.
`chelis reef install --from-lockfile --artifact-dir <DIR>` imports those files
from a directory with no network request. It checks each file against
`reef.lock` before any registry write and keeps the locked `remote_origin`.
A tool such as Nix can thus fetch locked packages itself and give them to
Reef. See [#3813](https://github.com/Chelis-Lang/chelis/issues/3813).
