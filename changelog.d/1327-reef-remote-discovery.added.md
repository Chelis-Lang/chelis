Reef now performs bounded GitHub release discovery. Normal commands reuse a
valid lock without listing releases. `chelis reef update [<package>]` performs
a full or targeted refresh, and `chelis reef outdated [<package>] [--json]`
reports available versions without final writes. Candidate scans, requests,
downloads, and resolver states have finite limits, and Reef publishes complete
verified cache entries before it replaces `reef.lock`. An explicit `chelis-std`
dependency uses the compiler bundle without network access. Cyclic path
dependency graphs fail with an ordered cycle report before resolution. See
[#1327](https://github.com/Chelis-Lang/chelis/pull/1327).
