The release workflow extracts the tagged release notes in a preflight job
before any platform build starts, so a tag whose commit still carries
`changelog.d/` fragments fails within a minute instead of after the builds.
Previously the check ran only in the publish job, and the v0.18.8 tag failed
there about 37 minutes in. See
[#2058](https://github.com/Chelis-Lang/chelis/pull/2058).
