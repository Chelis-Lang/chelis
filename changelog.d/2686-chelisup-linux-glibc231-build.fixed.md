On Linux, `chelisup install` now downloads a release's glibc-2.31 build
(`chelis-v<ver>-linux-x86_64-glibc2.31.tar.gz`) instead of its `linux-x86_64`
build, which needs glibc 2.39. Toolchains it installed failed on the first
`chelis` command on distributions with an older glibc; with glibc 2.31 or later,
such as Debian 11 and 12, Ubuntu 20.04 and 22.04, and RHEL 9, they now run.
Releases before 0.7.24, which mostly do not publish a glibc-2.31 build, still
install their `linux-x86_64` build. The release workflow's installed-artifact
canary now installs the same build. See
[#2686](https://github.com/Chelis-Lang/chelis/issues/2686).
