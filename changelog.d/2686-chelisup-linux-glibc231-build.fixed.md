On Linux, `chelisup install` now downloads a release's glibc-2.31 build
(`chelis-v<ver>-linux-x86_64-glibc2.31.tar.gz`) instead of its `linux-x86_64`
build, which needs glibc 2.39. Toolchains installed from that build failed on
the first `chelis` command on distributions with an older glibc. With glibc 2.31
or later, such as Debian 11 and 12, Ubuntu 20.04 and 22.04, and RHEL 9, a new
install now runs; a glibc older than 2.31, such as RHEL 8's, runs neither build.
A version an earlier `chelisup` already installed keeps its `linux-x86_64` build
until you run `chelisup uninstall <ver>` and install it again. Releases before
0.7.24, which mostly do not publish a glibc-2.31 build, still install their
`linux-x86_64` build. The release workflow's installed-artifact canary now
installs the same build. See [#2686](https://github.com/Chelis-Lang/chelis/issues/2686).
