Linux releases publish a static build, `chelis-v<ver>-linux-x86_64-static.tar.gz`,
and the bootstrap `chelisup-linux-x86_64` is static too. Both are static-pie
executables that need no program interpreter or system library, so they start on
any x86-64 Linux distribution, including NixOS without nix-ld and musl systems.
On Linux, `chelisup install` installs a release's static build when the release
publishes one, and its `linux-x86_64-glibc2.31` build otherwise. Releases still
publish the `linux-x86_64` and `linux-x86_64-glibc2.31` builds, so earlier
`chelisup` versions and existing downloads keep working. The static build ships
the same glibc runtime archive as the others, so native builds link it with a
glibc C toolchain. Its host, user, and group lookups use only `/etc/hosts`, the
name servers in `/etc/resolv.conf`, `/etc/passwd`, and `/etc/group`; it never
loads an NSS plugin such as systemd's `resolve` or `myhostname`, because a
plugin built for the host's own glibc crashes a static executable.
