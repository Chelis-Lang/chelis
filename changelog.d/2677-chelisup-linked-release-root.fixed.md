`chelisup install` refuses a release tarball whose top-level `chelis-v*` entry
is a link rather than a directory. Through such a link the runtime check read
files outside the release, and the install could leave a toolchain entry that
pointed outside the store or dangled and blocked reinstalling that version. See
[#2677](https://github.com/Chelis-Lang/chelis/issues/2677).
