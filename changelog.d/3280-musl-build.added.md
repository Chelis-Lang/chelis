Releases add a musl Linux build, `chelis-vX.Y.Z-linux-x86_64-musl.tar.gz`, whose
`chelis` is a static executable carrying a musl runtime archive.
`chelisup install` picks it on a musl system such as Alpine, recognized by
`/bin/sh` using musl's loader, and with it `chelis build` links with the system C
compiler after `apk add gcc musl-dev`. Releases without a musl build still
install the static glibc build there. See
[#3280](https://github.com/Chelis-Lang/chelis/issues/3280).
