Native Arb oracle builds now use the maintained `arb-sys` revision that explicitly
locates the bundled MPFR installation, rather than failing when `/usr/local/lib`
is absent. The revision also uses a standalone FLINT dependency, so a clean
checkout needs no sibling native-library repositories. The default Devenv
toolchain also supplies the `m4` processor needed for cold GMP builds.
The CI cache binds its fixed-output source hash to the reviewed Git revision,
rejecting lock updates that would otherwise reuse an older cached source.
