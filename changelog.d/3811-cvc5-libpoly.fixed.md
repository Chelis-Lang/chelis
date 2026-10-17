The Nix flake builds cvc5 with the configuration of a release build: LibPoly on,
the Production build type, and GPL off. Before, the flake built cvc5 without
LibPoly, and SMT results could differ from a release build. The cvc5 metadata now
names the BSD 3-clause license of this build. See
[#3811](https://github.com/Chelis-Lang/chelis/issues/3811).
