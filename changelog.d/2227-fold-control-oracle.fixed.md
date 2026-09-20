The static-condition regression matrix now scopes branch-presence checks to
the emitted `pick` body, rejects unused-function decoys, and checks exact
stderr for both taken and untaken exact-integer host branches. See
[#2227](https://github.com/Chelis-Lang/chelis/issues/2227).
