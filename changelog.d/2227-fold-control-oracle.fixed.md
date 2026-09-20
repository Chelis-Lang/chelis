The static-condition regression matrix now parses the exact emitted `pick`
definition and verifies fail and value markers in their respective
control-flow arms, rejects same-arm and unused-function decoys, and checks
exact stderr for taken and untaken exact-integer host branches. See
[#2227](https://github.com/Chelis-Lang/chelis/issues/2227).
