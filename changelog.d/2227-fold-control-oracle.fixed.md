The static-condition regression matrix now parses the exact emitted `pick`
definition, traces the exact `<` condition and integer operands, and verifies
exactly one fail invocation and value marker in their respective control-flow
arms. It rejects declarations, duplicates, same-arm and unused-function
decoys, and checks exact stderr for taken and untaken host branches. See
[#2227](https://github.com/Chelis-Lang/chelis/issues/2227).
