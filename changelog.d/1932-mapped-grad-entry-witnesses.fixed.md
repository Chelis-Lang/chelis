Mapped gradients now preserve authored runtime-extent entry witnesses, roots,
shape dependencies, and rendered dimension origins through vectorization and
splice. Constant tensors are batch-broadcast with an explicit IR identity, and
incomplete correspondence fails before Eval or artifact publication. Fixes
[#1932](https://github.com/Chelis-Lang/chelis/issues/1932).
