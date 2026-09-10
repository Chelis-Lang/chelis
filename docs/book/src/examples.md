# Examples

The repository keeps runnable examples in `examples/` and non-contract teaching sketches
in `examples/illustrative/`.

## Runnable Corpus

Use these files when you want examples that should survive the normal CLI loop:

- `examples/hello_tensor.ch`: tensor construction and elementwise addition.
- `examples/integer_functions.ch`: integer literal returns and nullary calls in eval and C builds.
- `examples/linreg.ch`: matmul, insert, copy, and reductions.
- `examples/vmap_relu.ch`: transform-oriented tensor flow.
- `examples/tensor_structural_ops.ch`: reshape, permute, pad, and related shape helpers.
- `examples/transformer_block.ch`: larger model-style composition.

Check one example:

```sh
chelis fmt --check examples/hello_tensor.ch
chelis lint --check examples/hello_tensor.ch
chelis check examples/hello_tensor.ch
```

Check the executable corpus:

```sh
python - <<'PY'
from pathlib import Path
import subprocess

for path in sorted(Path("examples").glob("*.ch")):
    subprocess.run(["chelis", "fmt", "--check", str(path)], check=True)
    subprocess.run(["chelis", "lint", "--check", str(path)], check=True)
    subprocess.run(["chelis", "check", str(path)], check=True)
PY
```

## Illustrative Examples

`examples/illustrative/` is for syntax and design sketches that are useful to read but
are not part of the executable corpus. Do not use those files as package acceptance
evidence unless the owning docs say a specific file is executable.

## Package Examples

For Reef package layout and shell authoring, read `packages/chelis-std/reef.toml` and the
source under `packages/chelis-std/src/`. `chelis-std` itself is bundled with the compiler;
use it as a runtime/package-layout reference, not something to install with Reef.
