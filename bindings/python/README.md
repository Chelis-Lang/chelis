# Chelis for Python

The `chelis` package exposes the Chelis compiler and evaluator to Python. It checks
and evaluates Surf or Deep source, converts between the two source forms, returns
generated build files, and loads callable compiled tensor models.

## Install from a checkout

The Python package requires Python 3.10 or later. Building it from a Chelis checkout
also requires Rust and a C toolchain.

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -e bindings/python
```

The editable install uses the compiler and runtime in the checkout. Rebuild the
extension after changing its declared runtime sources.

To build a wheel from the checkout:

```sh
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

## Check and evaluate source

```python
import chelis

source = """module Example
answer = add(20, 22)
"""

checked = chelis.check(source)
print(checked.score, checked.errors)

result = chelis.eval(source)
for root in result.roots:
    print(root.name, root.value)
```

`check` returns a fitness score, its components, and structured diagnostics.
`eval` returns named roots and their values. Pass NumPy arrays, `ChelisTensor`
objects, or CPU tensors that implement DLPack as named bindings when the source
declares tensor inputs.

The source conversion helpers are `desugar` (Surf to canonical Deep) and
`decompile` (Deep to canonical Surf). `validate` checks a source document in
Surf or Deep mode. `compile` returns generated target files and their compile
and link flags without executing them.

## Run compiled tensor models

`compile_and_load` compiles a source file and returns a callable `CompiledModel`.
Choose an entry with `entry_name=` when the file has more than one tensor
definition. Compiled entries use fully concrete tensor dimensions. The C target
supports `f32` and `f64`; the HIP target supports `f32` and requires a compatible
HIP environment. Scalar and host-program entries run through `eval`.

```python
import chelis

model = chelis.compile_and_load("model.ch", entry_name="main")
print(model.input_names, model.output_names)
```

`load` opens an existing compiled model. The loader checks the artifact's runtime
and library digests before loading it and warns when its source file no longer
matches the recorded source hash.

## Tensor data

`from_dlpack` wraps a CPU tensor without copying. `save_safetensors` and
`load_safetensors` read and write named tensor collections.

Evaluation preserves supported integer and floating-point storage values across
the Python boundary. Supported NumPy inputs include signed 8-, 16-, 32-, and
64-bit integers; `float16`, `float32`, `float64`, and `bfloat16`; and booleans.
Unsigned 8-, 16-, and 32-bit integers widen exactly to the next supported signed
width. Unsupported dtypes, including `uint64`, raise `ChelisError` and require
an explicit caller conversion.

`ChelisError` reports compiler, build, and runtime failures. `ValueError` reports
invalid inputs and attempts to pass GPU tensors through the CPU-only DLPack and
evaluation APIs.
