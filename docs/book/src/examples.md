# Examples

The commands on this page run from the root of a Chelis source checkout, where
the `examples/` directory is available. Install the toolchain first if `chelis`
is not on your path.

## Examples with results

These files have a value or an entry point that `chelis eval --file` can run:

- `examples/hello_tensor.ch`: constructs tensors and adds them elementwise.
- `examples/integer_functions.ch`: integer values, nullary calls, and a tensor gradient.
- `examples/recursive_cast_targets.ch`: recursive generic scalar functions at several dtypes.
- `examples/tensor_structural_ops.ch`: reshaping, concatenation, gather and scatter,
  window reductions, and convolution.

Try one:

```sh
chelis eval --file examples/hello_tensor.ch
```

## Definitions to check and reuse

These files define functions but have no value or entry point for the evaluator
to print on their own:

- `examples/linreg.ch`: matrix multiplication, explicit bias insertion, and reductions.
- `examples/vmap_relu.ch`: vectorized ReLU over a batch.
- `examples/transformer_block.ch`: a larger model function built from tensor operations.

Check one before adapting its functions in your own program:

```sh
chelis fmt --check examples/linreg.ch
chelis lint --check examples/linreg.ch
chelis check examples/linreg.ch
```

A successful `check` confirms that the source passes the checker; it does not
mean evaluating the file will print a result. See [First Program](first-program.md)
for a file that does both.

## Other example sources

`examples/illustrative/` contains sketches for reading and adapting. They are
outside the top-level checkable corpus, so check an individual file before
using it in a program.

For a Reef package layout, see `packages/chelis-std/reef.toml` and its `src/`
directory. `chelis-std` is bundled with the toolchain; see
[Reef and Packages](reef.md) to build your own package.
