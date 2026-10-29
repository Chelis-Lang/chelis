# Examples

Start with `hello_tensor.ch` for elementwise tensor operations,
`native_program.ch` for a small native executable, or
`recursive_cast_targets.ch` for dtype-polymorphic recursion. The
[guide](../docs/book/src/examples.md) walks through complete programs and
their output.

From the repository root:

```sh
chelis fmt --check examples/hello_tensor.ch
chelis check examples/hello_tensor.ch
chelis eval --file examples/hello_tensor.ch
chelis build examples/hello_tensor.ch --output out/hello_tensor/
./out/hello_tensor/hello_tensor
```

The evaluator and executable print the same result. Files that contain only
function definitions, such as `linreg.ch` and `native_library.ch`, check
without printing values and build to static libraries. Invoke a function
from a top-level binding to see its result.

`annotated_concat_softmax.ch` composes tensor concatenation and softmax.
`computed_tensor_consumers.ch` expands a tensor produced from a range and
reduces a symbolic input after reshape. Both examples run through the
evaluator and C executable with the same printed values.

## Package examples

Run package examples from their own directories so relative data paths and
Reef dependencies resolve:

- [Nautilus quantile contract](nautilus_quantile_contract/README.md) checks
  a property using a library contract. Its fixture dependency is included.
- [CSV and JSON pipeline](illustrative/io_pipeline/) reads `train.csv` and
  `config.json`: run `chelis reef setup`, then
  `chelis eval --file src/main.ch` from that directory.
- [Clarabel QP](clarabel_qp/) solves a constrained optimization problem and
  states properties about its ideal mathematical result. Follow the
  [provider setup](../packages/chelis-clarabel/README.md) to enable solver
  execution and the optional proof support.

`io/list_directory.ch` lists files under the working directory using
`Std.Io.Fs`. [Illustrative programs](illustrative/README.md) show additional
language forms and target-specific limits outside the executable corpus.
