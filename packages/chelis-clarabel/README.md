# Chelis Clarabel package

`Clarabel.Qp.solve` accepts dense `f64` tensors for `P`, `q`, `A`, and `b`,
a list of cone descriptions, and explicit solver settings. It returns a
`Solved` result with primal, dual, slack, and diagnostics, or a `Stopped`
result with a termination status. `Solved` is a numerical result; it does
not itself grant an exact optimality theorem.

The evaluator uses the optional in-process Rust provider. From the repository
root, build that evaluator with:

```sh
cargo build -p chelis-cli --features clarabel-provider
```

Run the examples from this package directory:

```sh
../../target/debug/chelis check tests/solve.ch
../../target/debug/chelis eval --file tests/solve.ch
../../target/debug/chelis eval --file tests/nonnegative.ch
../../target/debug/chelis eval --file tests/stopped.ch
```

The examples cover an unconstrained problem, a nonnegative cone, and a
non-solved termination status. Invalid cone dimensions fail before the
solver runs. The optional provider feature is needed for `eval`. The evaluator
calls the native provider only for this package's registered version and exact
`src/qp.ch` source; an edited package runs its own Chelis body instead.

`examples/illustrative/clarabel_qp` is a separate Chelis package with a
path dependency on this package. From that directory, run
`../../../target/debug/chelis eval --file src/main.ch` after the build above.

The package's compiled C binding and the
`clarabel.qp.ideal_optimality` proof contract are not implemented. A C build
of this package executes the unavailable-provider body if run, so it is not
a supported solver path. The exact mathematical proof view must not be
inferred from these runtime examples.
