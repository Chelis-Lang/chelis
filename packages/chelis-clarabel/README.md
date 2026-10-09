# Chelis Clarabel package

`Clarabel.Qp.solve` accepts dense `f64` tensors for `P`, `q`, `A`, and `b`,
a list of cone descriptions, and explicit solver settings. It returns a
`Solved` result with primal, dual, slack, and diagnostics, or a `Stopped`
result with a termination status. `Solved` is a numerical result; it does
not itself grant an exact optimality theorem.

The evaluator and compiled C backend use the optional Rust provider. From the
repository root, build the CLI with:

```sh
cargo build -p chelis-cli --features clarabel-provider
```

Run the examples from this package directory:

```sh
../../target/debug/chelis check tests/solve.ch
../../target/debug/chelis eval --file tests/solve.ch
../../target/debug/chelis eval --file tests/nonnegative.ch
../../target/debug/chelis eval --file tests/stopped.ch
mkdir -p build/clarabel-c
../../target/debug/chelis build --target c --output build/clarabel-c tests/solve.ch
build/clarabel-c/solve
```

The examples cover an unconstrained problem, a nonnegative cone, and a
non-solved termination status. Invalid cone dimensions fail before the
solver runs. The optional provider feature is needed for execution. The
provider call is admitted only for this package's registered version and exact
`src/qp.ch` source; an edited package runs its own Chelis body instead.

`examples/clarabel_qp` is a separate Chelis package with a
path dependency on this package. From that directory, run
`../../target/debug/chelis eval --file src/main.ch` after the build above.
Its `optimize(target)` function constructs a typed QP from an argument and
returns the solver's status-bearing result to `main`.

For the ideal mathematical proof view, build with both provider and SMT features:

```sh
cargo build -p chelis-cli --features clarabel-provider,smt
```

From `examples/clarabel_qp`, run:

```sh
../../target/debug/chelis prove tests/ideal_stationary.ch --tier smt-only --json
../../target/debug/chelis prove tests/ideal_baseline.ch --tier smt-only --json
../../target/debug/chelis prove tests/ideal_wrapper.ch --tier smt-only --json
../../target/debug/chelis prove tests/ideal_symbolic_constrained_quality.ch --tier smt-only --json
../../target/debug/chelis prove tests/ideal_symbolic_assumed_psd.ch --tier smt-only --json
```

These properties opt into `clarabel.qp.ideal_optimality` for the imported
`solve` call's `Solved` arm. The proof result is conditional on an asserted
ideal real optimizer axiom and carries `real_arithmetic`; it does not certify
the returned floating-point bits. `ideal_wrapper.ch` passes a typed tensor
through a Chelis helper that returns one `solve` call; the proof follows that
same dependency-owned call. The proof lowering handles fixed-size symbolic QP
data with zero and nonnegative cones, including a verified `B^T B` PSD
construction. A property using another symbolic `P` can explicitly add
`clarabel.qp.assume_psd`; the report lists that call-bound PSD premise as a
separate axiom. Runtime calls also support
second-order, exponential, power, and generalized-power cones. An unsupported
proof shape reports `unsupported`, and a changed provider source cannot grant
the axiom. Without the provider feature, the package declaration's fallback
body traps if called.
