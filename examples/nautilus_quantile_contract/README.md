# Nautilus quantile contract

This Reef package shows `chelis prove` relying on a library function's declared
contract. `src/proofs.ch` states that, for a fixed data set, the quantile at
level `p` never exceeds the quantile at a higher level `q`. The property cites
the `std.quantile.monotonicity` contract, which `Nautilus.Stats.quantile_vec`
declares, and the SMT solver proves the property with that contract as an
assumption.

The Nautilus dependency is a path dependency on `fixtures/nautilus`, an in-tree
copy of the functions this example uses from Nautilus 0.7.36, so the package
needs no download beyond the toolchain.

## Run it

From this directory, with the toolchain named by the `compiler` pin in
`reef.toml` installed:

```sh
chelis reef setup
chelis prove src/proofs.ch --tier smt-only --json
```

The SMT tier needs a compiler built with SMT support, as release binaries are.
Success is one `property` record with `status:"passed"`, `proof_tier:"smt"`,
and an assumption named `std.quantile.monotonicity` whose implementation is
`Nautilus.Stats.quantile_vec`. The compiler binds that assumption through the
linked dependency's declaration, so the import name alone does not decide which
implementation the proof trusts.
