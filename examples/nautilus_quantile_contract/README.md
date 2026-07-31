# Nautilus quantile contract

This executable Reef project demonstrates a Tier-B proof that consumes the
released `Nautilus.Stats.quantile_vec` monotonicity contract:

```sh
chelis reef setup
chelis prove src/proofs.ch --tier smt-only --json
```

Success is one `property` record with `status:"passed"`, `proof_tier:"smt"`,
and an assumption named `std.quantile.monotonicity` whose implementation is
`Nautilus.Stats.quantile_vec`. The compiler binds that assumption through the
linked dependency declaration; the author-facing import name alone is not a
trust signal.
