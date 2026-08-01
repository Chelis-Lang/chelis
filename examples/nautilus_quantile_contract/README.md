# Nautilus quantile contract

This executable Reef project demonstrates a Tier-B proof over the released
`Nautilus.Stats.quantile_vec` surface. Its path dependency is a hermetic,
byte-for-byte function-level mirror of Nautilus 0.7.36 under
`fixtures/nautilus`; keeping that fixture in-tree lets compiler release bumps
regenerate an honest lock without depending on a not-yet-released downstream
shell:

```sh
chelis reef setup
chelis prove src/proofs.ch --tier smt-only --json
```

Success is one `property` record with `status:"passed"`, `proof_tier:"smt"`,
and an assumption named `std.quantile.monotonicity` whose implementation is
`Nautilus.Stats.quantile_vec`. The compiler binds that assumption through the
linked dependency declaration; the author-facing import name alone is not a
trust signal. The chelis#979 CLI acceptance suite consumes these same fixture
files, so the executable example and its test cannot drift.
