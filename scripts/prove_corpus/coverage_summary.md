# Prove Corpus Coverage Map

Total properties harvested: 12

## Coverage Summary

| Bucket | Count | % |
|--------|-------|---|
| proved | 3 | 25.0% |
| disproved | 1 | 8.3% |
| fuzz_unsupported | 5 | 41.7% |
| fuzz_engine_failed | 2 | 16.7% |
| fuzz_transcendental | 0 | 0.0% |
| fuzz_shape | 1 | 8.3% |

## Per-Shell Breakdown

### chelis (0/7 prove)

| Property | Verdict | Bucket | Reason |
|----------|---------|--------|--------|
| prob_value_in_unit_interval | fuzz_validated | fuzz_unsupported | passed |
| simplex_binder_is_generated | fuzz_validated | fuzz_unsupported | passed |
| too_strong | failed | fuzz_engine_failed | fell to fuzz |
| tok_bounded | unsupported | fuzz_shape | t: type is not supported in L2 v1 |
| bounded | fuzz_validated | fuzz_unsupported | passed |
| always | unsupported | fuzz_unsupported | generator starvation for opaque type `Exact`: rejection samp |
| always | unsupported | fuzz_unsupported | generator starvation for opaque type `Exact`: rejection samp |

### shoals (3/5 prove)

| Property | Verdict | Bucket | Reason |
|----------|---------|--------|--------|
| put_call_parity_reflection | proven_modulo_fuzz_validated_contract | proved |  |
| call_upper_bounded_by_spot | proven_modulo_fuzz_validated_contract | proved |  |
| delta_in_unit_interval | proven_modulo_fuzz_validated_contract | proved |  |
| put_call_parity_corrupted | disproved_modulo_real_arithmetic | disproved |  |
| delta_unknown_contract | unsupported | fuzz_engine_failed | unknown contract `std.normal_cdf.not_a_contract` |

## Headline: Fuzz-Tier Analysis

**8/12 properties fall to fuzz** (67%)

Expected: interesting behavioral properties fall to fuzz; table-stakes polynomial/shape properties prove. This is the reach frontier map.

### Shape-blocked (1)

- `tok_bounded` (chelis corpus/prove_inject_free_not_injected): t: type is not supported in L2 v1

### Engine-failed (2)

- `delta_unknown_contract` (Shoals composites): unknown contract `std.normal_cdf.not_a_contract`
- `too_strong` (chelis corpus/prove_inject_false_fails): fell to fuzz

### Unsupported (5)

- `prob_value_in_unit_interval` (chelis opaque_invariants): passed
- `simplex_binder_is_generated` (chelis opaque_invariants_simplex): passed
- `bounded` (chelis corpus/prove_inject_pass): passed
- `always` (chelis corpus/prove_starve_exact_eq): generator starvation for opaque type `Exact`: rejection sampling accepted 0/300 (0.0000), constructor-based generation accepted 0/0 (0.0000, 0 distinct), both below the floor 0.0100; predicate shape `equality-atoms`; recommended route: Tier B (real semantics): exact float `==` starves both fuzz tiers by design; use a tolerance band over a module constant
- `always` (chelis corpus/prove_starve_min_rate_zero): generator starvation for opaque type `Exact`: rejection sampling accepted 0/300 (0.0000), constructor-based generation accepted 0/0 (0.0000, 0 distinct), both below the floor 0.0100; predicate shape `equality-atoms`; recommended route: Tier B (real semantics): exact float `==` starves both fuzz tiers by design; use a tolerance band over a module constant
