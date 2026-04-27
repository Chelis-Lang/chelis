# `chelis fuzz` Specification

## Purpose

`chelis fuzz` is the toolchain's executable-properties verification mechanism. It discovers `@property` annotations in Chelis source, generates type-directed random inputs, and verifies each property holds. Failure on any input produces a minimal counterexample.

This is the centerpiece of the Trust Stack Level 2 ("executable properties as spec"). For commercial demos to consequential-computing prospects, this is the load-bearing capability.

## Annotation Syntax

Properties are Chelis functions annotated with `@property`. They take typed parameters and return `bool`.

```chelis
@property fn price_is_positive(
  spot: f32, vol: f32, rate: f32, T: f32, strike: f32
) -> bool =
  gt(call_price(spot, vol, rate, T, strike), 0.0)

@property fn delta_in_unit_interval(
  spot: f32, vol: f32, rate: f32, T: f32, strike: f32
) -> bool =
  let d = grad(call_price, wrt=spot)(spot, vol, rate, T, strike)
  in and(gte(d, 0.0), lte(d, 1.0))
```

## Type-Directed Input Generation

The fuzzer generates random inputs from the parameter types:

| Type | Default range | Configurable |
|---|---|---|
| `f32` (no annotation) | `[-1e6, 1e6]` | Yes, via `@range(low, high)` |
| `f32` (positive context, e.g., named `vol`) | `[1e-6, 100.0]` | Yes |
| `i32`, `i64` | `[-1000, 1000]` | Yes |
| `bool` | uniform | n/a |
| `tensor[n, f32]` | element-wise from f32 distribution | Yes |
| `tensor[n, m, f32]` | same, with shape | Yes |
| ADTs | uniform across constructors, recursive on fields | n/a |

Parameter-level range overrides via annotation:

```chelis
@property
@range(spot, 0.01, 10000.0)
@range(vol, 0.001, 5.0)
@range(strike, 0.01, 10000.0)
fn price_bounded_by_spot(
  spot: f32, vol: f32, rate: f32, T: f32, strike: f32
) -> bool =
  lt(call_price(spot, vol, rate, T, strike), spot)
```

## CLI Surface

```
chelis fuzz                        # discover properties in current package, run all
chelis fuzz src/                   # explicit path
chelis fuzz src/pricer.ch          # single file
chelis fuzz --filter delta         # only properties matching "delta"
chelis fuzz --trials 100000        # control sample count (default 10000)
chelis fuzz --seed 42              # reproducible fuzzing run
chelis fuzz --json                 # machine-readable output for CI integration
chelis fuzz --shrink               # enable counterexample minimization (default on)
```

Exit codes:
- 0: all properties pass on all sampled inputs
- 1: at least one property failed (counterexamples in stdout)
- 2: error before fuzzing started (parse error, missing reef context, etc.)

## Counterexample Minimization

When a property fails, the fuzzer attempts to find a smaller failing input by:

1. For numeric inputs: binary-search toward zero or toward simpler values
2. For tensor inputs: try smaller dimensions, simpler element values
3. For ADTs: try simpler constructors first

Minimization runs for a bounded number of attempts (default 100) and reports the smallest failing input found.

Output format:

```
property: matches_textbook_reference (src/properties/pricing.ch:12)
FAILED at trial 4729 of 10000

failing input (minimized):
  spot   = 0.001
  vol    = 3.5
  rate   = -0.02
  T      = 0.001
  strike = 1000.0

property body returned false:
  close(
    call_price(0.001, 3.5, -0.02, 0.001, 1000.0),  -- = 1.234e-308 (subnormal)
    call_price_reference(0.001, 3.5, -0.02, 0.001, 1000.0),  -- = 0.0
    1e-6)  -- abs diff = 1.234e-308, exceeds tol when normalized

1 property failed of 47 sampled (10,000 trials each, 47 properties total)
```

## CI Integration

`chelis fuzz` is a CI gate. Standard CI configuration:

```yaml
jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: chelis fuzz --trials 100000 --seed ${{ github.run_id }}
```

Property regressions block merge. The seed parameter ensures reproducibility within a CI run while varying across runs to surface different failure modes over time.

## Implementation Components

### Compiler-side
- Parser: recognize `@property` and `@range(name, low, high)` annotations on `def` declarations
- Type checker: verify property functions return `bool` and have only Chelis-typed parameters (no effects allowed in the property body itself, though properties may CALL effectful functions if those effects can be handled by the property handler)
- Effect: the body of a property executes under a `Test`-like effect handler (or in eval mode), allowing assertion-style logic. The property itself is pure for purposes of fuzzing.

### Runtime-side
- Random input generator: type-directed, with range overrides from annotations
- Property runner: iterate trials, evaluate property body via `eval_in_context`, collect failures
- Counterexample minimizer: binary-search shrinker per type
- Output formatter: human-readable and JSON

### CLI integration
- `chelis fuzz` subcommand
- Discovery: walk `src/` and `properties/` directories, collect `@property`-annotated functions
- Filter, trial count, seed, JSON output flags

## Acceptance Criteria

1. `@property fn` annotations parse and type-check on `def` declarations.
2. Type-directed random input generation works for f32, i32, i64, bool, tensor[n, f32], tensor[n, m, f32], and basic ADTs.
3. `chelis fuzz` discovers and runs all properties in a reef package, with default 10,000 trials per property.
4. Property failures produce minimized counterexamples in human-readable form.
5. `--filter`, `--trials`, `--seed`, `--json` flags work as specified.
6. CI exit codes (0/1/2) work as specified.
7. End-to-end demo: a Shoals example where a buggy implementation (sign flipped on Greeks calculation) is rejected by `chelis fuzz` with a clear counterexample, and the same property passes on the corrected implementation.

## Relationship to `chelis test`

`chelis test` runs deterministic assertion-based tests via `Std.Test`. `chelis fuzz` runs property-based tests via `@property`. They are complementary, not competing. A typical reef package has both:

- `tests/` directory: `chelis test` for deterministic specific cases
- `properties/` directory: `chelis fuzz` for property-based random testing

CI runs both gates.
