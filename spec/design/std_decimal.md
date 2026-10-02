# Std.Decimal: exact decimal numbers

Tracker: chelis#2778.
Prior art: [`decimal_prior_art.md`](../../docs/investigations/decimal_prior_art.md),
cited below as "prior art §N".

This is the design of record for `Std.Decimal`. It records the reasoning behind the
surface; the normative text is the spec/05 atom [05-OP-76] and its registry rows (§12).
Where this document and a numbered chapter disagree, the chapter wins and this document
has a bug.

## 1. Why

The previous `Std.Decimal`, fenced by #2806 under #2778, could not meet its own contract,
the Decimal paragraphs of [05-OP-35]:

- The value was `Decimal { coefficient: i64, scale: i64 }`, and [05-OP-35] asked for exact
  rational arithmetic over every i64 scale, trapping only when the final canonical pair
  has no i64 representation. Every intermediate was checked i64 instead: parse
  accumulation, scale alignment, `pow10`, products, comparisons and `abs_int(i64::MIN)`.
  So `decimal("9223372036854775807.0")` trapped on a removable trailing zero.
- Public constructors accepted any field pair, including a negative scale, and no
  operation validated or canonicalized its inputs.
- `decimal_to_float` rounded the numerator and the denominator separately, then divided,
  so it was not correctly rounded.
- The contract said operations "trap `Domain`" or "trap `Overflow`", which a library
  definition cannot do: [04-NUM-9] traps are raised by primitives. The code used free-text
  `fail`, and a negative division scale leaked a raw string error (#681).
- `RoundDown` meant toward zero, which Swift spells for floor (prior art §5), and four
  `round_*()` functions duplicated the enum's own constructors.

The deeper problem was the representation. An exact-rational contract over an i64
coefficient and an unbounded scale needs multi-precision intermediates anyway, so the
i64 coefficient bought only overflow paths. It was also too small for the finance
arithmetic decimals exist for: an 18-decimal crypto amount above 9.2 units does not fit,
and neither does the exact product of a 9-digit notional and an 11-digit FX rate.

Decimal arithmetic is a solved problem in the sense that mature libraries agree on most
of the model (prior art §14). This design adopts that consensus and, where libraries
diverge, applies the Chelis tenets.

## 2. Principles as applied

- **Exact at its own width.** A `Decimal` is exact the way a Chelis `i64` is exact: every
  operation returns the exact result or fails loudly. Nothing rounds except an operation
  that is given a scale and a rounding mode.
- **No context, no default.** There is no precision context, global or otherwise (prior art
  §4), and no callable has a default rounding mode. Every rounding operation names its
  scale and its mode.
- **One value, one representation.** Values are canonical, so equality is structural and
  there is no cohort to trip over (prior art §3).
- **Opaque values, validating producers.** `Decimal` is `@opaque`; the exported callables
  are the only way to obtain one, so an invalid decimal cannot exist outside the module.
  This closes #2778's unvalidated-value class by construction, as Std.Datetime does for
  its types.
- **Bounded, so failures are functions of the inputs.** Every value fits a fixed envelope,
  every operation is constant-size, and no input can make a computation's cost or outcome
  depend on host memory (§3).
- **Small surface.** An operation that composes from others without loss is not added
  (§11).

## 3. The value

**Value set.** A `Decimal` is the rational `c / 10^s` for an integer coefficient `c` with
`|c| <= 10^38 - 1` and a scale `s` with `0 <= s <= 38`. There is no NaN, no infinity and no
negative zero.

**Canonical form.** A nonzero value's coefficient has no trailing decimal zero while
`s > 0`; zero is `(0, 0)`. So `1.50` and `1.5` are one value, `(15, 1)`, and `100` is
`(100, 0)` (scale never goes negative). The canonical scale is the number of fractional
digits the value needs.

**When a result is outside the set.** A rational is in the set exactly when its canonical
pair is: at most 38 significant digits and at most 38 fractional digits. A result can
therefore leave the set two ways:
- too many significant digits: `decimal_add(9e37, 1e37)` is `10^38`;
- too many fractional digits: `decimal_mul(1e-20, 1e-19)` is `1e-39`.

`decimal_div(10, 3, 38, r)` leaves the set (39 significant digits) and
`decimal_div(10, 3, 37, r)` does not. The product of two operands is exact whenever their
significant digits sum to at most 38 and their scales sum to at most 38, which covers
`round(amount * rate, 2)` for a 9-digit amount and an 11-digit rate.

**Why 38 digits.** 38 is the precision ceiling of SQL Server, Snowflake, BigQuery `NUMERIC`
and DuckDB decimals and of Arrow `decimal128` (prior art §9, §10; PostgreSQL's `numeric`
reaches far above it), so every `decimal128` value whose exact value has at most 38 significant digits
and 38 fractional digits is a `Decimal`. It holds 18-decimal crypto amounts and the exact product of
any two values that fit an i64 coefficient with scales summing to at most 38. An 18-digit
envelope (the old coefficient width, Arrow `decimal64`) overflows on ordinary rate
arithmetic; a 76-digit envelope (`decimal256`) doubles the cost of every operation for
range no surveyed workload needs.

**Why bounded, not arbitrary precision.** Arbitrary-precision decimals and their parsers
have a long resource-exhaustion record: Ruby, Go, Jackson, Haskell's `Data.Scientific` and
PostgreSQL added exponent, length or growth limits after advisories, `apd` bounds
exponents by design, and the rest leave the exposure to callers (prior art §12). Once such
limits exist the type is bounded anyway, with a bound chosen by an incident rather than a
design. Without them, repeated squaring or a large requested
division scale makes a result's cost, and whether it completes at all, depend on host
memory, which the determinism contract forbids: for fixed program text and declared
inputs, every result must be a function of those inputs. A bounded value also gives
`chelis prove` a linear range constraint to state.

**Why the scale lives in the value.** Systems that put precision and scale in the type
must derive a result type for every operation and cap it; at the cap they either round or
truncate silently (SQL Server, Snowflake) or fail (Arrow compute, DuckDB), and none defines
a natural scale for division (prior art §9). That needs type-level arithmetic over
naturals and still ends in implicit narrowing or in a failure the type did not predict. A
value-level scale with exact `+ - *` and an explicit scale on every rounding operation
says the same thing without type machinery. A property such as "`decimal_scale(x) <= 2`"
states what a type-level scale would.

**Why the scale is never negative.** A negative scale only buys magnitudes above 10^38,
which this envelope excludes, and Parquet forbids negative scales (prior art §10).
Integers keep their trailing zeros in the coefficient.

**Why trailing zeros are dropped.** Keeping cohorts produces two equalities and a
normalizing hash, and two of the surveyed libraries shipped bugs in exactly that seam
(prior art §3). TC39's Decimal made the same choice in 2024 and moved display precision
to a separate type. Display precision is an explicit argument here
(`decimal_to_fixed_string`).

**Representation.** The registered ADT identity is a fixed-arity record:

```
Decimal { negative: bool, limb0: i64, limb1: i64, limb2: i64, limb3: i64, limb4: i64, scale: i64 }
```

The magnitude is `limb0 + limb1·10^9 + limb2·10^18 + limb3·10^27 + limb4·10^36`, each limb
in `[0, 10^9)` and `limb4 < 100`; `negative` is false at zero; `scale` is the canonical
scale. Fixed fields rather than a `List` make the encoding unique without a length rule,
make structural equality plainly numeric equality, avoid a list allocation per value in
the compiled lane, and leave `@invariant` available, which a `List`-backed type cannot
carry. Base 10^9 keeps every limb product and carry below 2^63, the widest Chelis
integer, and makes scaling by a power of ten a limb shift plus one small multiply.

## 4. Failure contract

Decimal follows Std.Datetime's failure contract. A failure is [05-OP-60]'s `fail` with
the message `<function>: <kind>: <detail>`, where `<function>` is the exported callable's
name and `<detail>` names the offending value. `<kind>` is:

- `domain` when an argument or a text denotes no value of the operation's domain or of
  `Decimal`: malformed or over-long text, text or a float whose value is outside the value
  set, a scale outside 0..38, a zero divisor, a NaN or infinite float, or `RejectInexact`
  finding an inexact value;
- `overflow` when the exact result of arithmetic on decimals leaves the value set, or a
  decimal narrowed to i64 leaves i64.

Every range and validity check precedes the arithmetic it guards, so no [04-NUM-9] trap
of a primitive escapes a call for any arguments. When several checks fail, the first in
this order is reported, so the message is deterministic: each argument's own validity in
argument order, then `RejectInexact`, then the result's range.

A `try_` callable takes its twin's arguments and returns `Some` of the twin's result, or
`None` exactly where the twin fails `domain`. A twin exists for each callable with a
`domain` failure that a caller cannot rule out beforehand by composing other callables:
text that may be malformed or out of range (`decimal`), a float that may be non-finite,
out of range or inexact at the scale (`decimal_from_f64`), and a quotient that may be
inexact under `RejectInexact` (`decimal_div`). `decimal_round`'s and
`decimal_to_fixed_string`'s `domain` failures need none, because `decimal_scale(x) <= s`
decides them. `try_decimal_to_i64` also returns `None` where its twin fails `overflow`:
narrowing to i64 fails on ordinary data, as `to_int` returns `None` for out-of-range text.
Arithmetic overflow has no twin. A result beyond 38 digits is a program error, like i64
overflow, and fails in every form.

## 5. Rounding

Every rounding callable takes a `Rounding` from `Std.Rounding`, the rounding vocabulary
Std.Datetime shares, whose modes one atom, [05-OP-74], defines for every callable that
takes them. Each mode maps an exact value `v` and a quantum `q` to an integer multiple
`k·q`:

| Variant | Result |
|---|---|
| `RoundTowardNegative` | the largest multiple `<= v` |
| `RoundTowardPositive` | the smallest multiple `>= v` |
| `RoundTowardZero` | whichever of those two is nearer zero |
| `RoundAwayFromZero` | whichever is farther from zero |
| `RoundTiesToEven` | the nearest multiple; on an exact tie, the one with even `k` |
| `RoundTiesToAway` | the nearest multiple; on an exact tie, the one farther from zero |
| `RejectInexact` | `v` itself when it is a multiple; otherwise the callable fails `domain` |

The names are IEEE 754's rounding attributes (spec/05 already describes `round` as
roundTiesToEven), extended with away-from-zero, which decimal finance needs, and with a
require-exact policy (Java's `UNNECESSARY`, COBOL's `PROHIBITED`, prior art §5).
`RejectInexact` makes exact division, exact float conversion at a scale and exact
narrowing to an integer compositions of the rounding callables rather than separate
functions. Ties toward zero is omitted: no mandated rule in the survey uses it (EU
1103/97 and ISDA round halves up, HMRC rounds down, prior art §11), and IEEE 754, .NET and
TC39 omit it.

## 6. Text

**Grammar.** `decimal` accepts exactly one RFC 8259 number token:
`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`. That is the grammar [05-OP-2] already
applies to CSV numeric cells, without their tolerance for surrounding spaces and tabs.
There is no `+` sign, no leading integer zero, no bare `.5` or `1.`, no whitespace, no
digit separator and no NaN or infinity spelling. JSON, Python's `str(Decimal)` and Java's
`BigDecimal.toString` all emit exponents, and because the value is bounded an exponent
can be checked before anything is materialized.

**Edge cases the grammar admits.** `-0`, `-0.0e5` and `0e99999999999999999999` are zero:
an all-zero significand is zero whatever its exponent. Exponent digits may carry leading
zeros. The exponent is bounded by comparison against the significand's digit counts,
never by subtraction, so `1e-9223372036854775808` fails `domain` without an i64 trap.

**Length bound.** A token longer than 1000 characters fails `domain`. Every canonical
decimal needs at most 41 characters, so a longer token differs from some short one only
by zeros. Without a bound, parsing is quadratic in token length in the evaluator, whose
`string_slice` copies its input, which is the CVE-2020-10735 class; the bound fixes the
worst-case cost in every lane. Python bounds integer text at 4300 digits and Jackson
numbers at 1000 characters for the same reason (prior art §12).

**Rendering.** `decimal_to_string` emits the unique canonical plain form: an optional
`-`, at least one integer digit with no leading zero, and a `.` followed by exactly
`scale` digits when the scale is nonzero. It never emits an exponent, a `+` or a trailing
fractional zero, and it emits zero as `0`. `decimal_to_fixed_string(x, s)` emits exactly
`s` fractional digits (`"1.50"`), never rounds, and fails `domain` when `x` needs more
than `s`.

`decimal(decimal_to_string(x))` is `x` for every decimal, and every text `decimal` accepts
is also accepted by `to_float` ([05-OP-59]), with `decimal_to_f64(decimal(t)) =
to_float(t)`.

## 7. Callables

All callables are pure, outside AD, and have no accumulator.

| Callable | Meaning | Fails |
|---|---|---|
| `decimal(text)`, `try_decimal(text)` | the exact value of the token (§6) | `domain` |
| `decimal_to_string(x)` | canonical text (§6) | never |
| `decimal_to_fixed_string(x, scale)` | exactly `scale` fractional digits | `domain` |
| `decimal_from_i64(v)` | exact | never |
| `decimal_to_i64(x, r)`, `try_decimal_to_i64` | `x` rounded to an integer by `r` | `overflow` outside i64; `domain` for `RejectInexact` |
| `decimal_from_f64(x, scale, r)`, `try_decimal_from_f64` | the exact binary value of finite `x` rounded to a multiple of `10^-scale` by `r` | `domain` |
| `decimal_to_f64(x)` | correctly rounded once, ties to even | never |
| `decimal_to_f32(x)` | correctly rounded once, ties to even, directly | never |
| `decimal_scale(x)` | canonical fractional digit count, 0..38 | never |
| `decimal_add(a, b)`, `decimal_sub(a, b)`, `decimal_mul(a, b)` | exact | `overflow` |
| `decimal_round(x, scale, r)` | `x` rounded to a multiple of `10^-scale` by `r` | `domain` |
| `decimal_div(a, b, scale, r)`, `try_decimal_div` | the exact quotient rounded to a multiple of `10^-scale` by `r` | `domain`; `overflow` |
| `decimal_lt`, `decimal_lte`, `decimal_gt`, `decimal_gte` | exact order | never |

`decimal_round` never overflows: rounding to fewer fractional digits can add at most one
integer digit, and only to a value that had fewer than 38. `decimal_from_f64` takes an
f64; an f32 widens to f64 exactly first.

## 8. Equality and ordering

Equality is [05-OP-36]'s structural `eq`/`neq`. Because every value is canonical,
structural equality of two decimals is equality of the rationals they denote, and there is
no `decimal_eq`. This relies on structural equality on ADT values (#2587), which
[05-OP-36] states for the checker and both lanes. Ordered comparison of ADT
values stays a type error under [05-OP-36], so the four ordering callables exist.

## 9. Binary floats

**To f64.** `decimal_to_f64(x)` is `to_float(decimal_to_string(x))`: the canonical text is
inside [05-OP-59]'s grammar and `to_float` rounds it correctly, once, in both lanes. The
value set lies strictly inside f64's normal range, so the result is always finite and
nonzero for a nonzero decimal.

**To f32.** `decimal_to_f32` rounds the exact value to f32 directly. Rounding to f64 first
and then to f32 is double rounding and gives a different answer for some inputs (prior art
§8). The algorithm uses Clinger's fast path where it applies (|c| ≤ 2^24 and a scale whose
power of ten is exact in f32) and otherwise an exact bounded-limb quotient with a sticky
remainder and ties to even. 10^-38 is below f32's smallest normal (about 1.18e-38), so the
smallest decimals round to f32 subnormals; the exponent clamps at 2^-149. f64 may carry
the already-rounded f32 value before the final exact `cast`.

**From f64.** `decimal_from_f64(x, s, r)` decomposes a finite `x` as `m · 2^e` with exact
power-of-two scaling and an exact integral `cast`, then rounds `m · 2^e · 10^s` to an
integer by `r` in limb arithmetic, with magnitude shortcuts when `x` is far below one unit
in the last place or far above the range. NaN, an infinity, a scale outside 0..38 and a
rounded value outside the set fail `domain`. So `decimal_from_f64(0.1, 2, RoundTiesToEven)`
is `0.10`'s value `0.1`, and `decimal_from_f64(0.1, 2, RejectInexact)` fails because the
double nearest 0.1 is not a multiple of 0.01. No shortest-round-trip conversion is
offered: it guesses at the text a human meant, while exact-then-round states the intent.

**AD.** `decimal_from_f64` produces an exact, non-differentiable value. `grad` through it
is rejected structurally, never answered with a silent zero cotangent, as for the other
float-to-exact conversions ([05-OP-6]).

## 10. Interchange

The `decimal128` and `decimal256` dtype names stay reserved for the Arrow and Parquet
boundary (spec/04 §1.1.1). The boundary rule is a language decision, so it belongs in
spec/04: a `decimal128` or `decimal256` value whose exact value lies in the value set
(whatever its declared scale) ingests as that exact `Decimal`; any other value fails
`domain` and is never rounded; and export to a declared `(p, s)` rounds only by an explicit
`Rounding`. `Std.Io.Parquet` is a stub, so no conversion function is defined here.

## 11. What is absent, and where it lives

- **Composes without loss:** negation and absolute value (`decimal_sub` from zero and a
  comparison), minimum and maximum, integer division (`decimal_div` at scale 0 with
  `RoundTowardNegative`, whose failure is correct because such a quotient is outside the
  value set), and f32 ingress (exact widening to f64).
- **Composes except at the range edge:** a remainder `a - q·b` from that quotient. When
  `q` or `q·b` leaves the value set the composition fails although the remainder itself
  is representable (for `a = 1` and `b = 1e-38`, `q` is `10^38`; for `a = -(10^38 - 1)` and
  `b = 10^38 - 2`, `q·b` is about `-2·10^38`).
  A dedicated remainder is additive later if a consumer meets that edge.
- **Inexact by nature:** square roots, powers with fractional exponents, logarithms and
  exponentials; they go through `decimal_to_f64`.
- **Additive later if a consumer needs it:** a fused multiply-then-round for products
  beyond 38 digits.
- **Finance conventions:** currency, ISO 4217 minor units, cash rounding, sum-preserving
  allocation and remainder policy live in Shoals, built on this module, as day counts do on
  Std.Datetime. Their rules are conventions that change by market and jurisdiction (prior
  art §11).
- **Columns:** a decimal column is a fixed-scale integer tensor plus its scale, per spec/04
  §1.1.1's scaled-storage rule; it lands when Coral needs it.
- **JSON exactness:** a JSON float token ingests as `JsonFloat(f64)` under [05-OP-2], so a
  JSON price reaches `Decimal` only through f64. Keeping the token text is a [05-OP-2]
  decision, tracked by #2871.

## 12. Normative placement

- A new atom, [05-OP-76], `decimal(arguments...) -> result`, governs exactly the
  `decimal::*` identities of the [05-OP-34] and [05-OP-35] registries. It states the value
  set, canonical form, opacity, grammar and length bound, each callable, and the failure
  contract, citing [05-OP-74] for rounding, [05-OP-59] for `to_float`, [05-OP-60] for
  `fail` and [05-OP-36] for equality. Its neighbours are [05-OP-73] (Std.Datetime),
  [05-OP-74] (rounding) and [05-OP-75] (the clock).
- [05-OP-35] routes the `decimal::*` identities
  to [05-OP-76] and counts them in its manifest.
- [05-OP-34] lists `decimal::Decimal` among the opaque identities that hold their
  governing atom's invariants by construction.
- [05-OP-74] defines all seven `Rounding` modes, `RejectInexact` included, and [05-OP-73]
  states what each Std.Datetime callable that takes a `Rounding` does under
  `RejectInexact` (a `domain` failure when the value is not a whole number of units or
  increments).
- `stdlib_numeric_manifest.md` carries one row per `decimal::*` callable, and
  `stdlib_adt_identities.md` gives `decimal::Decimal` its fixed-limb shape.
- spec/04 §1.1.1 states the interchange rule (§10).

## 13. Implementation notes

- **Limb arithmetic.** Unexported helpers over `List[i64]` and tuples: normalize,
  compare, add, subtract, schoolbook multiply with the carry normalized after every
  multiply-add, multiply and divide by a small limb, multi-limb long division with a
  fixed-iteration quotient-digit search, shifts by powers of ten, trailing-zero count.
  Intermediates stay within a dozen limbs.
- **Loops.** Every loop is a `fold`, `scan`, `map`, `zip` or `enumerate` over a
  precomputed `range`. Nothing recurses over limbs or digits: the evaluator overflows its
  native stack at a modest recursion depth (#2471), and a self-recursive helper routes any
  `chelis prove` property touching it to the induction lane. There is no early exit, so data-dependent loops run
  to a precomputed bound.
- **Parsing** checks the length bound first and validates characters before trusting
  `to_int`, which trims whitespace and accepts signs.
- **Census.** Every numeric type declared in a stdlib file is a census row, exported or
  not, so the helpers declare no types; only exported definitions are rows.
- **Lint.** Helper names avoid two-to-four-letter prefixes other than the module's
  shorthand `dec_`, which `prefix-namespace` accepts.

## 14. Verification

- **Independent reference.** A Python reference built on `fractions.Fraction` implements
  every rounding mode and both float roundings (53- and 24-bit, subnormal-aware) directly
  on rationals. It does not use `numpy.float32(float)` or `decimal` division followed by
  `quantize`, both of which double-round. A differential driver runs golden vectors
  (envelope boundaries, limb carry chains such as `999999999·10^k`, ties in every mode and
  sign, subnormal f32 results, parser edge cases, random values) through the evaluator and
  the compiled C lane and compares them exactly. Nightly CI runs the edge corpus; a large
  random run is a manual gate.
- **Failure corpus.** Every `domain` and `overflow` path with its exact message, extreme
  arguments with no primitive trap escaping, construction and inspection of `Decimal`
  outside the module rejected, removed names unexported, `grad` through `decimal_from_f64`
  rejected.
- **Properties** for `chelis prove` at the fuzz tier, with `string`, `i64` and `f64`
  binders: the text round trip, commutativity of `decimal_add`, `decimal_sub` inverting
  it within range, `decimal_to_f64(decimal(t)) = to_float(t)`, idempotent rounding, and the
  i64 round trip.

## 15. Delivery and consumers

The design is #2872, structural equality on ADT values is #2876, and the implementation is
#2913.

The only code consumer is hello-chelis's decimal example, which constructs
`Decimal { coefficient, scale }` directly and stops compiling under opacity; it is
rewritten against the new surface when hello-chelis moves to a compiler that ships it.
Shoals plans an order book on `Std.Decimal`. Removing
the old callables, `RoundingMode` and the old field shape is a breaking change to a
surface that #2806 had fenced.
