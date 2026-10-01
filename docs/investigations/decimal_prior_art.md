# Decimal arithmetic prior art

This survey of decimal number implementations informs
[`spec/design/std_decimal.md`](../../spec/design/std_decimal.md). It is evidence, not a
design.

## 1. Scope and sources

Arbitrary-precision and floating decimals: Java `BigDecimal`, Python `decimal` and the
General Decimal Arithmetic specification (GDA) behind it, IEEE 754-2008 decimal formats,
.NET `System.Decimal`, the Rust crates `rust_decimal` and `bigdecimal`, Go's
`cockroachdb/apd` and `shopspring/decimal`, Swift Foundation `Decimal`, the TC39
JavaScript Decimal proposal, PostgreSQL `numeric`, and Haskell `Data.Scientific` and
`Data.Ratio`.

Declared-precision decimals: the SQL standard and the result-type rules of SQL Server,
Snowflake, PostgreSQL, BigQuery and DuckDB; Ada decimal fixed point; COBOL; Haskell
`Data.Fixed`; Julia `FixedPointDecimals.jl`.

Interchange, money and proof: Apache Arrow and Parquet decimals, Joda-Money, JSR 354,
Dinero.js, Fowler's Money pattern, ISO 4217, mandated rounding rules (EU, ISDA, FpML,
HMRC), binary/decimal conversion algorithms, SMT-LIB, Flocq, SPARK, Why3, Dafny and Lean.

Sources are primary (documentation, specifications, source code, issue trackers,
advisories) and are linked inline. "(not re-verified)" marks a claim that rests on a
secondary summary or on recall rather than a fetched primary text.

## 2. Representation and range

| Library | Coefficient | Scale or exponent | At the bound |
|---|---|---|---|
| Java `BigDecimal` | unbounded | int32 scale, may be negative | `ArithmeticException` when a result scale leaves int32; `pow(n)` requires 0 ≤ n ≤ 999999999 ([docs](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/math/BigDecimal.html)) |
| Python `decimal` / GDA | context precision (default 28 digits) | context Emin/Emax (default ±999999); `MAX_PREC` and `MAX_EMAX` are 999999999999999999 on 64-bit builds | rounds to the context; overflow signal ([docs](https://docs.python.org/3/library/decimal.html)) |
| IEEE 754 decimal64 / decimal128 | 16 / 34 digits | emax 384 / 6144 | rounds; overflow to infinity ([decimal128](https://en.wikipedia.org/wiki/Decimal128_floating-point_format)) |
| .NET `System.Decimal` | 96-bit integer | scale 0..28 | `OverflowException` ([docs](https://learn.microsoft.com/en-us/dotnet/api/system.decimal)) |
| `rust_decimal` | 96-bit integer | scale 0..28 | operators panic; `checked_*` returns `Option`; `saturating_*` clamps ([docs](https://docs.rs/rust_decimal/latest/rust_decimal/struct.Decimal.html)) |
| `bigdecimal` (Rust) | unbounded | i64 scale, may be negative | documented as "limited to 2^63 decimal places" ([docs](https://docs.rs/bigdecimal/latest/bigdecimal/)) |
| `apd` (Go) | unbounded | int32 exponent; context exponent bounds ±100000 | condition flags and errors ([docs](https://pkg.go.dev/github.com/cockroachdb/apd/v3)) |
| `shopspring/decimal` | unbounded `big.Int` | int32 exponent | "a maximum of 2^31 digits after the decimal point" ([repo](https://github.com/shopspring/decimal)) |
| Swift Foundation | 38 digits (8 × UInt16) | Int8 exponent | throws `.overflow` ([source](https://github.com/swiftlang/swift-foundation/blob/main/Sources/FoundationEssentials/Decimal/Decimal.swift)) |
| TC39 Decimal | decimal128, 34 digits | IEEE exponent | rounds to 34 digits ([proposal](https://tc39.es/proposal-decimal/)) |
| PostgreSQL `numeric` | up to 131072 digits before the point, 16383 after | declared scale −1000..1000 (negative since v15) | error ([docs](https://www.postgresql.org/docs/current/datatype-numeric.html)) |
| Haskell `Data.Scientific` | unbounded `Integer` | `Int` exponent | conversions can exhaust memory (section 12) ([docs](https://hackage.haskell.org/package/scientific/docs/Data-Scientific.html)) |
| Haskell `Data.Ratio` | unbounded numerator and denominator, always reduced | none | none ([docs](https://hackage.haskell.org/package/base/docs/Data-Ratio.html)) |

The two families are arbitrary precision (Java, Python, `bigdecimal`, `apd`,
`shopspring`, PostgreSQL, Haskell) and fixed width (.NET and `rust_decimal` at 96 bits,
Swift at 38 digits, IEEE and TC39 at 34). SQL and Arrow cap declared precision at 38
digits for 128-bit storage and 76 for 256-bit storage (section 10), which is the practical
ceiling for data pipelines.

## 3. Trailing zeros, cohorts and equality

A *cohort* is the set of representations of one value that differ only in trailing zeros
(1.5, 1.50, 1.500). Libraries that keep the scale of each result keep cohorts distinct.

- **Java** keeps the scale. "compareTo … considers members of the same cohort to be
  equal", while "equals … requires both the numerical value and representation", so
  `2.0.equals(2.00)` is false, `hashCode` differs, and the natural order is "inconsistent
  with equals": a `HashSet` and a `TreeSet` of the same values disagree. Before JDK 8,
  `stripTrailingZeros()` did not normalize `0.000`
  ([JDK-6480539](https://bugs.java.com/bugdatabase/view_bug.do?bug_id=6480539)).
- **GDA and Python** keep trailing zeros (`1.30 + 1.20 = 2.50`); `==` and `hash` ignore
  them; `compare_total` distinguishes cohorts. Cowlishaw's reasons: user expectation,
  euro-conversion rules ("All the digits must be present"), and measurement precision
  ([FAQ](https://speleotrove.com/decimal/decifaq1.html)).
- **Positive exponents.** Normalizing 100 gives `1E+2` in both Python (`normalize`) and
  Java (`stripTrailingZeros`).
- **.NET** "preserves any trailing zeros", which "do not affect the value … in arithmetic
  or comparison". Equal decimals with different scales once hashed differently
  ([dotnet/core#3398](https://github.com/dotnet/core/issues/3398)).
- **`rust_decimal`** keeps the scale; `Eq` compares values and `Hash` hashes the
  normalized form ([source](https://github.com/paupino/rust-decimal/blob/master/src/decimal.rs)).
- **`apd`**: `Cmp` compares values, `CmpTotal` representations, `Reduce` strips zeros.
  **`shopspring`** documents that `==` on the struct is wrong and `Equal`/`Cmp` compare
  values.
- **PostgreSQL** keeps a display scale and does not hash it: "two numerics can compare
  equal but have different scales"
  ([numeric.c](https://github.com/postgres/postgres/blob/master/src/backend/utils/adt/numeric.c)).
- **TC39** removed quanta and trailing zeros from the Decimal data model in 2024 and moved
  display precision to a separate Amount proposal ("Decimal deliberately does not track
  display precision")
  ([April 2024 notes](https://github.com/tc39/notes/blob/main/meetings/2024-04/april-11.md),
  [Amount](https://github.com/tc39/proposal-amount),
  [earlier debate](https://github.com/tc39/proposal-decimal/issues/12)).

Every library that keeps cohorts needs a second comparison or a normalizing hash, and two
of them shipped a bug in exactly that seam.

## 4. Precision context and division

- **Java** has no ambient context. Add, subtract, multiply and `divide(BigDecimal)` are
  exact; a non-terminating quotient throws `ArithmeticException`. Rounding happens only
  with an explicit `MathContext` or `(scale, RoundingMode)`
  ([MathContext](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/math/MathContext.html)).
- **Python** rounds every arithmetic result to a thread-local context's precision
  (default 28); inputs are not rounded, so `3.104 + 0.000 + 2.104 ≠ 3.104 + 2.104` at
  precision 3. Condition flags are sticky hidden state.
- **`apd`** takes an explicit `Context` value; methods return a condition and an error.
- **`shopspring`** divides at a mutable package global, `DivisionPrecision = 16`;
  `DivRound` takes explicit places ([docs](https://pkg.go.dev/github.com/shopspring/decimal)).
- **`bigdecimal`** divides at a compile-time environment default of 100 digits
  ([repo](https://github.com/akubera/bigdecimal-rs)).
- **.NET, `rust_decimal`, Swift and TC39** round to their fixed width (in .NET,
  `1/3*3 = 0.9999…`).
- **PostgreSQL** picks a division scale heuristically ("at least 16 significant digits …
  no worse than float8"), capped at 2000 (not re-verified against `select_div_scale`).

Exact addition, subtraction and multiplication are universal. Division is the one
operation that must round, and the libraries differ only in where the rounding parameter
comes from: an exception (Java), a context, a global, a type width, or an explicit
argument.

## 5. Rounding modes

| Meaning | Java | Python / GDA | .NET `MidpointRounding` | `rust_decimal` | Swift | TC39 | IEEE 754 / SMT-LIB |
|---|---|---|---|---|---|---|---|
| toward zero | `DOWN` | `ROUND_DOWN` | `ToZero` | `ToZero` | | `trunc` | roundTowardZero / RTZ |
| away from zero | `UP` | `ROUND_UP` | | `AwayFromZero` | | | |
| toward +∞ | `CEILING` | `ROUND_CEILING` | `ToPositiveInfinity` | `ToPositiveInfinity` | `.up` | `ceil` | roundTowardPositive / RTP |
| toward −∞ | `FLOOR` | `ROUND_FLOOR` | `ToNegativeInfinity` | `ToNegativeInfinity` | `.down` | `floor` | roundTowardNegative / RTN |
| nearest, ties even | `HALF_EVEN` | `ROUND_HALF_EVEN` | `ToEven` | `MidpointNearestEven` | `.bankers` | `halfEven` | roundTiesToEven / RNE |
| nearest, ties away | `HALF_UP` | `ROUND_HALF_UP` | `AwayFromZero` | `MidpointAwayFromZero` | `.plain` | `halfExpand` | roundTiesToAway / RNA |
| nearest, ties toward zero | `HALF_DOWN` | `ROUND_HALF_DOWN` | | `MidpointTowardZero` | | | |
| require exact | `UNNECESSARY` | | | | | | |
| other | | `ROUND_05UP` | | | | | |

Sources: [Java](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/math/RoundingMode.html),
[.NET](https://learn.microsoft.com/en-us/dotnet/api/system.midpointrounding),
[`rust_decimal`](https://docs.rs/rust_decimal/latest/rust_decimal/enum.RoundingStrategy.html),
[Swift source](https://github.com/swiftlang/swift-foundation/blob/main/Sources/FoundationEssentials/Decimal/Decimal+Math.swift),
[SMT-LIB FloatingPoint](https://smt-lib.org/theories-FloatingPoint.shtml).

- **"Down" and "up" are ambiguous.** Java and Python mean toward and away from zero;
  Swift's `.down` and `.up` are floor and ceiling. `rust_decimal` deprecated
  `RoundDown`/`RoundUp` (and `BankersRounding`, `RoundHalfUp`, `RoundHalfDown`) in 1.11 in
  favour of sign-unambiguous names. .NET's docs concede that "not every mode is only about
  how midpoints are handled" despite the enum's name.
- **Defaults disagree:** ties-to-even in Python, IEEE and TC39; ties-away in Java's
  `MathContext(int)` and `apd`; PostgreSQL's `round(numeric)` rounds ties away from zero
  while its `round(float8)` rounds ties to even. `rust_decimal`'s `round_dp` defaults to
  ties-to-even while its `rescale` uses ties-away.
- **Require-exact** exists as Java `UNNECESSARY` and COBOL `ROUNDED MODE PROHIBITED`; it
  turns any rounding operation into an assertion of representability.

## 6. Error model

- **GDA** defines the signals clamped, conversion syntax, division by zero, division
  impossible, division undefined, inexact, insufficient storage, invalid context, invalid
  operation, overflow, rounded, subnormal and underflow; untrapped, they produce NaN or
  ±Infinity ([GDA exceptions](https://speleotrove.com/decimal/daexcep.html)). Python traps
  overflow, invalid operation and division by zero by default.
- **Java, .NET and `rust_decimal`** are finite-only (no NaN, infinity or negative zero) and
  raise exceptions or panics (or return `Option` from `checked_*`).
- **Swift** has a NaN value plus `CalculationError` (loss of precision, overflow,
  underflow, divide by zero).
- **TC39** has NaN, ±Infinity and −0, which compares equal to 0.
- **PostgreSQL** has NaN and ±Infinity; NaN equals itself and sorts above everything.

## 7. Text conversion

- **Parsing.** Java accepts `[+-]digits[.digits][eE[+-]digits]` with no whitespace. GDA
  adds case-insensitive Inf, Infinity, NaN and sNaN, with "No blanks"
  ([GDA conversions](https://speleotrove.com/decimal/daconvs.html)). Python is looser than
  GDA: it strips whitespace and accepts underscores. .NET parsing is culture-sensitive and
  rejects exponents unless `NumberStyles.Float` is passed. `rust_decimal`'s `FromStr` falls
  back to scientific parsing on `e`; `from_str_exact` errors instead of rounding extra
  digits.
- **Rendering.** GDA, Java and Python switch to exponent form unless the exponent is ≤ 0
  and the adjusted exponent ≥ −6 (Java also offers `toPlainString`). TC39 uses exponent
  form outside 1e−6..1e34. .NET, `rust_decimal` and `shopspring` always print plain
  digits.
- **JSON** (RFC 8259) numbers are `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`: no
  `+`, no leading zeros, no bare `.5` or `1.`, exponent allowed.

## 8. Binary-float conversion

**From a binary float.** A finite double is a dyadic rational, so it has an exact decimal
expansion of up to about 767 significant digits.
- Exact expansion: Java `new BigDecimal(0.1)` and Python `Decimal(0.1)` both give
  `0.1000000000000000055511151231257827021181583404541015625`.
- Shortest round-trip string: Java `BigDecimal.valueOf` (via `Double.toString`, which was
  not shortest before JDK 19,
  [JDK-4511638](https://bugs.openjdk.org/browse/JDK-4511638)), TC39, and `shopspring`'s
  `NewFromFloat`.
- Rounded to a fixed digit count: .NET rounds "to 15 significant digits"
  ([docs](https://learn.microsoft.com/en-us/dotnet/api/system.decimal.-ctor)).
- Inexact: Swift's `Decimal(1.19)` gives `1.1899999999999997952`
  ([swift-corelibs-foundation#3913](https://github.com/swiftlang/swift-corelibs-foundation/issues/3913)).
- `rust_decimal` offers both `from_f64` (about 16 digits) and `from_f64_retain`.

**To a binary float.**
- Java's `doubleValue` went through `parseDouble(toString())`, correctly rounded but slow
  until JDK 21 ([JDK-8205592](https://bugs.openjdk.org/browse/JDK-8205592)). Haskell's
  `toRealFloat` guards against huge magnitudes. .NET says only that conversion "might lose
  precision".
- **Algorithms.** Clinger's fast path multiplies an exactly representable significand by
  an exactly representable power of ten in one IEEE operation (for f64, |c| ≤ 2^53 and
  |exponent| ≤ 22) and falls back to higher precision otherwise
  ([Clinger 1990](https://dl.acm.org/doi/10.1145/93542.93557)). Eisel–Lemire, used by
  `fast_float`, GCC, LLVM and Rust, rounds correctly including ties to even
  ([paper](https://arxiv.org/abs/2101.11408), [fast_float](https://github.com/fastfloat/fast_float));
  Mushtak and Lemire prove its big-integer fallback unnecessary for significands of at
  most 19 digits ([paper](https://arxiv.org/abs/2212.06644)). Both assume an unsigned
  64×64→128-bit multiply.
- **Shortest printing** (Steele–White, Grisu, Ryu, Schubfach, Dragonbox) yields at most 17
  significant digits for a double, with a decimal exponent of roughly −324..308
  ([Ryu](https://github.com/ulfjack/ryu), [Dragonbox](https://github.com/jk-jeon/dragonbox)).
- **Double rounding.** Converting a decimal to f64 and then to f32 can differ from
  converting it to f32 directly: `0.0691026858985424` gives `0x1.1b0b6ap-4` directly and
  `0x1.1b0b6cp-4` via double
  ([Exploring Binary](https://www.exploringbinary.com/double-rounding-errors-in-decimal-to-double-to-float-conversions/)).
  Rounding to odd at the intermediate precision is the proven fix
  ([Boldo and Melquiond](https://guillaume.melquiond.fr/doc/08-tc.pdf)).

**With only checked signed 64-bit integers.** Partial products of 32-bit limbs reach
(2^32 − 1)^2 > 2^63 − 1, so limbs must stay at or below 31 bits (or a decimal base such as
10^9, whose products stay below 10^18). Tables of 128-bit powers must be stored as limb
arrays. A bounded coefficient and scale admit a table-free path: Clinger's fast path where
it applies, otherwise an exact bounded-limb quotient `floor(c · 2^k / 10^s)` with the
remainder as a sticky bit, then ties-to-even.

## 9. Declared precision and scale

| System | + / − scale | × scale | ÷ scale | Above the cap |
|---|---|---|---|---|
| SQL standard | max(s1, s2) | s1 + s2 | implementation-defined | rounding or truncation implementation-defined (not re-verified for §6.12; [SQL-92 text](https://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt)) |
| SQL Server | max(s1, s2) | s1 + s2 | max(6, s1 + p2 + 1) | scale cut to 6 when the integral part exceeds 32 digits ([docs](https://learn.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql)) |
| Snowflake | | min(s1 + s2, max(s1, s2, 12)) | max(s1, min(s1 + 6, 12)) | rounded ([docs](https://docs.snowflake.com/en/sql-reference/operators-arithmetic)) |
| Arrow compute (Redshift rules) | max(s1, s2) | s1 + s2, precision p1 + p2 + 1 | max(4, s1 + p2 − s2 + 1) | error ([compute.rst](https://raw.githubusercontent.com/apache/arrow/main/docs/source/cpp/compute.rst)) |
| DuckDB | widths 1..38 | | division returns a float | error above 38 ([docs](https://duckdb.org/docs/current/sql/data_types/numeric)) |
| BigQuery | `NUMERIC` (38, 9), `BIGNUMERIC` scale 38 | | | rounds half away from zero ([docs](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions)) |

So in SQL Server `0.0000009 × 1.0` at `decimal(30,10)` yields `0.000001`. Every system
with precision and scale in the type has to derive a result type per operation and cap
it; at the cap SQL Server and Snowflake round or truncate silently while Arrow compute and
DuckDB fail. None gives division a natural scale, and DuckDB sidesteps it by leaving exact
arithmetic.

- **Ada** decimal fixed point (`delta 10.0**(-N) digits D`) truncates toward zero on
  conversion, `S'Round` rounds ties away from zero, and fixed × fixed must be converted to
  an explicit target type ([RM 4.6](https://www.adaic.org/resources/add_content/standards/05rm/html/RM-4-6.html),
  [RM 3.5.10](https://www.adaic.org/resources/add_content/standards/05rm/html/RM-3-5-10.html)).
- **COBOL** truncates by default; `ROUNDED MODE` offers away-from-zero, nearest-away,
  nearest-even, nearest-toward-zero, toward-greater, toward-lesser, truncation and
  prohibited (not re-verified against the 2014 standard;
  [IBM](https://www.ibm.com/docs/en/cobol-linux-x86/1.2.0?topic=errors-handling-in-arithmetic-operations)).
- **Haskell `Data.Fixed`** puts the resolution in the type and truncates results, with the
  documented warning "Multiplication is not associative or distributive"
  ([docs](https://hackage-content.haskell.org/package/base-4.22.0.0/docs/Data-Fixed.html)).
- **Julia `FixedPointDecimals.jl`** puts the integer type and digit count in the type;
  arithmetic other than division "will silently overflow"
  ([repo](https://github.com/JuliaMath/FixedPointDecimals.jl)).

## 10. Interchange: Arrow and Parquet

- **Arrow** stores decimals as two's-complement integers of 32, 64, 128 or 256 bits with
  maximum precision 9, 18, 38 and 76 ([Schema.fbs](https://raw.githubusercontent.com/apache/arrow/main/format/Schema.fbs);
  32- and 64-bit since Arrow 18,
  [release notes](https://arrow.apache.org/blog/2024/10/28/18.0.0-release/)). pyarrow
  allows negative scale (`decimal128(5, -3)` stores 12345000 as 12345,
  [docs](https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html)).
- **Parquet** DECIMAL uses INT32 (p ≤ 9), INT64 (p ≤ 18), fixed-length or variable byte
  arrays, big-endian two's complement, and "Scale must be zero or a positive integer less
  than or equal to the precision"
  ([LogicalTypes.md](https://raw.githubusercontent.com/apache/parquet-format/master/LogicalTypes.md)).
  Parquet forbids the negative scales Arrow allows.
- **Hosts.** pandas maps decimals to Python `decimal.Decimal` objects or keeps Arrow
  types; polars decimals are 128-bit with precision ≤ 38 and non-negative scale; DuckDB has
  read Parquet decimals above 38 digits as doubles
  ([duckdb#25058](https://github.com/duckdb/duckdb/pull/25058)) and `duckdb-rs` downcasts
  128- and 256-bit decimals to f64 ([duckdb-rs#305](https://github.com/duckdb/duckdb-rs/issues/305)).
  The ecosystem already narrows silently at this boundary in places.
- **Exact round trip** of `decimal128` needs a coefficient of at least 127 bits plus sign
  (|c| ≤ 10^38 − 1); `decimal256` needs 255 bits. A 64-bit coefficient covers `decimal32`,
  `decimal64` and Parquet INT32/INT64 exactly.

## 11. Money and mandated rounding

- **Libraries.** Joda-Money fixes a `Money`'s scale to its currency and requires an
  explicit rounding mode to convert from `BigMoney`; its ISO 4217 data is a user-extensible
  file because "the implementation in the JDK is too restrictive"
  ([guide](https://www.joda.org/joda-money/userguide.html)). JSR 354 (Moneta) is a separate
  library with query-based roundings, including cash rounding; its `FastMoney` is a 64-bit
  integer with scale 5 and default ties-to-even
  ([API](https://javamoney.github.io/api.html)). Dinero.js stores an integer amount, a
  scale and a currency, and `allocate` distributes indivisible minor units
  ([docs](https://www.dinerojs.com/docs/api/mutations/allocate)). Fowler: "it's easy to
  lose pennies … because of rounding errors"
  ([Money](https://martinfowler.com/eaaCatalog/money.html)).
- **ISO 4217** minor units are 0 (JPY, KRW), 2, 3 (BHD, KWD, JOD, OMR, TND) or 4 (CLF), and
  the list is amended over time ([ISO 4217](https://en.wikipedia.org/wiki/ISO_4217)).
- **Mandated rounding.** EU Regulation 1103/97: conversion rates have six significant
  figures, inverse rates "shall not be used", and an amount exactly half-way "shall be
  rounded up" ([1103/97](https://eur-lex.europa.eu/eli/reg/1997/1103/oj/eng)). The 2006
  ISDA Definitions round percentages to 1e-5 percentage points and currency amounts to
  two decimals with ".005 being rounded upwards"
  ([ISDA](https://www.isda.org/a/smMDE/Blackline-2000-v-2006-ISDA-Definitions.pdf)). FpML's
  `RoundingDirectionEnum` has Up, Down and Nearest with a precision, with no stated sign
  convention ([FpML](https://www.fpml.org/spec/fpml-5-5-4-tr-1/html/reporting/schemaDocumentation/schemas/fpml-enum-5-5_xsd/simpleTypes/RoundingDirectionEnum.html)).
  HMRC allows VAT totals to be rounded down to a penny
  ([VATREC12010](https://www.gov.uk/hmrc-internal-manuals/vat-trader-records/vatrec12010)).
  None of the surveyed rules mandates ties toward zero.
- **Ownership.** No mainstream standard library ships a Money type. Currency tables, cash
  rounding, remainder policy and jurisdiction rules change over time and live in
  libraries above the language's decimal type.

## 12. Resource exhaustion

Arbitrary precision combined with exponent notation or unbounded scale has a long
advisory history:

- Ruby `BigDecimal`, CVE-2009-1904: a large-exponent string crashed the interpreter; the
  advised workaround was to forbid scientific notation
  ([NVD](https://nvd.nist.gov/vuln/detail/CVE-2009-1904)).
- Go `math/big` `Rat.SetString`, CVE-2022-23772: exponent overflow led to uncontrolled
  memory use ([advisory](https://github.com/advisories/GHSA-q99m-p7hq-5v4f)).
- Jackson, CVE-2018-1000873: `1e100000000` converted to a long hung
  ([issue](https://github.com/FasterXML/jackson-modules-java8/issues/90)); Jackson then
  added a number-length limit of 1000 characters, later bypassed in its async parser
  ([advisory](https://github.com/advisories/GHSA-72hv-8253-57qq)), and notes that the JDK's
  `BigInteger(String)` and `BigDecimal(String)` are quadratic in digit count
  ([advisory](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-q4xh-88c3-wmh7)).
- Haskell `Data.Scientific` documents that converting `1e1000000000` to a `Rational` will
  "fill up all space and crash your program"; HSEC-2026-0007 found aeson's bound check
  rejected only large positive exponents, so `1e-999999999` exhausted memory
  ([OSV](https://osv.dev/vulnerability/HSEC-2026-0007)).
- PostgreSQL bounded repeated squaring in `power_var_int` after it produced "ridiculously
  enormous intermediate values"
  ([commit message](https://www.postgresql.org/message-id/E1XSHa1-00086a-2A%40gemulon.postgresql.org)).
- Python: int/str conversion is limited to 4300 digits by default since CVE-2020-10735,
  because the conversion is quadratic in digit count
  ([docs](https://docs.python.org/3/library/stdtypes.html)); the limit covers `int`, and
  conversions between `int` and the pure-Python `decimal` implementation interact with it
  ([cpython#96589](https://github.com/python/cpython/issues/96589)).
- Lean's JSON number parser panicked on `3E9999999993` because it built 10^e
  ([lean4#13987](https://github.com/leanprover/lean4/issues/13987)).
- `rust_decimal`'s recursive parser overflows the stack on a long run of leading zeros
  (about 27,000 on an 8 MiB stack, fewer on a smaller thread stack)
  ([rust-decimal#840](https://github.com/paupino/rust-decimal/issues/840)).
- `apd` states its design goal as operations that "will produce an error if they will be
  slow" ([repo](https://github.com/cockroachdb/apd)).

Ruby, Go, Jackson, Haskell's `Data.Scientific` and PostgreSQL added exponent, length or
growth bounds after advisories; `apd` bounds exponents by design; the remaining
arbitrary-precision libraries leave the exposure to callers. TC39 chose decimal128 over an arbitrary-precision BigDecimal partly
because unbounded division needs a mandatory rounding parameter
([proposal](https://github.com/tc39/proposal-decimal)).

## 13. Verification

- **SMT-LIB** has theories of integers, reals, mixed integer-real arithmetic, binary
  floating point, bit-vectors, arrays and strings, and no decimal or fixed-point theory
  ([theories](https://smt-lib.org/theories.shtml)); its floating-point theory is binary
  only. Nonlinear integer arithmetic is undecidable, so a symbolic 10^s is hard while a
  literal one is a constant.
- **Flocq** (Coq) supports any radix including 10, with fixed-exponent (FIX) formats and a
  generic rounding operator; "a decimal at scale s rounded by mode m" is FIX in radix 10
  with exponent −s ([Flocq](https://flocq.gitlabpages.inria.fr/theos.html)).
- **SPARK** treats fixed-point values as integer multiples of a small, so addition and
  subtraction become integer addition
  ([GNATprove](https://docs.adacore.com/spark2014-docs/html/ug/en/usage_scenarios.html)).
  **Why3**'s `mach.fxp` pairs a machine integer and exponent with a ghost real and an
  invariant ([stdlib](https://why3.org/stdlib/mach.fxp.html)).
- **Dafny** has exact `real` with decimal literals; Leino notes that bounded types make
  "the necessary precondition specification … very complicated"
  ([paper](https://leino.science/papers/krml289.html)).
- **Lean** core's `JsonNumber` is a mantissa and a natural exponent over unbounded
  integers; Mathlib has rationals but no decimal type.

The simplest statement of decimal semantics for proof is the exact rational c · 10^−s:
add, subtract and multiply are rational operations guarded by a representability
condition, and rounding at a literal scale is linear integer-real arithmetic.

## 14. Convergence and divergence

The libraries converge on exact addition, subtraction and multiplication; value-based
ordering; both ties-to-even and ties-away offered; an explicit way to round to a scale;
text as the recommended input path; and, eventually, hard bounds.

Where they diverge, the cause is traceable:
- cohorts and trailing zeros come from IEEE/GDA hardware lineage and SQL/COBOL round
  trips;
- 96-bit and 34- or 38-digit widths come from hardware and performance;
- ambient contexts come from REXX/GDA heritage and convenience, and package globals from
  ergonomics;
- float-conversion behaviour follows each host's legacy number printing;
- exponent-form rendering comes from the GDA specification;
- NaN and infinity mirror IEEE binary hardware.
