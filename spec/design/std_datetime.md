# Std.Datetime: dates, times, instants and zones

Tracker: chelis#2858.
Prior art: [`datetime_prior_art.md`](../../docs/investigations/datetime_prior_art.md),
cited below as "prior art §N".

This is the design of record for the `Std.Datetime` family of standard-library modules,
which replaces `Std.Time`. It decides the whole surface now and delivers it in stages
(§16). Normative text lands in spec/05 one stage at a time, together with that stage's
registry rows (§15). Where this document and a numbered chapter disagree, the chapter
wins and this document has a bug.

## 1. Why

`Std.Time`, which this design replaces, was fenced by #2803: each of its 16 callables
failed with #2779. The code the fence replaced had several defects:
- duration normalization was not Euclidean;
- parsing accepted only four-digit years;
- `sub_days` negated `i64::MIN`;
- the ordinal came from a recursive walk of one year at a time, which also overflows the
  evaluator's stack far from 1970;
- raw `Date` fields were never validated;
- negative years rendered as `00-1`;
- years above 9999 did not round-trip.

Its contract, the calendar paragraph of [05-OP-35], asked for exact arithmetic over every
i64 year. The fence was the honest answer to an implementation that could not deliver
that.

Shoals carries its own date layer on top of `Std.Time`, with its own defect list:
- shoals#87: business-day rolls ignore the holiday calendar, and ACT/ACT is computed as
  `days / 365.25`.
- A tenor month is 30 days and a year 365, and schedules step in 30-day months.
- The weekend is hardcoded to Saturday and Sunday.
- Holiday tables answer "not a holiday" for years they do not cover.

Coral's time-series index (coral#41) and Nautilus's econometrics (nautilus#92) are
waiting on a date type that does not exist.

Dates and times are a solved problem in the sense that mature libraries agree on most
of the model (prior art §2 to §6). This design adopts that consensus. Where the libraries
diverge, Chelis applies its own tenets, and in practice that means replacing defaults
with required arguments.

## 2. What belongs in Std

Two tests decide where a piece lives.

**Algorithm or legislated data.** Std holds computations over the proleptic Gregorian
calendar and the caller's arguments, and nothing whose answer is decided by a
legislature, a regulator or an exchange:
- The time zone database is data: IANA republishes it several times a year.
- Holiday lists are data: the US created Juneteenth with one day's notice in 2021, and
  the UK added bank holidays at short notice in 2022 and 2023.
- Even the weekend is data: Saudi Arabia moved its weekend in 2013, and the UAE in 2022.

Data lives in separately versioned packages pinned by `reef.lock`, so it is a declared
input under the determinism contract. Std defines the value types that data packages
construct (`BusinessCalendar`, `TimeZone`). It holds no tz database, no holiday list and
no default weekend.

**General convention or domain convention.** Among the pure algorithms, Std holds what is
general: Easter, the nth weekday of a month, business-day rolls (NumPy ships the same
machinery in its core; prior art §11). Shoals holds what only finance uses: day counts,
tenors including ON/TN/SN, spot lags, schedules with stubs and the end-of-month rule,
the Nearest and HalfMonthModifiedFollowing rolls whose definitions differ between
libraries, and IMM dates. This follows the canonical reference §8.5 rule that domain
knowledge belongs in external libraries.

| Piece | Home |
|---|---|
| Civil dates and times, instants, durations, periods, fixed offsets, text forms | `Std.Datetime` |
| Rounding modes shared with `Std.Decimal` | `Std.Rounding` |
| Business calendar values, rolls, business-day offsets and counts | `Std.Datetime.Business` |
| Vectorized date and instant columns | `Std.Datetime.Columns` |
| Zone rules as values; zone-aware conversion | `Std.Datetime.Zone` |
| Reading the wall clock and a monotonic clock | `Std.Datetime.Clock` |
| IANA tz data; public, bank and market holiday calendars | the `bed` repository (§13) |
| Day counts, tenors, spot lags, schedules, finance-only rolls | Shoals (§14) |

## 3. Principles as applied here

- **Opaque values, validating producers.** Every value type is `@opaque`. The only way to
  obtain one is an exported function that validates its arguments, so an invalid date
  cannot exist outside the module. This closes #2779's unvalidated-constructor class by
  construction. No type carries `@invariant` (§18).
- **Required policies, never defaults.** Every point where the surveyed libraries choose
  silently is a required argument here. Where a surveyed library justifies its default
  at all, the justification is compatibility with older libraries (prior art §5), which
  Chelis does not need. This covers:
  - month-end overflow;
  - DST disambiguation;
  - offset/zone conflict;
  - rounding;
  - what a business-day offset does when it starts on a holiday.
- **Exact arithmetic in a bounded domain.** Every intermediate stays provably inside
  checked i64 (§4). Nothing clamps, wraps or saturates silently. A narrowing conversion
  states its rounding.
- **Legislated data is a declared input.** Zone rules and calendars are values built
  from pinned packages. Nothing reads the host's tz database, locale or clock implicitly.
- **Only the clock is an effect.** All of `Std.Datetime` is pure except
  `Std.Datetime.Clock`, whose reads are `IO` and visible in every caller's signature.
- **Small surface.** Where an operation composes from others without loss, it is not
  added. There is no `date_sub_days`, because `date_add_days(d, n)` with a negative `n`
  says the same thing.

## 4. Timeline model and range

**Calendar.** The calendar is proleptic Gregorian with astronomical year numbering: year
0 exists, and year −1 is 2 BCE.

**Timescale.** The timescale is POSIX. Every day has exactly 86 400 seconds, leap seconds
are not representable, and text containing second `60` is rejected. This matches
Temporal, jiff, Go and Python (prior art §4). A TAI or GPS timescale would be a separate
type with a leap-second table supplied as data. It is out of scope, and nothing here
blocks it.

**Supported years are −9999 through 9999.** This replaces [05-OP-35]'s "all i64 years".
The narrower range is chosen on its own merits:
- It is the range of RFC 3339's four-digit year (0000 to 9999, extended here to negative
  years) and of jiff (prior art §3).
- Every intermediate of every algorithm below provably fits in checked i64, so
  correctness rests on ordinary integer arithmetic.
- The whole domain is 7 304 484 days, few enough to check exhaustively against an
  independent oracle (§17).

No numerical or financial workload is known to need civil dates beyond ten thousand
years. Astronomical work at that scale uses Julian dates in floating point, which is a
different model.

| Quantity | Exact bounds |
|---|---|
| Date epoch day (days since 1970-01-01) | −4 371 587 to 2 932 896 |
| Civil seconds (start of −9999-01-01 to end of 9999-12-31) | −377 705 116 800 to 253 402 300 799 |
| Offset seconds | −86 399 to 86 399 (±23:59:59) |
| Instant unix second | −377 705 030 401 to 253 402 214 400 |
| Largest instant difference | 631 107 244 801 s (< 2^40) |

**Instant range.** The instant range is the civil range shrunk by the largest offset at
each end. Every instant therefore has a civil reading under every offset, so formatting
an instant and converting it under any offset never fail for lack of range. jiff makes the
same choice for the same reason. Zone conversion can still fail where a zone's data has
no coverage (§11).

**Offset range.** An offset is less than a day in magnitude. That matches RFC 3339's
two-digit hour and Temporal, and covers every offset in the IANA database, historical
local mean time included.

## 5. Failure contract

[04-NUM-9]'s numeric traps are raised by primitives and name the primitive. A library
definition cannot raise one under its own name. The standard library's existing modules
report invalid input through `fail` ([05-OP-60]); for example, `parse_json` fails with
`parse_json failed: malformed JSON`. `Std.Datetime` follows that practice and pins the
message grammar so that every failure is deterministic and machine-readable:

```
<function>: <kind>: <detail>
```

`<function>` is the exported callable's name. `<kind>` is one of:

| Kind | Meaning | `try_` form |
|---|---|---|
| `domain` | An input is outside the operation's value set: an invalid field, a year outside the range, malformed text, a query outside a calendar's horizon, a `Reject…` policy firing. | returns `None` |
| `overflow` | An arithmetic result leaves the type's range. | still fails |
| `io` | The host could not supply a clock reading (`Std.Datetime.Clock` only, §12). | no `try_` form |

Two `try_` forms have a second `None` case: `try_instant_to_unix_count` and
`try_duration_to_count` return `None` where their twin fails `overflow` because the count
does not fit in i64, as well as where it fails `domain` under `RejectInexact` (§8.5,
§8.6).

**Coverage rule.** One rule decides every edge case, and per-function text states only
exceptions to it:
- An argument, a count or a text that denotes no value of its type fails `domain`. So
  do `duration_from_count`, `instant_from_unix_count` and the column constructors when
  the value their arguments denote lies outside the type.
- Consulting data outside its coverage fails `domain`. Coverage means a calendar's
  horizon, or a zone's transitions past the last one when its footer is empty.
- A result computed by arithmetic on values of these types that leaves its type fails
  `overflow`. That arithmetic is adding, subtracting, negating or multiplying dates,
  times, instants, durations and periods, including reading a civil value at an offset
  (`dt − o`) and rounding an instant to a multiple (`instant_round_to`). `period_mul` and
  `duration_mul` are such arithmetic. A count conversion (`instant_to_unix_count`,
  `duration_to_count`) whose count leaves i64 also fails `overflow`. Nothing else does.
- Between a range failure and a coverage failure, the range check runs first, so
  `overflow` wins over a coverage `domain`.
- Within one call, checks run in the order the defining computation produces the
  quantity each one checks, so a call that could fail both ways reports the earlier
  check:
  - `date_add_months`: the target year-month's range, then the day under the policy;
  - `date_add_period` and `datetime_add_period`: the month step as above, then the day
    step's range, so `try_date_add_period(date(2024, 1, 31), period(1, 4000000),
    RejectInvalidDay)` returns `None` before the day step could overflow;
  - `duration_to_count`: the rounding policy, then representability in i64;
  - `instant_round_to`: the increment, then the rounding policy, then the result's range.
  No other S1 callable can fail both ways for one input; `instant_to_unix_count` follows
  the same order as `duration_to_count`, but only its nanosecond counts can leave i64,
  and those are always exact.

**No primitive trap escapes.** Every range and validity check runs before the arithmetic
it protects, so no primitive numeric trap ever escapes a `Std.Datetime` call. A test
enforces this (§17).

`<detail>` names the offending value, for example
`date: domain: day 30 is outside 1..29 for 2024-02`.

## 6. Value types

Each type below is `@opaque`. "Equality" is what structural `eq` ([05-OP-36]) means
outside the module. It always compares the representation, and the representation is
canonical, so equal representations mean equal values. Structural `eq` and `neq`
([05-OP-36]) are the equality of every value type, and no per-type equality function is
added.

| Type | Meaning | Representation | Equality |
|---|---|---|---|
| `Date` | civil calendar date | `{ epoch_day: i64 }` | same day |
| `Time` | civil time of day | `{ nanosecond_of_day: i64 }`, in `[0, 86 400 × 10^9)` | same time |
| `DateTime` | civil date and time, no zone | `{ epoch_day: i64, nanosecond_of_day: i64 }` | same civil reading |
| `Instant` | point on the POSIX timescale | `{ unix_second: i64, nanosecond: i64 }`, nanosecond in `[0, 10^9)` | same instant |
| `Offset` | fixed UTC offset | `{ seconds: i64 }` | same offset |
| `OffsetDateTime` | an instant with the offset it was written in | `{ instant: Instant, offset: Offset }` | same instant **and** same offset |
| `Duration` | exact elapsed time | `{ second: i64, nanosecond: i64 }`, nanosecond in `[0, 10^9)` | same length |
| `Period` | calendar period | `{ months: i64, days: i64 }`, never of mixed sign | same months and days |
| `Dates[n]` | a column of dates | `{ epoch_days: tensor[n, i64] }` | same days, elementwise and in order |
| `Instants[n]` | a column of instants | `{ unix_seconds: tensor[n, i64], nanoseconds: tensor[n, i64] }` | likewise |

**Duration** is Euclidean-normalized: −1.5 s is `{ second: −2, nanosecond: 500 000 000 }`.
The seconds field spans all of i64. Negating `{ i64::MIN, 0 }` fails with `overflow`;
every other duration negates exactly.

**Period** stores years as 12 months. Adding one year and adding twelve months always
give the same result (prior art §6), so `period(12, 0)` and a one-year period are one
value and `eq` is semantic equality. A period of mixed sign, such as one month minus one
day, is rejected:
- it has no ISO 8601 text form;
- two periods applied in sequence say the same thing explicitly;
- Temporal makes the same rule.

A period has no ordering (is 1 month longer than 30 days?), so none is offered.
`date_add_period(d, p)` followed by adding `period_negate(p)` need not return `d`
(31 January + 1 month − 1 month is 28 January). This is a property of calendars, not of
the implementation.

**OffsetDateTime** exists so that RFC 3339 text round-trips: `2026-10-01T09:30:00-04:00`
keeps its `-04:00`. Two values for the same instant with different offsets are not `eq`.
Compare their instants for that.

**`Weekday`** is a plain enum, `Monday | Tuesday | Wednesday | Thursday | Friday |
Saturday | Sunday`. Every constructor is valid, so it is not opaque.

**Columns** have no missing-value sentinel (no NaT; prior art §10). Where a vectorized
producer can fail per element, its `try_` form returns the column together with a
`tensor[n, bool]` validity mask. Invalid positions then hold 1970-01-01 (or the unix
epoch), which the mask marks as meaningless.

**Internal helpers use tuples, never private record types.** The capacity census registers
every numeric `deftype`, exported or not.

## 7. Policy types

Each policy is a plain enum. Every constructor name is unique across `Std.Datetime`, the
rest of `Std`, and the Shoals types that import these modules. An unqualified nullary
constructor resolves to the last declaration of that name in scope, so a shared name such
as `Reject` would silently change meaning.

| Type | Variants | Used by |
|---|---|---|
| `DayOverflow` | `ClampToMonthEnd`, `RejectInvalidDay` | month arithmetic: 31 January + 1 month |
| `TimeUnit` | `Hours`, `Minutes`, `Seconds`, `Milliseconds`, `Microseconds`, `Nanoseconds` | counts of exact time |
| `BusinessDayRoll` | `Unadjusted`, `Following`, `Preceding`, `ModifiedFollowing`, `ModifiedPreceding` | §9 |
| `NonBusinessStart` | `RejectNonBusinessStart`, `RollStartForward`, `RollStartBackward` | §9 |
| `Disambiguation` | `EarlierInstant`, `LaterInstant`, `CompatibleInstant`, `RejectNonUniqueLocal` | §11 |
| `OffsetConflict` | `UseWrittenOffset`, `UseZoneRules`, `RejectOffsetMismatch` | §11 |

Each policy type lands with the stage whose functions use it; every constructor name in
the table is reserved from S1 on.

There is no roll-over option for month arithmetic. 31 January + 1 month rolling over to
2 or 3 March is `date_add_days` applied to a clamped result, so it composes.

**Rounding is shared, not time-specific.** The conversions that round to a quantum take a
`Rounding` from `Std.Rounding`: in S1, `instant_to_unix_count`, `duration_to_count` and
`instant_round_to`, with the `try_` forms of the first two, and in S3 the column forms
of §10. `duration_to_seconds_f64` is the one other conversion that drops precision; it is
a named lossy boundary with fixed nearest-even rounding (§8.6). `Std.Rounding` is a
small module that S1 introduces and that `Std.Decimal` adopts when it is redesigned, so
the standard library has one rounding vocabulary. Its
variants use IEEE 754's attribute names, which spec/05 already uses for `round`, wherever
IEEE 754 has one; `RoundAwayFromZero` and `RejectInexact` have no IEEE 754 counterpart.
For an exact value `v` and a positive quantum `q`, each returns a multiple `k·q`:
- `RoundTowardNegative`: the largest `k·q ≤ v`.
- `RoundTowardPositive`: the smallest `k·q ≥ v`.
- `RoundTowardZero`: whichever of those two is nearer zero.
- `RoundAwayFromZero`: whichever is farther from zero, or `v` itself when it is already
  a multiple.
- `RoundTiesToEven`: the nearest multiple; an exact tie takes the even `k`.
- `RoundTiesToAway`: the nearest multiple; an exact tie takes the one farther from zero.
- `RejectInexact`: `v` itself, which must already be a multiple; otherwise the callable
  fails `domain` and its `try_` twin returns `None`.

On the instant and duration line, rounding toward the past is `RoundTowardNegative` and
rounding toward the future is `RoundTowardPositive`; for times before 1970 these differ
from `RoundTowardZero`. The mode meanings are defined once, in their own spec/05 atom
that both the datetime and decimal atoms cite (§15).

## 8. `Std.Datetime` (stage S1)

Signatures are Surf. "Fails" uses the §5 grammar. Every function is total on its stated
domain and pure. A `try_` function has the same signature with an `Option` result, and
returns `None` exactly where its trapping twin fails with `domain`. The column `try_`
forms are the exception: they return the column with a validity mask (§6).

### 8.1 Calendar queries

These are total over every i64 year, because their arithmetic cannot overflow.

| Function | Result |
|---|---|
| `is_leap_year(year: i64) -> bool` | Gregorian rule |
| `days_in_year(year: i64) -> i64` | 365 or 366 |
| `days_in_month(year: i64, month: i64) -> i64` | 28 to 31; fails `domain` unless month is 1..12 |
| `weekday_iso_number(w: Weekday) -> i64` | Monday 1 … Sunday 7 |
| `weekday_from_iso_number(n: i64) -> Weekday` | inverse; `try_` form |
| `weekday_name(w: Weekday) -> string` | `"monday"` … `"sunday"` (lowercase ASCII) |

### 8.2 Date

**Construction and access:**
- `date(year, month, day) -> Date` and `try_date`. The arguments are i64. Fails `domain`
  for an invalid field or a year outside −9999..9999.
- `date_year`, `date_month`, `date_day`, all `(Date) -> i64`.
- `date_epoch_day(d) -> i64`; `date_from_epoch_day(n) -> Date` and its `try_` form.
- `date_weekday(d) -> Weekday`.
- `date_day_of_year(d) -> i64`, numbered from 1.
- `date_iso_week(d) -> (i64, i64)` returns the ISO 8601 week-year and week number 1..53.
  The week-year never leaves the supported range: −9999-01-01 is a Monday and
  9999-12-31 a Friday, so the first and last ISO weeks of the range lie inside it.
- `date_from_iso_week(iso_year, week, weekday) -> Date` and its `try_` form. Fails
  `domain` for a week that does not exist in that week-year, or a result outside the
  range.

**Arithmetic.** "Fails overflow" means the result leaves the range.
- `date_add_days(d, n: i64) -> Date`. Fails `overflow`. Any i64 `n` is accepted; the
  range check runs before any addition.
- `date_days_until(a, b) -> i64` is `epoch_day(b) − epoch_day(a)`. It never fails.
- `date_add_months(d, n: i64, overflow: DayOverflow) -> Date` and its `try_` form.
  - The year and month move by `n` months, computed on the total month count
    `12·year + (month − 1) + n`.
  - The day is kept if it exists in the target month. Otherwise `ClampToMonthEnd` takes
    the month's last day, and `RejectInvalidDay` fails `domain`.
  - Fails `overflow` when the target year leaves the range.
- `date_add_period(d, p, overflow) -> Date` and its `try_` form apply
  `date_add_months(d, months, overflow)` and then `date_add_days(…, days)`. This is the
  order every surveyed library uses (prior art §6).
- `date_period_until(a, b) -> Period` returns the single-sign period whose months have the
  largest magnitude such that `date_add_period(a, result, ClampToMonthEnd) = b`.
  - For `a ≤ b`, months is the largest `m ≥ 0` with
    `date_add_months(a, m, ClampToMonthEnd) ≤ b`, and days is the remaining gap.
  - For `a > b`, the same rule applies with signs mirrored.
  - The round-trip property defines the function, and §17 tests it over the domain.

**Comparison:** `date_lt`, `date_lte`, `date_gt`, `date_gte`, all `(Date, Date) -> bool`.

**Text:** `date_to_string`, `parse_date`, `try_parse_date` (§8.8).

**Algorithms.** The conversions between epoch days and fields are Hinnant's
`days_from_civil` and `civil_from_days` (prior art §16), written with `floor_div` and a
Euclidean remainder. `mod` in Chelis is a truncating remainder, so the remainder is
written `r = mod(x, y); if r < 0 then r + y`. Over the supported range every intermediate
is below 2^33. The weekday is `(epoch_day + 3)` taken Euclidean modulo 7, with 0 as
Monday (1970-01-01 is a Thursday). No algorithm recurses over days, months or years.

### 8.3 Holiday-rule helpers

These are calendar computations that holiday data packages compose. None of them consults
data.

- `nth_weekday_in_month(year, month, w: Weekday, n: i64) -> Option[Date]` returns the
  `n`-th `w` of the month. It is `None` when the month has fewer than `n` of them, and
  fails `domain` for `n ≤ 0`, an invalid month, or a year outside the range.
- `last_weekday_in_month(year, month, w) -> Date`.
- `weekday_on_or_after(d, w) -> Date` and `weekday_on_or_before(d, w) -> Date` fail
  `overflow` at the range edges. "Strictly after" composes as
  `weekday_on_or_after(date_add_days(d, 1), w)`.
- `easter_sunday_gregorian(year) -> Date` is the Gregorian computus (the anonymous
  Gregorian algorithm in Meeus's form).
- `easter_sunday_orthodox(year) -> Date` is the Julian computus, converted to its
  proleptic Gregorian date.

Both Easter functions use floor division and Euclidean remainders, so negative years are
correct. Both fail `domain` for a year outside the range. For every year inside it the
result lies inside the range: Gregorian Easter falls between 22 March and 25 April of its
year, and Orthodox Easter between −9999-01-21 and 9999-06-27. So neither fails
`overflow`. Observance rules such as "moved to Monday when it falls on a Sunday" are
jurisdiction rules, so they belong to the data packages, not here.

### 8.4 Time and DateTime

**`Time`:**
- `time(hour, minute, second, nanosecond) -> Time` and its `try_` form. Ranges: 0..23,
  0..59, 0..59, 0..999 999 999. There is no 24:00 and no second 60.
- Accessors: `time_hour`, `time_minute`, `time_second`, `time_nanosecond`,
  `time_nanosecond_of_day`; and `time_from_nanosecond_of_day` with its `try_` form.
- `time_add_duration(t, d) -> (i64, Time)` returns the whole days carried and the new
  time. Wrapping past midnight is therefore never hidden: a caller that wants the wrapped
  time takes the second component and ignores the carry explicitly.
- `time_until(a, b) -> Duration` is `b − a` within one day and may be negative.
- Comparisons `time_lt`, `time_lte`, `time_gt`, `time_gte`. Text forms as in §8.8.

**`DateTime`:**
- `datetime(d: Date, t: Time) -> DateTime` is total. Accessors: `datetime_date`,
  `datetime_time`.
- `datetime_add_duration(dt, d) -> DateTime` treats each civil day as 86 400 s, which is
  exact: a civil datetime has no zone, so it has no DST. Fails `overflow`.
- `datetime_add_period(dt, p, overflow) -> DateTime` and its `try_` form change only the
  date part.
- `datetime_until(a, b) -> Duration` is exact and never fails.
- Comparisons, plus text forms.

**Arithmetic precision.** A civil reading in nanoseconds reaches about 6.3 × 10^20, which
exceeds i64. DateTime arithmetic is therefore carried in (second, nanosecond) pairs and
never as a single nanosecond count.

### 8.5 Offset, Instant, OffsetDateTime

**`Offset`:**
- `offset_from_seconds(s) -> Offset` and its `try_` form, with `|s| ≤ 86 399`.
- `offset_seconds(o) -> i64`.
- Text forms `offset_to_string`, `parse_offset`, `try_parse_offset`.

**`Instant`:**
- `instant_from_unix(second, nanosecond) -> Instant` and its `try_` form. Fails `domain`
  for a nanosecond outside `[0, 10^9)` or a result outside the instant range.
- Accessors `instant_unix_second`, `instant_nanosecond`.
- `instant_from_unix_count(count: i64, unit: TimeUnit) -> Instant` and its `try_` form
  read, for example, a millisecond timestamp column.
- `instant_to_unix_count(i, unit, rounding: Rounding) -> i64`.
  - Seconds, milliseconds and microseconds always fit in i64 over the range.
  - Nanoseconds fit only within about ±292 years of 1970; outside that this fails
    `overflow`.
  - Under `RejectInexact`, a count that is not whole fails `domain`.
  - `try_instant_to_unix_count` returns `None` in exactly those two cases (§5).
- `instant_add_duration(i, d) -> Instant` fails `overflow`.
- `instant_until(a, b) -> Duration` is exact and never fails.
- `instant_round_to(i, increment: Duration, rounding) -> Instant` rounds to a multiple of
  `increment`, counted from the unix epoch. The increment must be positive and must
  divide 86 400 s exactly, so buckets align with UTC days (Temporal's rule). Otherwise it
  fails `domain`; a result past the range fails `overflow`.
- Comparisons `instant_lt`, `instant_lte`, `instant_gt`, `instant_gte`.
- `instant_to_datetime_at(i, o: Offset) -> DateTime` is total (§4).
- `datetime_to_instant_at(dt, o) -> Instant` fails `overflow` at the edges of the range.
- Text: `instant_to_string` emits `Z`. `parse_instant` and `try_parse_instant` accept any
  offset and convert exactly.

**`OffsetDateTime`:**
- `offset_datetime(i, o) -> OffsetDateTime` is total.
- Accessors `offset_datetime_instant`, `offset_datetime_offset`, and
  `offset_datetime_local(odt) -> DateTime`.
- Text forms as in §8.8.

**Values not offered.** There is no instant-plus-calendar-unit arithmetic: "one month
after an instant" has no meaning without a zone (§11).

### 8.6 Duration and Period

**`Duration`:**
- `duration(second, nanosecond) -> Duration` normalizes the exact total. Fails `domain`
  when the denoted value lies outside the type (its normalized second leaves i64).
- `duration_from_count(count, unit: TimeUnit) -> Duration`, plus accessors
  `duration_second` and `duration_nanosecond`.
- `duration_to_count(d, unit, rounding) -> i64` fails `overflow` when the result does not
  fit and, under `RejectInexact`, `domain` when the count is not whole; its `try_` form
  returns `None` in both cases.
- `duration_to_seconds_f64(d) -> f64` is the named lossy boundary for putting time on a
  numerical axis. It is a fixed composition, each step rounded to nearest-even, so every
  lane returns the same bits:
  - First split the duration into a whole part `w` and a fraction `f` **of the same
    sign**: `(w, f) = (second, nanosecond)` when `second ≥ 0` or `nanosecond = 0`, and
    `(second + 1, nanosecond − 10^9)` otherwise. The Euclidean form would make the two
    terms of a small negative duration cancel; −1 ns is stored as `{−1, 999 999 999}`.
  - The result is `f64(w) + f64(f) / 1e9`.
  - Correct rounding of the exact value would need integers wider than i64.
  - Whenever `|w| < 2^53` the two terms share a sign, so the result is within one unit
    in the last place. Every difference of two instants satisfies `|w| < 2^53`. §17
    tests the bound, negative sub-second durations included.
- Arithmetic: `duration_add`, `duration_sub`, `duration_negate`, and
  `duration_mul(d, k: i64)`. Each is exact and fails `overflow`; intermediate products
  are split so that only an unrepresentable result fails.
  - There is no duration division. Its exact quotient needs a numerator wider than i64.
    An average interval composes as `duration_to_count` in a unit that fits, followed
    by integer division with the caller's rounding.
- Comparisons `duration_lt`, `duration_lte`, `duration_gt`, `duration_gte`, and text forms.

**`Period`:**
- `period(months, days) -> Period` and its `try_` form fail `domain` for mixed signs.
- Accessors `period_months`, `period_days`.
- `period_negate`, which fails `overflow` only at `i64::MIN`, and `period_mul(p, k)`.
- Text forms.

There is no unit "day" for a duration, and no unit "hour" for a period. A duration of a
day would be ambiguous across a DST change; an hour is not a calendar unit. Each type
therefore has only the units that are exact for it.

### 8.7 Column types

S1 defines the column types, so that the business and columnar stages proceed
independently.
- `dates_from_epoch_days(t: tensor[n, i64]) -> Dates[n]` fails `domain` naming the first
  element outside the range. Its `try_` form follows §6's column rule:
  `try_dates_from_epoch_days(t) -> (Dates[n], tensor[n, bool])`.
- `dates_epoch_days(ds) -> tensor[n, i64]`.
- `instants_from_unix(seconds: tensor[n, i64], nanoseconds: tensor[n, i64]) -> Instants[n]`,
  likewise with `try_instants_from_unix(...) -> (Instants[n], tensor[n, bool])`.
- `instants_unix_seconds(is) -> tensor[n, i64]` and `instants_nanoseconds(is) -> tensor[n, i64]`.

Ownership follows spec/04's rules for tensor-carrying ADTs:
- The constructors and their masked `try_` forms consume their tensors, which become the
  column's storage, because a borrow cannot be stored in an aggregate (spec/04 §8).
- The accessors `dates_epoch_days`, `instants_unix_seconds` and `instants_nanoseconds`
  consume the column and return its storage. An owned tensor-carrying ADT is linear
  (spec/04 §8.4), and a borrowing accessor would have to copy, because a borrow cannot be
  returned. A caller that uses a column again receives implicit linearity's inserted copy.

### 8.8 Text profile

One named profile, built on RFC 3339 and RFC 9557, governs every text form. It is a
profile, not RFC 3339 itself:
- It adds six-digit signed years.
- It rejects RFC 3339's `:60`.
- It has no strftime-style patterns and no locale formatting. Agents and data pipelines
  needing another layout parse it with string operations into the constructors above.

**Grammar:**

```
year     = 4DIGIT / ("+" / "-") 6DIGIT       ; "-000000" is invalid; value within -9999..9999
date     = year "-" 2DIGIT "-" 2DIGIT
time     = 2DIGIT ":" 2DIGIT ":" 2DIGIT [ "." 1*9DIGIT ]
datetime = date ( "T" / "t" / " " ) time
offset   = "Z" / "z" / ( "+" / "-" ) 2DIGIT ":" 2DIGIT [ ":" 2DIGIT ]
instant  = datetime offset                   ; OffsetDateTime uses the same form
duration = [ "-" ] "PT" [ 1*DIGIT "H" ] [ 1*DIGIT "M" ] [ 1*DIGIT [ "." 1*9DIGIT ] "S" ]
period   = [ "-" ] "P" [ 1*DIGIT "Y" ] [ 1*DIGIT "M" ] [ 1*DIGIT "W" ] [ 1*DIGIT "D" ]
```

**Parsing:**
- Field values are range-checked after the grammar: month 1..12, a valid day for the
  month, hour 0..23, minute and second 0..59, offset hour 0..23.
- A duration or period needs at least one component.
- Text the grammar admits is accepted exactly when it denotes a value of the type.
  Anything else fails `domain`, and every `try_parse_` form returns `None` for it:
  - text outside the grammar, including more than nine fractional digits (there is no
    rounding at ingress);
  - grammar-valid text whose value lies outside the type's value set: a year past
    ±9999, a duration or period whose value lies outside its type, or an instant outside
    the instant range of §4.
  - Each value is judged as a whole, not component by component. So
    `-PT9223372036854775808S` (the canonical text of the most negative duration) parses,
    while `PT2562047788015216H`, whose components each fit in i64 but whose total does
    not, fails.
  `overflow` stays reserved for arithmetic results (§5).
- The instant range excludes the last and first offset-width of civil time (§4). So
  `9999-12-31T23:59:59Z`, a common end-of-time sentinel, fails `parse_instant` with
  `domain`; its civil part `9999-12-31T23:59:59` parses as a `DateTime`.
- `-00:00` and `+00:00` parse as offset zero.
- `parse_offset_datetime("…Z")` gives offset zero.
- A period's `Y` counts 12 months and `W` counts 7 days. A duration has no `D`, and a
  period has no time part.

**Formatting** emits exactly one canonical string per value:
- years 0..9999 as four digits, negative years as `-` and six digits;
- `T` as the separator;
- the fraction with trailing zeros removed, omitted entirely when zero;
- offset zero as `Z`, others as `±HH:MM`, with `:SS` only when the seconds are nonzero;
- a duration as total seconds (`PT3661S`, `-PT0.5S`, `PT0S`);
- a period as months and days with zero parts dropped (`P14M3D`, `P0D`).

Formatting and parsing round-trip for every value, and `parse(format(v)) = v` is tested
over the corpus (§17).

## 9. `Std.Datetime.Business` (stage S2)

**`Weekmask`** is a plain record of seven named booleans, `{ monday, tuesday, wednesday,
thursday, friday, saturday, sunday }`. There is no built-in weekend (§2).

**`BusinessCalendar`** is opaque. It holds a weekmask, the sorted unique holiday epoch
days, and a horizon `[valid_from, valid_until]`, both ends inclusive. Its constructor is
`business_calendar(weekmask, holidays: List[Date], valid_from: Date, valid_until: Date)`
with a `try_` form, and it fails `domain` when:
- the weekmask has no business day, which would make offsets non-terminating;
- `valid_from > valid_until`;
- a holiday lies outside the horizon.

Duplicates and holidays falling on non-business weekdays are normalized away, as NumPy
does. Accessors return the weekmask, the normalized holiday list, and both horizon ends.

**Every query is answerable only inside the horizon.** A calendar has no information
outside the dates its data covers. A query there fails `domain`, and the `try_` forms
return `None`. Every operation that takes a calendar, `Unadjusted` rolls included,
requires its date arguments inside the horizon. The one exception is the ends of
`business_day_count` and `dates_business_day_count`: either may be the day after
`valid_until`, which is the exclusive end of a range reaching the horizon's last day and
the first argument of that range's reversed count. The surveyed libraries instead error,
silently degrade to weekends-only, or silently project their rules forward (prior
art §14). Shoals' tables degraded silently, and this rule closes that class.

**Operations:**
- `is_business_day(cal, d) -> bool`.
- `business_day_roll(cal, d, roll: BusinessDayRoll) -> Date`.
  - `Unadjusted` returns `d` once `d` is inside the horizon.
  - `Following` returns the first business day on or after `d`, and `Preceding` the last
    one on or before it.
  - `ModifiedFollowing` is `Following` unless that changes the month, and then
    `Preceding`. `ModifiedPreceding` is the mirror.
  - Fails `domain` when the answer would lie outside the horizon. An answer that days
    outside the horizon could change is treated the same way. `ModifiedFollowing` with no
    business day left in the horizon is answered only when the day after `valid_until`
    already lies in a later month, because the following business day then changes the
    month wherever it falls; `ModifiedPreceding` mirrors this at `valid_from`. [05-OP-73]
    states the rule once: a result exists when every choice of business days outside the
    horizon gives the same answer inside it.
- `business_day_offset(cal, d, n: i64, start: NonBusinessStart) -> Date`.
  - If `d` is not a business day, `start` decides: `RejectNonBusinessStart` fails
    `domain`; `RollStartForward` and `RollStartBackward` roll first.
  - Then it moves `n` business days, forward for positive `n` and backward for negative
    `n`; `n = 0` returns the rolled start.
  - Rolling first and then offsetting is NumPy's rule. The surveyed libraries give three
    different answers for "Saturday plus one business day" (prior art §11), so no answer
    is implied.
- `business_day_count(cal, begin, end) -> i64` counts business days in `[begin, end)`
  when `begin ≤ end`, and is `−count(end, begin)` otherwise. Both ends must lie in
  `[valid_from, valid_until + 1 day]`.
- `business_in_all(a, b)` and `business_in_any(a, b)` combine two calendars.
  - With `business_in_all`, a day is a business day when it is one in both calendars.
    This is a settlement calendar: weekmasks are intersected and holidays united.
  - With `business_in_any`, a day is a business day when it is one in either calendar.
  - The horizon of the result is the intersection of the two horizons, and the
    constructor fails `domain` when that intersection is empty. `business_in_all` also
    fails `domain` when the two weekmasks share no weekday, because a calendar's weekmask
    must name a business day; a union of two valid weekmasks always does.
  - QuantLib and Strata spell these as "join", "combine" and "link", which does not say
    which is which (prior art §11); the names here say what they compute.
- **Vectorized forms** over `Dates[n]`:
  - `dates_is_business_day`, which returns `tensor[n, bool]`;
  - `dates_business_day_roll`;
  - `dates_business_day_offset`, with a `tensor[n, i64]` of offsets;
  - `dates_business_day_count`.

  Each fails `domain` with its scalar twin's detail for the lowest element where that
  twin fails, out-of-horizon elements included.
  - They consume their `Dates[n]` columns. A sibling module reaches a column's storage only
    through the consuming accessor `dates_epoch_days`, so a borrowed column would cost a
    copy on every call. A caller that keeps the column receives implicit linearity's copy,
    as with S1's accessors.
  - They borrow the offsets tensor, which they only read.
- **`try_` forms.** `business_calendar`, `is_business_day`, `business_day_roll`,
  `business_day_offset`, `business_day_count`, `business_in_all`, and `business_in_any`
  each have a `try_` twin. The four calendar readers cannot fail, and the vectorized forms
  have no `try_` twin: §6's masked `try_` rule governs column producers, and these are
  queries over a column.

**Complexity.** No operation recurses or loops per day. A scalar query is O(log h) in the
number of holidays.

**Representation (decided in S2).** A `BusinessCalendar` holds exactly what it denotes:
- the weekmask;
- the holidays, as ascending unique epoch days in a `List[i64]`, each inside the horizon and
  on a business weekday;
- both horizon ends as epoch days.

Nothing proportional to the horizon's length is stored or built:
- Weekmask days are counted and selected in closed form, through a per-week table of the
  business weekdays before each weekday.
- The holidays before a day come from a binary search.
- The business day with a given number of business days before it comes from a second
  binary search, over how many holidays it passes. That predicate is monotone, because
  every holiday is a weekmask day.

The representation is O(h), construction is O(h log h) for the sort, and each scalar query
reads O(log h) list elements.

**Vectorized forms (decided in S2).** Each vectorized form applies the scalar algorithm to
every element, at O(n log h) per call with nothing the horizon's length. That also keeps
each element's failure identical to its scalar twin's by construction. The prefix-count
gather planned above was built and measured, and four problems ruled it out:
- A tensor cannot be sized from a scalar (#469). Every call would therefore build a
  horizon-length `List` first, 7 304 485 elements for a full-range calendar.
- `chelis eval` rejects `scatter_replace` (#2892).
- Compiled C rejected the table program at internal shape invariants (#2893, #2907).
- Compiled C panicked on one variant of it (#2906).

A tensor kernel can replace the per-element loop, without changing the contract, once both
lanes lower such programs.

**Measured cost.** These costs were measured on one shared Apple-silicon workstation under
other load, so they bound orders of magnitude rather than fix figures. Compiled C means
`chelis build --target c` followed by `clang -O2`. `chelis eval` is a release build with the
module loaded from source. Calendar A is Monday to Friday from 1950-01-01 to 2100-12-31
(55 152 days) with 1 064 holidays left after normalization. Calendar B is Monday to Friday
over the whole range (7 304 484 days) with no holidays.

| Operation | Compiled C | `chelis eval` |
|---|---|---|
| construct A (1 490 candidate holidays) | 10 ms | 30 ms |
| `is_business_day` on A | 8 µs | 0.4 ms |
| `business_day_roll(…, ModifiedFollowing)` on A | 69 µs | 4.6 ms |
| `business_day_offset(…, 250, RollStartForward)` on A | 101 µs | 6.2 ms |
| `business_day_count` over 300 days on A | 44 µs | 3.1 ms |
| `dates_business_day_offset` on A, per element of a 1 000-element column | 99 µs | 5.5 ms |
| `business_day_offset(…, 10^6, RollStartForward)` on B | 51 µs | 3.0 ms |

Calendar B's offsets cost no more than A's, because no cost scales with the horizon.

Two costs outside this module show up in the measurements:
- In a debug build of the runtime, compiled C rescans a `bool` tensor's whole storage on
  every element read (#2903), so there building a column with `dates_from_epoch_days` and
  reading a `bool` result grow quadratically in the column's length beyond about 10 000
  elements. `dates_business_day_roll` and `dates_business_day_offset` pay it again building
  their result columns; each form's per-element work stays linear. A release build
  of the runtime does not rescan.
- In `chelis eval`, each `index` into a `List` copies the list (#2335), so the binary
  searches cost O(h) per probe there.

## 10. `Std.Datetime.Columns` (stage S3)

These are vectorized forms of §8 over `Dates[n]` and `Instants[n]`, plus a `Durations[n]`
column (`{ seconds, nanoseconds }` tensors). Every kernel is composed from i64 tensor
primitives (`floor_div`, `mod`, `where`, `gather`, `cumsum`, comparisons), so no new dtype
or builtin is introduced.

**Families:**
- Field extraction to `tensor[n, i64]`: year, month, day, ISO weekday number, day of year.
- `dates_from_ymd` and its masked `try_` form.
- `dates_add_days` (with a `tensor[n, i64]`), `dates_add_months` with `DayOverflow`, and
  `dates_days_until`.
- Comparisons, returning `tensor[n, bool]`.
- `try_parse_dates(List[string])` returns a column and a validity mask; this is coral#41's
  ingestion path. Also `dates_to_strings`.
- For instants:
  - `instants_from_unix_count` and `instants_to_unix_count` take a `TimeUnit` and a
    `Rounding`;
  - `instants_add_duration` and `instants_until`;
  - `instants_round_to` buckets instants for resampling;
  - `instants_to_dates_at(is, o: Offset)`;
  - `instants_seconds_since_f64(is, origin: Instant) -> tensor[n, f64]` gives a
    numerical time axis, applying `duration_to_seconds_f64`'s composition per element.

A failure of any element fails the whole call, except in the masked `try_` forms.

**Backend claims cite cells.** The S3 PR cites, for each family, its cell in
`spec/design/capability_table.md`. For example, i64 tensors on HIP are `Unimplemented`
(#689), so HIP execution of these kernels is claimed only when that cell turns green.

## 11. `Std.Datetime.Zone` (stage S4a)

**`TimeZone`** is opaque. It holds the IANA identifier, the offset in force before the
first transition, a strictly increasing list of transitions (unix second and new offset),
and an optional structured POSIX rule for instants after the last transition (TZif's
footer, RFC 8536 §3.3).

**Constructors:**
- `time_zone_from_tzif(name: string, bytes: List[i64]) -> TimeZone` and its `try_` form
  parse a TZif version 2 or later file and fail `domain` when:
  - the file is malformed;
  - an offset lies outside ±23:59:59;
  - the transitions are not strictly increasing;
  - the footer is inconsistent with the last transition.

  TZif is the interchange format every tz consumer reads, so Std commits to it and is
  independent of how the data package ships the bytes (§13).
- `time_zone_fixed(o: Offset) -> TimeZone`.
- `time_zone_utc()`.
- Accessors `time_zone_name` and `time_zone_offset_at(tz, i: Instant) -> Offset`, with a
  `try_` form for the latter.
  - Instants before the first transition use the initial offset (RFC 8536's time type 0).
  - Instants after the last transition use the footer rule.
  - A zone whose footer is empty has no information past its last transition (RFC 8536
    §3.3). There `time_zone_offset_at` fails `domain` rather than carrying the last
    offset forward, which would be the silent extrapolation §3 forbids.
  - Every other instant has exactly one offset.
  - By §5's coverage rule, every Zone operation that needs the offset at an uncovered
    instant fails `domain`. This includes `zoned`, `zoned_from_local`,
    `zoned_add_duration`, `zoned_add_period` and `zoned_from_text`, and they fail when
    they construct the value, so an existing `Zoned` always has an offset. In
    `zoned_from_local`'s candidate test, an uncovered `dt − o` fails the whole call
    rather than excluding `o`.

**`Zoned`** is opaque and holds `{ instant, zone }`. Values are immutable, so the zone is
shared, not copied. `eq` on two zoned values compares the instant and the zone's
representation: its name, initial offset, transitions and footer. Two encodings of the
same rules (for example a TZif file with redundant transitions and one without) can
therefore compare unequal. A program that means "same instant in the same named zone"
compares `zoned_instant` and `time_zone_name`.
- `zoned(i, tz) -> Zoned` and its `try_` form fail `domain` exactly where
  `time_zone_offset_at` does.
- `zoned_from_local(dt, tz, disambiguation: Disambiguation) -> Zoned` and its `try_`
  form.
  - **Candidates.** The candidates are every instant whose local reading in `tz` is
    `dt`: the set of `dt − o` over the zone's offsets `o` for which
    `time_zone_offset_at(tz, dt − o) = o`. One candidate means `dt` is unique and is the
    result under every policy.
  - **Range edges.** A trial `dt − o` outside the instant range fails the whole call with
    `overflow`, as `datetime_to_instant_at` does, before any coverage check. This can
    happen only within one offset width of the civil range's ends, for example the
    `9999-12-31T23:59:59` sentinel in any zone. The same holds for
    `zoned_add_period`'s re-resolution and for `zoned_from_text` under `UseZoneRules`.
  - **Fold** (two or more candidates): `EarlierInstant` takes the earliest and
    `LaterInstant` the latest.
  - **Gap** (no candidate): there is exactly one transition at instant `T` whose
    pre-transition offset `o_b` and post-transition offset `o_a` satisfy
    `T + o_b ≤ dt < T + o_a` (local times). `EarlierInstant` gives `dt − o_a` and
    `LaterInstant` gives `dt − o_b`, the instants just before and after the gap as
    Temporal defines them.
  - `CompatibleInstant` is earlier in a fold and later in a gap: Temporal's
    `'compatible'`, offered by name only.
  - `RejectNonUniqueLocal` fails `domain` in either case.
- Accessors `zoned_instant`, `zoned_zone`, `zoned_local(z) -> DateTime`, `zoned_offset`.
- `zoned_add_duration(z, d) -> Zoned` moves along the instant timeline.
- `zoned_add_period(z, p, overflow: DayOverflow, disambiguation) -> Zoned` adds `p` to the
  local date and keeps the local time, then re-resolves with `disambiguation`. Date units
  move on the wall clock and time units on the instant line (prior art §6).
- **Text.** `zoned_to_string` emits RFC 9557: `2026-10-01T09:30:00-04:00[America/New_York]`.
  - Parsing is split so it needs no lookup function and no effect.
    `parse_zoned_text` / `try_parse_zoned_text` return a plain `ZonedText` record:
    `{ local: DateTime, offset: Offset, zone_name: string, critical: bool }`.
  - `zoned_from_text(zt, tz, conflict: OffsetConflict) -> Zoned` and its `try_` form
    resolve it against a `TimeZone` the caller obtained.
    - `UseWrittenOffset` keeps the instant the text names.
    - `UseZoneRules` re-reads the local time in the zone. Its gap/fold cases fail
      `domain`; a caller wanting a policy goes through `zoned_from_local`.
    - `RejectOffsetMismatch` accepts the written offset when `local − offset` is one of
      `zoned_from_local`'s candidates for that local time, and fails `domain` otherwise.
      So in a fold, a written offset that is one of the two valid offsets selects that
      occurrence, as Temporal's `'reject'` does.
  - A critical (`!`) zone annotation fails `domain` under `UseWrittenOffset` when the
    written offset fails `RejectOffsetMismatch`'s test (`time_zone_offset_at(tz,
    local − offset) ≠ offset`), as RFC 9557 requires.
  - The suffix tags `u-ca=iso8601` and `u-ca=gregory` are accepted. Other calendars are
    rejected, unknown elective tags are ignored, and unknown critical tags are rejected.

## 12. `Std.Datetime.Clock` (stage S5)

**API:**
- `clock_now() -> Instant ! { IO }` reads the host wall clock on the POSIX timescale.
- `monotonic_now() -> MonotonicInstant ! { IO }` reads a clock that never runs backwards.
  - `MonotonicInstant` is opaque. Its only operation is
    `monotonic_until(a, b) -> Duration`, and nothing converts it to an `Instant`.
  - This is the separation Go merges into one type and then documents around (prior
    art §9).

**Failures.** A host reading outside the instant range, or a host clock error, fails
with kind `io` (§5), naming the reading.

**Builtins.**
- Each clock is one host-lane builtin under [05-HOST-2], in the same family as
  `process_run`. Each returns `(i64, i64)` (seconds and nanoseconds) from a single
  read, so the two halves cannot tear.
- Each needs an atom in spec/05, a row in `builtin_semantic_identities.md`, and
  `generate_rejection_registries.py --write`. It changes no published C header.
- Compiled host execution of IO builtins follows chelis#1297. S5 delivers the eval lane
  and names #1297 for the compiled lane.

**Why a separate module.** The clock lives in its own module so that a program, or a
shell's policy, can import all of pure `Std.Datetime` and provably never read the clock.

## 13. The `bed` repository: external data (stages S4b and S7)

`bed` (as in seabed) is one new Chelis-Lang repository, private until chelis itself is
public. It is framed broadly: it holds external data that Chelis programs consume as
declared inputs, meaning data whose truth is set outside any program by a legislature, a
standards body or an exchange, and synced from that upstream on the upstream's schedule.
Date and time data is its first content, not its definition.

Every package in `bed` follows the same rules:
- a sync generator (Python under uv, with tests) builds the package from one pinned
  upstream release;
- the package records its provenance (source, upstream release or retrieval date) and
  exposes its data version as a value;
- the package's version is derived from the upstream release, independently of the
  compiler's;
- where the data covers a span of time, the values it produces carry that span, and
  nothing is extrapolated silently.

Other data of this kind fits the same frame later (currency codes, market identifier
codes, the IERS leap-second table), but this design commits only to the two packages
below.

**tzdata (S4b):**
- A generator (Python under uv, with tests) builds the package from a pinned IANA
  release.
- The package's version records that release (for example `2026b`), and the package
  exposes it as `tzdata_version() -> string`.
- Lookup is by exact IANA name, case-sensitive. Backward-compatible link names resolve
  to their target zone and keep the target's canonical name.
- How the bytes reach `time_zone_from_tzif` is decided by an S4b spike. Generated Chelis
  source keeps lookup pure. TZif files read at run time make loading `IO` but keep the
  package small. The spike measures compile and evaluation cost, and records the choice
  here.

**Holidays (S7):**
- One `BusinessCalendar` producer per jurisdiction or market:
  - bank and public holidays, e.g. England and Wales, US federal, Japan, New South Wales,
    Hong Kong;
  - market calendars, e.g. NYSE, SIFMA, TARGET2.
- Shoals' current tables are the starting list. They are rebuilt with observance rules
  and corrected.
- **Provenance:** each calendar names its source and retrieval date.
- **Horizon:** each calendar's horizon is the span its source actually publishes.
- **Projections are named, never silent.** Extending a calendar past its published
  years by applying its rules is offered only as a separately named producer
  (`…_projected(until_year)`), so a projection is visible in the program text.
- Lunar and announced holidays have no rule, so a projected calendar cannot include them
  and says so.

**Versioning.** Under the shell contract both packages pin `compiler = "=X.Y.Z"`, so they
are re-released for every compiler release as well as for every data release. This is
accepted, since `reef conform bump` automates it. It will be revisited if the churn proves
costly.

## 14. Shoals on Std.Datetime (stage S6)

Once the downstream hold lifts (§16), Shoals rebuilds its finance layer on
`Std.Datetime` and the holidays package in one release, and deletes its generic copies:
`add_months`, `days_in_month`, the Easter computus, the weekend predicates, and the
holiday tables.

**Day counts:**
- Each day count returns an exact rational, plus one correctly rounded conversion to the
  caller's float dtype. The surveyed libraries all convert to float immediately (prior
  art §12).
- Conventions carry unambiguous names. ACT/365 and "30/360 ISDA" are not accepted names,
  because libraries disagree on what they mean.
- Disputed definitions are named by their source (ACT/ACT AFB).
- Inputs beyond two dates are required arguments:
  - ACT/ACT ICMA takes the reference period and frequency;
  - 30E/360 ISDA takes the maturity date;
  - 30/360 US takes the end-of-month flag;
  - BUS/252 takes a `BusinessCalendar`.

**Tenors, lags and schedules:**
- `Tenor` is built on `Period`: one month is a month, not 30 days.
- ON, TN and SN are pairs of business-day lag and length, not calendar days.
- Spot lag follows Strata's two-calendar form (count in one calendar, adjust in another).
- Schedules step from a fixed anchor by k × tenor, so there is no drift, with explicit
  stub, roll and end-of-month arguments.

**Fixes.** This closes shoals#87 and the audit items the planning review recorded against
`date.ch`, `tenor.ch`, `holidaycal.ch`, and the date properties.

**Pin.** Shoals moves to the chelis release that carries S2.

## 15. Normative placement and registration

**Registries.** Datetime identities live in the existing registries:
- functions in `spec/registry/stdlib_numeric_manifest.md` under [05-OP-35];
- every new type with a reachable numeric field, opaque or not (`ZonedText` included),
  in `spec/registry/stdlib_adt_identities.md` under [05-OP-34].

[05-OP-35] gains one sentence routing their semantics to a new atom, "`datetime::*`
identities follow [05-OP-73]", as JSON access already follows [05-OP-2..5]. S2 extends the
sentence and the atom's scope to the `datetime/business::*` identities. The capacity
census, the frozen-contract oracle (`scripts/dtype_phase4b_oracle.py`) and the registry
bijection test then need no new structure.

**Atom prose lands per stage.** The new atom's prose is written stage by stage, each
stage adding the paragraphs for the identities it exports. The atom is the contract for
"exactly" the registered identities, so prose about identities not yet registered would
contradict it. This document holds the rest of the decided design until then.

**Rounding atom.** S1 also adds [05-OP-74], one short atom defining the seven
`Std.Rounding` modes for an exact value and a positive quantum (§7). The datetime atom cites it, and so does
`Std.Decimal`'s atom when Decimal adopts the shared type.

**S1 also amends the existing atoms.**
- In [05-OP-34]:
  - admit opaque standard-library ADTs whose only construction path is exported
    validating producers;
  - replace the `time::*` rows;
  - rewrite the sentence on Decimal, date and duration invariants;
  - correct the count ("five" against four rows).
- In [05-OP-35]:
  - delete the calendar paragraph, its "time values" mention, and the "calendar ordinal
    computations" clause of the atom's second paragraph;
  - remove the 16 `time::*` rows;
  - correct the count.

**Clock atoms.** S5 adds the clock builtins' atom next to the other host IO identities,
and extends [05-HOST-2]'s list of host operations with them.

**Release.** Removing `Std.Time` removes census rows, which the remediation roadmap's
invariant 7 makes 0.19 payload by default. S1 ships in a 0.18.x patch rather than
waiting for 0.19, by release decision, as 0.18.4 did for its ABI change. Every `Std.Time`
callable already fails, so the removal stops no running computation. A program that only
builds `Std.Time` records or names its types still type-checks against the fenced module
and breaks when the module is removed. The known importers are Shoals and hello-chelis,
whose cut-overs are S6 and S8.

## 16. Stages

| Stage | Repository | Delivers | Needs |
|---|---|---|---|
| S0 | chelis | this document; the prior-art survey; tracker issues | nothing |
| S1 (#2859) | chelis | `Std.Datetime` (§8) and `Std.Rounding` (§7); atom amendments and registries; `Std.Time` deleted | S0 |
| S2 (#2860) | chelis | `Std.Datetime.Business` (§9) | S1 |
| S3 (#2861) | chelis | `Std.Datetime.Columns` (§10) | S1 |
| S4a (#2862) | chelis | `Std.Datetime.Zone` (§11) | S1 |
| S4b | bed | tzdata package (§13) | S4a's constructor |
| S5 (#2863) | chelis | `Std.Datetime.Clock` (§12) | S1 |
| S6 (shoals#104) | shoals | finance layer on Std (§14) | S2 and S7 released; maintainer go-ahead |
| S7 | bed | holidays package (§13) | S2 in a release |
| S8 (hello-chelis#40) | hello-chelis, coral | `datetimecal` example rewritten; coral time-index issue filed | S1 released; maintainer go-ahead |

**Parallelism.** S2, S3, S4a and S5 are written in parallel once S1 merges.
- They merge one at a time, because each touches the same registries, census, count
  literals and standard-library bundle.
- Each bundle regeneration builds the compiler, so their builds are sequenced.

**Downstream hold.** The Shoals and hello-chelis cut-overs (S6, S8) wait until the chelis
and `bed` stages are finished, or until a maintainer approves an earlier cut-over. Until
S6, Shoals keeps its own date layer.

## 17. Verification

- **Calendar bijection.**
  - An independent oracle in Python, under uv, generates golden vectors: from
    `datetime.date.toordinal` for years 1 to 9999, and from a separately written
    Hinnant reference for years −9999 to 0.
  - CI checks the range edges, every day from 1900 to 2100, and seeded random samples.
  - A manual gate in `docs/manual_gates.md` checks every one of the 7 304 484 days in
    compiled C. It covers field round trips, weekday, day of year, ISO week, and
    `date_period_until`'s defining property.
- **Text.** `parse(format(v)) = v` and `format(parse(s)) = canonical(s)` over a corpus
  that includes:
  - negative and four-digit years, and both range edges;
  - every offset form, fractions of 1 to 9 digits, and the space and lowercase
    separators.
- **Negative corpus.** Every `domain` and `overflow` path, with its exact message:
  - `:60`, `-000000`, ten fractional digits;
  - mixed-sign periods, `RejectInvalidDay`, out-of-horizon queries;
  - each `Reject…` policy;
  - construction of every opaque type outside its module.
- **Failure contract.** Wrapping each failing call with the expected message shows that
  no primitive numeric trap escapes (§5).
- **Lanes.** `chelis eval` and compiled C agree on the whole corpus.
- **Float boundary.** `duration_to_seconds_f64` is compared with the exact rational
  (Python `fractions`) over the corpus, negative sub-second durations such as −1 ns and
  −1 µs included, and must lie within one unit in the last place wherever its whole
  part satisfies `|w| < 2^53` (§8.6).
- **Business days.** Differential against `numpy.busday_offset` and `busday_count`
  under matching roll modes. NumPy's `busday_count` counts `(end, begin]` when
  `begin > end`, which is not antisymmetric, so reversed counts are compared through
  `count(a, b) = −count(b, a)`.
- **Zones.** Differential against Python's `zoneinfo` loaded from the same tzdata release,
  including every DST gap and fold under each `Disambiguation`.
- **Shoals.** Day counts are checked against QuantLib and Strata reference values, with
  disputed conventions named.
- **Removal.** Importing `Std.Time` fails as an unresolved import, as the existing
  removed-module test does for `Std.Init` and `Std.Tokenizer`. The changelog fragment
  and docs name the replacement.
- **Downstream.** Coral and nautilus pass `chelis check` against each new
  standard-library bundle. Shoals and hello-chelis import `Std.Time`, so they fail
  `chelis check` against bundles without it until their cut-overs (S6, S8, §15).

  The first opaque type in the standard library can turn an unannotated accessor lambda
  into an `OpaqueTypeViolation` in any program that reaches the module (spec/04 §2.5), so
  S1 adds a fixture for that case.

## 18. Alternatives rejected

- **Repairing `Std.Time` in place.** Its public record fields admit invalid dates, and
  its all-i64 year contract needs multi-limb arithmetic in a language whose widest
  integer is i64. The defect class is the representation, so the representation changes.
- **All i64 years.** Rejected for the reasons in §4.
- **One duration type covering calendar and clock units (Temporal).** It needs a reference
  date to answer basic questions. Two types (Java, jiff, Arrow) make the distinction a
  type error instead.
- **Defaults for policies.** Rejected for the reasons in §3.
- **Instants as a single i64 nanosecond count.** That covers only 1677 to 2262; pandas
  added other units to escape it (prior art §10).
- **A datetime tensor dtype.** The column types compose i64 tensors and need nothing from
  the compiler, which is the small-core rule.
- **A NaT sentinel.** NumPy's NaT compares inconsistently, sorts against its own `<`, and
  leaks into integer views (prior art §10). Masks say the same thing explicitly.
- **The host's tz database, locale or clock read implicitly.** Results would vary between
  machines for identical program text.
- **strftime patterns in Std.** Pattern letters are a known bug source (`YYYY` against
  `yyyy`; prior art §8). Ingestion of other layouts composes from string operations.
- **Holiday data in Std.** Holiday data is legislated data (§2), and Std ships with the
  compiler.
- **`@invariant` on the value types.**
  - Opacity already makes invalid values unconstructible, and an invariant is never
    checked at run time (spec/04 §2.5).
  - Under `chelis prove`, an invariant-carrying type makes every exported producer that
    returns it inside a list or record an error (the opaque-invariants RFC's
    covered-or-rejected rule). `business_calendar_holidays` and `Zoned` are such
    producers.
  - It can be added later if the prover learns those containers.

## 19. Decisions owned by a stage

Each item below is decided in the named stage's PR and recorded in this document there:
- the ownership form of each column signature (S1, S3);
- `BusinessCalendar`'s internal representation, with its measured cost (S2);
- how tzdata bytes reach `time_zone_from_tzif`, decided by the S4b spike;
- the holidays package's provenance format and its list of calendars (S7);
- the clock builtins' names (S5);
