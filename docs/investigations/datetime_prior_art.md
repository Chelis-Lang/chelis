# Date and time library prior art

This survey of mature date and time libraries informs
[`spec/design/std_datetime.md`](../../spec/design/std_datetime.md).

## 1. Scope and sources

Libraries: java.time, TC39 Temporal, the Rust crates jiff, chrono and `time`, Go's
`time`, C++20 `<chrono>`, Python `datetime`/`zoneinfo`, Haskell `time`,
kotlinx-datetime, Swift Foundation; NumPy, pandas, Apache Arrow, Polars, Julia `Dates`,
kdb+/q; QuantLib, OpenGamma Strata, BusinessDays.jl, DayCounts.jl. Also RFC 3339, RFC
9557, the ISDA 30/360 memo, and Hinnant's and Neri and Schneider's day-number
algorithms. Sources are primary (documentation, specifications, or source at the
version in section 17, else main or master, which also backs uncited finance facts).
Survey date: 2026-10-01. "(not re-verified)" marks a claim not checked against a
primary source; "verified on NumPy 2.4.6" marks behaviour executed on that release.
This is evidence, not a design. Where the general and the numerical halves of the
survey state the same fact, they agree.

## 2. Type taxonomy

Roles: *instant* (an exact point on a global timeline), *civil* (a wall-clock date or
time with no zone), *zoned* (an instant with a zone identity), *exact duration* (fixed
seconds), and *calendar period* (months, days, or similar units whose length depends on
where they apply). An empty cell means no dedicated type was recorded.

| Library | Instant | Civil | Zoned | Exact duration | Calendar period |
|---|---|---|---|---|---|
| java.time | `Instant` | `LocalDate`, `LocalTime`, `LocalDateTime` | `OffsetDateTime`, `ZonedDateTime` | `Duration` | `Period` |
| Temporal (Stage 4, March 2026 [Stage4]) | `Instant` | `PlainDate`, `PlainTime`, `PlainDateTime`, `PlainYearMonth`, `PlainMonthDay` | `ZonedDateTime` | one `Duration` for both roles; calendar units need `relativeTo` | (same `Duration`) |
| jiff | `Timestamp` | `civil::{Date, Time, DateTime}` | `Zoned` | `SignedDuration` | `Span` (calendar and clock units) |
| chrono | `DateTime<Tz>` | `Naive*` | `DateTime<Tz>` | `TimeDelta` | `Months`, `Days` |
| `time` crate | `OffsetDateTime`, `UtcDateTime` | `PrimitiveDateTime`, `Date`, `Time` | | `Duration` | none (no month arithmetic) |
| Go | one type, `Time`: instant plus `*Location` plus an optional monotonic reading | (same `Time`) | (same `Time`) | `Duration`, i64 nanoseconds | month arithmetic on `Time` normalizes (section 6) |
| C++20 | `sys_time` (Unix time); `utc_clock`, `tai_clock`, `gps_clock` | `local_time`, `year_month_day` | `zoned_time` | `duration`, generic over its representation | month and year arithmetic on `year_month_day` |
| Python | one type, `datetime`, naive or aware by `tzinfo` | naive `datetime`; `date`, `time` | aware `datetime` | `timedelta` | none (no month arithmetic) |
| Haskell `time` | `UTCTime`; TAI `AbsoluteTime` | `Day`, `LocalTime` | `ZonedTime` (fixed offset) | `NominalDiffTime` | `CalendarDiffDays` |
| kotlinx-datetime | `kotlin.time.Instant` | `LocalDate`, `LocalTime`, `LocalDateTime`, `YearMonth` | zone types `TimeZone`, `FixedOffsetTimeZone` | | `DateTimePeriod`, `DatePeriod` |
| Swift Foundation | `Date`, a `Double` count of seconds since 2001 (range not re-verified) | `DateComponents` with a `Calendar` | | | |

Columnar systems carry the split as dtypes (section 10): pandas `Timedelta` against
`DateOffset`, Polars `"24h"` against `offset_by("1d")`, Arrow `duration` against
`interval`, NumPy `timedelta64` (month units do not mix with day units), Julia periods.

**Convergence and divergence.** The general libraries agree on separate instant, civil,
and zoned concepts (the single-type designs, Go and Python, carry most of the bug
classes in section 15); immutable, opaque values; a proleptic Gregorian calendar
(Hinnant's algorithms include year 0 and negative years; Python starts at year 1); the
POSIX timescale (but see section 4); distinct exact and calendar durations; one order
for applying a mixed period (section 6); date units on the wall clock and time units on
the instant timeline for zoned values; IANA zone data and RFC 3339/9557 strings; an
injectable clock; and Hinnant or Neri-Schneider day counting. NumPy and Julia carry no
zone. They differ on defaults (clamp, roll over, or invalid value for month overflow;
compatible or throwing for DST), on what sets the range (storage layout for chrono and
pandas, legacy `Date` for Temporal, four-digit years for jiff and Python, an `int` year
for Java), on zone data (system data, fresh but host-dependent, against embedded data,
pinned but going stale), on durations (Temporal's one type with `relativeTo` against two
in Java and jiff), and on parsing `:60` (constrain against reject).

## 3. Representation, resolution, and range

| Library | Representation and resolution | Range | Stated reason |
|---|---|---|---|
| java.time | `Instant`: i64 epoch seconds plus an `int` of nanoseconds in 0..999,999,999 [J-Inst] | ±1,000,000,000 years | "one year earlier than the minimum LocalDateTime... sufficient... to handle the range of ZoneOffset... year fits in an int" [J-Inst] |
| Temporal | BigInt epoch nanoseconds | ±10^8 days | "the same as the old-style JavaScript Date" [T-inst]; `PlainDate` is slightly wider so that `toPlainDate()` succeeds on any `PlainDateTime` [T-pd]. `Duration`: years, months, weeks < 2^32; time part < 2^53 s [T-spec duration.html] |
| jiff | i64 seconds plus nanoseconds | years -9999..=9999 | the year range is "the primary point" from which the rest is derived; `Timestamp` is shrunk by the maximum offset (±25:59:59) so every `Timestamp` converts to a civil value, "because getting a civil datetime is important for formatting" [jiff-src util/b.rs] |
| chrono | `NaiveDate` packs `(year<<13)\|ordinal\|flags` into an i32 [chrono-src] | ±262,143 years | a consequence of the bit packing |
| `time` crate | private fields | ±9999; ±999,999 with the `large-dates` feature | the feature has "performance implications and introduces some ambiguities when parsing" [time-date] |
| Go | `wall` u64 plus `ext` i64 seconds since year 1 [Go src] | effectively unbounded internally | RFC 3339 output requires years 0000..9999 [Go] |
| C++20 | user-chosen | `year` is -32767..32767 (not re-verified) | |
| Python | fields; microsecond resolution | years 1..9999 [py-dt] | none given |
| Haskell `time` | `Day` is a Modified Julian Day as an unbounded `Integer` [hs] | unbounded | |

Reference points: pandas nanosecond timestamps cover only 1677-09-21 to 2262-04-11,
"limited by the underlying 64-bit integer", and second resolution widens this to about
±2.9e11 years [pandas]. PostgreSQL stores `timestamp` as 8 bytes of microseconds
covering 4713 BC to 294276 AD [pg]. Section 10 covers columnar types; section 14 gives
QuantLib's `Date` range.

**i64 arithmetic at years -9999..=9999** (proleptic Gregorian, Unix epoch). At
nanosecond resolution this range does not fit one i64; java.time and jiff store seconds
and nanoseconds separately, Temporal uses a BigInt, and Go splits a u64 and an i64.

| Quantity | Magnitude at ±9999 years | Fits in i64 |
|---|---|---|
| Unix seconds | -377,705,116,800 (-9999-01-01T00:00:00Z) to 253,402,300,799 (9999-12-31T23:59:59Z), both below 2^39 | yes |
| Day count from 1970-01-01 | -4,371,587 to 2,932,896, below 2^23 | yes (also fits i32) |
| Month index (year × 12 + month) | about ±120,000 | yes |
| 400-year era index × 146,097 | about 3.7 million | yes |
| Difference between any two instants | about 6.3e11 s | yes |
| Total nanoseconds of an instant | about 3.8e20, about 70 bits with sign | no: i64 holds 9.22e18 ns, which is ±292 years |

A single i64 of microseconds spans about ±292,277 years. jiff bounds each `Span` field
by the full span of its range (years ≤ 19,998; seconds ≤ about 6.3e11); at those bounds,
converting a span to seconds stays inside i64.

## 4. Timescale and leap seconds

Temporal ("Like Unix time... ignores leap seconds" [T-inst]), jiff ("behaves as if they
don't exist" [jiff-ts]), Python [py-dt], and Go ("cannot be parsed... not
representable" [Go]) use POSIX days of 86,400 s. java.time defines its timescale as
UTC-SLS, with a leap second "spread equally over the last 1000 seconds of the day", but
implementations "are not required to actually perform the UTC-SLS slew" [J-Inst].
chrono represents `:60` as nanoseconds ≥ 1e9, but "there is absolutely no guarantee
that the leap second read has actually happened" and "Associativity does not generally
hold" [chrono-nt]. C++20 has real `utc_clock` and `tai_clock` backed by the tzdb
leap-second list; Haskell has a TAI `AbsoluteTime` with a leap-second map (not
re-verified). NumPy's "Datetime64 shortcomings" section concedes that positive leap
seconds cannot be parsed and that intervals in seconds across UTC can be off by an
integer number of seconds [np-dt]. Section 8 covers parsing `:60`.

## 5. Time zones: data source, versioning, and DST gap/fold resolution

**Data source and version.** No surveyed library guarantees that zone data is a
versioned, explicit input; C++20, jiff, and chrono-tz each provide part of it.

| Library | Zone data source | Version exposed |
|---|---|---|
| java.time | rules provider bundled with the JDK | `ZoneRulesProvider.getVersions` (not re-verified) |
| Temporal | the engine's own IANA data; ECMA-402 requires that, once observed, offsets and transitions stay "consistent with results previously observed by that agent" [402] | |
| jiff | system tzdb on Unix; embedded `jiff-tzdb` where the platform has none; `tz::include!` at compile time; explicit `TimeZoneDatabase::from_dir` [jiff-src tz] | `jiff_tzdb::VERSION` [jiff-tzdb] |
| chrono | `Local` reads the system; `chrono-tz` compiles tzdb in at build time | `IANA_TZDB_VERSION` [chrono-tz] |
| Go | `ZONEINFO` environment variable, then the system, then the `$GOROOT` zip, then embedded `time/tzdata` only as a fallback [Go] | |
| C++20 | implementation-defined | `tzdb::version`, `reload_tzdb`, `remote_version` [cpp-tzdb] |
| Python | system `TZPATH`, falling back to the `tzdata` package; configurable through `PYTHONTZPATH` and `reset_tzpath` [py-zi] | nothing in the documentation |
| kotlinx-datetime | java.time on the JVM; JavaScript needs `@js-joda/timezone` [kt] | |
| pandas 3.0 | `zoneinfo`, replacing pytz; results depend on the system's tzdata version | |

**Gap and fold resolution.** In a *gap* (clocks spring forward) a wall-clock time does
not exist; in a *fold* (clocks fall back) it occurs twice.

| Library | Default | Caller choice |
|---|---|---|
| java.time | gap: "shifted forwards by the length of the Gap"; overlap: keep the previous offset, otherwise the earlier one [J-ZDT] | `withEarlierOffsetAtOverlap`, `withLaterOffsetAtOverlap`; `ofStrict` throws |
| Temporal | `'compatible'`: earlier in a fold, later in a gap [T-tz] | `disambiguation`: `'compatible'`, `'earlier'`, `'later'`, `'reject'` |
| jiff | `Compatible` [jiff-dis] | `Earlier`, `Later`, `Reject` |
| Python | `fold=0`: the earlier time in a fold; in a gap "the greater of the two" [PEP495], the same as compatible | `fold=1` |
| kotlinx-datetime | gap: the earlier offset; overlap: the earlier instant [kt-toInstant] | none |
| Go | "correct in one of the two zones... does not guarantee which" [Go] | none |
| chrono | none: returns `MappedLocalTime::{Single, Ambiguous, None}` | `.single()`, `.earliest()`, `.latest()` |
| C++20 | none: `to_sys(tp)` throws `ambiguous_local_time` or `nonexistent_local_time` [cpp-tosys] | `choose::earliest`, `choose::latest`; a gap then maps to the transition instant |

The defaults converge on "compatible" behaviour, and every one traces to compatibility
with older libraries. Temporal's only stated justification is that `'compatible'`
"matches the behavior of legacy Date as well as... moment.js, Luxon, and date-fns"
[T-tz]. C++20 (throws without `choose`) and chrono (returns `MappedLocalTime`) have none.

**Offset conflicts and stored values.** Temporal's `offset` option defaults to `'reject'`
in `from()` "because there is no obvious default solution" [T-tz]. jiff's parse default
is `OffsetConflict::Reject`; its docs show a 2018 tzdb and the current one giving
different offsets for the same string [jiff-src tz/db]. Stored future zoned values can
become wrong after a tzdb update (Brazil 2019) [jiff-src tz/db; T-tz]. RFC 9557: on an
offset and zone mismatch, a critical zone means the application MUST act; otherwise it
MAY [RFC9557]. Section 10 has Arrow's zone semantics.

## 6. Calendar arithmetic

**Month overflow.** Adding months can land on a day the target month lacks.

| Library | Jan 31 + 1 month | Policy |
|---|---|---|
| Temporal | Feb 28 [T-pd] | `overflow: 'constrain'` (default) or `'reject'` |
| java.time | Feb 28 or 29 | always clamps; `ResolverStyle` applies to parsing only |
| jiff | Feb 28 or 29 | always clamps (`new_constrain`) [jiff-src civil/date.rs] |
| chrono | clamps; `None` only when out of range [chrono-nd] | |
| Julia, QuantLib | clamp to the end of the month | |
| Haskell | `addGregorianMonthsClip` gives Feb 28; `addGregorianMonthsRollOver` gives Mar 2 [hs] | two separately named functions, no default |
| Go | Oct 31 + 1 month = Dec 1 [Go] | silently normalizes |
| C++20 | June 31: a value that exists but where `ok()` is false; the caller picks `ymd.year()/ymd.month()/last` (clamp) or `sys_days{ymd}` (roll over) [cpp-ymd] | the caller must choose |
| NumPy | month and year deltas cannot be added to day units: "Cannot cast ufunc 'add' input 1 from dtype('<m8[M]')... 'same_kind'" (verified on NumPy 2.4.6) | a forced conversion uses the 400-year average [np-dt] |
| `time` crate, Python | no month arithmetic | |

**Order of applying a mixed period.** Every library with month arithmetic uses one
order: (1) years and months together, (2) resolve the day of month, (3) weeks × 7 plus
days, (4) time units: Temporal's `CalendarDateAdd` [T-spec calendar.html], Java
`Period.addTo` ("adds years and months together... ensures correct behaviour at the end
of the month" [J-Per]), jiff [jiff-src], Haskell ("Add months... then add days" [hs]).
Julia applies combined periods "by the Periods' *types*" (Year, then Month, then Week,
and so on), not in the order written [jl-dates].

**Zoned values and the two kinds of day.** Date units move along the local (wall-clock)
timeline, time units along the instant timeline [J-ZDT]. pandas: "A Timedelta day will
always increment datetimes by 24 hours, while a DateOffset day will increment ... to the
same time the next day" [pandas]. Polars: `offset_by("1d")` is a calendar day, while
`"24h"` is a duration [pl-offset].

**Differences.** Temporal `until` and `since` take `largestUnit`, `smallestUnit`,
`roundingIncrement`, and `roundingMode`, which has 9 modes and defaults to `trunc`;
calendar units require `relativeTo` [T-pd, T-dur]. Java
`Period.between(2010-01-15, 2011-03-18)` is P1Y2M3D, and a month counts only "if the end
day-of-month is ≥ the start day-of-month" [J-Per]. Haskell guarantees
`addGregorianDurationClip (diffGregorianDurationClip d2 d1) d1 = d2` [hs].

**Algebraic caveats.** Clamping makes iterated month addition drift: (Jan 31 + 1M) +
1M = Mar 28, but Jan 31 + 2M = Mar 31. Julia is not associative:
`(Date(2014,1,29)+Day(1))+Month(1)` = 2014-02-28, but
`(Date(2014,1,29)+Month(1))+Day(1)` = 2014-03-01 [jl-dates]. Java compares periods
structurally, "15 Months is not equal to 1 Year and 3 Months" [J-Per], as does Strata's
`Tenor` (section 13). Java allows mixed signs within one period; Temporal requires that
"All non-zero values must... have the same sign" [T-dur]. pandas anchored offsets depend
on whether the input is on the anchor: 2014-01-02 - `MonthEnd(1)` = 2013-12-31, but
2014-01-31 + `MonthEnd(1)` = 2014-02-28.

## 7. Validation and opacity

Opaque types with fallible constructors: java.time (throws), Temporal (constrains by
default, rejects optionally), jiff, `time` (returns `Result`), and chrono (`_opt`
constructors; the panicking `from_ymd` and similar were deprecated in 0.4.23
[chrono-nd]), though chrono's `TimeDelta::seconds` and `TimeDelta::days` still panic
when out of bounds [chrono-src time_delta.rs]. Invalid values held or silently fixed:
C++ `year_month_day` can hold an invalid date, checked with `ok()`; Go normalizes
silently; Haskell's `fromGregorian` clips ("Invalid values will be clipped... month
first, then day"), while `fromGregorianValid` returns `Nothing` [hs]. Immutability is
now standard; Moment ("moments are mutable" [moment]) and `java.util.Date`/`Calendar`
(not re-verified) are the documented counterexamples. Arrow `date64` stores milliseconds
whose "values are evenly divisible by 86400000", a redundant invariant every writer
must keep [arrow-schema].

## 8. Text formats

- **RFC 3339** requires a four-digit year ("between 0000AD and 9999AD"), allows lowercase
  `t` and `z`, and allows `:60` [RFC3339]. Go's RFC 3339 output requires years
  0000..9999 [Go].
- **RFC 9557** redefines `Z` as "UTC known, local offset unknown" and adds bracketed
  suffixes; `!` marks one critical, and a critical suffix that is not understood or is
  inconsistent "MUST" be rejected [RFC9557]. Offset and zone mismatch: section 5.
- **ISO 8601 expanded years and other parser choices:** Temporal accepts ±6-digit years
  and makes `-000000` a syntax error [T-spec abstractops.html]; it accepts no localized
  formats and no week dates, and rejects `Z` when parsing into a Plain type [T-strings].
  Python 3.11 `fromisoformat` does not support `±YYYYYY` and lets "any single unicode
  character" stand in for `T` [py-dt]. The `time` crate's `large-dates` feature
  "introduces some ambiguities when parsing" [time-date].
- **Parsing `:60`:** Temporal and jiff silently turn it into `:59`
  [T-spec abstractops.html; jiff-ts]. The `time` crate turns it into 59.999999999 and
  allows it only as the last second of a month in UTC [time src parsing/parsable.rs].
  Go rejects it [Go]; NumPy cannot parse positive leap seconds [np-dt].
- **Locale formatting:** Temporal delegates it to Intl, outside the core. Pattern
  letters in Java and Swift format strings (`YYYY` week-year against `yyyy`/`uuuu`) are
  a well-known source of bugs (not re-verified).

## 9. Clock access

Java: "Best practice for applications is to pass a Clock into any method that requires
the current instant", with `Clock.fixed`, `Clock.offset`, and `Clock.tick` [J-Clock].
Temporal exposes the current time only through the separate `Temporal.Now` namespace.
kotlinx-datetime moved `Instant` and `Clock` to `kotlin.time` in 0.7.0 [kt]. Haskell's
`getCurrentTime` runs in `IO`. Go combines the wall and monotonic clocks in one value:
comparisons use the monotonic reading, `Round(0)` removes it, and `==` compares the
`Location` and the monotonic reading, so the docs say to "prefer t.Equal(u)" [Go].
jiff's `Timestamp::now()` panics if the system clock is outside the supported range
[jiff-ts]. QuantLib's `Settings::instance().evaluationDate()` is a singleton: "Today's
date is returned if the evaluation date is set to the null date (its default value)",
and no notification is sent "as the clock strikes midnight". The opt-in
`QL_REQUIRE_EXPLICIT_EVALUATION_DATE` was added later [ql-settings].

## 10. Columnar and array representation

| System | Date | Timestamp | Where unit and zone live | Missing values |
|---|---|---|---|---|
| NumPy | `datetime64[D]` | `datetime64[unit]`, unit from Y down to as, i64, epoch 1970 | unit in the dtype; no zone ("NumPy does not store timezone information") | NaT, the sentinel INT64_MIN |
| pandas | as NumPy | `datetime64[unit, tz]` | unit and zone on the dtype | `pd.NaT` sentinel |
| Arrow | `date32` (i32 days); `date64` (ms; "values are evenly divisible by 86400000") | `timestamp[unit, tz?]` | type parameters | validity bitmap |
| Polars | `Date`: i32 days since 1970 | `Datetime(time_unit in {ns, us, ms}, default us; time_zone)`, i64 | on the dtype | validity bitmap |
| Julia | `Date`: Int64 Rata Die, epoch 0000-12-31 | `DateTime` in ms | type only; no zone | none built in |
| QuantLib | `Date`: `int_fast32_t` serial, Excel-compatible | optional `QL_HIGH_RESOLUTION_DATE` | | null `Date()` |
| kdb+/q | `d`: i32 days since 2000.01.01 | `p`: i64 ns since 2000.01.01 | type letter | typed nulls such as `0Nd`, plus infinities such as `0Wd` |

Sources: [np-dt], [arrow-schema], [pl-date], [pl-datetime], [jl-dates], [kdb-types].
kdb+ also has a float `datetime` (`z`), "deprecated in favour of the timestamp
datatype", and a month ordinal type, as NumPy has `datetime64[M]`.

**Arrow semantics** [arrow-schema]. A timestamp with a zone counts from the UTC epoch,
so changing the zone only changes metadata. Without one, the value is "wall clock time"
in an unknown zone; it "cannot be reliably compared or ordered", and "it is *not*
possible to interpret an unset or empty timezone as the same as 'UTC'". A zone string is
a tz database name or a fixed offset `+XX:XX`. The three calendar interval kinds are
`YEAR_MONTH`, `DAY_TIME`, and `MONTH_DAY_NANO`; in the last "each field is
independent", so months, days, and nanoseconds never normalize into each other.
`duration` is "an absolute length of time unrelated to any calendar artifacts".

**NaT, promotion, and overflow footguns** (all verified on NumPy 2.4.6):
- date - date gives a timedelta, and units are promoted implicitly:
  `datetime64('2020-01-01') + timedelta64(1,'h')` gives `2020-01-01T01`.
- Overflow wraps silently: `datetime64[D]('3000-01-01').astype('M8[ns]')` gives
  `1830-11-23T00:50:52.580896768`; `datetime64('2020-01-01T00:00:00.000000001')` minus
  `datetime64('1700-01-01')` gives `-8348571273709551615 ns`, because both operands are
  promoted to ns and then wrap.
- NaT is INT64_MIN, so it leaks into any integer view.
- Every ordered comparison with NaT is False, yet `np.sort` puts NaT last, so sort order
  disagrees with `<`.
- The business-day functions handle NaT inconsistently: `is_busday(NaT)` silently
  returns False; `busday_count` raises; `busday_offset` raises under `roll='raise'` but
  passes NaT through under `roll='forward'`.

**Resolution chosen by the data.** pandas' nanosecond range is 1677-09-21 to 2262-04-11;
pandas 2 added s, ms, and us units, and pandas 3.0 *infers* the unit from the input (us
for parsed strings), warning that "converting to integers ... will give integers 1000x
smaller" [pd-3.0]. The data chooses the type, silently changing `astype("int64")`.

## 11. Business-day machinery

**NumPy** [np-busoff; np-buscount] has `busday_offset(dates, offsets, roll='raise',
weekmask='1111100', holidays=None, busdaycal=None)`, with rolls `raise`, `nat`,
`forward`/`following`, `backward`/`preceding`, and
`modifiedfollowing`/`modifiedpreceding` ("...unless it is across a Month boundary"). It
rolls first, then offsets: Saturday 2011-06-25 + 2 is Wednesday 06-29 under `forward`
and Tuesday 06-28 under `backward`, and `offsets=0, roll='forward'` gives the first
business day on or after. `busday_count` counts the half-open range [begin, end) and
goes negative for reversed dates. `busdaycalendar` sorts holidays and drops duplicates,
NaT, and holidays on weekend days (verified on NumPy 2.4.6). A weekmask can be written
`"1111100"`, `[1,1,1,1,1,0,0]`, or `"Mon Tue ..."`.

**"Saturday + 1 business day" has no agreed answer.**

| Library | Rule | Answer |
|---|---|---|
| QuantLib `Calendar::advance(d, n, Days)` | steps `++d1` while the day is a holiday; with n = 0 it calls `adjust(d, c)`; when n ≠ 0 and the unit is Days it ignores the convention argument [ql-cal] | Monday |
| pandas `CustomBusinessDay` | uses `roll="backward"` if n > 0 and `"forward"` if n ≤ 0, passed to `np.busday_offset`; plain `BusinessDay` has its own weekday arithmetic (`_adjust_ndays`) [pd-offsets] | Monday |
| NumPy `busday_offset` | raises by default | Tuesday under `forward`, Monday under `backward` |
| BusinessDays.jl `advancebdays` | "Computation starts by next Business Day if `dt` is not a Business Day", an implicit forward roll [bdays-jl] | Tuesday |
| Polars `add_business_days` | `roll` in {raise, forward, backward}, default `raise`; marked "unstable" [pl-addbd] | raises by default |

**Counting conventions.** QuantLib `businessDaysBetween(from, to, includeFirst=true,
includeLast=false)`. Strata throws "if the end is before the start". BusinessDays.jl
`bdayscount` rolls both ends forward, then "the first Business Day is excluded"; for
d0 ≤ d1 this agrees with [b, e), but the convention is described differently.

**Roll conventions** as QuantLib documents them [ql-bdc]: Following, Preceding,
ModifiedFollowing, ModifiedPreceding, Unadjusted; `HalfMonthModifiedFollowing`
("...unless that day crosses the mid-month (15th) or the end of month"); and `Nearest`
("If both ... equally far away, default to following"). Strata's `NEAREST` is a
different rule: it "adjusts Sunday and Monday forward, and other days backward ...
despite the name, the algorithm may not return the business day that is actually
nearest" [st-bdc]. The same name denotes two rules.

**Joint calendars and their ambiguous names.** QuantLib `JoinHolidays` makes a day a
holiday if it is one in *any* calendar (business days are the intersection);
`JoinBusinessDays` makes it a business day if it is one in any calendar (the union)
[ql-joint]. Strata's `combinedWith` (`"GBLO+USNY"`) means a business day in both, and
`linkedWith` (`"GBLO~USNY"`) in either [st-hcal]. The verbs "join", "combine", and
"link" do not say which of the two a caller gets.

## 12. Day count conventions

The 30/360 family applies DCF = [360(Y2 - Y1) + 30(M2 - M1) + (D2 - D1)] / 360 after
each convention's day adjustments [isda-30360; st-dc; ql-30360].

| Convention | Exact definition | Inputs beyond (d1, d2) |
|---|---|---|
| ACT/360 | (d2 - d1) / 360 | none |
| ACT/365F | (d2 - d1) / 365 | none |
| ACT/ACT ISDA (4.16b) | days in the non-leap part / 365 + days in the leap part / 366. QuantLib: `(y2-y1-1) + days(d1, Jan1(y1+1))/dib1 + days(Jan1(y2), d2)/dib2` | none |
| ACT/ACT ICMA (4.16c, Rule 251) | days / (F × days in the coupon period); a stub is split into notional regular periods generated with an EOM flag, and the parts are summed | reference period or schedule, frequency F, EOM flag, stub direction |
| ACT/ACT AFB | step whole years back from d2 (with a Feb-29 roll rule), then add the remainder / 366 if the period contains 29 Feb, else / 365 | none, but the definition is disputed (below) |
| 30/360 Bond Basis (4.16f) | D1 = 31 becomes 30; if D2 = 31 and D1 is 30 or 31, D2 becomes 30 | none |
| 30E/360 Eurobond (4.16g) | D1 = 31 becomes 30; D2 = 31 becomes 30 | none |
| 30E/360 ISDA, German (4.16h) | as 30E/360, plus the last day of February becomes 30, except D2 when D2 is the maturity date | maturity date |
| 30/360 US, SIA (30U/360) | if EOM and both dates are the last day of February, D2 = 30; if EOM and D1 is the last day of February, D1 = 30; then the Bond Basis rules. Rule order matters | EOM flag (Strata defaults it to true) |
| BUS/252 | business days in [d1, d2) / 252 | a calendar (the Brazilian one); Strata `DayCount.ofBus252(calendarId)` |
| ACT/365L | (definition not recorded) | frequency and period end |

**Every one is an exact rational.** The denominators are 360, 365, 252, F × L, or
lcm(365, 366) = 133,590 for ISDA and AFB (365 and 366 are coprime). Every surveyed
library converts to floating point at once: QuantLib's `Time` is a `Real` (double),
Strata's `yearFraction` returns a double, and DayCounts.jl computes
`Dates.value(enddate-startdate)/365` as a Float64 and *defines* `yearfrac(a,b) =
-yearfrac(b,a)`, a modelling choice that is not part of the conventions [dc-jl].

**Naming collisions.** Strata's `THIRTY_360_ISDA` is Bond Basis (4.16f); QuantLib's
`Thirty360::ISDA` is 30E/360 ISDA (4.16h). Strata says ACT/365F is "also known as
'Act/365'", while QuantLib says ISDA uses "Actual/365, Act/365, A/365" as aliases for
ACT/ACT ISDA [ql-aa-hpp]. AFB is disputed: Strata knowingly diverges from ISDA's 1999
"clarification" (2004-02-28 to 2008-02-28 is 4 + 1/366 under the ISDA rule and 4 under
Strata's), noting that the ISDA rule leaves "one day receiving two days interest and the
next receiving no interest".

**ICMA needs a reference period.** Without a schedule, QuantLib's ICMA uses
`Old_ISMA_Impl`, which defaults the reference period to (d1, d2) and *infers* the
frequency as `lround(12*days/365)` months, falling back to `d1 + 1*Years` when that is 0
[ql-aa-cpp]. Strata's `yearFraction(d1, d2)` "will throw an exception because schedule
information is required" for ICMA, ACT/365L, and 30E/360 ISDA.

## 13. Periods, tenors, schedules, stubs, and the end-of-month rule

- **1M is not 30D.** Iterated clamped month addition drifts (section 6); QuantLib
  `Schedule` avoids it by advancing from a fixed seed by `periods*tenor` [ql-sched].
- **Tenors.** Strata's `Tenor` is "any non-negative non-zero period"; months and years
  are "not normalized", so 12M and 1Y are unequal as values but add identically
  [st-tenor].
- **End-of-month rule.** QuantLib's `advance(..., endOfMonth=true)` moves "to the last
  calendar day if d is the last calendar day" under Unadjusted, otherwise "to the last
  *business* day if d is the last business day", and silently turns EOM off
  (`allowsEndOfMonth`, `endOfMonth_ = false`) for tenors below 1M or not in months or
  years. Strata uses a `RollConvention` (EOM, IMM, DAY_1 through DAY_30, and others).
- **ON, TN, SN, and spot lag** are not periods. By market convention (not a library
  quote) each is a business-day start lag plus a business-day length: ON is T to T+1bd,
  TN is T+1 to T+2, SN is spot to spot+1bd. Strata's `DaysAdjustment` makes the steps
  explicit, "first add 2 London business days, and then adjust the result to be a valid
  New York business day using 'ModifiedFollowing'", with different counting and
  adjusting calendars [st-daysadj].
- **Stubs.** QuantLib `DateGeneration`: Backward, Forward, Zero,
  ThirdWednesday(Inclusive), Twentieth, TwentiethIMM, OldCDS, CDS, CDS2015 [ql-dgr].
  Strata `StubConvention`: NONE, SHORT/LONG/SMART_INITIAL (generated backwards from the
  end date), SHORT/LONG/SMART_FINAL (forwards), and BOTH. SMART uses a hidden threshold:
  a stub "of less than 7 days ... will be combined with the next period", measured on
  unadjusted dates. Explicit `firstRegularStartDate` or `lastRegularEndDate` that
  contradict the convention make Strata throw [st-stub].

## 14. Holiday calendar data: rule or table, provenance, and horizon

| Library | Rule-based or table | Data horizon | Outside the horizon |
|---|---|---|---|
| NumPy | table (array) | none | treated as weekmask-only, silently |
| pandas `AbstractHolidayCalendar` | rules with observance (`nearest_workday`, `next_monday`, ...) | 1970-01-01 to 2200-12-31 by default | rules project known holidays forward; one-off closures are unknowable |
| QuantLib | hand-coded rules per calendar, with year-specific exceptions | the `Date` range, 1901-01-01 to 2199-12-31 (serials 367 to 109574) | error outside the `Date` range; inside it the rules extrapolate silently |
| Strata | table (a bitmask per month), generated for years 1950 to 2099 | from "start of the year of the earliest holiday" to the end of the latest | "Beyond the range of known holiday dates, weekend days are used", silently; queries are allowed from year 0 to 10,000 |
| BusinessDays.jl | rules; optional global cache, default 1980-01-01 to 2150-12-20 | the cache bounds | `@assert` "Date out of cache bounds" when cached, computed when not: the same call behaves differently depending on hidden state |

On provenance, Strata says its calendar data was "obtained by direct research" and "was
not derived from a vendor ... may or may not be sufficient for your production needs"
[st-calids]. On mutation, QuantLib `TARGET` holds a `static shared_ptr<Calendar::Impl>`
and `addHoliday` writes into `impl_->addedHolidays`, so `addHoliday` on one `TARGET`
object changes every `TARGET` object in the process [ql-target; ql-calhpp].

## 15. Documented regrets and recurring bug classes

- **Naive values read as local time.** Python deprecated `utcnow()` and
  `utcfromtimestamp()` in 3.12 because "naive datetime objects are treated by many
  datetime methods as local times". Naive and aware values "are never equal", ordering
  them raises `TypeError`, and subtracting two aware values with the same `tzinfo`
  ignores the zone and compares wall times [py-dt]. Inter-zone comparisons involving
  fold-dependent times return False [PEP495].
- **Mutable values or shared mutable state.** Moment and `java.util.Date` (section 7;
  the latter also has 0-based months and years offset from 1900, not re-verified);
  QuantLib `addHoliday` (section 14).
- **Ambient "today" or host-dependent results.** QuantLib `evaluationDate` (section 9);
  pandas 3.0 zone results follow system tzdata, not program text alone; Go embeds zone
  data only as a fallback; tzdb updates break stored future zoned values (section 5).
- **Silent normalization, guessing, and defaults later withdrawn.** Go (sections 5, 6);
  QuantLib's ICMA inference (section 12) and EOM disablement (section 13). QuantLib's
  history [ql-hist]: 1.23 "Fixed implementation of U.S. 30/360 convention (the old one is
  still available as 30/360 NASD)", made Bond Basis an alias of 30/360 ISMA, "Deprecated
  default constructor for actual/actual and 30/360", and let ISDA take the termination
  date; 1.18 "A bug in the 30/360 German day counter was fixed"; 1.15 "Fix
  implementation of Actual/Actual (ISMA) ... when a schedule is provided". The
  conventions with defaults were wrong for years, and the defaults were removed.
- **Silent overflow, sentinels, and data-chosen types.** NumPy (section 10); pandas'
  1677 to 2262 range led it to add resolutions [pandas], then 3.0 made the unit depend
  on the input [pd-3.0]; Arrow `date64` (section 7).
- **Ambiguous names.** pandas 2.2 deprecated `M`, `Q`, and `Y` for `ME`, `QE`, and `YE`
  because "M" meant MonthEnd for offsets but "month" for Period [pd-2.2]; day count
  names and AFB (section 12); `Nearest` and joint-calendar verbs (section 11); `YYYY`
  against `yyyy` pattern letters (not re-verified).
- **Results that hinge on order, an anchor, or a hidden threshold.** Julia's addition
  and pandas anchored offsets (section 6); Strata's SMART stub threshold (section 13).
- **API shape and equality.** The author of jiff calls chrono's API "overengineered",
  with mixed `Option` and `Result` error handling and incomplete DST-safe arithmetic
  [jiff-design]; chrono deprecated its panicking constructors and admits leap-second
  arithmetic breaks associativity [chrono-nt]. Go's `==` trap (section 9).

## 16. Day-count algorithms

These convert between a civil date and a day number (not the conventions of section 12).
Hinnant's [H] use 400-year eras and, internally, a year starting in March; they are
proleptic Gregorian, including year 0 and negative years. The valid range is
"[civil_from_days(min), civil_from_days(max-719468)]": about ±5.8 million years with 32
bits, and with 64 bits overflow is "far beyond +/- the age of the universe". The stated
intent is "to make range checking superfluous". jiff uses them, calling them "much more
straight-forward" than Rata Die [jiff-src], as do libc++ and MSVC [H]. Neri and
Schneider's "Euclidean affine functions" [NS] are used in libstdc++ since GCC 11, in
Linux since 5.14, and in Go's `time` [Go src].

## 17. References

- [RFC3339] https://www.rfc-editor.org/rfc/rfc3339.html
- [RFC9557] https://www.rfc-editor.org/rfc/rfc9557.html
- [402] https://tc39.es/ecma402/#sec-use-of-iana-time-zone-database
- [Stage4] https://www.igalia.com/2026/03/13/Temporal-Reaches-Stage-4.html
- [T-spec] https://github.com/tc39/proposal-temporal/tree/main/spec
- [T-pd] https://tc39.es/proposal-temporal/docs/plaindate.html
- [T-tz] https://tc39.es/proposal-temporal/docs/timezone.html
- [T-inst] https://tc39.es/proposal-temporal/docs/instant.html
- [T-dur] https://tc39.es/proposal-temporal/docs/duration.html
- [T-strings] https://tc39.es/proposal-temporal/docs/strings.html
- [isda-30360] https://www.isda.org/2008/12/22/30-360-day-count-conventions/
- [J-Inst] https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/time/Instant.html
- [J-ZDT] https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/time/ZonedDateTime.html
- [J-Per] https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/time/Period.html
- [J-Clock] https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/time/Clock.html
- [jiff-design] https://github.com/BurntSushi/jiff/blob/master/DESIGN.md
- [jiff-ts] https://docs.rs/jiff/latest/jiff/struct.Timestamp.html
- [jiff-dis] https://docs.rs/jiff/latest/jiff/tz/enum.Disambiguation.html
- [jiff-src] https://github.com/BurntSushi/jiff at 0.2.31: `src/util/b.rs`, `src/civil/date.rs`, `src/tz/db/mod.rs`
- [jiff-tzdb] https://docs.rs/jiff-tzdb
- [chrono-nd] https://docs.rs/chrono/latest/chrono/naive/struct.NaiveDate.html
- [chrono-nt] https://docs.rs/chrono/latest/chrono/naive/struct.NaiveTime.html
- [chrono-src] chrono 0.4.45 source: `src/naive/date/mod.rs`, `src/time_delta.rs`
- [chrono-tz] https://docs.rs/chrono-tz
- [time-date] https://docs.rs/time/latest/time/struct.Date.html
- [time src] `time` 0.3.53 source: `src/parsing/parsable.rs`
- [Go] https://pkg.go.dev/time
- [Go src] https://github.com/golang/go/blob/master/src/time/time.go
- [cpp-ymd] https://en.cppreference.com/w/cpp/chrono/year_month_day/operator_arith
- [cpp-tosys] https://en.cppreference.com/w/cpp/chrono/time_zone/to_sys
- [cpp-tzdb] https://en.cppreference.com/w/cpp/chrono/tzdb_functions
- [py-dt] https://docs.python.org/3/library/datetime.html
- [py-zi] https://docs.python.org/3/library/zoneinfo.html
- [PEP495] https://peps.python.org/pep-0495/
- [hs] https://hackage.haskell.org/package/time/docs/Data-Time-Calendar.html
- [kt] https://github.com/Kotlin/kotlinx-datetime
- [kt-toInstant] https://kotlinlang.org/api/kotlinx-datetime/kotlinx-datetime/kotlinx.datetime/to-instant.html
- [moment] https://momentjs.com/docs/
- [pg] https://www.postgresql.org/docs/current/datatype-datetime.html
- [np-dt] https://numpy.org/doc/stable/reference/arrays.datetime.html
- [np-busoff] https://numpy.org/doc/stable/reference/generated/numpy.busday_offset.html
- [np-buscount] https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html
- [pandas] https://pandas.pydata.org/docs/user_guide/timeseries.html
- [pd-2.2] https://pandas.pydata.org/docs/whatsnew/v2.2.0.html
- [pd-3.0] https://pandas.pydata.org/docs/whatsnew/v3.0.0.html
- [pd-offsets] https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/tslibs/offsets.pyx
- [arrow-schema] https://github.com/apache/arrow/blob/main/format/Schema.fbs
- [pl-date] https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Date.html
- [pl-datetime] https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Datetime.html
- [pl-offset] https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.offset_by.html
- [pl-addbd] https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.add_business_days.html
- [jl-dates] https://docs.julialang.org/en/v1/stdlib/Dates/
- [kdb-types] https://code.kx.com/q/basics/datatypes/
- [ql-cal] https://github.com/lballabio/QuantLib/blob/master/ql/time/calendar.cpp
- [ql-calhpp] https://github.com/lballabio/QuantLib/blob/master/ql/time/calendar.hpp
- [ql-bdc] https://github.com/lballabio/QuantLib/blob/master/ql/time/businessdayconvention.hpp
- [ql-30360] https://github.com/lballabio/QuantLib/blob/master/ql/time/daycounters/thirty360.hpp
- [ql-aa-hpp] https://github.com/lballabio/QuantLib/blob/master/ql/time/daycounters/actualactual.hpp
- [ql-aa-cpp] https://github.com/lballabio/QuantLib/blob/master/ql/time/daycounters/actualactual.cpp
- [ql-sched] https://github.com/lballabio/QuantLib/blob/master/ql/time/schedule.cpp
- [ql-dgr] https://github.com/lballabio/QuantLib/blob/master/ql/time/dategenerationrule.hpp
- [ql-joint] https://github.com/lballabio/QuantLib/blob/master/ql/time/calendars/jointcalendar.hpp
- [ql-target] https://github.com/lballabio/QuantLib/blob/master/ql/time/calendars/target.cpp
- [ql-settings] https://github.com/lballabio/QuantLib/blob/master/ql/settings.hpp
- [ql-hist] https://rkapl123.github.io/QLAnnotatedSource/dc/dd1/history.html
- [st-bdc] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/BusinessDayConventions.java
- [st-dc] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/DayCounts.java
- [st-tenor] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/Tenor.java
- [st-daysadj] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/DaysAdjustment.java
- [st-stub] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/schedule/StubConvention.java
- [st-calids] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/HolidayCalendarIds.java
- [st-hcal] https://github.com/OpenGamma/Strata/blob/main/modules/basics/src/main/java/com/opengamma/strata/basics/date/HolidayCalendar.java
- [bdays-jl] https://github.com/JuliaFinance/BusinessDays.jl/blob/master/src/bdays.jl
- [dc-jl] https://github.com/JuliaFinance/DayCounts.jl/blob/master/src/DayCounts.jl
- [H] https://howardhinnant.github.io/date_algorithms.html
- [NS] https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3172
