//! chelis#2859: the `Std.Datetime` failure contract of [05-OP-73].
//!
//! Every failure is a `fail` whose message is exactly
//! `<function>: <kind>: <detail>`, and every range and validity check runs
//! before the arithmetic it protects, so no primitive numeric trap escapes a
//! call made with extreme i64 arguments. Each suite runs as one `chelis test`
//! invocation over a generated fixture whose every test calls one failing (or
//! extreme) expression; the runner reports each call's failure message on its
//! FAIL line. Opaque construction outside the module and the unannotated
//! accessor-lambda interaction with the first standard-library opaque type are
//! checked through `chelis check`.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::collections::BTreeMap;
use std::path::Path;

const IMPORTS: &str = "import Std.Datetime (Weekday, Monday, Sunday, ClampToMonthEnd, RejectInvalidDay, Hours, Minutes, Seconds, Milliseconds, Microseconds, Nanoseconds, Date, Duration, Period, is_leap_year, days_in_year, days_in_month, weekday_from_iso_number, date, date_from_epoch_day, date_iso_week, date_day_of_year, date_from_iso_week, date_add_days, date_add_months, try_date_add_months, date_add_period, try_date_add_period, date_period_until, parse_date, nth_weekday_in_month, last_weekday_in_month, weekday_on_or_after, weekday_on_or_before, easter_sunday_gregorian, easter_sunday_orthodox, time, time_from_nanosecond_of_day, time_add_duration, datetime, datetime_add_duration, datetime_add_period, try_datetime_add_period, datetime_until, parse_time, parse_datetime, offset_from_seconds, parse_offset, instant_from_unix, instant_from_unix_count, instant_to_unix_count, instant_add_duration, instant_until, instant_round_to, instant_to_datetime_at, datetime_to_instant_at, parse_instant, parse_offset_datetime, duration, duration_from_count, duration_to_count, duration_to_seconds_f64, duration_add, duration_sub, duration_negate, duration_mul, parse_duration, try_parse_duration, period, period_negate, period_mul, parse_period, try_parse_period, dates_from_epoch_days, instants_from_unix)\nimport Std.Rounding (RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)";

/// (test name, expression, exact failure message). The `order_*` rows pin
/// [05-OP-73]'s check order for every callable that could fail both ways.
const EXACT_FAILURES: &[(&str, &str, &str)] = &[
    (
        "days_in_month_month",
        "days_in_month(2024i64, 13i64)",
        "days_in_month: domain: month 13 is outside 1..12",
    ),
    (
        "weekday_number",
        "weekday_from_iso_number(8i64)",
        "weekday_from_iso_number: domain: weekday number 8 is outside 1..7",
    ),
    (
        "date_day",
        "date(2024i64, 2i64, 30i64)",
        "date: domain: day 30 is outside 1..29 for 2024-02",
    ),
    (
        "date_year",
        "date(10000i64, 1i64, 1i64)",
        "date: domain: year 10000 is outside -9999..9999",
    ),
    (
        "date_month",
        "date(2024i64, 0i64, 1i64)",
        "date: domain: month 0 is outside 1..12",
    ),
    (
        "date_from_epoch_day_range",
        "date_from_epoch_day(2932897i64)",
        "date_from_epoch_day: domain: epoch day 2932897 is outside -4371587..2932896",
    ),
    (
        "date_from_iso_week_missing",
        "date_from_iso_week(2025i64, 53i64, Monday)",
        "date_from_iso_week: domain: week 53 is outside 1..52 for ISO week-year 2025",
    ),
    (
        "date_from_iso_week_range",
        "date_from_iso_week(10000i64, 2i64, Monday)",
        "date_from_iso_week: domain: ISO week-year 10000 is outside -9999..9999",
    ),
    (
        "date_from_iso_week_end",
        "date_from_iso_week(9999i64, 52i64, Sunday)",
        "date_from_iso_week: domain: ISO week date 9999-W52-7 is outside the supported date range",
    ),
    (
        "date_add_days_end",
        "date_add_days(date(9999i64, 12i64, 31i64), 1i64)",
        "date_add_days: overflow: 9999-12-31 plus 1 days is outside the supported date range",
    ),
    (
        "date_add_days_max",
        "date_add_days(date(1970i64, 1i64, 1i64), 9223372036854775807i64)",
        "date_add_days: overflow: 1970-01-01 plus 9223372036854775807 days is outside the supported date range",
    ),
    (
        "date_add_days_min",
        "date_add_days(date(1970i64, 1i64, 1i64), i64_minimum())",
        "date_add_days: overflow: 1970-01-01 plus -9223372036854775808 days is outside the supported date range",
    ),
    (
        "date_add_months_reject",
        "date_add_months(date(2024i64, 1i64, 31i64), 1i64, RejectInvalidDay)",
        "date_add_months: domain: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "date_add_months_end",
        "date_add_months(date(9999i64, 12i64, 1i64), 1i64, ClampToMonthEnd)",
        "date_add_months: overflow: 9999-12-01 plus 1 months is outside the supported date range",
    ),
    (
        "date_add_months_min",
        "date_add_months(date(2024i64, 1i64, 1i64), i64_minimum(), ClampToMonthEnd)",
        "date_add_months: overflow: 2024-01-01 plus -9223372036854775808 months is outside the supported date range",
    ),
    (
        "try_date_add_months_end",
        "try_date_add_months(date(9999i64, 12i64, 1i64), 1i64, RejectInvalidDay)",
        "try_date_add_months: overflow: 9999-12-01 plus 1 months is outside the supported date range",
    ),
    (
        "date_add_period_reject",
        "date_add_period(date(2024i64, 1i64, 31i64), period(1i64, 0i64), RejectInvalidDay)",
        "date_add_period: domain: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "date_add_period_days",
        "date_add_period(date(9999i64, 12i64, 31i64), period(0i64, 1i64), ClampToMonthEnd)",
        "date_add_period: overflow: 9999-12-31 plus 1 days is outside the supported date range",
    ),
    (
        "try_date_add_period_months",
        "try_date_add_period(date(2024i64, 1i64, 31i64), period(9223372036854775807i64, 0i64), RejectInvalidDay)",
        "try_date_add_period: overflow: 2024-01-31 plus 9223372036854775807 months is outside the supported date range",
    ),
    (
        "parse_date_field",
        "parse_date(\"2024-02-30\")",
        "parse_date: domain: day 30 is outside 1..29 for 2024-02",
    ),
    (
        "parse_date_negative_zero",
        "parse_date(\"-000000-01-01\")",
        "parse_date: domain: \"-000000-01-01\" is not a date in the text profile",
    ),
    (
        "nth_weekday_zero",
        "nth_weekday_in_month(2026i64, 2i64, Monday, 0i64)",
        "nth_weekday_in_month: domain: occurrence 0 is not positive",
    ),
    (
        "nth_weekday_month",
        "nth_weekday_in_month(2026i64, 13i64, Monday, 1i64)",
        "nth_weekday_in_month: domain: month 13 is outside 1..12",
    ),
    (
        "nth_weekday_year",
        "nth_weekday_in_month(10000i64, 1i64, Monday, 1i64)",
        "nth_weekday_in_month: domain: year 10000 is outside -9999..9999",
    ),
    (
        "last_weekday_month",
        "last_weekday_in_month(2026i64, 0i64, Monday)",
        "last_weekday_in_month: domain: month 0 is outside 1..12",
    ),
    (
        "weekday_on_or_after_end",
        "weekday_on_or_after(date(9999i64, 12i64, 31i64), Monday)",
        "weekday_on_or_after: overflow: 9999-12-31 plus 3 days is outside the supported date range",
    ),
    (
        "weekday_on_or_before_start",
        "weekday_on_or_before(date(-9999i64, 1i64, 1i64), Sunday)",
        "weekday_on_or_before: overflow: -009999-01-01 plus -1 days is outside the supported date range",
    ),
    (
        "easter_gregorian_year",
        "easter_sunday_gregorian(10000i64)",
        "easter_sunday_gregorian: domain: year 10000 is outside -9999..9999",
    ),
    (
        "easter_orthodox_year",
        "easter_sunday_orthodox(-10000i64)",
        "easter_sunday_orthodox: domain: year -10000 is outside -9999..9999",
    ),
    (
        "time_hour",
        "time(24i64, 0i64, 0i64, 0i64)",
        "time: domain: hour 24 is outside 0..23",
    ),
    (
        "time_minute",
        "time(0i64, 60i64, 0i64, 0i64)",
        "time: domain: minute 60 is outside 0..59",
    ),
    (
        "time_second",
        "time(0i64, 0i64, 60i64, 0i64)",
        "time: domain: second 60 is outside 0..59",
    ),
    (
        "time_nanosecond",
        "time(0i64, 0i64, 0i64, 1000000000i64)",
        "time: domain: nanosecond 1000000000 is outside 0..999999999",
    ),
    (
        "time_from_nanosecond_of_day_range",
        "time_from_nanosecond_of_day(86400000000000i64)",
        "time_from_nanosecond_of_day: domain: nanosecond of day 86400000000000 is outside 0..86399999999999",
    ),
    (
        "datetime_add_duration_end",
        "datetime_add_duration(datetime(date(9999i64, 12i64, 31i64), time(23i64, 59i64, 59i64, 0i64)), duration(1i64, 0i64))",
        "datetime_add_duration: overflow: 9999-12-31T23:59:59 plus PT1S is outside the supported range",
    ),
    (
        "datetime_add_duration_min",
        "datetime_add_duration(datetime(date(1970i64, 1i64, 1i64), time(0i64, 0i64, 0i64, 0i64)), duration(i64_minimum(), 0i64))",
        "datetime_add_duration: overflow: 1970-01-01T00:00:00 plus -PT9223372036854775808S is outside the supported range",
    ),
    (
        "datetime_add_period_reject",
        "datetime_add_period(datetime(date(2024i64, 1i64, 31i64), time(0i64, 0i64, 0i64, 0i64)), period(1i64, 0i64), RejectInvalidDay)",
        "datetime_add_period: domain: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "parse_time_second",
        "parse_time(\"23:59:60\")",
        "parse_time: domain: second 60 is outside 0..59",
    ),
    (
        "parse_time_fraction",
        "parse_time(\"09:30:00.1234567890\")",
        "parse_time: domain: \"09:30:00.1234567890\" is not a time in the text profile",
    ),
    (
        "parse_datetime_short",
        "parse_datetime(\"2026-10-01T09:30\")",
        "parse_datetime: domain: \"2026-10-01T09:30\" is not a datetime in the text profile",
    ),
    (
        "offset_range",
        "offset_from_seconds(86400i64)",
        "offset_from_seconds: domain: offset 86400 s is outside -86399..86399",
    ),
    (
        "parse_offset_hour",
        "parse_offset(\"+24:00\")",
        "parse_offset: domain: offset hour 24 is outside 0..23",
    ),
    (
        "parse_offset_basic",
        "parse_offset(\"+0530\")",
        "parse_offset: domain: \"+0530\" is not an offset in the text profile",
    ),
    (
        "instant_nanosecond",
        "instant_from_unix(0i64, 1000000000i64)",
        "instant_from_unix: domain: nanosecond 1000000000 is outside 0..999999999",
    ),
    (
        "instant_second",
        "instant_from_unix(253402214401i64, 0i64)",
        "instant_from_unix: domain: unix second 253402214401 is outside -377705030401..253402214400",
    ),
    (
        "instant_count_seconds",
        "instant_from_unix_count(9223372036854775807i64, Seconds)",
        "instant_from_unix_count: domain: 9223372036854775807 seconds since the unix epoch is outside the supported instant range",
    ),
    (
        "instant_count_hours",
        "instant_from_unix_count(9223372036854775807i64, Hours)",
        "instant_from_unix_count: domain: 9223372036854775807 hours since the unix epoch is outside the supported instant range",
    ),
    (
        "instant_to_count_nanos",
        "instant_to_unix_count(instant_from_unix(253402214400i64, 0i64), Nanoseconds, RoundTowardNegative)",
        "instant_to_unix_count: overflow: 9999-12-31T00:00:00Z in nanoseconds does not fit in i64",
    ),
    (
        "instant_to_count_inexact",
        "instant_to_unix_count(instant_from_unix(-1i64, 500000000i64), Seconds, RejectInexact)",
        "instant_to_unix_count: domain: 1969-12-31T23:59:59.5Z is not a whole number of seconds",
    ),
    (
        "instant_add_duration_end",
        "instant_add_duration(instant_from_unix(253402214400i64, 0i64), duration(1i64, 0i64))",
        "instant_add_duration: overflow: 9999-12-31T00:00:00Z plus PT1S is outside the supported instant range",
    ),
    (
        "instant_add_duration_max",
        "instant_add_duration(instant_from_unix(0i64, 1i64), duration(9223372036854775807i64, 999999999i64))",
        "instant_add_duration: overflow: 1970-01-01T00:00:00.000000001Z plus PT9223372036854775807.999999999S is outside the supported instant range",
    ),
    (
        "instant_round_seven",
        "instant_round_to(instant_from_unix(0i64, 0i64), duration(7i64, 0i64), RoundTowardNegative)",
        "instant_round_to: domain: increment PT7S does not divide one day",
    ),
    (
        "instant_round_zero",
        "instant_round_to(instant_from_unix(0i64, 0i64), duration(0i64, 0i64), RoundTowardNegative)",
        "instant_round_to: domain: increment PT0S is not positive",
    ),
    (
        "instant_round_negative",
        "instant_round_to(instant_from_unix(0i64, 0i64), duration(-60i64, 0i64), RoundTowardNegative)",
        "instant_round_to: domain: increment -PT60S is not positive",
    ),
    (
        "instant_round_huge",
        "instant_round_to(instant_from_unix(0i64, 0i64), duration(9223372036854775807i64, 0i64), RoundTowardNegative)",
        "instant_round_to: domain: increment PT9223372036854775807S does not divide one day",
    ),
    (
        "instant_round_end",
        "instant_round_to(instant_from_unix(253402214400i64, 1i64), duration(86400i64, 0i64), RoundTowardPositive)",
        "instant_round_to: overflow: rounding 9999-12-31T00:00:00.000000001Z to a multiple of PT86400S leaves the supported instant range",
    ),
    (
        "instant_round_inexact",
        "instant_round_to(instant_from_unix(90i64, 0i64), duration(60i64, 0i64), RejectInexact)",
        "instant_round_to: domain: 1970-01-01T00:01:30Z is not a multiple of PT60S",
    ),
    (
        "datetime_to_instant_end",
        "datetime_to_instant_at(datetime(date(9999i64, 12i64, 31i64), time(23i64, 59i64, 59i64, 0i64)), offset_from_seconds(-1i64))",
        "datetime_to_instant_at: overflow: 9999-12-31T23:59:59 at offset -00:00:01 is outside the supported instant range",
    ),
    (
        "parse_instant_range",
        "parse_instant(\"9999-12-31T23:59:59-00:01\")",
        "parse_instant: domain: unix second 253402300859 is outside -377705030401..253402214400",
    ),
    (
        "parse_instant_no_offset",
        "parse_instant(\"2026-10-01T09:30:00\")",
        "parse_instant: domain: \"2026-10-01T09:30:00\" is not an instant in the text profile",
    ),
    (
        "parse_offset_datetime_hour",
        "parse_offset_datetime(\"2026-10-01T09:30:00+24:00\")",
        "parse_offset_datetime: domain: offset hour 24 is outside 0..23",
    ),
    (
        "duration_max",
        "duration(9223372036854775807i64, 1000000000i64)",
        "duration: domain: second 9223372036854775807 with nanosecond 1000000000 is outside the duration range",
    ),
    (
        "duration_min",
        "duration(i64_minimum(), -1i64)",
        "duration: domain: second -9223372036854775808 with nanosecond -1 is outside the duration range",
    ),
    (
        "duration_from_count_hours",
        "duration_from_count(9223372036854775807i64, Hours)",
        "duration_from_count: domain: 9223372036854775807 hours is outside the duration range",
    ),
    (
        "duration_to_count_nanos",
        "duration_to_count(duration(9223372037i64, 0i64), Nanoseconds, RoundTowardNegative)",
        "duration_to_count: overflow: PT9223372037S in nanoseconds does not fit in i64",
    ),
    (
        "duration_to_count_ceiling",
        "duration_to_count(duration(9223372036854775807i64, 1i64), Seconds, RoundTowardPositive)",
        "duration_to_count: overflow: PT9223372036854775807.000000001S in seconds does not fit in i64",
    ),
    (
        "duration_to_count_inexact",
        "duration_to_count(duration(90i64, 0i64), Minutes, RejectInexact)",
        "duration_to_count: domain: PT90S is not a whole number of minutes",
    ),
    (
        "duration_add_max",
        "duration_add(duration(9223372036854775807i64, 500000000i64), duration(0i64, 500000000i64))",
        "duration_add: overflow: PT9223372036854775807.5S plus PT0.5S does not fit in the duration range",
    ),
    (
        "duration_sub_min",
        "duration_sub(duration(i64_minimum(), 0i64), duration(0i64, 1i64))",
        "duration_sub: overflow: -PT9223372036854775808S minus PT0.000000001S does not fit in the duration range",
    ),
    (
        "duration_negate_min",
        "duration_negate(duration(i64_minimum(), 0i64))",
        "duration_negate: overflow: the negation of -PT9223372036854775808S does not fit in the duration range",
    ),
    (
        "duration_mul_max",
        "duration_mul(duration(2i64, 0i64), 9223372036854775807i64)",
        "duration_mul: overflow: PT2S times 9223372036854775807 does not fit in the duration range",
    ),
    (
        "duration_mul_min",
        "duration_mul(duration(i64_minimum(), 0i64), -1i64)",
        "duration_mul: overflow: -PT9223372036854775808S times -1 does not fit in the duration range",
    ),
    (
        "parse_duration_empty",
        "parse_duration(\"PT\")",
        "parse_duration: domain: \"PT\" is not a duration in the text profile",
    ),
    (
        "parse_duration_seconds",
        "parse_duration(\"PT9223372036854775808S\")",
        "parse_duration: domain: \"PT9223372036854775808S\" is outside the duration range",
    ),
    (
        "parse_duration_hours",
        "parse_duration(\"PT2562047788015216H\")",
        "parse_duration: domain: \"PT2562047788015216H\" is outside the duration range",
    ),
    (
        "period_mixed",
        "period(1i64, -1i64)",
        "period: domain: months 1 and days -1 have mixed signs",
    ),
    (
        "period_negate_min",
        "period_negate(period(i64_minimum(), 0i64))",
        "period_negate: overflow: the negation of -P9223372036854775808M does not fit in i64",
    ),
    (
        "period_mul_max",
        "period_mul(period(2i64, 0i64), 9223372036854775807i64)",
        "period_mul: overflow: P2M times 9223372036854775807 does not fit in i64",
    ),
    (
        "parse_period_inner_sign",
        "parse_period(\"P1M-1D\")",
        "parse_period: domain: \"P1M-1D\" is not a period in the text profile",
    ),
    (
        "parse_period_years",
        "parse_period(\"P768614336404564651Y\")",
        "parse_period: domain: \"P768614336404564651Y\" is outside the period range",
    ),
    (
        "parse_period_days",
        "parse_period(\"P9223372036854775808D\")",
        "parse_period: domain: \"P9223372036854775808D\" is outside the period range",
    ),
    (
        "parse_duration_negative_past_minimum",
        "parse_duration(\"-PT9223372036854775808.5S\")",
        "parse_duration: domain: \"-PT9223372036854775808.5S\" is outside the duration range",
    ),
    (
        "dates_element",
        "dates_from_epoch_days(to_tensor([0i64, 2932897i64]))",
        "dates_from_epoch_days: domain: element 1: epoch day 2932897 is outside -4371587..2932896",
    ),
    (
        "instants_element",
        "instants_from_unix(to_tensor([0i64, 1i64]), to_tensor([0i64, 1000000000i64]))",
        "instants_from_unix: domain: element 1: nanosecond 1000000000 is outside 0..999999999",
    ),
    (
        "order_date_add_months_range_before_policy",
        "date_add_months(date(9999i64, 1i64, 31i64), 13i64, RejectInvalidDay)",
        "date_add_months: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_try_date_add_months_range_before_policy",
        "try_date_add_months(date(9999i64, 1i64, 31i64), 13i64, RejectInvalidDay)",
        "try_date_add_months: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_date_add_period_range_before_policy",
        "date_add_period(date(9999i64, 1i64, 31i64), period(13i64, 0i64), RejectInvalidDay)",
        "date_add_period: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_try_date_add_period_range_before_policy",
        "try_date_add_period(date(9999i64, 1i64, 31i64), period(13i64, 0i64), RejectInvalidDay)",
        "try_date_add_period: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_date_add_period_policy_before_day_range",
        "date_add_period(date(2024i64, 1i64, 31i64), period(1i64, 4000000i64), RejectInvalidDay)",
        "date_add_period: domain: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "order_datetime_add_period_range_before_policy",
        "datetime_add_period(datetime(date(9999i64, 1i64, 31i64), time(0i64, 0i64, 0i64, 0i64)), period(13i64, 0i64), RejectInvalidDay)",
        "datetime_add_period: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_try_datetime_add_period_range_before_policy",
        "try_datetime_add_period(datetime(date(9999i64, 1i64, 31i64), time(0i64, 0i64, 0i64, 0i64)), period(13i64, 0i64), RejectInvalidDay)",
        "try_datetime_add_period: overflow: 9999-01-31 plus 13 months is outside the supported date range",
    ),
    (
        "order_datetime_add_period_policy_before_day_range",
        "datetime_add_period(datetime(date(2024i64, 1i64, 31i64), time(0i64, 0i64, 0i64, 0i64)), period(1i64, 4000000i64), RejectInvalidDay)",
        "datetime_add_period: domain: day 31 is outside 1..29 for 2024-02",
    ),
    (
        "order_duration_to_count_policy_before_range",
        "duration_to_count(duration(9223372036854775807i64, 1i64), Milliseconds, RejectInexact)",
        "duration_to_count: domain: PT9223372036854775807.000000001S is not a whole number of milliseconds",
    ),
    (
        "order_instant_round_to_increment_before_range",
        "instant_round_to(instant_from_unix(253402214400i64, 1i64), duration(7i64, 0i64), RoundTowardPositive)",
        "instant_round_to: domain: increment PT7S does not divide one day",
    ),
];

const MAX: &str = "9223372036854775807i64";
const MIN: &str = "i64_minimum()";

fn run_chelis(app_pkg: &Path, reef_home: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(args)
        .output()
        .expect("chelis runs");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// Runs one `chelis test` over a fixture with one test per expression and
/// returns each test's outcome: `None` for PASS, `Some(message)` for FAIL.
fn run_expression_suite(
    dir_name: &str,
    expressions: &[(String, String)],
) -> BTreeMap<String, Option<String>> {
    let (_dir, reef_home, app_pkg) = make_app(dir_name);
    let mut source = format!(
        "module Demo.Tests.Datetime\n{IMPORTS}\ndef i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)\n"
    );
    for (name, expression) in expressions {
        source.push_str(&format!(
            "def test_{name}() -> unit ! {{ Test }} = {{\n  _value = {expression}\n  ()\n}}\n"
        ));
    }
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let path = app_pkg.join("tests/datetime_cases.ch");
    write_file(&path, &source);
    let (_, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &["test", "--batch-mode", "file", path.to_str().unwrap()],
    );
    let mut outcomes = BTreeMap::new();
    for line in rendered.lines() {
        let Some(rest) = line.trim_start().strip_prefix("test_") else {
            continue;
        };
        let Some((name, verdict)) = rest.split_once(' ') else {
            continue;
        };
        let verdict = verdict.trim_start_matches(['.', ' ']);
        if verdict == "PASS" {
            outcomes.insert(name.to_string(), None);
        } else if let Some(message) = verdict
            .strip_prefix("FAIL (")
            .and_then(|tail| tail.strip_suffix(')'))
        {
            outcomes.insert(name.to_string(), Some(message.to_string()));
        }
    }
    assert_eq!(
        outcomes.len(),
        expressions.len(),
        "every generated test must report exactly one verdict:\n{rendered}"
    );
    outcomes
}

#[test]
fn std_datetime_failures_report_their_exact_message() {
    let expressions: Vec<(String, String)> = EXACT_FAILURES
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("datetime-failures-2859", &expressions);
    for (name, expression, expected) in EXACT_FAILURES {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.to_string())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
}

/// Every call below passes, or fails `domain` or `overflow` under its own
/// callable's name; a primitive trap would surface under another message.
fn extreme_cases() -> Vec<(&'static str, String)> {
    let ints = [MAX, MIN];
    let date_low = "date(-9999i64, 1i64, 1i64)";
    let date_high = "date(9999i64, 12i64, 31i64)";
    let late_time = "time(23i64, 59i64, 59i64, 999999999i64)";
    let datetime_low = format!("datetime({date_low}, time(0i64, 0i64, 0i64, 0i64))");
    let datetime_high = format!("datetime({date_high}, {late_time})");
    let instant_low = "instant_from_unix(-377705030401i64, 0i64)";
    let instant_high = "instant_from_unix(253402214400i64, 999999999i64)";
    let durations = [
        format!("duration({MAX}, 999999999i64)"),
        format!("duration({MIN}, 0i64)"),
        format!("duration({MIN}, 1i64)"),
        "duration(-1i64, 999999999i64)".to_string(),
    ];
    let periods = [
        format!("period({MAX}, {MAX})"),
        format!("period({MIN}, {MIN})"),
        format!("period(0i64, {MAX})"),
    ];
    let offsets = [
        "offset_from_seconds(86399i64)",
        "offset_from_seconds(-86399i64)",
    ];
    let units = [
        "Hours",
        "Minutes",
        "Seconds",
        "Milliseconds",
        "Microseconds",
        "Nanoseconds",
    ];
    let roundings = [
        "RoundTowardNegative",
        "RoundTowardPositive",
        "RoundTowardZero",
        "RoundAwayFromZero",
        "RoundTiesToEven",
        "RoundTiesToAway",
        "RejectInexact",
    ];
    let mut cases: Vec<(&'static str, String)> = Vec::new();
    for a in ints {
        cases.push(("is_leap_year", format!("is_leap_year({a})")));
        cases.push(("days_in_year", format!("days_in_year({a})")));
        cases.push((
            "weekday_from_iso_number",
            format!("weekday_from_iso_number({a})"),
        ));
        cases.push(("date_from_epoch_day", format!("date_from_epoch_day({a})")));
        cases.push((
            "time_from_nanosecond_of_day",
            format!("time_from_nanosecond_of_day({a})"),
        ));
        cases.push(("offset_from_seconds", format!("offset_from_seconds({a})")));
        cases.push((
            "easter_sunday_gregorian",
            format!("easter_sunday_gregorian({a})"),
        ));
        cases.push((
            "easter_sunday_orthodox",
            format!("easter_sunday_orthodox({a})"),
        ));
        for b in ints {
            cases.push(("days_in_month", format!("days_in_month({a}, {b})")));
            cases.push(("instant_from_unix", format!("instant_from_unix({a}, {b})")));
            cases.push(("duration", format!("duration({a}, {b})")));
            cases.push(("period", format!("period({a}, {b})")));
            cases.push((
                "nth_weekday_in_month",
                format!("nth_weekday_in_month({a}, {b}, Monday, {MAX})"),
            ));
            cases.push((
                "last_weekday_in_month",
                format!("last_weekday_in_month({a}, {b}, Sunday)"),
            ));
            cases.push((
                "date_from_iso_week",
                format!("date_from_iso_week({a}, {b}, Sunday)"),
            ));
            cases.push(("date", format!("date({a}, {b}, {a})")));
            cases.push(("time", format!("time({a}, {b}, {a}, {b})")));
        }
        for d in [date_low, date_high] {
            cases.push(("date_add_days", format!("date_add_days({d}, {a})")));
            cases.push((
                "date_add_months",
                format!("date_add_months({d}, {a}, ClampToMonthEnd)"),
            ));
            cases.push((
                "try_date_add_months",
                format!("try_date_add_months({d}, {a}, RejectInvalidDay)"),
            ));
        }
        for unit in units {
            cases.push((
                "instant_from_unix_count",
                format!("instant_from_unix_count({a}, {unit})"),
            ));
            cases.push((
                "duration_from_count",
                format!("duration_from_count({a}, {unit})"),
            ));
        }
        for d in &durations {
            cases.push(("duration_mul", format!("duration_mul({d}, {a})")));
        }
        for p in &periods {
            cases.push(("period_mul", format!("period_mul({p}, {a})")));
        }
    }
    for d in [date_low, date_high] {
        cases.push(("date_iso_week", format!("date_iso_week({d})")));
        cases.push(("date_day_of_year", format!("date_day_of_year({d})")));
        cases.push((
            "weekday_on_or_after",
            format!("weekday_on_or_after({d}, Sunday)"),
        ));
        cases.push((
            "weekday_on_or_before",
            format!("weekday_on_or_before({d}, Monday)"),
        ));
        let other = if d == date_low { date_high } else { date_low };
        cases.push((
            "date_period_until",
            format!("date_period_until({d}, {other})"),
        ));
        for p in &periods {
            cases.push((
                "date_add_period",
                format!("date_add_period({d}, {p}, ClampToMonthEnd)"),
            ));
            cases.push((
                "try_date_add_period",
                format!("try_date_add_period({d}, {p}, RejectInvalidDay)"),
            ));
        }
    }
    for dt in [&datetime_low, &datetime_high] {
        cases.push((
            "datetime_until",
            format!("datetime_until({datetime_low}, {dt})"),
        ));
        for o in offsets {
            cases.push((
                "datetime_to_instant_at",
                format!("datetime_to_instant_at({dt}, {o})"),
            ));
        }
        for d in &durations {
            cases.push((
                "datetime_add_duration",
                format!("datetime_add_duration({dt}, {d})"),
            ));
        }
        for p in &periods {
            cases.push((
                "datetime_add_period",
                format!("datetime_add_period({dt}, {p}, ClampToMonthEnd)"),
            ));
        }
    }
    for i in [instant_low, instant_high] {
        cases.push((
            "instant_until",
            format!("instant_until({instant_low}, {i})"),
        ));
        for o in offsets {
            cases.push((
                "instant_to_datetime_at",
                format!("instant_to_datetime_at({i}, {o})"),
            ));
        }
        for d in &durations {
            cases.push((
                "instant_add_duration",
                format!("instant_add_duration({i}, {d})"),
            ));
            cases.push((
                "instant_round_to",
                format!("instant_round_to({i}, {d}, RoundTowardPositive)"),
            ));
        }
        cases.push((
            "instant_round_to",
            format!("instant_round_to({i}, duration(86400i64, 0i64), RoundTowardPositive)"),
        ));
        for unit in ["Hours", "Seconds", "Milliseconds", "Nanoseconds"] {
            for rounding in roundings {
                cases.push((
                    "instant_to_unix_count",
                    format!("instant_to_unix_count({i}, {unit}, {rounding})"),
                ));
            }
        }
    }
    for d in &durations {
        cases.push((
            "time_add_duration",
            format!("time_add_duration({late_time}, {d})"),
        ));
        cases.push(("duration_negate", format!("duration_negate({d})")));
        cases.push((
            "duration_to_seconds_f64",
            format!("duration_to_seconds_f64({d})"),
        ));
        for unit in ["Hours", "Seconds", "Milliseconds", "Nanoseconds"] {
            for rounding in ["RoundTowardPositive", "RoundTiesToEven", "RejectInexact"] {
                cases.push((
                    "duration_to_count",
                    format!("duration_to_count({d}, {unit}, {rounding})"),
                ));
            }
        }
        for e in &durations {
            cases.push(("duration_add", format!("duration_add({d}, {e})")));
            cases.push(("duration_sub", format!("duration_sub({d}, {e})")));
        }
    }
    for p in &periods {
        cases.push(("period_negate", format!("period_negate({p})")));
    }
    for text in [
        "PT9223372036854775807H",
        "-PT9223372036854775808S",
        "-PT9223372036854775808.5S",
        "PT99999999999999999999999M",
        "-PT2562047788015215H30M8S",
    ] {
        cases.push(("parse_duration", format!("parse_duration(\"{text}\")")));
    }
    for text in [
        "P9223372036854775807Y",
        "-P9223372036854775808M",
        "P1317624576693539401W",
        "-P768614336404564650Y8M",
    ] {
        cases.push(("parse_period", format!("parse_period(\"{text}\")")));
    }
    for text in [
        "+999999-01-01",
        "-999999-12-31",
        "9999-12-31T23:59:59.999999999-23:59:59",
        "-009999-01-01T00:00:00+23:59:59",
    ] {
        cases.push(("parse_instant", format!("parse_instant(\"{text}\")")));
        cases.push((
            "parse_offset_datetime",
            format!("parse_offset_datetime(\"{text}\")"),
        ));
        cases.push(("parse_date", format!("parse_date(\"{text}\")")));
        cases.push(("parse_datetime", format!("parse_datetime(\"{text}\")")));
    }
    cases
}

#[test]
fn std_datetime_extreme_arguments_raise_no_primitive_trap() {
    let cases = extreme_cases();
    let expressions: Vec<(String, String)> = cases
        .iter()
        .enumerate()
        .map(|(index, (_, expression))| (format!("case_{index:04}"), expression.clone()))
        .collect();
    let outcomes = run_expression_suite("datetime-extremes-2859", &expressions);
    let mut failures = 0usize;
    for (index, (function, expression)) in cases.iter().enumerate() {
        let outcome = &outcomes[&format!("case_{index:04}")];
        if let Some(message) = outcome {
            failures += 1;
            assert!(
                message.starts_with(&format!("{function}: domain: "))
                    || message.starts_with(&format!("{function}: overflow: ")),
                "`{expression}` escaped the failure contract with `{message}`"
            );
        }
    }
    assert!(
        failures > 0 && failures < cases.len(),
        "the sweep must exercise both accepted and rejected extremes ({failures} of {})",
        cases.len()
    );
}

/// [05-OP-73] opaque value types: construction, inspection, and field access
/// outside `Std.Datetime` are `OpaqueTypeViolation`s, one per type.
#[test]
fn std_datetime_opaque_types_reject_outside_construction() {
    let constructions = [
        ("Date", "Date { epoch_day: 0i64 }"),
        ("Time", "Time { nanosecond_of_day: 0i64 }"),
        (
            "DateTime",
            "DateTime { epoch_day: 0i64, nanosecond_of_day: 0i64 }",
        ),
        ("Instant", "Instant { unix_second: 0i64, nanosecond: 0i64 }"),
        ("Offset", "Offset { seconds: 0i64 }"),
        (
            "OffsetDateTime",
            "OffsetDateTime { instant: instant_from_unix(0i64, 0i64), offset: offset_from_seconds(0i64) }",
        ),
        ("Duration", "Duration { second: 0i64, nanosecond: 0i64 }"),
        ("Period", "Period { months: 0i64, days: 0i64 }"),
        ("Dates", "Dates { epoch_days: to_tensor([0i64]) }"),
        (
            "Instants",
            "Instants { unix_seconds: to_tensor([0i64]), nanoseconds: to_tensor([0i64]) }",
        ),
    ];
    let (_dir, reef_home, app_pkg) = make_app("datetime-opaque-2859");
    let mut source = String::from(
        "module Demo.Main\nimport Std.Datetime (Date, Time, DateTime, Instant, Offset, OffsetDateTime, Duration, Period, Dates, Instants, date, instant_from_unix, offset_from_seconds)\n",
    );
    for (index, (_, construction)) in constructions.iter().enumerate() {
        source.push_str(&format!("forged_{index} = {construction}\n"));
    }
    source.push_str("peeked = date(2024i64, 1i64, 1i64).epoch_day\n");
    let path = app_pkg.join("src/main.ch");
    write_file(&path, &source);
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "forging opaque datetime values must not check:\n{rendered}"
    );
    for (type_name, construction) in constructions {
        assert!(
            rendered.contains(&format!(
                "record construction of opaque type `{type_name}` outside its defining module"
            )),
            "`{construction}` was not rejected as an opaque construction:\n{rendered}"
        );
    }
    assert!(
        rendered.contains("field access") && rendered.contains("opaque type `Date`"),
        "field access on an opaque Date was not rejected:\n{rendered}"
    );
}

/// spec/04 §2.5: an accessor lambda whose target type is never determined is
/// already a [04-INF-9] error; a check unit that reaches `Std.Datetime`, the
/// first standard-library opaque module, additionally reports it as an
/// `OpaqueTypeViolation`, and one that does not reach it is unchanged. A
/// lambda that is applied, so its target is determined, still checks.
#[test]
fn std_datetime_accessor_lambdas_follow_the_opaque_target_rule() {
    let (_dir, reef_home, app_pkg) = make_app("datetime-lambda-2859");
    let path = app_pkg.join("src/main.ch");
    write_file(
        &path,
        "module Demo.Main\nimport Std.Datetime (date, date_year)\ntype Point =\n  | Point { x: i64 }\ndef applied() -> i64 = {\n  get = fn (r) -> r.x\n  add(get(Point { x: 1i64 }), date_year(date(2024i64, 1i64, 1i64)))\n}\n",
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        success,
        "an applied accessor lambda must check:\n{rendered}"
    );

    write_file(
        &path,
        "module Demo.Main\nimport Std.Datetime (date, date_year)\ntype Point =\n  | Point { x: i64 }\ndef unapplied() -> i64 = {\n  get = fn (r) -> r.x\n  date_year(date(2024i64, 1i64, 1i64))\n}\n",
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "an unapplied accessor lambda must not check:\n{rendered}"
    );
    assert!(
        rendered.contains("[04-INF-9]"),
        "the undetermined operand is an [04-INF-9] error:\n{rendered}"
    );
    assert!(
        rendered.contains("OpaqueTypeViolation"),
        "a check unit reaching Std.Datetime reports the unresolved access as an opaque violation:\n{rendered}"
    );

    write_file(
        &path,
        "module Demo.Main\nimport Std.Text (join)\ntype Point =\n  | Point { x: i64 }\ndef unapplied() -> i64 = {\n  get = fn (r) -> r.x\n  1i64\n}\n",
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "an unapplied accessor lambda must not check:\n{rendered}"
    );
    assert!(
        rendered.contains("[04-INF-9]") && !rendered.contains("OpaqueTypeViolation"),
        "a check unit that does not reach Std.Datetime reports only the [04-INF-9] error:\n{rendered}"
    );
}
