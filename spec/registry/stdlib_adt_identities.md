# Registry: exported stdlib numeric ADT identities ([05-OP-34])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-34] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| identity | exact variants and fields |
|---|---|
| `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(i64) | JsonBigInt(string) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
| `decimal::Decimal` | `Decimal { negative: bool, limb0: i64, limb1: i64, limb2: i64, limb3: i64, limb4: i64, scale: i64 }` |
| `datetime/business::BusinessCalendar` | `BusinessCalendar { weekmask: Weekmask, holidays: List[i64], valid_from: i64, valid_until: i64 }` |
| `datetime::Date` | `Date { epoch_day: i64 }` |
| `datetime::Time` | `Time { nanosecond_of_day: i64 }` |
| `datetime::DateTime` | `DateTime { epoch_day: i64, nanosecond_of_day: i64 }` |
| `datetime::Instant` | `Instant { unix_second: i64, nanosecond: i64 }` |
| `datetime::Offset` | `Offset { seconds: i64 }` |
| `datetime::OffsetDateTime` | `OffsetDateTime { instant: Instant, offset: Offset }` |
| `datetime::Duration` | `Duration { second: i64, nanosecond: i64 }` |
| `datetime::Period` | `Period { months: i64, days: i64 }` |
| `datetime::Dates` | `Dates { epoch_days: tensor[n,i64] }` |
| `datetime::Instants` | `Instants { unix_seconds: tensor[n,i64], nanoseconds: tensor[n,i64] }` |
| `datetime/clock::MonotonicInstant` | `MonotonicInstant { second: i64, nanosecond: i64 }` |
| `datetime/zone::TimeZone` | `TimeZone { name: string, initial_offset: i64, transitions: List[(i64,i64)], footer: Option[(i64,Option[(i64,(i64,i64,i64,i64,i64),(i64,i64,i64,i64,i64))])] }` |
| `datetime/zone::Zoned` | `Zoned { instant: Instant, zone: TimeZone }` |
| `datetime/zone::ZonedText` | `ZonedText { local: DateTime, offset: Offset, zone_name: string, critical: bool }` |
