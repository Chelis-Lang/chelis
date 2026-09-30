# Registry: exported stdlib numeric ADT identities ([05-OP-34])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-34] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| identity | exact variants and fields |
|---|---|
| `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(i64) | JsonBigInt(string) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
| `decimal::Decimal` | `Decimal { coefficient: i64, scale: i64 }` |
| `time::Date` | `Date { year: i64, month: i64, day: i64 }` |
| `time::Duration` | `Duration { days: i64, hours: i64, minutes: i64, seconds: i64 }` |
