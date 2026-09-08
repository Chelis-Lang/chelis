# Registry: exported stdlib numeric ADT identities ([05-OP-34])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-34] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| identity | exact variants and fields |
|---|---|
| `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(int64) | JsonBigInt(string) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
| `decimal::Decimal` | `Decimal { coefficient: int64, scale: int64 }` |
| `time::Date` | `Date { year: int64, month: int64, day: int64 }` |
| `time::Duration` | `Duration { days: int64, hours: int64, minutes: int64, seconds: int64 }` |
| `tokenizer::Tokenizer` | `BpeTokenizer(Dict[string,int64], Dict[string,int64], Dict[int64,string], int64)` |
