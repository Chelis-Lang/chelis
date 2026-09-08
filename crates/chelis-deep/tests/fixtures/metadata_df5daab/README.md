These are outputs from the generic AST at commit
`df5daab0a574823ca67d39cbd0f78dacaa36f329`, generated with that checkout's
`chelis_deep::parser::parse_str` on each string in `sources.json`.
`ast.json` used `serde_json::to_vec_pretty`; `ast.bin` used
`bincode::serialize`, each on `Vec<Vec<Expr>>`. The producer ran as a
standalone integration test in `chelis-deep` (JSON) and `chelis-types`
(binary), using their existing dev dependencies. The temporary producer
tests were removed after successful execution.

The corpus has all 30 registered keys, prefix metadata, a nested extension,
and duplicate keys inside preserved source data. These are old-producer
fixtures: do not regenerate them with the new representation. The consumer
test compares decoded values and spans with current text ingestion.
