**checker/CLI: `chelis check`'s error objects change shape in
four ways (chelis#886).** The report's error objects are now produced by
serializing one typed carrier instead of four hand-written `format!`
templates. `suggestions` is emitted when non-empty, for effect diagnostics
as well as check ones; `span_offset` becomes `span`, a tagged carrier that
is `{"span":"range","offset":N,"len":M}` when the producer measured an
extent and `{"span":"point","offset":N}` when it held only a coordinate
(chelis#1395); and member order places `suggestions` between `severity`
and `span`. `kind`, `message`, `severity`, `expected`, `got` and `span_id` are
unchanged in spelling, presence rule and position. `spec/04-type-system.md`
§6.4 states the transport contract as [04-FIT-11] through [04-FIT-17].
A consumer that reads `span_offset` from the check report must move to
`span.offset`, which is a sibling key in both variants and so needs no
matching; reading `len` requires matching the `span` discriminator. The
untagged shape is
hard-rejected by the consumer type rather than silently accepted. No
in-repo consumer read `span_offset`; `chelis-lsp` reads the carrier and
renders a point as a caret-position range.
