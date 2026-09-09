`chelis check` builds its report document by serializing one typed value,
`chelis_compiler_api::schema::CheckResult`, instead of the two `format!`
templates that previously assembled it beside a type describing the same
shape. `inferred_signatures` becomes a member of that type rather than a JSON
string spliced between two literal keys. The emitted bytes are unchanged,
verified against the prior binary across 6324 `chelis check` invocations over
1581 files. Failures that short-circuit before the checker still emit the
report; failures that bypass it entirely, such as an unreadable file or a
style-gate rejection, remain outside it and are tracked by
[#886](https://github.com/Chelis-Lang/chelis/issues/886). See
[#1672](https://github.com/Chelis-Lang/chelis/pull/1672).
