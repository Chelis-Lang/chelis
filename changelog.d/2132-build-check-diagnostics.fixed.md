`chelis build`, `chelis eval --file` and `chelis deep --annotate` render a
type-check rejection as one line per diagnostic, carrying its kind, message,
location, source identity and suggestions as `chelis check` publishes them.
Previously `chelis build` printed a Rust `Debug` dump of the internal error
list. spec/04 [04-FIT-26] states the rule. See
[#1853](https://github.com/Chelis-Lang/chelis/issues/1853) and
[#2132](https://github.com/Chelis-Lang/chelis/pull/2132).
