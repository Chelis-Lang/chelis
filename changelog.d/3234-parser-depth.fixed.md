Surf expression, type, and pattern parsing now grows the native stack at
recursive entries. Valid sources with 40,000 grouping parentheses pass direct
validation and the CLI on a small worker stack; rejected variants return parse
errors. See [#3234](https://github.com/Chelis-Lang/chelis/issues/3234).
