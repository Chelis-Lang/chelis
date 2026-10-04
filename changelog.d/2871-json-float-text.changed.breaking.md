`JsonFloat` keeps a JSON number token's exact text beside its correctly rounded
`f64`, as `JsonFloat(value, text)` under [05-OP-2]. Parsing fills both fields
from the token, so `decimal(text)` from `Std.Decimal` reaches the value the
producer wrote, and `to_json` writes the stored text, so tokens such as `1E5`,
`19.950`, and `-0e0` keep their spelling. See
[#2871](https://github.com/Chelis-Lang/chelis/issues/2871).

- Construct a `JsonFloat` from a computed value with
  `JsonFloat(x, to_string(x))`, and match it with two fields, for example
  `JsonFloat(x, _)`.
- `to_json`, `try_to_json`, `write_json`, and `try_write_json` refuse a
  `JsonFloat` whose text is not an RFC 8259 number token with a fraction or
  exponent, or whose correctly rounded `f64` is not bit-identical to the stored
  value, signed zero included. `to_json` and `write_json` fail, and the `try_`
  forms return `None`.
- Parsing rejects number tokens outside the RFC 8259 grammar that it previously
  accepted, such as `01.5`, `1.`, `1.e5`, and `-.5`.
- Documents that spell one number differently, such as `1.0` and `1.00`, are no
  longer equal, and `to_json` no longer rewrites float tokens in shortest
  round-trip form.
