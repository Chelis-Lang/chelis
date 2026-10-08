Ordinary embedding serializers can encode deeply nested execution values, and
input decoding accepts or rejects deeply nested values without overflowing a
small worker stack. The existing execution wire format is unchanged. See
[#2601](https://github.com/Chelis-Lang/chelis/issues/2601).
