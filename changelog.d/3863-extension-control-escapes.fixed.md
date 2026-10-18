`chelis fmt` no longer changes extension data, and canonical Deep spells
every control character in a string with its canonical escape. A raw control
character inside an extension-data string, or in a string literal, now prints
as its named escape (`\n`, `\t`, `\r`, `\0`) or, for the others, minimal
lowercase `\u{h}`, at every layout position. The Deep reader accepts `\u{h}`,
with one to six hexadecimal digits in any case or padding. Previously a
carriage return before a line feed in extension data was silently dropped
wherever an enclosing form broke across lines, so formatting and re-parsing
gave different data, and a raw carriage return made the file fail the
`fmt --check` style gate. Decoded values are unchanged
([#3863](https://github.com/Chelis-Lang/chelis/issues/3863)).
