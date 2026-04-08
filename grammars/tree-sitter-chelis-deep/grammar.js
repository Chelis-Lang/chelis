module.exports = grammar({
  name: "chelis_deep",

  extras: ($) => [/\s/, $.comment],
  word: ($) => $.symbol,

  rules: {
    source_file: ($) => repeat($.expr),
    comment: () => token(seq(";", /.*/)),
    expr: ($) => choice($.list, $.map, $.symbol, $.keyword, $.number_literal, $.string_literal),
    list: ($) => seq("(", repeat($.expr), ")"),
    map: ($) => seq("{", repeat($.map_entry), "}"),
    map_entry: ($) => seq(choice($.symbol, $.keyword, $.string_literal), $.expr),
    keyword: () => /:[A-Za-z_][A-Za-z0-9_\-]*/,
    number_literal: () => /-?\d+(\.\d+)?/,
    string_literal: () => token(seq('"', repeat(choice(/[^"\\]+/, /\\./)), '"')),
    symbol: () => /[A-Za-z_+\-*\/<>=!?][A-Za-z0-9_+\-*\/<>=!?\.]*/,
  },
});
