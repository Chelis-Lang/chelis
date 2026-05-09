const PREC = {
  pipe: 1,
  or: 2,
  and: 3,
  compare: 4,
  add: 5,
  multiply: 6,
  unary: 7,
  annotation: 8,
  call: 9,
  field: 10,
  arrow: 1,
};

module.exports = grammar({
  name: "chelis_surf",

  extras: ($) => [/\s/, $.line_comment, $.block_comment],
  word: ($) => $.identifier,

  conflicts: ($) => [
    [$.tuple_expression, $.parenthesized_expression],
    [$.tuple_type, $.parenthesized_type],
    [$.call_expression, $.field_expression],
    [$.record_expression, $.block_expression],
    [$.let_definition, $.let_in_expression],
    [$.variant_tuple_fields, $.parenthesized_type],
    [$.primary_expression, $.record_expression],
    [$.infer_type, $.wildcard],
    [$.primary_expression, $.type_expression],
  ],

  rules: {
    source_file: ($) => repeat($.declaration),
    declaration: ($) =>
      choice(
        $.module_declaration,
        $.import_declaration,
        $.export_declaration,
        $.dim_declaration,
        $.signature_declaration,
        $.property_declaration,
        $.function_definition,
        $.let_definition,
      ),
    module_declaration: ($) => seq("module", field("name", $.module_path)),
    import_declaration: ($) =>
      prec.right(
        seq(
          "import",
          field("module", $.module_path),
          optional(choice($.import_names, $.import_all)),
        ),
      ),
    import_names: ($) => seq("(", commaSep1(choice($.identifier, $.type_identifier)), ")"),
    import_all: ($) => seq("(", "..", ")"),
    export_declaration: ($) => seq("export", "(", commaSep1(choice($.identifier, $.type_identifier)), ")"),
    dim_declaration: ($) => seq("dim", commaSep1($.identifier)),
    signature_declaration: ($) => seq("sig", field("name", $.identifier), ":", field("type", $.type_expression)),
    property_declaration: ($) =>
      seq(
        "@",
        "property",
        field("name", $.identifier),
        "forall",
        "(",
        commaSep($.parameter),
        ")",
        optional(seq("where", commaSep1(field("precondition", $.expression)))),
        ":",
        field("body", $.expression),
        repeat($.property_option),
      ),
    property_option: ($) =>
      seq("with", choice("tolerance", "seed", "samples"), "=", field("value", $.expression)),
    function_definition: ($) =>
      seq(
        "def",
        field("name", $.identifier),
        optional(field("dim_parameters", $.dimension_parameters)),
        "(",
        commaSep($.parameter),
        ")",
        optional(seq(":", field("return_type", $.type_expression))),
        "=",
        field("body", $.expression),
      ),
    let_definition: ($) =>
      seq(
        "let",
        field("pattern", $.let_pattern),
        optional(seq(":", field("type", $.type_expression))),
        "=",
        field("value", $.expression),
      ),
    dimension_parameters: ($) => seq("[", commaSep1($.identifier), "]"),
    parameter: ($) => seq(field("name", $.identifier), optional(seq(":", field("type", $.type_expression)))),
    let_pattern: ($) => choice($.identifier, $.wildcard, seq("(", commaSep1($.let_pattern), ")")),
    module_path: ($) => seq($.type_identifier, repeat(seq(".", $.type_identifier))),
    expression: ($) =>
      choice(
        $.if_expression,
        $.match_expression,
        $.let_in_expression,
        $.lambda_expression,
        $.block_expression,
        $.par_expression,
        $.pipe_expression,
        $.logical_or_expression,
        $.logical_and_expression,
        $.comparison_expression,
        $.additive_expression,
        $.multiplicative_expression,
        $.unary_expression,
        $.annotation_expression,
        $.field_expression,
        $.call_expression,
        $.primary_expression,
      ),
    primary_expression: ($) =>
      choice(
        $.identifier,
        $.type_identifier,
        $.wildcard,
        $.number,
        $.string,
        $.boolean,
        $.tuple_expression,
        $.parenthesized_expression,
        $.record_expression,
      ),
    parenthesized_expression: ($) => seq("(", $.expression, ")"),
    tuple_expression: ($) => seq("(", $.expression, ",", commaSep1($.expression), ")"),
    record_expression: ($) =>
      seq(
        field("constructor", $.type_identifier),
        "{",
        commaSep(choice($.record_field, $.record_pun)),
        "}",
      ),
    record_field: ($) => seq(field("name", $.identifier), ":", field("value", $.expression)),
    record_pun: ($) => field("name", $.identifier),
    call_expression: ($) =>
      prec.left(PREC.call, seq(field("function", $.expression), "(", commaSep($.expression), ")")),
    field_expression: ($) =>
      prec.left(PREC.field, seq(field("value", $.expression), ".", field("field", choice($.identifier, $.number)))),
    annotation_expression: ($) =>
      prec.left(PREC.annotation, seq(field("value", $.expression), ":", $.type_expression)),
    unary_expression: ($) =>
      prec(PREC.unary, seq(field("operator", choice("-", "!", "grad", "jit", "realize", "copy")), $.expression)),
    multiplicative_expression: ($) =>
      prec.left(PREC.multiply, seq($.expression, field("operator", choice("*", "/", "%")), $.expression)),
    additive_expression: ($) =>
      prec.left(PREC.add, seq($.expression, field("operator", choice("+", "-")), $.expression)),
    comparison_expression: ($) =>
      prec.left(PREC.compare, seq($.expression, field("operator", choice("==", "!=", "<", ">", "<=", ">=")), $.expression)),
    logical_and_expression: ($) => prec.left(PREC.and, seq($.expression, "&&", $.expression)),
    logical_or_expression: ($) => prec.left(PREC.or, seq($.expression, "||", $.expression)),
    pipe_expression: ($) => prec.left(PREC.pipe, seq($.expression, "|>", $.expression)),
    if_expression: ($) => seq("if", $.expression, "then", $.expression, "else", $.expression),
    let_in_expression: ($) => seq("let", commaSep1($.let_binding), "in", $.expression),
    let_binding: ($) => seq(field("pattern", $.let_pattern), optional(seq(":", $.type_expression)), "=", field("value", $.expression)),
    lambda_expression: ($) => seq("fn", "(", commaSep($.parameter), ")", "->", $.expression),
    block_expression: ($) => seq("{", repeat($.let_definition), field("result", $.expression), optional(";"), "}"),
    par_expression: ($) => seq("par", "{", commaSep1($.expression), optional(";"), "}"),
    match_expression: ($) => seq("match", $.expression, "with", "{", repeat1($.match_arm), "}"),
    match_arm: ($) => seq(optional("|"), field("pattern", $.pattern), optional(seq("if", $.expression)), "=>", field("body", $.expression)),
    pattern: ($) =>
      choice(
        $.wildcard,
        $.identifier,
        $.type_identifier,
        $.number,
        $.string,
        $.boolean,
        $.constructor_pattern,
        $.tuple_pattern,
        $.record_pattern,
        $.as_pattern,
      ),
    constructor_pattern: ($) => seq(field("name", $.type_identifier), "(", commaSep($.pattern), ")"),
    tuple_pattern: ($) => seq("(", $.pattern, ",", commaSep1($.pattern), ")"),
    record_pattern: ($) => seq(field("name", $.type_identifier), "{", commaSep(choice($.record_pattern_field, $.record_pattern_pun)), "}"),
    record_pattern_field: ($) => seq(field("name", $.identifier), ":", field("pattern", $.pattern)),
    record_pattern_pun: ($) => field("name", $.identifier),
    as_pattern: ($) => seq(field("name", $.identifier), "@", field("pattern", $.pattern)),
    type_expression: ($) =>
      choice(
        $.function_type,
        $.tensor_type,
        $.tuple_type,
        $.applied_type,
        $.parenthesized_type,
        $.infer_type,
        $.type_identifier,
        $.identifier,
      ),
    parenthesized_type: ($) => seq("(", $.type_expression, ")"),
    tuple_type: ($) => seq("(", $.type_expression, ",", commaSep1($.type_expression), ")"),
    infer_type: ($) => "_",
    function_type: ($) => prec.right(PREC.arrow, seq($.non_arrow_type, "->", $.type_expression)),
    non_arrow_type: ($) => choice($.tensor_type, $.tuple_type, $.applied_type, $.parenthesized_type, $.infer_type, $.type_identifier, $.identifier),
    applied_type: ($) => prec.left(PREC.call, seq(field("name", $.type_identifier), repeat1($.non_arrow_type))),
    tensor_type: ($) => seq("tensor", "[", commaSep1($.type_expression), "]"),
    boolean: ($) => choice("true", "false"),
    wildcard: ($) => "_",
    number: ($) =>
      token(
        choice(
          /0[xX][0-9a-fA-F_]+/,
          /0[bB][01_]+/,
          /[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?/,
          /[0-9][0-9_]*([eE][+-]?[0-9_]+)?/
        )
      ),
    string: ($) => token(seq('"', repeat(choice(/[^"\\]+/, /\\./)), '"')),
    identifier: ($) => /[_a-z][_A-Za-z0-9]*/,
    type_identifier: ($) => /[A-Z][_A-Za-z0-9]*/,
    line_comment: () => token(seq("--", /.*/)),
    block_comment: () => token(seq("{-", repeat(choice(/[^-]/, /-[^}]/)), "-}")),
    variant_tuple_fields: ($) => seq("(", commaSep1($.type_expression), ")"),
  },
});

function commaSep(rule) {
  return optional(commaSep1(rule));
}

function commaSep1(rule) {
  return seq(rule, repeat(seq(",", rule)), optional(","));
}
