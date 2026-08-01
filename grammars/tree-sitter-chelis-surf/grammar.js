const PREC = {
  pipe: 1,
  recordUpdate: 2,
  or: 3,
  and: 4,
  equality: 5,
  compare: 6,
  add: 7,
  multiply: 8,
  unary: 9,
  annotation: 10,
  call: 11,
  field: 12,
  arrow: 1,
};

module.exports = grammar({
  name: "chelis_surf",

  externals: ($) => [
    $._canonical_number,
    $._canonical_pattern_number,
    $._canonical_axis_integer,
    $._canonical_nonzero_axis_integer,
  ],

  extras: ($) => [/\s/, $.line_comment, $.block_comment],
  word: ($) => $.identifier,

  conflicts: ($) => [
    [$.tuple_expression, $.parenthesized_expression],
    [$.tuple_type, $.parenthesized_type],
    [$.unit_expression, $.unit_type],
    [$.let_pattern, $.unit_expression],
    [$.tuple_pattern, $.parenthesized_pattern],
    [$.record_expression, $.block_expression],
    [$.record_update_expression, $.with_handler_expression],
    [$.type_definition, $.type_alias],
    [$.call_expression, $.field_expression],
    [$.primary_expression, $.type_expression],
    [$.primary_expression, $.non_arrow_type],
    [$.primary_expression, $.qualified_type_name],
    [$.value_identifier, $.primary_expression],
    [$.infer_type, $.wildcard],
  ],

  rules: {
    source_file: ($) => repeat($.declaration),

    declaration: ($) =>
      choice(
        $.module_declaration,
        $.import_declaration,
        $.export_declaration,
        $.dim_declaration,
        $.opaque_type_declaration,
        $.type_definition,
        $.type_alias,
        $.signature_declaration,
        $.property_declaration,
        $.function_definition,
        $.macro_definition,
        $.binding_definition,
      ),

    module_declaration: ($) => seq("module", field("name", $.module_path)),
    module_path: ($) => seq($.type_identifier, repeat(seq(".", $.type_identifier))),

    import_declaration: ($) =>
      seq(
        "import",
        field("module", $.module_path),
        optional(choice($.import_names, $.import_all)),
      ),
    import_names: ($) => seq("(", commaSep1($.declared_name), ")"),
    import_all: () => seq("(", "..", ")"),
    export_declaration: ($) => seq("export", "(", commaSep1($.declared_name), ")"),
    declared_name: ($) => choice($.identifier, $.type_identifier),

    dim_declaration: ($) => seq("dim", commaSepNoTrail1($.identifier)),

    opaque_type_declaration: ($) =>
      seq(
        "@",
        "opaque",
        optional($.invariant_declaration),
        field("definition", $.type_definition),
      ),
    invariant_declaration: ($) =>
      seq(
        "@",
        "invariant",
        "(",
        field("binder", $.identifier),
        ")",
        field("predicate", $.expression),
      ),
    type_definition: ($) =>
      seq(
        "type",
        field("name", $.type_identifier),
        optional($.type_parameters),
        "=",
        repeat1(seq("|", $.variant)),
      ),
    type_alias: ($) =>
      seq(
        "type",
        field("name", $.type_identifier),
        optional($.type_parameters),
        "=",
        field("type", $.type_expression),
      ),
    type_parameters: ($) => seq("[", commaSep1($.identifier), "]"),
    variant: ($) =>
      seq(
        field("name", $.type_identifier),
        optional(choice($.variant_record_fields, $.variant_tuple_fields)),
      ),
    variant_record_fields: ($) => seq("{", commaSep1($.field_declaration), "}"),
    variant_tuple_fields: ($) => seq("(", commaSep1($.type_expression), ")"),
    field_declaration: ($) =>
      seq(field("name", $.identifier), ":", field("type", $.type_expression)),

    signature_declaration: ($) =>
      seq(
        "sig",
        field("name", $.identifier),
        ":",
        field("type", $.type_expression),
        optional($.effect_clause),
      ),

    property_declaration: ($) =>
      seq(
        "@",
        "property",
        field("name", $.identifier),
        "forall",
        "(",
        commaSep($.typed_parameter),
        ")",
        optional(seq("where", commaSep1(field("precondition", $.property_precondition)))),
        ":",
        field("body", $.expression),
        repeat($.property_option),
      ),
    property_precondition: ($) => choice($.call_expression, $.primary_expression),
    property_option: ($) =>
      seq(
        "with",
        choice("tolerance", "seed", "samples", "contract"),
        "=",
        field("value", $.expression),
      ),

    function_definition: ($) =>
      seq(
        "def",
        field("name", $.identifier),
        optional(field("quantifiers", $.dimension_parameters)),
        "(",
        commaSep($.parameter),
        ")",
        optional(seq("->", field("return_type", $.type_expression))),
        optional($.effect_clause),
        "=",
        field("body", $.expression),
      ),
    dimension_parameters: ($) => seq("[", commaSep1($.identifier), "]"),
    parameter: ($) =>
      seq(
        field("name", $.value_identifier),
        optional(seq(":", field("type", $.type_expression))),
      ),
    typed_parameter: ($) =>
      seq(field("name", $.value_identifier), ":", field("type", $.type_expression)),
    effect_clause: ($) => seq("!", "{", commaSep($.effect_expression), "}"),
    effect_expression: ($) =>
      choice("Diff", "Random", "Accum", "IO", "Test", seq("Resource", "(", $.string, ")")),

    macro_definition: ($) =>
      seq(
        "macro",
        field("name", $.identifier),
        "(",
        commaSep($.identifier),
        ")",
        "=",
        field("body", $.expression),
      ),
    binding_definition: ($) =>
      seq(
        field("name", $.value_identifier),
        optional(seq(":", field("type", $.type_expression))),
        "=",
        field("value", $.expression),
      ),
    value_identifier: ($) => choice($.identifier, $.single_upper_identifier),

    expression: ($) =>
      choice(
        $.if_expression,
        $.match_expression,
        $.lambda_expression,
        $.with_handler_expression,
        $.block_expression,
        $.par_expression,
        $.do_expression,
        $.record_update_expression,
        $.pipe_expression,
        $.logical_or_expression,
        $.logical_and_expression,
        $.equality_expression,
        $.comparison_expression,
        $.additive_expression,
        $.multiplicative_expression,
        $.unary_expression,
        $.annotation_expression,
        $.call_expression,
        $.field_expression,
        $.transform_expression,
        $.quote_expression,
        $.primary_expression,
      ),

    if_expression: ($) =>
      seq(
        "if",
        field("condition", $.expression),
        "then",
        field("consequence", $.expression),
        "else",
        field("alternative", $.expression),
      ),
    match_expression: ($) =>
      seq("match", field("value", $.expression), "with", "{", repeat1($.match_arm), "}"),
    match_arm: ($) =>
      seq(
        "|",
        field("pattern", $.pattern),
        optional(seq("if", field("guard", $.expression))),
        "=>",
        field("body", $.expression),
      ),
    lambda_expression: ($) =>
      seq("fn", "(", commaSep($.parameter), ")", "->", field("body", $.expression)),

    with_handler_expression: ($) =>
      seq(
        "with",
        field("handler", choice("seed", "device")),
        "(",
        field("argument", $.expression),
        ")",
        field("body", $.handler_block_expression),
      ),
    handler_block_expression: ($) =>
      seq(
        "{",
        choice(
          prec.dynamic(1, $.expression),
          seq(repeat1($.let_binding), field("result", prec.dynamic(1, $.expression))),
        ),
        "}",
      ),

    block_expression: ($) =>
      seq("{", repeat1($.let_binding), field("result", prec.dynamic(1, $.expression)), "}"),
    let_binding: ($) =>
      seq(
        field("pattern", $.let_pattern),
        optional(seq(":", field("type", $.type_expression))),
        "=",
        field("value", $.expression),
      ),
    let_pattern: ($) =>
      choice(
        $.value_identifier,
        $.wildcard,
        seq("(", ")"),
        seq("(", $.let_pattern, ",", optional(commaSep1($.let_pattern)), ")"),
      ),

    par_expression: ($) => seq("par", "{", semicolonSep1($.expression), "}"),
    do_expression: ($) => seq("do", "{", semicolonSep1($.expression), "}"),

    record_update_expression: ($) =>
      prec.right(
        PREC.recordUpdate,
        seq(
          field("value", $.expression),
          "with",
          "{",
          commaSep1(choice($.record_field, $.record_pun)),
          "}",
        ),
      ),

    pipe_expression: ($) =>
      prec.left(PREC.pipe, seq($.expression, "|>", field("stage", $.expression))),
    logical_or_expression: ($) =>
      prec.left(PREC.or, seq($.expression, field("operator", "||"), $.expression)),
    logical_and_expression: ($) =>
      prec.left(PREC.and, seq($.expression, field("operator", "&&"), $.expression)),
    equality_expression: ($) =>
      prec.left(
        PREC.equality,
        seq($.expression, field("operator", choice("==", "!=")), $.expression),
      ),
    comparison_expression: ($) =>
      prec.left(
        PREC.compare,
        seq($.expression, field("operator", choice("<", ">", "<=", ">=")), $.expression),
      ),
    additive_expression: ($) =>
      prec.left(PREC.add, seq($.expression, field("operator", choice("+", "-")), $.expression)),
    multiplicative_expression: ($) =>
      prec.left(
        PREC.multiply,
        seq($.expression, field("operator", choice("*", "/", "%")), $.expression),
      ),
    unary_expression: ($) =>
      prec(PREC.unary, seq(field("operator", choice("-", "!", "&")), $.expression)),
    annotation_expression: ($) =>
      prec.left(
        PREC.annotation,
        seq(field("value", $.expression), ":", field("type", $.type_expression)),
      ),

    call_expression: ($) =>
      choice(
        prec.left(
          PREC.call,
          seq(
            field(
              "function",
              choice(
                $.identifier,
                $.callable_access_expression,
                $.transform_expression,
                $.quote_expression,
                $.parenthesized_expression,
              ),
            ),
            $.call_arguments,
            repeat($.access_step),
          ),
        ),
        prec.left(
          PREC.call,
          seq(
            field("function", $.qualified_type_name),
            $.nonempty_call_arguments,
            repeat($.access_step),
          ),
        ),
      ),
    call_arguments: ($) => seq(token.immediate("("), commaSep($.expression), ")"),
    nonempty_call_arguments: ($) =>
      seq(token.immediate("("), commaSep1($.expression), ")"),
    callable_access_expression: ($) =>
      prec.left(
        PREC.field,
        seq(
          field("value", $.primary_expression),
          repeat($.access_step),
          ".",
          field("field", $.identifier),
        ),
      ),
    field_expression: ($) =>
      prec.left(
        PREC.field,
        seq(field("value", $.primary_expression), repeat1($.access_step)),
      ),
    access_step: ($) => seq(".", field("field", choice($.identifier, $.type_identifier, $.axis_integer))),

    transform_expression: ($) =>
      choice(
        $.grad_expression,
        $.vmap_expression,
        $.cast_expression,
        $.unary_transform_expression,
        $.bare_unary_transform,
      ),
    grad_expression: ($) =>
      seq(
        "grad",
        "(",
        field("function", $.expression),
        optional(seq(",", "wrt", "=", $.grad_targets)),
        ")",
      ),
    grad_targets: ($) =>
      choice($.identifier, seq("(", $.identifier, ",", commaSep1($.identifier), ")")),
    vmap_expression: ($) =>
      seq(
        "vmap",
        "(",
        field("function", $.expression),
        optional(seq(",", "axis", "=", field("axis", $.nonzero_axis_integer))),
        ")",
      ),
    cast_expression: ($) =>
      seq("cast", "(", field("value", $.expression), ",", field("precision", $.identifier), ")"),
    unary_transform_expression: ($) =>
      prec(1, seq(field("transform", choice("jit", "realize", "copy")), "(", $.expression, ")")),
    bare_unary_transform: () => prec(-1, choice("realize", "copy")),
    quote_expression: ($) =>
      seq(field("form", choice("quote", "unquote", "splice")), "(", $.expression, ")"),

    primary_expression: ($) =>
      choice(
        $.identifier,
        $.type_identifier,
        $.number,
        $.string,
        $.boolean,
        $.unit_expression,
        $.tuple_expression,
        $.parenthesized_expression,
        $.list_expression,
        $.record_expression,
      ),
    unit_expression: () => seq("(", ")"),
    parenthesized_expression: ($) => seq("(", $.expression, ")"),
    tuple_expression: ($) =>
      prec.dynamic(
        2,
        seq("(", $.expression, ",", optional(seq(commaSep1($.expression))), ")"),
      ),
    list_expression: ($) => seq("[", commaSep($.expression), "]"),
    record_expression: ($) =>
      seq(
        field("constructor", $.qualified_type_name),
        "{",
        commaSep(choice($.record_field, $.record_pun)),
        "}",
      ),
    record_field: ($) => seq(field("name", $.identifier), ":", field("value", $.expression)),
    record_pun: ($) => field("name", $.identifier),

    pattern: ($) => choice($.as_pattern, $.pattern_atom),
    as_pattern: ($) =>
      seq(field("name", $.identifier), "@", field("pattern", $.pattern)),
    pattern_atom: ($) =>
      choice(
        $.wildcard,
        $.identifier,
        $.pattern_number,
        $.negative_pattern_number,
        $.string,
        $.boolean,
        $.constructor_pattern,
        $.unit_pattern,
        $.tuple_pattern,
        $.parenthesized_pattern,
        $.record_pattern,
      ),
    constructor_pattern: ($) =>
      choice(
        field("name", $.qualified_type_name),
        seq(field("name", $.qualified_type_name), "(", commaSep1($.pattern), ")"),
      ),
    unit_pattern: () => seq("(", ")"),
    tuple_pattern: ($) => seq("(", $.pattern, ",", optional(commaSep1($.pattern)), ")"),
    parenthesized_pattern: ($) => seq("(", $.pattern, ")"),
    record_pattern: ($) =>
      seq(
        field("name", $.qualified_type_name),
        "{",
        commaSep(choice($.record_pattern_field, $.record_pattern_pun)),
        "}",
      ),
    record_pattern_field: ($) =>
      seq(field("name", $.identifier), ":", field("pattern", $.pattern)),
    record_pattern_pun: ($) => field("name", $.identifier),

    type_expression: ($) => choice($.function_type, $.non_arrow_type),
    function_type: ($) =>
      prec.right(PREC.arrow, seq($.non_arrow_type, "->", $.type_expression)),
    non_arrow_type: ($) =>
      choice(
        $.reference_type,
        $.tensor_type,
        $.tuple_type,
        $.unit_type,
        $.applied_type,
        $.parenthesized_type,
        $.infer_type,
        $.qualified_type_name,
        $.identifier,
      ),
    reference_type: ($) => seq("&", $.non_arrow_type),
    tensor_type: ($) =>
      seq("tensor", "[", commaSep1(choice($.dimension_expression, $.identifier)), "]"),
    dimension_expression: ($) => choice($.axis_integer, "*", seq("..", $.identifier)),
    applied_type: ($) =>
      prec(1, seq(field("name", $.qualified_type_name), "[", commaSep1($.type_expression), "]")),
    unit_type: () => seq("(", ")"),
    tuple_type: ($) =>
      seq("(", $.type_expression, ",", optional(commaSep1($.type_expression)), ")"),
    parenthesized_type: ($) => seq("(", $.type_expression, ")"),
    infer_type: () => "_",
    qualified_type_name: ($) =>
      seq($.type_identifier, repeat(seq(".", $.type_identifier))),

    boolean: () => choice("true", "false"),
    wildcard: () => "_",
    axis_integer: ($) => $._canonical_axis_integer,
    nonzero_axis_integer: ($) => $._canonical_nonzero_axis_integer,
    // Canonical numeric spelling is value-sensitive: it is the shortest
    // decimal that round-trips to the decoded f64. A regex can validate only
    // the token's shape, so the external scanner performs the same shortest
    // spelling check as the canonical Rust parser.
    number: ($) => $._canonical_number,
    pattern_number: ($) => $._canonical_pattern_number,
    negative_pattern_number: ($) => seq("-", $.pattern_number),
    string: () => token(seq('"', repeat(choice(/[^"\\\n\r]+/, /\\[nrt0"\\]/)), '"')),
    identifier: () => /[_a-z][_A-Za-z0-9]*/,
    single_upper_identifier: () => /[A-Z]/,
    type_identifier: () => /[A-Z][_A-Za-z0-9]*/,
    line_comment: () => token(seq("--", /[^\n\r]*/)),
    block_comment: () => token(seq("{-", repeat(choice(/[^-]/, /-[^}]/)), "-}")),
  },
});

function commaSep(rule) {
  return optional(commaSep1(rule));
}

function commaSep1(rule) {
  return seq(rule, repeat(seq(",", rule)));
}

function semicolonSep1(rule) {
  return seq(rule, repeat(seq(";", rule)));
}

function commaSepNoTrail1(rule) {
  return seq(rule, repeat(seq(",", rule)));
}
