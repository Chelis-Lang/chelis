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
    $._canonical_min_integer_magnitude,
    $._canonical_min_pattern_magnitude,
    $._canonical_string,
    $._canonical_identifier,
    $._canonical_record_field_name,
    $._canonical_record_pattern_field_name,
    $._canonical_then,
    $._canonical_else,
    $._property_body_colon,
    $._whitespace,
    $._canonical_declaration_end,
    $._canonical_block_binding_end,
    $._canonical_block_expression_end,
  ],

  extras: ($) => [$._whitespace, $.line_comment, $.block_comment],
  word: ($) => $._canonical_identifier,

  conflicts: ($) => [
    [$.tuple_expression, $.parenthesized_expression],
    [$.tuple_type, $.parenthesized_type],
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
    [$.non_lambda_expression, $._open_form_expression],
    [$.expression, $._open_form_expression],
  ],

  rules: {
    source_file: ($) =>
      optional(
        seq(
          $.declaration,
          repeat(seq($._canonical_declaration_end, $.declaration)),
          optional($._canonical_declaration_end),
        ),
      ),

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
        optional(","),
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
        optional(field("quantifiers", $.type_binders)),
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
        choice(
          ":",
          seq(
            "where",
            commaSep1(field("precondition", $._property_precondition)),
            $._property_body_colon,
          ),
        ),
        field("body", $.expression),
        repeat($.property_option),
      ),
    property_option: ($) =>
      choice(
        seq(
          "with",
          choice("tolerance", "seed", "samples"),
          "=",
          field("value", $.expression),
        ),
        seq("with", "contract", "=", field("value", $.string)),
      ),

    function_definition: ($) =>
      seq(
        "def",
        field("name", $.identifier),
        optional(field("quantifiers", $.type_binders)),
        "(",
        commaSep($.parameter),
        ")",
        optional(seq("->", field("return_type", $.type_expression))),
        optional($.effect_clause),
        "=",
        field("body", $.expression),
      ),
    // spec/02-surf-syntax.md §P4b/§P4c: an unkinded binder list, each entry
    // optionally bounded by one of the three closed dtype families.
    type_binders: ($) => seq("[", commaSep1($.type_binder), "]"),
    type_binder: ($) =>
      seq(
        field("name", $.identifier),
        optional(seq(":", field("bound", $.dtype_family))),
      ),
    dtype_family: ($) => choice("Float", "Int", "Numeric"),
    parameter: ($) =>
      seq(
        field("name", $.value_identifier),
        optional(seq(":", field("type", $.type_expression))),
      ),
    typed_parameter: ($) =>
      seq(field("name", $.value_identifier), ":", field("type", $.type_expression)),
    effect_clause: ($) => seq("!", "{", commaSep($.effect_expression), "}"),
    effect_expression: ($) =>
      choice(
        "Diff",
        "Accum",
        "IO",
        "Test",
        seq("Resource", "(", $.string, optional(","), ")"),
      ),

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

    expression: ($) => choice($.lambda_expression, $.non_lambda_expression),
    non_lambda_expression: ($) =>
      choice(
        $.pipe_expression,
        $._non_pipe_operand,
      ),
    _non_pipe_operand: ($) =>
      choice(
        $.if_expression,
        $.match_expression,
        $.with_handler_expression,
        $.block_expression,
        $.par_expression,
        $.do_expression,
        $.record_update_expression,
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

    _pipe_operand: ($) => choice(
      $.with_handler_expression, $.block_expression, $.par_expression,
      $.do_expression, alias($._pipe_call_expression, $.call_expression),
      $.transform_expression, $.quote_expression, $.primary_expression,
    ),
    _open_form_expression: ($) => choice($.lambda_expression, $._non_pipe_operand),

    // Postfix access outside a delimited argument would mix with the pipe.
    _pipe_call_expression: ($) => prec.left(PREC.call, seq(
      field("function", choice($.primary_expression, $.transform_expression, $.quote_expression)),
      $.call_arguments,
    )),

    if_expression: ($) =>
      seq(
        "if",
        field("condition", $._open_form_expression),
        alias($._canonical_then, "then"),
        field("consequence", $._open_form_expression),
        alias($._canonical_else, "else"),
        field("alternative", $._open_form_expression),
      ),
    match_expression: ($) =>
      seq("match", field("value", $._open_form_expression), "with", "{", repeat1($.match_arm), "}"),
    match_arm: ($) =>
      seq(
        "|",
        field("pattern", $.pattern),
        optional(seq("if", field("guard", $.expression))),
        "=>",
        field("body", $.expression),
      ),
    lambda_expression: ($) =>
      seq("fn", "(", commaSep($.parameter), ")", "->", field("body", $._open_form_expression)),

    with_handler_expression: ($) =>
      seq(
        "with",
        field("handler", "device"),
        "(",
        field("argument", $.expression),
        optional(","),
        ")",
        field("body", $.handler_block_expression),
      ),
    handler_block_expression: ($) =>
      seq(
        "{",
        choice(
          seq(
            prec.dynamic(1, $.expression),
            optional($._canonical_block_expression_end),
          ),
          seq(
            repeat1($.let_binding),
            field("result", prec.dynamic(1, $.expression)),
            optional($._canonical_block_expression_end),
          ),
        ),
        "}",
      ),

    block_expression: ($) =>
      seq(
        "{",
        repeat1($.let_binding),
        field("result", prec.dynamic(1, $.expression)),
        optional($._canonical_block_expression_end),
        "}",
      ),
    let_binding: ($) =>
      seq(
        field("pattern", $.let_pattern),
        optional(seq(":", field("type", $.type_expression))),
        "=",
        field("value", $.expression),
        $._canonical_block_binding_end,
      ),
    let_pattern: ($) =>
      choice(
        $.value_identifier,
        $.wildcard,
        seq("(", ")"),
        seq("(", $.let_pattern, ",", optional(commaSep1($.let_pattern)), ")"),
      ),

    par_expression: ($) => seq("par", "{", blockExpressionSep1($, $.expression), "}"),
    do_expression: ($) => seq("do", "{", blockExpressionSep1($, $.expression), "}"),

    record_update_expression: ($) =>
      prec.right(
        PREC.recordUpdate,
        seq(
          field("value", $._open_form_expression),
          "with",
          "{",
          commaSep1(choice($.record_field, $.record_pun)),
          "}",
        ),
      ),

    pipe_expression: ($) =>
      prec.left(
        PREC.pipe,
        seq($._pipe_operand, repeat1($.pipe_stage)),
      ),
    pipe_stage: ($) =>
      prec.left(
        PREC.pipe,
        seq(
          "|>",
          field(
            "value",
            choice(
              $.cast_pipe_stage,
              $._pipe_operand,
            ),
          ),
        ),
      ),
    // The Rust parser admits `x |> cast(f32)` and each named rung, such as
    // `x |> cast_trunc(i32)`, as call-stage sugar for `cast(x, f32)` /
    // `cast_trunc(x, i32)`. Keep this
    // syntax scoped to pipe stages; ordinary cast expressions still require
    // both the value and precision arguments below.
    cast_pipe_stage: ($) =>
      seq(
        field("mode", choice("cast", "cast_trunc", "cast_saturate", "cast_wrap")),
        "(",
        field("precision", $.identifier),
        ")",
      ),
    logical_or_expression: ($) =>
      prec.left(
        PREC.or,
        seq(
          $._logical_and_level,
          repeat1(seq(field("operator", "||"), $._logical_and_level)),
        ),
      ),
    _logical_or_level: ($) => choice($.logical_or_expression, $._logical_and_level),
    logical_and_expression: ($) =>
      prec.left(
        PREC.and,
        seq(
          $._equality_level,
          repeat1(seq(field("operator", "&&"), $._equality_level)),
        ),
      ),
    _logical_and_level: ($) => choice($.logical_and_expression, $._equality_level),
    equality_expression: ($) =>
      prec(
        PREC.equality,
        seq(
          $._comparison_level,
          field("operator", choice("==", "!=")),
          $._comparison_level,
        ),
      ),
    _equality_level: ($) => choice($.equality_expression, $._comparison_level),
    comparison_expression: ($) =>
      prec(
        PREC.compare,
        seq(
          $._additive_level,
          field("operator", choice("<", ">", "<=", ">=")),
          $._additive_level,
        ),
      ),
    _comparison_level: ($) => choice($.comparison_expression, $._additive_level),
    additive_expression: ($) =>
      prec.left(
        PREC.add,
        seq(
          $._multiplicative_level,
          repeat1(seq(field("operator", choice("+", "-")), $._multiplicative_level)),
        ),
      ),
    _additive_level: ($) => choice($.additive_expression, $._multiplicative_level),
    multiplicative_expression: ($) =>
      prec.left(
        PREC.multiply,
        seq(
          $._unary_level,
          repeat1(seq(field("operator", choice("*", "/", "%")), $._unary_level)),
        ),
      ),
    _multiplicative_level: ($) => choice($.multiplicative_expression, $._unary_level),
    unary_expression: ($) =>
      prec.right(
        PREC.unary,
        choice(
          seq(field("operator", "-"), choice($._unary_level, $._canonical_min_integer_magnitude)),
          seq(field("operator", choice("!", "&")), $._unary_level),
        ),
      ),
    _unary_level: ($) => choice($.unary_expression, $._annotation_level),
    annotation_expression: ($) =>
      prec.left(
        PREC.annotation,
        seq(field("value", $._postfix_expression), ":", field("type", $.type_expression)),
      ),
    _annotation_level: ($) => choice($.annotation_expression, $._postfix_expression),
    _postfix_expression: ($) =>
      choice(
        $.call_expression,
        $.field_expression,
        $.transform_expression,
        $.quote_expression,
        $.primary_expression,
      ),

    call_expression: ($) =>
      choice(
        prec.left(
          PREC.call,
          seq(
            field(
              "function",
              choice(
                $.primary_expression,
                $.callable_access_expression,
                $.transform_expression,
                $.quote_expression,
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
            $.call_arguments,
            repeat($.access_step),
          ),
        ),
      ),
    call_arguments: ($) => seq("(", commaSep($.expression), ")"),
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
        optional(","),
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
        optional(","),
        ")",
      ),
    cast_expression: ($) =>
      seq(
        field("mode", choice("cast", "cast_trunc", "cast_saturate", "cast_wrap")),
        "(",
        field("value", $.expression),
        ",",
        field("precision", $.identifier),
        optional(","),
        ")",
      ),
    unary_transform_expression: ($) =>
      prec(
        1,
        seq(
          field("transform", choice("jit", "realize", "copy")),
          "(",
          $.expression,
          optional(","),
          ")",
        ),
      ),
    bare_unary_transform: () => prec(-1, choice("realize", "copy")),
    quote_expression: ($) =>
      seq(
        field("form", choice("quote", "unquote", "splice")),
        "(",
        $.expression,
        optional(","),
        ")",
      ),

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
    record_field: ($) =>
      seq(
        field("name", alias($._canonical_record_field_name, $.identifier)),
        ":",
        field("value", $.expression),
      ),
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
      seq(
        field("name", alias($._canonical_record_pattern_field_name, $.identifier)),
        ":",
        field("pattern", $.pattern),
      ),
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
    // A type-application argument admits an integer literal: the concrete
    // dimension instantiation of a dimension-parameterized ADT
    // (`Frame[2]`, chelis#940). Bare type positions do not (chelis#1179).
    applied_type: ($) =>
      prec(
        1,
        seq(
          field("name", $.qualified_type_name),
          "[",
          commaSep1(choice($.type_expression, $.axis_integer)),
          "]",
        ),
      ),
    unit_type: () => "unit",
    tuple_type: ($) =>
      seq("(", $.type_expression, ",", optional(commaSep1($.type_expression)), ")"),
    parenthesized_type: ($) => seq("(", $.type_expression, ")"),
    infer_type: () => "_",
    qualified_type_name: ($) =>
      seq($.type_identifier, repeat(seq(".", $.type_identifier))),

    // The comma/colon delimiters already group a complete property
    // precondition. Canonical Surf therefore rejects a redundant pair around
    // the whole clause while retaining parentheses inside the expression,
    // such as `(a + b) < c`.
    _property_precondition: ($) =>
      choice(
        $.lambda_expression,
        $.if_expression,
        $.match_expression,
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
        $.identifier,
        $.type_identifier,
        $.number,
        $.string,
        $.boolean,
        $.unit_expression,
        $.tuple_expression,
        $.list_expression,
        $.record_expression,
      ),

    boolean: () => choice("true", "false"),
    wildcard: () => "_",
    axis_integer: ($) => $._canonical_axis_integer,
    nonzero_axis_integer: ($) => $._canonical_nonzero_axis_integer,
    // Numeric input is value-sensitive. The scanner accepts the deliberately
    // small alias family (radices, digit separators, and exponent spellings)
    // while rejecting malformed or lossy forms; `chelis fmt` owns canonical
    // output spelling.
    number: ($) => $._canonical_number,
    pattern_number: ($) => $._canonical_pattern_number,
    negative_pattern_number: ($) =>
      seq("-", choice($.pattern_number, $._canonical_min_pattern_magnitude)),
    string: ($) => $._canonical_string,
    identifier: ($) => $._canonical_identifier,
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
  return seq(rule, repeat(seq(",", rule)), optional(","));
}

function semicolonSep1(rule) {
  return seq(rule, repeat(seq(";", rule)), optional(";"));
}

function blockExpressionSep1($, rule) {
  const bounded = seq(rule, optional($._canonical_block_expression_end));
  return seq(bounded, repeat(seq(";", bounded)), optional(";"));
}

function commaSepNoTrail1(rule) {
  return seq(rule, repeat(seq(",", rule)));
}
