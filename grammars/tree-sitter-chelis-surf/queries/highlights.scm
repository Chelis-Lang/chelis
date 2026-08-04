(line_comment) @comment.line
(block_comment) @comment.block

[
  "def"
  "sig"
  "property"
  "forall"
  "where"
  "type"
  "opaque"
  "invariant"
  "macro"
  "dim"
  "module"
  "import"
  "export"
  "match"
  "with"
  "fn"
  "if"
  "then"
  "else"
  "grad"
  "vmap"
  "jit"
  "cast"
  "realize"
  "copy"
  "par"
  "do"
  "quote"
  "unquote"
  "splice"
] @keyword

"tensor" @type.keyword

[
  "->"
  "=>"
  "|>"
  "="
  "+"
  "-"
  "*"
  "/"
  "%"
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  ":"
  "."
] @operator

(module_declaration
  name: (module_path
    (type_identifier) @namespace))

(import_declaration
  module: (module_path
    (type_identifier) @namespace))

(function_definition name: (identifier) @function)
(property_declaration name: (identifier) @function)
(signature_declaration name: (identifier) @function)
(macro_definition name: (identifier) @function.macro)
(call_expression
  function: (primary_expression
    (identifier) @function.call))
(call_expression
  function: (callable_access_expression
    field: (identifier) @function.call))
(parameter name: (value_identifier (identifier) @parameter))
(typed_parameter name: (value_identifier (identifier) @parameter))
(access_step field: (identifier) @property)
(record_field name: (identifier) @property)
(record_pattern_field name: (identifier) @property)
(record_expression constructor: (qualified_type_name (type_identifier) @type))
(record_pattern name: (qualified_type_name (type_identifier) @type))
(constructor_pattern name: (qualified_type_name (type_identifier) @type))
((identifier) @type.builtin
  (#match? @type.builtin "^(f16|bf16|f32|f64|int8|int16|int32|int64|bool|string)$"))
(type_identifier) @type
(identifier) @variable
(string) @string
(number) @number
(pattern_number) @number
(axis_integer) @number
(nonzero_axis_integer) @number
(boolean) @boolean
