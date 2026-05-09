(line_comment) @comment.line
(block_comment) @comment.block

[
  "def"
  "sig"
  "property"
  "forall"
  "where"
  "let"
  "in"
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
  "jit"
  "realize"
  "copy"
  "par"
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
(call_expression
  function: (expression
    (primary_expression
      (identifier) @function.call)))
(call_expression
  function: (expression
    (field_expression field: (identifier) @function.call)))
(parameter name: (identifier) @parameter)
(field_expression field: (identifier) @property)
(record_field name: (identifier) @property)
(record_pattern_field name: (identifier) @property)
(record_expression constructor: (type_identifier) @type)
(record_pattern name: (type_identifier) @type)
(constructor_pattern name: (type_identifier) @type)
((identifier) @type.builtin
  (#match? @type.builtin "^(f16|f32|f64|i8|i16|i32|i64|u8|u16|u32|u64|bool)$"))
(type_identifier) @type
(identifier) @variable
(string) @string
(number) @number
(boolean) @boolean
