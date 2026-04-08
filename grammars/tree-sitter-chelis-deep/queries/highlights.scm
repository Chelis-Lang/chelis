(comment) @comment
(number_literal) @number
(string_literal) @string
(keyword) @constant.builtin
(map_entry (symbol) @property)
(map_entry (keyword) @property)
(list
  .
  (expr (symbol) @_form)
  (#match? @_form "^(def|sig)$")
  (expr (map))
  (expr (symbol) @function))
(list
  .
  (expr (symbol) @_form)
  (#match? @_form "^(var|load)$")
  (expr (map))
  (expr (symbol) @parameter))
(list
  .
  (expr (symbol) @_form)
  (#eq? @_form "lit")
  (expr (map))
  (expr (symbol) @type))
(symbol) @identifier
(list . (expr (symbol) @function.special))
