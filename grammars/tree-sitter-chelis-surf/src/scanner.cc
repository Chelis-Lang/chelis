#include <tree_sitter/parser.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>

namespace {

enum TokenType {
  CANONICAL_NUMBER,
  CANONICAL_PATTERN_NUMBER,
  CANONICAL_AXIS_INTEGER,
  CANONICAL_NONZERO_AXIS_INTEGER,
  CANONICAL_MIN_INTEGER_MAGNITUDE,
  CANONICAL_MIN_PATTERN_MAGNITUDE,
  CANONICAL_STRING,
  CANONICAL_IDENTIFIER,
  CANONICAL_RECORD_FIELD_NAME,
  CANONICAL_RECORD_PATTERN_FIELD_NAME,
  CANONICAL_PIPE_LAMBDA_FN,
  CANONICAL_THEN,
  CANONICAL_ELSE,
  PROPERTY_BODY_COLON,
  WHITESPACE,
  CANONICAL_DECLARATION_END,
  CANONICAL_BLOCK_BINDING_END,
  CANONICAL_BLOCK_EXPRESSION_END,
};

bool is_digit(int32_t value) { return value >= '0' && value <= '9'; }

bool is_hex_digit(int32_t value) {
  return is_digit(value) || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

bool is_identifier_start(int32_t value) {
  return value == '_' || (value >= 'a' && value <= 'z');
}

bool is_identifier_continue(int32_t value) {
  return is_identifier_start(value) || (value >= 'A' && value <= 'Z') ||
         is_digit(value);
}

std::string consume_identifier(TSLexer *lexer) {
  std::string identifier;
  while (is_identifier_continue(lexer->lookahead)) {
    identifier.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
  }
  return identifier;
}

bool is_reserved_identifier(const std::string &identifier) {
  return identifier == "_" || identifier == "def" ||
         identifier == "sig" || identifier == "type" || identifier == "dim" ||
         identifier == "macro" || identifier == "match" || identifier == "with" ||
         identifier == "fn" || identifier == "module" || identifier == "import" ||
         identifier == "if" || identifier == "then" || identifier == "else" ||
         identifier == "grad" || identifier == "vmap" || identifier == "jit" ||
         identifier == "realize" || identifier == "copy" ||
         identifier == "tensor" || identifier == "cast" ||
         identifier == "cast_trunc" ||
         identifier == "export" || identifier == "par" || identifier == "do" ||
         identifier == "quote" || identifier == "unquote" ||
         identifier == "splice" || identifier == "true" || identifier == "false" ||
         identifier == "effect" || identifier == "handler" ||
         identifier == "perform" || identifier == "resume" ||
         identifier == "borrow";
}

void skip_line_comment(TSLexer *lexer) {
  while (!lexer->eof(lexer) && lexer->lookahead != '\n' &&
         lexer->lookahead != '\r') {
    lexer->advance(lexer, false);
  }
}

bool skip_trivia_for_lookahead(TSLexer *lexer) {
  while (!lexer->eof(lexer)) {
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
           lexer->lookahead == '\n' || lexer->lookahead == '\r') {
      lexer->advance(lexer, false);
    }
    if (lexer->lookahead == '-') {
      lexer->advance(lexer, false);
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      skip_line_comment(lexer);
      continue;
    }
    if (lexer->lookahead == '{') {
      lexer->advance(lexer, false);
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      unsigned depth = 1;
      while (!lexer->eof(lexer) && depth > 0) {
        if (lexer->lookahead == '{') {
          lexer->advance(lexer, false);
          if (lexer->lookahead == '-') {
            lexer->advance(lexer, false);
            ++depth;
          }
        } else if (lexer->lookahead == '-') {
          lexer->advance(lexer, false);
          if (lexer->lookahead == '}') {
            lexer->advance(lexer, false);
            --depth;
          }
        } else {
          lexer->advance(lexer, false);
        }
      }
      if (depth != 0) {
        return false;
      }
      continue;
    }
    return true;
  }
  return false;
}

bool consume_arrow_after_trivia(TSLexer *lexer) {
  while (!lexer->eof(lexer)) {
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
           lexer->lookahead == '\n' || lexer->lookahead == '\r') {
      lexer->advance(lexer, false);
    }
    if (lexer->lookahead == '-') {
      lexer->advance(lexer, false);
      if (lexer->lookahead == '>') {
        lexer->advance(lexer, false);
        return true;
      }
      if (lexer->lookahead != '-') {
        return false;
      }
      lexer->advance(lexer, false);
      skip_line_comment(lexer);
      continue;
    }
    if (lexer->lookahead != '{') {
      return false;
    }
    lexer->advance(lexer, false);
    if (lexer->lookahead != '-') {
      return false;
    }
    lexer->advance(lexer, false);
    unsigned depth = 1;
    while (!lexer->eof(lexer) && depth > 0) {
      if (lexer->lookahead == '{') {
        lexer->advance(lexer, false);
        if (lexer->lookahead == '-') {
          lexer->advance(lexer, false);
          ++depth;
        }
      } else if (lexer->lookahead == '-') {
        lexer->advance(lexer, false);
        if (lexer->lookahead == '}') {
          lexer->advance(lexer, false);
          --depth;
        }
      } else {
        lexer->advance(lexer, false);
      }
    }
    if (depth != 0) {
      return false;
    }
  }
  return false;
}

bool finish_identifier(TSLexer *lexer, const bool *valid_symbols,
                       const std::string &name) {
  lexer->mark_end(lexer);
  if (is_reserved_identifier(name)) {
    return false;
  }

  const bool expression_field = valid_symbols[CANONICAL_RECORD_FIELD_NAME];
  const bool pattern_field = valid_symbols[CANONICAL_RECORD_PATTERN_FIELD_NAME];
  if (expression_field || pattern_field) {
    bool explicit_field = skip_trivia_for_lookahead(lexer) && lexer->lookahead == ':';
    bool redundant_pun = false;
    if (explicit_field) {
      lexer->advance(lexer, false);
      if (skip_trivia_for_lookahead(lexer) &&
          is_identifier_start(lexer->lookahead)) {
        const std::string value_name = consume_identifier(lexer);
        redundant_pun = value_name == name && skip_trivia_for_lookahead(lexer) &&
                        (lexer->lookahead == ',' || lexer->lookahead == '}');
      }
    }
    if (explicit_field && !redundant_pun) {
      lexer->result_symbol = pattern_field && !expression_field
                                 ? CANONICAL_RECORD_PATTERN_FIELD_NAME
                                 : CANONICAL_RECORD_FIELD_NAME;
      return true;
    }
  }

  if (valid_symbols[CANONICAL_IDENTIFIER]) {
    lexer->result_symbol = CANONICAL_IDENTIFIER;
    return true;
  }
  return false;
}

bool scan_identifier(TSLexer *lexer, const bool *valid_symbols) {
  if (!is_identifier_start(lexer->lookahead)) {
    return false;
  }
  return finish_identifier(lexer, valid_symbols, consume_identifier(lexer));
}

bool scan_pipe_lambda_fn(TSLexer *lexer, const bool *valid_symbols) {
  if (lexer->lookahead != 'f') {
    return false;
  }
  std::string name = "f";
  lexer->advance(lexer, false);
  if (lexer->lookahead != 'n') {
    while (is_identifier_continue(lexer->lookahead)) {
      name.push_back(static_cast<char>(lexer->lookahead));
      lexer->advance(lexer, false);
    }
    return finish_identifier(lexer, valid_symbols, name);
  }
  name.push_back('n');
  lexer->advance(lexer, false);
  if (is_identifier_continue(lexer->lookahead)) {
    while (is_identifier_continue(lexer->lookahead)) {
      name.push_back(static_cast<char>(lexer->lookahead));
      lexer->advance(lexer, false);
    }
    return finish_identifier(lexer, valid_symbols, name);
  }
  lexer->mark_end(lexer);
  lexer->result_symbol = CANONICAL_PIPE_LAMBDA_FN;

  if (!skip_trivia_for_lookahead(lexer) || lexer->lookahead != '(') {
    return true;
  }
  lexer->advance(lexer, false);
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }
  const std::string parameter = consume_identifier(lexer);
  if (!skip_trivia_for_lookahead(lexer) || lexer->lookahead != ')') {
    return true;  // typed or multiple parameters are never call-stage aliases
  }
  lexer->advance(lexer, false);
  if (!consume_arrow_after_trivia(lexer)) {
    return true;
  }
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }

  std::string callable = consume_identifier(lexer);
  if (!skip_trivia_for_lookahead(lexer)) {
    return true;
  }
  while (lexer->lookahead == '.') {
    lexer->advance(lexer, false);
    if (!skip_trivia_for_lookahead(lexer) ||
        !(is_identifier_start(lexer->lookahead) ||
          (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
      return true;
    }
    callable = consume_identifier(lexer);
    if (!skip_trivia_for_lookahead(lexer)) {
      return true;
    }
  }
  if (lexer->lookahead != '(') {
    return true;
  }
  lexer->advance(lexer, false);
  if (!skip_trivia_for_lookahead(lexer) ||
      !(is_identifier_start(lexer->lookahead) ||
        (lexer->lookahead >= 'A' && lexer->lookahead <= 'Z'))) {
    return true;
  }
  const std::string first_argument = consume_identifier(lexer);
  if (first_argument != parameter || !skip_trivia_for_lookahead(lexer)) {
    return true;
  }

  const bool direct_argument_end = lexer->lookahead == ')' || lexer->lookahead == ',';
  const bool transform_alias =
      (callable == "realize" || callable == "copy") && lexer->lookahead == ')';
  const bool cast_alias =
      (callable == "cast" || callable == "cast_trunc") &&
      lexer->lookahead == ',';
  if (transform_alias || cast_alias ||
      (callable != "realize" && callable != "copy" && callable != "cast" &&
       callable != "cast_trunc" &&
       direct_argument_end)) {
    return false;
  }

  lexer->result_symbol = CANONICAL_PIPE_LAMBDA_FN;
  return true;
}

bool scan_conditional_keyword(TSLexer *lexer, const bool *valid_symbols) {
  if (lexer->lookahead != 't' && lexer->lookahead != 'e') {
    return false;
  }
  const std::string name = consume_identifier(lexer);
  lexer->mark_end(lexer);
  if (name == "then" && valid_symbols[CANONICAL_THEN]) {
    lexer->result_symbol = CANONICAL_THEN;
    return true;
  }
  if (name == "else" && valid_symbols[CANONICAL_ELSE]) {
    lexer->result_symbol = CANONICAL_ELSE;
    return true;
  }
  return finish_identifier(lexer, valid_symbols, name);
}

bool scan_whitespace_or_binding_end(TSLexer *lexer, bool whitespace,
                                    bool declaration_end,
                                    bool block_binding_end,
                                    bool block_expression_end) {
  bool saw_whitespace = false;
  bool saw_newline = false;
  while (lexer->lookahead == ' ' || lexer->lookahead == '\t' ||
         lexer->lookahead == '\n' || lexer->lookahead == '\r') {
    saw_whitespace = true;
    if (!saw_newline &&
        (lexer->lookahead == '\n' || lexer->lookahead == '\r')) {
      if (lexer->lookahead == '\r') {
        lexer->advance(lexer, false);
        if (lexer->lookahead == '\n') {
          lexer->advance(lexer, false);
        }
      } else {
        lexer->advance(lexer, false);
      }
      saw_newline = true;
      continue;
    }
    lexer->advance(lexer, false);
  }
  if (!saw_whitespace) {
    return false;
  }
  lexer->mark_end(lexer);
  if ((declaration_end || block_binding_end || block_expression_end) &&
      saw_newline) {
    const bool has_next = skip_trivia_for_lookahead(lexer);
    const bool starts_bar = has_next && lexer->lookahead == '|';
    bool continues_pipe = false;
    if (starts_bar) {
      lexer->advance(lexer, false);
      continues_pipe = lexer->lookahead == '>';
    }
    // Declaration bodies are permissive: a leading variant `|` is not a
    // declaration start, so it continues just like every other non-start.
    // The closed block boundary admits only the exact `|>` token.
    const bool continues_declaration_bar = declaration_end && starts_bar;
    bool continues_property = false;
    if (declaration_end && has_next && lexer->lookahead == 'w') {
      continues_property = consume_identifier(lexer) == "with";
    }
    if (!continues_pipe && !continues_declaration_bar && !continues_property) {
      lexer->result_symbol = block_expression_end
                                 ? CANONICAL_BLOCK_EXPRESSION_END
                             : block_binding_end ? CANONICAL_BLOCK_BINDING_END
                                                 : CANONICAL_DECLARATION_END;
      return true;
    }
  }
  if (whitespace) {
    lexer->result_symbol = WHITESPACE;
    return true;
  }
  return false;
}

template <typename DigitPredicate>
bool consume_digit_run(TSLexer *lexer, DigitPredicate is_valid_digit,
                       std::string &spelling, std::string &clean,
                       bool &saw_underscore, bool already_has_digit = false) {
  bool previous_was_digit = already_has_digit;
  bool saw_digit = already_has_digit;
  while (is_valid_digit(lexer->lookahead) || lexer->lookahead == '_') {
    const char ch = static_cast<char>(lexer->lookahead);
    spelling.push_back(ch);
    lexer->advance(lexer, false);
    if (ch == '_') {
      saw_underscore = true;
      if (!previous_was_digit || !is_valid_digit(lexer->lookahead)) {
        return false;
      }
      previous_was_digit = false;
      continue;
    }
    clean.push_back(ch);
    saw_digit = true;
    previous_was_digit = true;
  }
  return saw_digit && previous_was_digit;
}

bool is_valid_unicode_escape(const std::string &digits) {
  if (digits.empty() || digits.size() > 6) {
    return false;
  }
  uint32_t scalar = 0;
  const auto parsed = std::from_chars(
      digits.data(), digits.data() + digits.size(), scalar, 16);
  if (parsed.ec != std::errc() || parsed.ptr != digits.data() + digits.size()) {
    return false;
  }
  return scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff);
}

bool scan_string(TSLexer *lexer) {
  lexer->advance(lexer, false);  // opening quote
  while (!lexer->eof(lexer)) {
    const int32_t lookahead = lexer->lookahead;
    if (lookahead == '"') {
      lexer->advance(lexer, false);
      lexer->mark_end(lexer);
      lexer->result_symbol = CANONICAL_STRING;
      return true;
    }
    if (lookahead == '\\') {
      lexer->advance(lexer, false);
      if (lexer->eof(lexer)) {
        return false;
      }
      if (lexer->lookahead == '\\' || lexer->lookahead == '"' ||
          lexer->lookahead == 'n' || lexer->lookahead == 't' ||
          lexer->lookahead == 'r' || lexer->lookahead == '0') {
        lexer->advance(lexer, false);
        continue;
      }
      if (lexer->lookahead != 'u') {
        return false;
      }
      lexer->advance(lexer, false);
      if (lexer->lookahead != '{') {
        return false;
      }
      lexer->advance(lexer, false);
      std::string digits;
      while (is_hex_digit(lexer->lookahead)) {
        digits.push_back(static_cast<char>(lexer->lookahead));
        lexer->advance(lexer, false);
      }
      if (lexer->lookahead != '}' || !is_valid_unicode_escape(digits)) {
        return false;
      }
      lexer->advance(lexer, false);
      continue;
    }
    if (lookahead < 0x20 || (lookahead >= 0x7f && lookahead <= 0x9f)) {
      return false;
    }
    lexer->advance(lexer, false);
  }
  return false;
}

bool parse_float_classic(const std::string &text, double &value) {
  // Callers pass strings whose decimal grammar was already validated. Some
  // standard libraries set `failbit` while still returning the correctly
  // rounded subnormal value. libstdc++ 10 also saturates overflow at
  // `double::max()` while setting `failbit`, whereas libc++ returns infinity.
  // Reject the saturating form explicitly; infinity still fails the caller's
  // `std::isfinite` check and underflow-to-zero is a finite decode.
  std::istringstream input(text);
  input.imbue(std::locale::classic());
  input >> std::noskipws >> value;
  return !input.bad() &&
         !(input.fail() && value == std::numeric_limits<double>::max());
}

bool is_integer_suffix(const std::string &suffix) {
  return suffix == "i8" || suffix == "i16" || suffix == "i32" ||
         suffix == "i64";
}

bool is_float_suffix(const std::string &suffix) {
  return suffix == "f16" || suffix == "bf16" || suffix == "f32" ||
         suffix == "f64";
}

bool parse_unsigned_integer(const std::string &digits, int base,
                            uint64_t &value) {
  const auto parsed = std::from_chars(
      digits.data(), digits.data() + digits.size(), value, base);
  return parsed.ec == std::errc() && parsed.ptr == digits.data() + digits.size();
}

bool scan_number(TSLexer *lexer, const bool *valid_symbols) {
  const bool expression_context = valid_symbols[CANONICAL_NUMBER];
  const bool pattern_context = valid_symbols[CANONICAL_PATTERN_NUMBER];
  const bool axis_context = valid_symbols[CANONICAL_AXIS_INTEGER];
  const bool nonzero_axis_context =
      valid_symbols[CANONICAL_NONZERO_AXIS_INTEGER];
  const bool min_integer_context =
      valid_symbols[CANONICAL_MIN_INTEGER_MAGNITUDE];
  const bool min_pattern_context =
      valid_symbols[CANONICAL_MIN_PATTERN_MAGNITUDE];
  if (!expression_context && !pattern_context && !axis_context &&
      !nonzero_axis_context && !min_integer_context && !min_pattern_context) {
    return false;
  }

  if (!is_digit(lexer->lookahead)) {
    return false;
  }

  std::string spelling;
  std::string clean;
  bool saw_underscore = false;
  bool is_radix = false;
  int radix = 10;

  if (lexer->lookahead == '0') {
    spelling.push_back('0');
    clean.push_back('0');
    lexer->advance(lexer, false);
    if (lexer->lookahead == 'x' || lexer->lookahead == 'X' ||
        lexer->lookahead == 'b' || lexer->lookahead == 'B') {
      const char prefix = static_cast<char>(lexer->lookahead);
      is_radix = true;
      radix = prefix == 'x' || prefix == 'X' ? 16 : 2;
      spelling.push_back(prefix);
      clean.clear();
      lexer->advance(lexer, false);
      const auto radix_digit = [radix](int32_t value) {
        return radix == 16 ? is_hex_digit(value)
                           : value == '0' || value == '1';
      };
      if (!consume_digit_run(lexer, radix_digit, spelling, clean,
                             saw_underscore)) {
        return false;
      }
      if (radix == 16) {
        for (const std::string &float_suffix :
             {"bf16", "f32", "f64", "f16"}) {
          if (clean.size() > float_suffix.size() &&
              clean.compare(clean.size() - float_suffix.size(),
                            float_suffix.size(), float_suffix) == 0) {
            return false;
          }
        }
      }
    } else if (!consume_digit_run(lexer, is_digit, spelling, clean,
                                  saw_underscore, true)) {
      return false;
    }
  } else if (!consume_digit_run(lexer, is_digit, spelling, clean,
                                saw_underscore)) {
    return false;
  }

  bool has_decimal_point = false;
  bool has_exponent = false;
  bool field_access_dot = false;
  if (!is_radix && lexer->lookahead == '.') {
    lexer->mark_end(lexer);
    lexer->advance(lexer, false);
    if (!is_digit(lexer->lookahead)) {
      field_access_dot = true;
    } else {
      has_decimal_point = true;
      spelling.push_back('.');
      clean.push_back('.');
      if (!consume_digit_run(lexer, is_digit, spelling, clean,
                             saw_underscore)) {
        return false;
      }
    }
  }

  if (!field_access_dot && !is_radix &&
      (lexer->lookahead == 'e' || lexer->lookahead == 'E')) {
    has_exponent = true;
    spelling.push_back(static_cast<char>(lexer->lookahead));
    clean.push_back(static_cast<char>(lexer->lookahead));
    lexer->advance(lexer, false);
    if (lexer->lookahead == '-' || lexer->lookahead == '+') {
      spelling.push_back(static_cast<char>(lexer->lookahead));
      clean.push_back(static_cast<char>(lexer->lookahead));
      lexer->advance(lexer, false);
    }
    if (!consume_digit_run(lexer, is_digit, spelling, clean,
                           saw_underscore)) {
      return false;
    }
  }

  std::string suffix;
  if (!field_access_dot) {
    while ((lexer->lookahead >= 'A' && lexer->lookahead <= 'Z') ||
           (lexer->lookahead >= 'a' && lexer->lookahead <= 'z') ||
           is_digit(lexer->lookahead) || lexer->lookahead == '_') {
      suffix.push_back(static_cast<char>(lexer->lookahead));
      lexer->advance(lexer, false);
    }
    lexer->mark_end(lexer);
  }

  const bool use_pattern = pattern_context && !expression_context;
  if (use_pattern && !suffix.empty()) {
    return false;
  }

  if (!is_radix && !has_decimal_point && !has_exponent &&
      clean == "9223372036854775808") {
    if (min_pattern_context && suffix.empty()) {
      lexer->result_symbol = CANONICAL_MIN_PATTERN_MAGNITUDE;
      return true;
    }
    if (min_integer_context && (suffix.empty() || suffix == "i64")) {
      lexer->result_symbol = CANONICAL_MIN_INTEGER_MAGNITUDE;
      return true;
    }
    return false;
  }

  if (!expression_context && !pattern_context &&
      (axis_context || nonzero_axis_context)) {
    uint64_t value = 0;
    const bool accepted_axis = suffix.empty() && !has_decimal_point &&
                               !has_exponent &&
                               parse_unsigned_integer(clean, radix, value) &&
                               value <=
                                   static_cast<uint64_t>(
                                       std::numeric_limits<int64_t>::max());
    if (!accepted_axis || (nonzero_axis_context && value == 0)) {
      return false;
    }
    lexer->result_symbol = nonzero_axis_context
                               ? CANONICAL_NONZERO_AXIS_INTEGER
                               : CANONICAL_AXIS_INTEGER;
    return true;
  }

  if (has_decimal_point || has_exponent) {
    if ((!suffix.empty() && !is_float_suffix(suffix)) || is_radix) {
      return false;
    }
    // Every finite decimal float body decodes to the value its canonical
    // spelling round-trips to, so the scanner admits the whole family and
    // leaves canonicalization to `chelis fmt` (spec/02 §P10, chelis#2119).
    // Only a non-finite decode is refused here.
    double value = 0.0;
    if (!parse_float_classic(clean, value) || !std::isfinite(value)) {
      return false;
    }
  } else {
    uint64_t value = 0;
    if (!parse_unsigned_integer(clean, radix, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    if (!suffix.empty() && !is_integer_suffix(suffix) &&
        !is_float_suffix(suffix)) {
      return false;
    }
    if (is_float_suffix(suffix)) {
      // spec/02 §P10a: an integer body under a float suffix keeps its integer
      // spelling, and no radix form carries a float suffix.
      if (is_radix || std::to_string(value) != clean) {
        return false;
      }
      // The body binds at the suffix width, so a magnitude that rounds to
      // infinity there is not a literal of that type. f16 is the only width an
      // integer body can overflow: the rest have finite ranges above
      // `int64::max`, which the bound above already refuses. 65520 is the
      // round-half-to-even boundary above f16's largest finite value, 65504.
      if (suffix == "f16" && value >= 65520) {
        return false;
      }
    } else if (!is_radix && std::to_string(value) != clean) {
      return false;
    }
  }

  lexer->result_symbol =
      use_pattern ? CANONICAL_PATTERN_NUMBER : CANONICAL_NUMBER;
  return true;
}

}  // namespace

extern "C" {

void *tree_sitter_chelis_surf_external_scanner_create() { return nullptr; }

void tree_sitter_chelis_surf_external_scanner_destroy(void *) {}

unsigned tree_sitter_chelis_surf_external_scanner_serialize(void *, char *) {
  return 0;
}

void tree_sitter_chelis_surf_external_scanner_deserialize(void *, const char *,
                                                          unsigned) {}

bool tree_sitter_chelis_surf_external_scanner_scan(void *, TSLexer *lexer,
                                                   const bool *valid_symbols) {
  if ((valid_symbols[WHITESPACE] ||
       valid_symbols[CANONICAL_DECLARATION_END] ||
       valid_symbols[CANONICAL_BLOCK_BINDING_END] ||
       valid_symbols[CANONICAL_BLOCK_EXPRESSION_END]) &&
      scan_whitespace_or_binding_end(
          lexer, valid_symbols[WHITESPACE],
          valid_symbols[CANONICAL_DECLARATION_END],
          valid_symbols[CANONICAL_BLOCK_BINDING_END],
          valid_symbols[CANONICAL_BLOCK_EXPRESSION_END])) {
    return true;
  }
  if (valid_symbols[CANONICAL_STRING] && lexer->lookahead == '"') {
    return scan_string(lexer);
  }
  if (valid_symbols[PROPERTY_BODY_COLON] && lexer->lookahead == ':') {
    lexer->advance(lexer, false);
    lexer->mark_end(lexer);
    lexer->result_symbol = PROPERTY_BODY_COLON;
    return true;
  }
  if (valid_symbols[CANONICAL_PIPE_LAMBDA_FN] && lexer->lookahead == 'f') {
    return scan_pipe_lambda_fn(lexer, valid_symbols);
  }
  if ((valid_symbols[CANONICAL_THEN] || valid_symbols[CANONICAL_ELSE]) &&
      (lexer->lookahead == 't' || lexer->lookahead == 'e')) {
    return scan_conditional_keyword(lexer, valid_symbols);
  }
  if (is_identifier_start(lexer->lookahead) &&
      (valid_symbols[CANONICAL_IDENTIFIER] ||
       valid_symbols[CANONICAL_RECORD_FIELD_NAME] ||
       valid_symbols[CANONICAL_RECORD_PATTERN_FIELD_NAME])) {
    return scan_identifier(lexer, valid_symbols);
  }
  return scan_number(lexer, valid_symbols);
}

}  // extern "C"
